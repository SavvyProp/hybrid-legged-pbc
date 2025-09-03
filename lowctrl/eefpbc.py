import jax.numpy as jnp
from mujoco import mjx
import jax
import lowctrl.math as lmath
import lowctrl.pd as lpd
from lowctrl.qp_cons import qp_cons
from flax import linen as nn

def get_jac(mjx_model, mjx_data, site_name, ids):
    """
    Gets the jacobian of position and rotation for a site
    """
    site_id = ids["eef"][site_name]["site_id"]
    body_id = ids["eef"][site_name]["body_id"]
    point = mjx_data.site_xpos[site_id]
    jacp, jacr = mjx.jac(mjx_model, mjx_data, point, body_id)
    j = jnp.vstack((jacp.T, jacr.T))
    return j

def jac_stack(mjx_model, mjx_data, ids):
    """
    Stacks the jacobians of all end effectors vertically
    """
    jacs = []
    for eef_name in ids["eef"].keys():
        j = get_jac(mjx_model, mjx_data, eef_name, ids)
        jacs.append(j)
    return jnp.vstack(jacs)

def jdot(mjx_model, mjx_data, jac_func, ids):
    """
    Gets the directional derivative of the jacobian in the direction of qvel
    The reby getting j dot
    """
    qpos = mjx_data.qpos
    qdot_qpos = lmath.qposdot_from_qvel(mjx_model, 
                                   mjx_data.qpos, 
                                   mjx_data.qvel)
    
    @jax.jit
    def _f(qpos_61):
        d = mjx_data.replace(qpos = qpos_61,
                                             qvel = jnp.zeros_like(mjx_data.qvel))
        #d = mjx.forward(mjx_model, d)
        d = mjx.kinematics(mjx_model, d)
        #d = mjx.com_pos(mjx_model, d)
        jac = jac_func(mjx_model, d, ids)
        return jac
    
    _, jdot = jax.jvp(_f, (qpos,), (qdot_qpos, ))

    return jdot


def get_djp_old(mjx_model, mjx_data, ids):
    djac_fik = jdot(mjx_model, mjx_data, jac_stack, ids)
    v = mjx_data.qvel[ids["joint_vel_ids"]]
    return djac_fik @ v 

def get_djp(mjx_model, mjx_data, ids):
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel
    qdot_qpos = lmath.qposdot_from_qvel(mjx_model, qpos, qvel)

    orienlist = []
    for eef_name in ids["eef"].keys():
        orienmat = mjx_data.site_xmat[ids["eef"][eef_name]["site_id"]]
        orienlist.append(orienmat)

    def get_vel(qpos1):
        def get_fik(qpos2):
            d = mjx_data.replace(qpos=qpos2, 
                                        qvel = jnp.zeros_like(mjx_data.qvel))
            d = mjx.kinematics(mjx_model, d)
            poslist = []
            orienlist = []
            for eef_name in ids["eef"].keys():
                site_id = ids["eef"][eef_name]["site_id"]
                point = d.site_xpos[site_id]
                orien = d.site_xmat[site_id]
                poslist.append(point)
                orienlist.append(orien)
            return poslist, orienlist
        _, (poslist, angmatlist) = jax.jvp(get_fik, (qpos1, ), (qdot_qpos, ))

        def rotmat2angvel(R, dR):
            K = R.T @ dR
            K = 0.5 * (K - K.T)
            omega = jnp.array([
                K[2, 1], K[0, 2], K[1, 0]
            ])
            return omega
        angvellist = []
        for c in range(ids["eef_num"]):
            angvel = rotmat2angvel(orienlist[c], angmatlist[c])
            angvellist.append(angvel)
        
        return poslist, angvellist
    
    _, (acclist, angacclist) = jax.jvp(get_vel, (qpos, ), (qdot_qpos, ))

    combined_list = []
    for c in range(ids["eef_num"]):
        combined_list += [acclist[c].flatten(), angacclist[c].flatten()]

    return jnp.concatenate(combined_list, axis = 0)

def get_mh(mjx_model, mjx_data, ids):
    """
    Gets the mass matrix and bias force for the 36 digit model
    """
    m = mjx.full_m(mjx_model, mjx_data)
    vel_ids = ids["joint_vel_ids"]
    m_36 = m[jnp.ix_(vel_ids, vel_ids)]
    h_36 = mjx_data.qfrc_bias[vel_ids]
    return m_36, h_36

def logit2acc(logit):
    max_acc = 1000.0
    return jnp.tanh(logit) * max_acc

def logit2gndacc(logit, ids):
    max_linacc = 10.0
    eef_num = ids["eef_num"]
    linacc = jnp.tanh(logit) * max_linacc
    linacc = jnp.reshape(linacc, (eef_num, 6))
    return linacc

def ctrl2logits(ctrl, ids):
    jnt_num = ids["ctrl_num"]
    eef_num = ids["eef_num"]
    des_pos_logit = ctrl[:jnt_num]
    gnd_acc_logit = ctrl[jnt_num:jnt_num + eef_num * 6]
    qp_weight_logit = ctrl[jnt_num + eef_num * 6:jnt_num + eef_num * 6 + 2]
    tau_mix_logit = ctrl[jnt_num + eef_num * 6 + 2: 2 * jnt_num + eef_num * 6 + 2]
    w = ctrl[2 * jnt_num + eef_num * 6 + 2: 2 * jnt_num + eef_num * 7 + 2]
    oriens = ctrl[2 * jnt_num + eef_num * 7 + 2: 2 * jnt_num + eef_num * 10 + 2]
    base_acc = ctrl[2 * jnt_num + eef_num * 10 + 2: 2 * jnt_num + eef_num * 10 + 8]
    select = ctrl[-1]
    return des_pos_logit, gnd_acc_logit, qp_weight_logit, tau_mix_logit, w, oriens, base_acc, select

def ctrl2components(ctrl, ids):
    (des_pos_logit, 
     gnd_acc_logit, qp_weight_logit, tau_mix_logit, 
     w, oriens_logit, base_acc, select) = ctrl2logits(ctrl, ids)
    des_pos = lpd.logit2limit(des_pos_logit, ids)
    #des_pos = des_pos_logit
    gnd_acc = logit2gndacc(gnd_acc_logit, ids)
    base_acc = jnp.tanh(base_acc) * 10.0
    qp_weights = nn.sigmoid(qp_weight_logit)
    tau_mix = nn.sigmoid(tau_mix_logit)
    oriens_logit = jnp.reshape(oriens_logit, [ids["eef_num"], 3])
    eps = 1e-6
    oriens = oriens_logit / (jnp.linalg.norm(oriens_logit, axis=1, keepdims=True) + eps)
    return des_pos, gnd_acc, qp_weights, tau_mix, w, oriens, base_acc, nn.sigmoid(select)

def get_joint_traj(qpos, des_pos, T):
    qpos_c = qpos[7:]

    des_vel = (des_pos - qpos_c) / T

    return des_pos, des_vel

def get_eef_acc(jvp, 
                w, ground_acc, 
                base_acc, select, ids):
    eef_num = ids["eef_num"]
    #acc_float = j_c @ des_acc + jvp
    acc_float = jvp
    select_gnd = nn.softmax(w)
    select_acc = jnp.sum((ground_acc - acc_float.reshape(eef_num, -1)) * select_gnd[:, None], axis = 0)
    
    base_acc = select_acc * select + base_acc * (1 - select)
    base_acc = jnp.tile(base_acc, [eef_num])

    eef_acc = base_acc + acc_float
    #eef_acc = eef_acc * 0.0
    return eef_acc

def pbc_pinv(qpos, qvel, m_uc, h_uc, joint_traj, eef_acc, 
        jacs, jvp, cons_stack, ids):
    ju = jacs[:, :6]
    jc = jacs[:, 6:]

    fac = 0.1

    eef_num = ids["eef_num"]
    ctrl_num = ids["ctrl_num"]

    def make_dsub(cons_m, ju, jc, m):
        m_uu = m[:6, :6]
        m_uc = m[:6, 6:]
        m_cu = m[6:, :6]
        m_cc = m[6:, 6:]
        d11 = jnp.block([
            [cons_m[:, :eef_num * 6], cons_m[:, eef_num * 6: eef_num * 6 + 6]]
            [jnp.zeros([eef_num * 6, eef_num * 6]), ju],
            [ju.T, m_uu]
        ])


        d12 = jnp.block([
            [cons_m[:, eef_num * 6 + 6:]],
            [jc],
            [m_uc]
        ])

        d21 = jnp.block([
            [jc.T, m_cu]
        ])

        d22 = m_cc

        return d11, d12, d21, d22
    
    def make_hsub(cons_h, joint_a_cons, h_uc):
        h1 = jnp.concatenate([
            joint_a_cons + cons_h * fac, h_uc[:6]
        ], axis = 0)
        h2 = h_uc[6:]
        return h1, h2
    
    joint_a_cons = jvp - eef_acc
    
    d11, d12, d21, d22 = make_dsub(cons_stack[0], ju, jc, m_uc)

    h1, h2 = make_hsub(cons_stack[1], joint_a_cons, h_uc)

    #bf_sub = jnp.vstack([jnp.zeros([6, ctrl_num]), jnp.eye(23)])

    hbar = h2 - d21 @ jnp.linalg.solve(d11, h1)
    b = -h1

    lmbda = jnp.linalg.solve(d11, b)

    ec_ik = qpos[7:] - joint_traj[0]
    ec_ik_dot = qvel[6:] - joint_traj[1]

    u_b_ff_grv = hbar

    u_b_ff = u_b_ff_grv # + u_b_ff_acc * 1.0

    u_b_fb = ids["p_gains"] * ec_ik # + ids["d_gains"] * ec_ik_dot
    return u_b_ff, u_b_fb, lmbda


def pbc(qpos, qvel, m_uc, h_uc, joint_traj, eef_acc, 
        jacs, jvp, cons_stack, ids):
    ju = jacs[:, :6]
    jc = jacs[:, 6:]

    fac = 0.1

    eef_num = ids["eef_num"]
    ctrl_num = ids["ctrl_num"]

    def make_dsub(cons_m, ju, jc, m):
        m_uu = m[:6, :6]
        m_uc = m[:6, 6:]
        m_cu = m[6:, :6]
        m_cc = m[6:, 6:]
        d11 = jnp.block([
            [cons_m[:, :eef_num * 6] * fac, 
             ju + cons_m[:, eef_num * 6: eef_num * 6 + 6] * fac],
            [ju.T, m_uu]
        ])


        d12 = jnp.block([
            [jc + cons_m[:, eef_num * 6 + 6:] * fac],
            [m_uc]
        ])

        d21 = jnp.block([
            [jc.T, m_cu]
        ])

        d22 = m_cc

        return d11, d12, d21, d22
    
    def make_hsub(cons_h, joint_a_cons, h_uc):
        h1 = jnp.concatenate([
            joint_a_cons + cons_h * fac, h_uc[:6]
        ], axis = 0)
        h2 = h_uc[6:]
        return h1, h2
    
    joint_a_cons = jvp - eef_acc
    
    d11, d12, d21, d22 = make_dsub(cons_stack[0], ju, jc, m_uc)

    h1, h2 = make_hsub(cons_stack[1], joint_a_cons, h_uc)

    #bf_sub = jnp.vstack([jnp.zeros([6, ctrl_num]), jnp.eye(23)])

    hbar = h2 - d21 @ jnp.linalg.solve(d11, h1)
    b = -h1

    lmbda = jnp.linalg.solve(d11, b)

    ec_ik = qpos[7:] - joint_traj[0]
    ec_ik_dot = qvel[6:] - joint_traj[1]

    u_b_ff_grv = hbar

    u_b_ff = u_b_ff_grv # + u_b_ff_acc * 1.0

    u_b_fb = ids["p_gains"] * ec_ik # + ids["d_gains"] * ec_ik_dot
    return u_b_ff, u_b_fb, lmbda

def step(mjx_model, state, act, ids):

    jacs = jac_stack(mjx_model, state, ids)
    m_uc, h_uc = get_mh(mjx_model, state, ids)

    (des_pos, gnd_acc, 
     qp_weights, tau_mix, 
     w, oriens, 
     base_acc, select) = ctrl2components(act, ids)
    
    s = nn.sigmoid(w)
    
    qpos = state.qpos[ids["joint_pos_ids"]]
    qvel = state.qvel[ids["joint_vel_ids"]]

    jvp = get_djp(mjx_model, state, ids)

    joint_traj = get_joint_traj(qpos, des_pos, 0.1)

    cons_stack = qp_cons(m_uc[:6, :], h_uc[:6], qp_weights,
                         oriens, s, w, jacs[:, :6], ids)
    
    eef_acc = get_eef_acc(jvp, 
                          w, 
                          gnd_acc, base_acc, select, ids)
    
    u_b_ff, u_b_fb, lmbda = pbc(qpos, qvel, m_uc, h_uc, joint_traj, eef_acc, 
                               jacs, jvp, cons_stack, ids)
    
    u = u_b_ff * tau_mix - u_b_fb
    #u = u_b_ff
    #u = -u_b_fb

    #f_stc = lmbda[: ids["eef_num"] * 6]

    #state = state.replace(f_stc = f_stc)
    tau_limits = ids["tau_limits"]
    u = jnp.clip(u, -tau_limits, tau_limits)

    return u



def default_act(ids):
	pos = ids["default_qpos"][7:]
	gnd_acc = jnp.zeros((ids["eef_num"], 6))
	qp_weights = jnp.array([4, 4])
	tau_mix = jnp.ones([ids["ctrl_num"]]) * 5.0
	w = jnp.array([10., 10., -5., -5.])
	target_orien = jnp.array([
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 0., 1.],
	   [0., 0., 1.]
    ]).flatten()
	base_acc = jnp.zeros([6,])
	select = jnp.array([-10.0])
	act = jnp.concatenate([pos, 
                           jnp.reshape(gnd_acc, (-1,)), 
                           qp_weights, 
                           tau_mix, 
                           w, 
                           target_orien, 
                           base_acc,
                           select])
	return act