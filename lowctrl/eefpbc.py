import jax.numpy as jnp
from mujoco import mjx
import jax
import lowctrl.math as lmath
import lowctrl.pd as lpd
from lowctrl.qp_cons import qp_solve
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

W_SIZE = 3

def ctrl2logits(ctrl, ids):
    jnt_num = ids["ctrl_num"]
    eef_num = ids["eef_num"]
    des_pos_logit = ctrl[:jnt_num]
    qp_weight_logit = ctrl[jnt_num :jnt_num + W_SIZE]
    w = ctrl[jnt_num + W_SIZE: jnt_num + eef_num + W_SIZE]
    oriens = ctrl[jnt_num + eef_num + W_SIZE: jnt_num + eef_num * 4 + W_SIZE]
    
    return des_pos_logit, qp_weight_logit, w, oriens

def ctrl2components(ctrl, ids):
    (des_pos_logit, 
     qp_weight_logit, 
     w, oriens_logit) = ctrl2logits(ctrl, ids)
    des_pos = lpd.logit2limit(des_pos_logit, ids)
    #des_pos = des_pos_logit
    qp_weights = nn.sigmoid(qp_weight_logit)
    oriens_logit = jnp.reshape(oriens_logit, [ids["eef_num"], 3])
    eps = 1e-6
    oriens = oriens_logit / (jnp.linalg.norm(oriens_logit, axis=1, keepdims=True) + eps)
    return des_pos, qp_weights, w, oriens

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

    #eef_acc = base_acc + acc_float
    #eef_acc = eef_acc * 0.0
    return jvp
    #return acc_float


def pbc(qpos, m_uc, h_uc, des_pos, eef_acc, 
        jacs, jvp, cons_stack, w, ids):

    eef_num = ids["eef_num"]

    select_weights = nn.softmax(w)
    jacs_2 = jnp.reshape(jacs, (eef_num, 6, -1))
    jacs_2_weighted = jnp.sum(
        jacs_2 * select_weights[:, None, None], axis = 0)
    
    jvp2 = jnp.reshape(jvp, (eef_num, 6))
    jvp2_weighted = jnp.sum(jvp2 * select_weights[:, None],
                            axis = 0)
    
    joint_a_cons = jvp2_weighted - eef_acc

    ju2 = jacs_2_weighted[:, :6]
    jc2 = jacs_2_weighted[:, 6:]

    m_frc = jacs.T @ cons_stack[0]
    h_frc = jacs.T @ cons_stack[1]

    def make_dsub(ju, jc, m):
        m_uu = m[:6, :6]
        m_uc = m[:6, 6:]
        m_cu = m[6:, :6]
        m_cc = m[6:, 6:]
        d11 = jnp.block([
            [jnp.zeros([6, 6]), 
            ju],
            [ju.T, m_uu]
        ])
        d12 = jnp.block([
            [jc],
            [m_uc]
        ])

        d21 = jnp.block([
            [jc.T, m_cu]
        ])

        d22 = m_cc

        return d11, d12, d21, d22
    
    def make_hsub(joint_a_cons, h_uc):
        h1 = jnp.concatenate([joint_a_cons,
                              h_uc[:6]], axis = 0)
        h2 = h_uc[6:]
        return h1, h2
    
    d11, d12, d21, d22 = make_dsub(ju2, jc2, m_uc - m_frc)

    h1, h2 = make_hsub(joint_a_cons, h_uc - h_frc)
    #bf_sub = jnp.vstack([jnp.zeros([6, ctrl_num]), jnp.eye(23)])
    
    noselect_const = jnp.clip(jnp.sum(nn.sigmoid(w)), 0.0, 1.0)

    hbar = h2 - noselect_const * d21 @ jnp.linalg.solve(d11, h1)
    #lmbda = jnp.linalg.solve(d11, b)

    ec_ik = qpos[7:] - des_pos[0]

    u_b_ff_grv = jnp.nan_to_num(hbar, posinf = 0.0, neginf = 0.0, nan = 0.0)

    u_b_ff = u_b_ff_grv # + u_b_ff_acc * 1.0

    u_b_fb = ids["p_gains"] * ec_ik # + ids["d_gains"] * ec_ik_dot
    return u_b_ff, u_b_fb

def ff_only(qpos, des_pos, h_uc, 
        jacs, cons_stack, ids):
    
    h2 = h_uc[6:]

    jc = jacs[:, 6:]

    F = jc.T @ cons_stack[1]

    ec_ik = qpos[7:] - des_pos

    u_b_fb = ids["p_gains"] * ec_ik

    u_b_ff = jnp.nan_to_num(h2 - F, posinf = 0.0, neginf = 0.0, nan = 0.0)

    return u_b_ff, u_b_fb

def step(mjx_model, state, act, ids, override_pos = None):

    jacs = jac_stack(mjx_model, state, ids)
    jvp = get_djp(mjx_model, state, ids)
    m_uc, h_uc = get_mh(mjx_model, state, ids)

    (des_pos, 
     qp_weights, 
     w, oriens, 
     ) = ctrl2components(act, ids)
    
    if override_pos is not None:
        des_pos = override_pos
    
    qpos = state.qpos[ids["joint_pos_ids"]]

    qacc_gain = 400.0
    qc = qacc_gain * (des_pos - qpos[7:])
    
    qc0 = jnp.zeros_like(qc)
    eef_acc = jnp.zeros([6 * ids["eef_num"]])

    f, q_u = qp_solve(m_uc, h_uc, 
                qp_weights, oriens, w, 
                jacs, jvp, 
                eef_acc, qc0,
                ids)
    
    h_c = h_uc[6:]
    j_c = jacs[:, 6:]
    m_cu = m_uc[6:, :6]
    m_cc = m_uc[6:, 6:]
    u_b_ff = -j_c.T @ f + m_cu @ q_u + h_c
    u_b_ff *= qp_weights[2]
    u_b_ff = jnp.nan_to_num(u_b_ff, posinf = 0.0, neginf = 0.0, nan = 0.0)
    #u_b_fb = -m_cc @ qc
    ec_ik = qpos[7:] - des_pos

    u_b_fb = ids["p_gains"] * ec_ik

    u = u_b_ff - u_b_fb
    #u = -u_b_fb
    tau_limits = ids["tau_limits"]
    u = jnp.clip(u, -tau_limits, tau_limits)
    return u


def debug_step(mjx_model, state, act, ids):
    jacs = jac_stack(mjx_model, state, ids)
    jvp = get_djp(mjx_model, state, ids)
    m_uc, h_uc = get_mh(mjx_model, state, ids)

    (des_pos, 
     qp_weights, 
     w, oriens, 
     ) = ctrl2components(act, ids)
    
    qpos = state.qpos[ids["joint_pos_ids"]]

    qacc_gain = 400.0
    qc = qacc_gain * (des_pos - qpos[7:])
    
    qc0 = jnp.zeros_like(qc)
    eef_acc = jnp.zeros([6 * ids["eef_num"]])

    print("jvp", jvp)

    f, q_u = qp_solve(m_uc, h_uc, 
                qp_weights, oriens, w, 
                jacs, jvp, 
                eef_acc, qc0,
                ids)
    
    return f, q_u

def default_act(ids):
	pos = ids["default_qpos"][7:]
	qp_weights = jnp.ones([W_SIZE]) * 4
	w = jnp.array([10., 10., -5., -5.])
	target_orien = jnp.array([
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 0., 1.],
	   [0., 0., 1.]
    ]).flatten()
	act = jnp.concatenate([pos, 
                           qp_weights, 
                           w, 
                           target_orien])
	return act