import jax.numpy as jnp
import jax
import jax.scipy as jsp
from lowctrl import math as lmath
from lowctrl import model as lmodel
from flax import linen as nn

def make_centroidal_a(m, eefpos, com_pos, ids):
    m_lin = m[:3, :3]
    m_ang = m[3:6, 3:6]
    anginv = jnp.linalg.inv(m_ang)
    mass = jnp.trace(m_lin) / 3.0
    r = eefpos - com_pos[None, :]
    f_blocks = []
    for i in range(ids["eef_num"]):
        r_skew = lmath.skew(r[i, :])
        f_block = jnp.block([
            [jnp.eye(3) / mass, jnp.zeros([3, 3])],
            [anginv @ r_skew, anginv]
        ])
        f_blocks.append(f_block)
    a = jnp.concatenate(f_blocks, axis = 1)
    g = jnp.array([0, 0, -9.81, 0, 0, 0])
    return a, g
# Functions to build minimization objectives

def centroidal_acc_q(q_ddot_com_ref):
    weight_vec = jnp.array([1.0, 1.0, 1.0, 1e-1, 1e-1, 1e-1])
    big_c = jnp.eye(6) * weight_vec[None, :]
    big_q = big_c.T @ big_c
    small_q = big_c.T @ q_ddot_com_ref
    return big_q, small_q

def centroidal_cons_q(a, g):
    m_temp = jnp.concatenate([
        jnp.eye(6), -a
    ], axis = 1)
    big_q = m_temp.T @ m_temp
    small_q = m_temp.T @ g
    return big_q, small_q

def f_mag_q(w, ids):
    logits = -jnp.clip(w, -6.0, 6.0)
    big_qp = lmath.vec2diags(jnp.exp(logits), ids)
    big_qp += jnp.eye(6 * ids["eef_num"]) * 1
    tau_cost = lmath.torqueCost(20.0, ids)
    big_qp = tau_cost @ big_qp 
    return big_qp, jnp.zeros(6 * ids["eef_num"])

def q_ddot_c_q(ids):
    big_q = jnp.eye(ids["ctrl_num"])
    small_q = jnp.zeros(ids["ctrl_num"])
    return big_q, small_q

def qu_mag_q():
    return jnp.eye(6), jnp.zeros(6)

def eef_acc_q(j_i, jvp, a_i, qc):
    j_i_u = j_i[:, :6]
    j_i_c = j_i[:, 6:]

    big_q = j_i_u.T @ j_i_u
    small_q = j_i_u.T @ (a_i - jvp - j_i_c @ qc)
    return big_q, small_q

def eefs_acc_q(s, jacs, jvp, a_stc, qc, ids):
    big_q = jnp.zeros((6, 6))
    small_q = jnp.zeros((6))
    for c in range(ids["eef_num"]):
        j_i = jacs[c*6:(c+1)*6, :]
        jvp_i = jvp[c*6:(c+1)*6]
        a_i = a_stc[c*6:(c+1)*6]
        big_q_i, small_q_i = eef_acc_q(j_i, jvp_i, a_i, qc)
        big_q_i *= s[c]
        small_q_i *= s[c]
        big_q = big_q + big_q_i
        small_q = small_q + small_q_i
    return big_q, small_q

def u_pd_q(qc_weight, u_ref, ids):
    big_c = jnp.eye(ids["ctrl_num"]) * qc_weight[:, None]
    big_c = big_c + jnp.eye(ids["ctrl_num"]) * 1e-2
    big_q = big_c.T @ big_c
    small_q = big_c.T @ u_ref
    return big_q, small_q

# Functions to build the constraint matrices

# Provide a dict of selection matrixes:
# q_ddot_com, F, q_ddot_uc

def centroidal_qacc_cons(select, big_a):
    lhs = select["q_ddot_com"] - big_a @ select["F"]
    rhs = jnp.array([0, 0, -9.81, 0, 0, 0])
    return lhs, rhs

def centroidal_quc_cons(select, com_jvp, com_jac, qc):
    select_u = select["q_ddot_u"]
    com_jac_u = com_jac[:, :6]
    com_jac_c = com_jac[:, 6:]
    lhs = -com_jac_u @ select_u + select["q_ddot_com"][:3, :]
    rhs = com_jvp + com_jac_c @ qc
    return lhs, rhs

def zero_force_cons(select, s, ids):
    s_mat = lmath.vec2diags((1 - s), ids)
    lhs = s_mat @ select["F"]
    return lhs, jnp.zeros(ids["eef_num"] * 6)

def fullbody_u_cons(select, m, h, jacs, qc):
    m_u_uc = m[:6, :]
    m_u_u = m_u_uc[:, :6]
    m_u_c = m_u_uc[:, 6:]
    h_u = h[:6]
    ju = jacs[:, :6]
    
    lhs = ju.T @ select["F"] - m_u_u @ select["q_ddot_u"]
    rhs = h_u + m_u_c @ qc
    return lhs, rhs


def schur_solve(qp_q, qp_c, cons_lhs, cons_rhs):
    Q = 0.5 * (qp_q + qp_q.T)
    A = cons_lhs
    c = qp_c
    b = cons_rhs
    Z = jnp.zeros((A.shape[0], A.shape[0]), dtype=jnp.float32)
    KKT = jnp.block([[Q, A.T],
                     [A, Z]])
    rhs = jnp.concatenate([c, b], axis=0)
    sol_all = jnp.linalg.solve(KKT, rhs)
    sol = sol_all[:Q.shape[0]]
    return sol

def maqp(m, h, w, a_stc,
         eefpos, com_pos, 
         jacs, jvp, 
         com_jac, com_jvp, 
         com_ref, qc, ids, is_mjx = True, debug = False):
    
    s = nn.sigmoid(w)
    
    # q_ddot_com, F, q_ddot_uc, u_b
    # cent acc, cent com, f mag, q_ddot_c, qu_mag, eef_acc, u_pd
    weights = jnp.array([1e1, 1e1, 1e-5, 5e-3, 1e-3])
    mat_height = 6 + 6 + 6 * ids["eef_num"]
    u_size = 6
    F_size = ids["eef_num"] * 6
    select = {
        "q_ddot_com": jnp.block([jnp.eye(6), jnp.zeros((6, mat_height - 6))]),
        "F": jnp.block([jnp.zeros([F_size, 6]), jnp.eye(F_size), jnp.zeros((F_size, mat_height - F_size - 6))]),
        "q_ddot_u": jnp.block([jnp.zeros([u_size, mat_height - u_size]), 
                                jnp.eye(u_size)]),
    }

    # Prebuild 0

    a, g = make_centroidal_a(m,
                          eefpos,
                          com_pos,
                          ids
                          )

    # Make qp optimization objective

    qp_q = jnp.zeros((mat_height, mat_height))
    qp_c = jnp.zeros((mat_height, ))

    big_q_com, small_q_com = centroidal_acc_q(com_ref)

    big_q_com *= weights[0]
    small_q_com *= weights[0]

    qp_q = qp_q.at[:6, :6].add(big_q_com)
    qp_c = qp_c.at[:6].add(small_q_com)

    big_q_cent, small_q_cent = centroidal_cons_q(a, g)
    big_q_cent *= weights[1]
    small_q_cent *= weights[1]

    qp_q = qp_q.at[:(6 + F_size), :(6 + F_size)].add(big_q_cent)
    qp_c = qp_c.at[:(6 + F_size)].add(small_q_cent)

    big_q_f, small_q_f = f_mag_q(w, ids)

    big_q_f *= weights[2]
    small_q_f *= weights[2]

    qp_q = qp_q.at[6: 6 + F_size, 6: 6 + F_size].add(big_q_f)
    qp_c = qp_c.at[6: 6 + F_size].add(small_q_f)

    big_q_qu, small_q_qu = qu_mag_q()
    big_q_qu *= weights[4]
    small_q_qu *= weights[4]

    qp_q = qp_q.at[6 + F_size: 12 + F_size, 6 + F_size: 12 + F_size].add(big_q_qu)
    qp_c = qp_c.at[6 + F_size: 12 + F_size].add(small_q_qu)

    big_q_acc, small_q_acc = eefs_acc_q(s, jacs, jvp, a_stc, qc, ids)
    big_q_acc *= weights[5]
    small_q_acc *= weights[5]

    qp_q = qp_q.at[6 + F_size:mat_height, 
                   6 + F_size:mat_height].add(big_q_acc)
    qp_c = qp_c.at[6 + F_size:mat_height].add(small_q_acc)


    # Make qp constraints 

    cons_lhs_list = []
    cons_rhs_list = []

    centroid_lhs2, centroid_rhs2 = centroidal_quc_cons(select, com_jvp, com_jac, qc)
    cons_lhs_list.append(centroid_lhs2)
    cons_rhs_list.append(centroid_rhs2)

    fullbody_lhs, fullbody_rhs = fullbody_u_cons(select, m, h, jacs, qc)
    cons_lhs_list.append(fullbody_lhs)
    cons_rhs_list.append(fullbody_rhs)

    cons_lhs = jnp.vstack(cons_lhs_list)
    cons_rhs = jnp.concatenate(cons_rhs_list, axis = 0)

    sol = schur_solve(qp_q, qp_c, cons_lhs, cons_rhs)

    q_ddot_com = sol[:6]
    f = sol[6:6 + ids["eef_num"] * 6]
    q_ddot_u = sol[6 + ids["eef_num"] * 6:12 + ids["eef_num"] * 6]

    m_c_uc = m[6:, :]
    m_c_u = m_c_uc[:, :6]
    m_c_uc = m_c_uc[:, 6:]
    h_c = h[6:]
    jc = jacs[:, 6:]
    u = m_c_u @ q_ddot_u + m_c_uc @ qc - jc.T @ f + h_c

    debug_dict = {}

    if debug:
        big_q_com, small_q_com = centroidal_acc_q(com_ref)
        centroidal_error = eval_qp(big_q_com, small_q_com, q_ddot_com)
        big_q_com, small_q_com = centroidal_cons_q(a, g)
        centroidal_cons_error = eval_qp(big_q_com, small_q_com, 
                                        jnp.concatenate([q_ddot_com, f], axis = 0))
        big_q_f, small_q_f = f_mag_q(w, ids)
        f_mag_error = eval_qp(big_q_f, small_q_f, f)
        big_q_qu, small_q_qu = qu_mag_q()
        qu_mag_error = eval_qp(big_q_qu, small_q_qu, q_ddot_u)
        big_q_acc, small_q_acc = eefs_acc_q(s, jacs, jvp, a_stc, qc, ids)
        eef_accs_error = eval_qp(big_q_acc, small_q_acc, 
                                q_ddot_u)
        errors = jnp.array([centroidal_error, centroidal_cons_error, f_mag_error, 
                            qu_mag_error, eef_accs_error])
        debug_dict["errors"] = errors * weights
    return u, f, q_ddot_com, debug_dict

def eval_qp(big_q, small_q, x):
    res = 0.5 * x.T @ big_q @ x - small_q.T @ x
    return res

def ctrl2logits(act, ids):
    des_pos = act[0:ids["ctrl_num"]]
    des_com_vel = act[ids["ctrl_num"]:ids["ctrl_num"] + 3]
    des_com_angvel = act[ids["ctrl_num"] + 3 : ids["ctrl_num"] + 6]
    w = act[ids["ctrl_num"] + 6 : ids["ctrl_num"] + ids["eef_num"] + 6]
    des_pos_pd = act[ids["ctrl_num"] + ids["eef_num"] + 6 : ids["ctrl_num"] * 2 + ids["eef_num"] + 6]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "des_pos_pd": des_pos_pd,
    }
    return logits

def ctrl2components(act, ids):
    # des_pos, des_com_pos, w
    logits = ctrl2logits(act, ids)
    des_pos = ids["default_qpos"][7:] + jnp.tanh(logits["des_pos"]) * 1.0
    #des_angvel = jnp.tanh(logits["des_com_angvel"]) * 1.0
    des_angvel = logits["des_com_angvel"] * 0.20
    des_angvel_mag = jnp.clip(jnp.linalg.norm(des_angvel), 0.0, 3.0)
    des_angvel = des_angvel * (des_angvel_mag / (1e-6 + jnp.linalg.norm(des_angvel)))
    #des_com_vel = jnp.tanh(logits["des_com_vel"]) * 0.7
    des_com_vel = logits["des_com_vel"] * 0.05
    des_com_vel_mag = jnp.clip(jnp.linalg.norm(des_com_vel), 0.0, 0.7)
    des_com_vel = des_com_vel * (des_com_vel_mag / (1e-6 + jnp.linalg.norm(des_com_vel)))
    des_pos_pd = ids["default_qpos"][7:] + jnp.tanh(logits["des_pos_pd"]) * 1.0
    w = logits["w"]
    outputs = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_angvel,
        "w": w,
        "des_pos_pd": des_pos_pd,
    }
    return outputs

def highlvlPD(data, des_pos, des_com_vel, des_angvel, ids):
    qpos = data.qpos[ids["joint_pos_ids"]]
    qvel = data.qvel[ids["joint_vel_ids"]]

    jp_gain = 200.0
    jd_gain = 10.0

    world_com_vel = lmath.rotate_des_com_vel(des_com_vel, data)

    qacc = jp_gain * (des_pos - qpos[7:]) - jd_gain * qvel[6:]

    c_lin_p_gain = 5.0
    com_acc = c_lin_p_gain * (world_com_vel - qvel[0:3])
    
    c_ang_p_gain = 4.0
    com_angacc = c_ang_p_gain * (des_angvel - qvel[3:6])

    com_accs = jnp.concatenate([com_acc, com_angacc], axis = 0)

    return qacc, com_accs, world_com_vel

def make_filt_state(ids):
    state = {
        "prev_u": jnp.zeros([ids["ctrl_num"]]),
    }
    return state

def step(model, data, act, ids, is_mjx = False, 
         debug = False, filt_state = None):
    output = ctrl2components(act, ids)
    des_pos = output["des_pos"]
    des_pos_pd = output["des_pos_pd"]
    des_com_vel = output["des_com_vel"]
    des_angvel = output["des_com_angvel"]
    w = output["w"]
    
    p_weight = ids["p_gains"]
    d_weight = ids["d_gains"]

    qpos = data.qpos[ids["joint_pos_ids"]][7:]
    qvel = data.qvel[ids["joint_vel_ids"]][6:]

    pd_tau = p_weight * (des_pos_pd - qpos) + d_weight * (0.0 - qvel)
    
    qacc_c, com_accs, world_com_vel = highlvlPD(data, des_pos, des_com_vel, des_angvel, ids)
    m, h, jacs, jvp, jac_com, com_jvp, eefpos, com_pos = lmodel.get_kin_values(
        model, data, ids, is_mjx = is_mjx)
    a_stc = jnp.zeros(6 * ids["eef_num"])
    #s = jnp.where(nn.sigmoid(w) > 0.5, 1.0, 0.0)
    u, f, q_ddot_com, norm_dict = maqp(m, h, w, a_stc,
                eefpos, com_pos,
                jacs, jvp,
                jac_com, com_jvp,
                com_accs,
                qacc_c, ids, is_mjx = is_mjx, debug = debug)
    u = jnp.nan_to_num(u, posinf = 0.0, neginf = 0.0, nan = 0.0)
    u = u * 0.1 + pd_tau

    #u_final = u * (pd_weight) + pd_tau * (1.0 - pd_weight)

    tau_limits = ids["tau_limits"]

    if filt_state is not None:
        alpha = 0.95
        u_filt = alpha * filt_state["prev_u"] + (1 - alpha) * u
        filt_state["prev_u"] = u_filt
        u_final = jnp.clip(u_filt, -tau_limits, tau_limits)
    else:
        u_final = jnp.clip(u, -tau_limits, tau_limits)
    if debug:
        debug_info = {
            "pd_tau": pd_tau,
            "u": u,
            "u_final": u_final,
            "f": f,
            "q_ddot_com": q_ddot_com,
            "com_ref": com_accs,
            "des_com_vel": world_com_vel,
            "des_angvel": des_angvel,
            "real_com_vel": data.qvel[0:3],
            "real_angvel": data.qvel[3:6],
            "qp_errors": norm_dict["errors"],
            "des_pos": des_pos,
        }
        if filt_state is not None:
            debug_info["u_filt"] = u_filt
        return debug_info
    else:
        if filt_state is not None:
            return u_final, filt_state
        return u_final

def default_act(ids):
    des_pos = jnp.zeros([ids["ctrl_num"]])
    des_com_vel = jnp.zeros([3])
    des_com_angvel = jnp.zeros([3])
    #w = jnp.ones([ids["eef_num"]]) * 4.0
    w = jnp.array([10., 10., -5., -5.])
    des_pos_pd = jnp.zeros([ids["ctrl_num"]])
    act = jnp.concatenate([des_pos, des_com_vel, des_com_angvel, w, des_pos_pd], axis = 0)
    return act

def default_act_lock_com(com_pos, data, ids):
    act = default_act(ids)
    # Set controller to lock com position
    current_com = data.subtree_com[0]
    vel_mag = 0.5
    point_vec = com_pos - current_com
    point_vec = vel_mag * point_vec / jnp.linalg.norm(point_vec + 1e-6)
    des_com_vel = point_vec
    act = act.at[ids["ctrl_num"]:ids["ctrl_num"] + 3].set(des_com_vel)

    # Set controller to lock base orientation

    A = data.qpos[3:7]

    B = ids["default_qpos"][3:7]

    ang_disp = lmath.angular_displacement_from_A_to_B(A, B)

    ang_vel = ang_disp * 1.0

    ang_vel_norm = jnp.clip(jnp.linalg.norm(ang_vel), min = 0.0, max = 3.0)

    ang_vel = ang_vel * ang_vel_norm / (jnp.linalg.norm(ang_vel)  + 1e-6)


    act = act.at[ids["ctrl_num"] + 3: ids["ctrl_num"] + 6].set(ang_vel)

    return act


def test_act_move_com(com_pos, data, t, ids):
    act = default_act(ids)

    # Set desired joint pos
    pos = default_act(ids)[:ids["ctrl_num"]]
    pos = pos.at[2].set(jnp.sin(t) * 0.4)
    pos = pos.at[6].set(jnp.sin(-t) * 0.4)

    act = act.at[0:ids["ctrl_num"]].set(pos)


    # Set controller to lock com position
    current_com = data.subtree_com[0]

    
    delta = jnp.array([0.0, t, 0.0])
    delta = jnp.sin(delta) * 0.03

    com_pos += delta

    point_vec = com_pos - current_com

    vel_ = point_vec * 200.0
    vel_mag_norm = jnp.clip(jnp.linalg.norm(vel_), min = 0.0, max = 3.0)
    vel_ = vel_ * vel_mag_norm / (jnp.linalg.norm(vel_) + 1e-6)

    des_com_vel = vel_
    act = act.at[ids["ctrl_num"]:ids["ctrl_num"] + 3].set(des_com_vel)

    # Set controller to lock base orientation

    A = data.qpos[3:7]

    B = ids["default_qpos"][3:7]

    ang_disp = lmath.angular_displacement_from_A_to_B(A, B)

    ang_vel = ang_disp * 9.0

    ang_vel_norm = jnp.clip(jnp.linalg.norm(ang_vel), min = 0.0, max = 3.0)

    ang_vel = ang_vel * ang_vel_norm / (jnp.linalg.norm(ang_vel)  + 1e-6)


    act = act.at[ids["ctrl_num"] + 3: ids["ctrl_num"] + 6].set(ang_vel)

    return act


def shift_com_pos(com_pos, data, t, tmax, ids, delta = jnp.array([0.0, 0.04, 0.0])):
    act = default_act(ids)
    current_com = data.subtree_com[0]

    
    com_pos += delta * jnp.clip(t / tmax, 0.0, 1.0)

    point_vec = com_pos - current_com

    vel_ = point_vec * 200.0
    vel_mag_norm = jnp.clip(jnp.linalg.norm(vel_), min = 0.0, max = 3.0)
    vel_ = vel_ * vel_mag_norm / (jnp.linalg.norm(vel_) + 1e-6)

    des_com_vel = vel_
    act = act.at[ids["ctrl_num"]:ids["ctrl_num"] + 3].set(des_com_vel)

    # Set controller to lock base orientation

    A = data.qpos[3:7]

    B = ids["default_qpos"][3:7]

    ang_disp = lmath.angular_displacement_from_A_to_B(A, B)

    ang_vel = ang_disp * 9.0

    ang_vel_norm = jnp.clip(jnp.linalg.norm(ang_vel), min = 0.0, max = 3.0)

    ang_vel = ang_vel * ang_vel_norm / (jnp.linalg.norm(ang_vel)  + 1e-6)


    act = act.at[ids["ctrl_num"] + 3: ids["ctrl_num"] + 6].set(ang_vel)
    return act

def raise_right_leg(com_pos, data, t, tmax, ids):
    target_pose = jnp.zeros([ids["ctrl_num"]])
    target_pose = target_pose.at[17].set(-1.0)
    target_pose = target_pose.at[20].set(2.0)
    
    des_pos = jnp.zeros([ids["ctrl_num"]])
    # -0.7 on right hip, +1 on right knee
    fac = jnp.clip(t / tmax, 0.0, 1.0)
    des_pos = des_pos * (1.0 - fac) + target_pose * fac
    
    des_com_vel = jnp.zeros([3])
    des_com_angvel = jnp.zeros([3])
    #w = jnp.ones([ids["eef_num"]]) * 4.0
    w = jnp.array([10., -5., -5., -5.])
    des_pos_pd = jnp.zeros([ids["ctrl_num"]])
    act = jnp.concatenate([des_pos, des_com_vel, des_com_angvel, w, des_pos_pd], axis = 0)
    
    com_pos = com_pos + jnp.array([0.0, 0.05, 0.0])
    current_com = data.subtree_com[0]

    # Set qc weights to 1 for right leg joints
    point_vec = com_pos - current_com

    vel_ = point_vec * 200.0
    vel_mag_norm = jnp.clip(jnp.linalg.norm(vel_), min = 0.0, max = 3.0)
    vel_ = vel_ * vel_mag_norm / (jnp.linalg.norm(vel_) + 1e-6)

    des_com_vel = vel_
    act = act.at[ids["ctrl_num"]:ids["ctrl_num"] + 3].set(des_com_vel)

    # Set controller to lock base orientation

    A = data.qpos[3:7]

    B = ids["default_qpos"][3:7]

    ang_disp = lmath.angular_displacement_from_A_to_B(A, B)

    ang_vel = ang_disp * 9.0

    ang_vel_norm = jnp.clip(jnp.linalg.norm(ang_vel), min = 0.0, max = 3.0)

    ang_vel = ang_vel * ang_vel_norm / (jnp.linalg.norm(ang_vel)  + 1e-6)

    act = act.at[ids["ctrl_num"] + 3: ids["ctrl_num"] + 6].set(ang_vel)
    return act

def ds2ss(com_pos, data, t, ids):
    act1 = shift_com_pos(com_pos, data, t, 2, ids)
    act2 = raise_right_leg(com_pos, data, t - 2, 0.5, ids)
    act = jnp.where(t < 2.0, act1, act2)
    return act