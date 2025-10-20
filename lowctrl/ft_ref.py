import jax.numpy as jnp
import jax
import jax.scipy as jsp
from lowctrl import math as lmath
from lowctrl import model as lmodel
from lowctrl.qp_solve import schur_solve
from lowctrl import qp_solve
from flax import linen as nn

def make_centroidal_a(eefpos, com_pos, ids):
    m_ang = ids["angular_inertia"]
    anginv = jnp.linalg.inv(m_ang)
    mass = ids["mass"]
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

# Costs

def f_mag_q(w, ids):
    logits = -jnp.clip(w, -6.0, 6.0)
    big_qp = lmath.vec2diags(jnp.exp(logits), ids)
    tau_cost = lmath.torqueCost(40.0, ids)
    big_qp = tau_cost @ big_qp 
    return big_qp, jnp.zeros(6 * ids["eef_num"])

def f_ref_q(f_ref, ids):
    # f_ref is a ids["eef_num"] x 3 vector of desired forces
    big_q = jnp.zeros((6 * ids["eef_num"], 6 * ids["eef_num"]))
    small_q = jnp.zeros(6 * ids["eef_num"])
    for c in range(ids["eef_num"]):
        small_q = small_q.at[c * 6:c * 6 + 3].set(f_ref[c, :])
        big_q = big_q.at[c * 6:c * 6 + 3, 
                         c * 6:c * 6 + 3].set(jnp.eye(3))
    return big_q, small_q

def centroidal_acc_q(q_ddot_com_ref):
    weight_vec = jnp.array([1.0, 1.0, 1.0, 1e-1, 1e-1, 1e-1])
    big_c = jnp.eye(6) * weight_vec[None, :]
    big_q = big_c.T @ big_c
    small_q = big_c.T @ q_ddot_com_ref
    return big_q, small_q

def joint_torque_q(jacs, tau_ref):
    mat = -jacs[:, 6:].T
    big_q = mat.T @ mat
    small_q = mat.T @ tau_ref
    return big_q, small_q

# Constraints

def centroidal_qacc_cons(select, big_a, g, com_ref):
    # big_a @ F + g = com_ref
    lhs = big_a @ select["F"]
    rhs = com_ref - g
    return lhs, rhs
# QP Solver

def eval_qp(big_q, small_q, x):
    res = 0.5 * x.T @ big_q @ x - small_q.T @ x
    return res

def ft_ref(eefpos, com_pos, 
         jacs, tau_ref, com_ref, w, ids, debug,
         barrier = True):
    weights = jnp.array([1e-3, 1e-2])
    F_size = ids["eef_num"] * 6
    select = {
        "F": jnp.eye(F_size),
    }

    a, g = make_centroidal_a(
                          eefpos,
                          com_pos,
                          ids
                          )
    
    qp_q = jnp.zeros((F_size, F_size))
    qp_c = jnp.zeros((F_size, ))

    # Make Costs

    big_q_mag, small_c_mag = f_mag_q(w, ids)
    big_q_mag *= weights[0]
    small_c_mag *= weights[0]
    qp_q = big_q_mag
    qp_c = small_c_mag

    big_q_tau, small_c_tau = joint_torque_q(jacs, tau_ref)
    big_q_tau *= weights[1]
    small_c_tau *= weights[1]

    qp_q += big_q_tau
    qp_c += small_c_tau

    # Make Cons

    cons_lhs_list = []
    cons_rhs_list = []

    centroid_lhs, centroid_rhs = centroidal_qacc_cons(select, a, g, com_ref)

    cons_lhs_list.append(centroid_lhs)
    cons_rhs_list.append(centroid_rhs)

    cons_lhs = jnp.vstack(cons_lhs_list)
    cons_rhs = jnp.concatenate(cons_rhs_list, axis = 0)

    sol = schur_solve(qp_q, qp_c, cons_lhs, cons_rhs)

    f = sol

    tau = -jacs[:, 6:].T @ f

    debug_dict = {}
    if debug:
        mag_err = eval_qp(big_q_mag, small_c_mag, f)
        tau_err = eval_qp(big_q_tau, small_c_tau, f)
        debug_dict = {
            "errors": jnp.array([mag_err, tau_err])
        }
    return tau, f, debug_dict

def make_filt_state(ids):
    state = {
        "prev_u": jnp.zeros([ids["ctrl_num"]]),
    }
    return state

def ctrl2logits(act, ids):
    des_pos = act[0:ids["ctrl_num"]]
    des_com_vel = act[ids["ctrl_num"]:ids["ctrl_num"] + 3]
    des_com_angvel = act[ids["ctrl_num"] + 3 : ids["ctrl_num"] + 6]
    w = act[ids["ctrl_num"] + 6 : ids["ctrl_num"] + ids["eef_num"] + 6]
    torque = act[ids["ctrl_num"] + ids["eef_num"] + 6:
              ids["ctrl_num"] * 2 + ids["eef_num"] + 6]
    d_gain = act[-2:]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "torque": torque,
        "d_gain": d_gain
    }
    return logits

def ctrl2components(data, act, ids):
    # des_pos, des_com_pos, w
    logits = ctrl2logits(act, ids)
    des_pos = ids["default_qpos"][7:] + jnp.tanh(logits["des_pos"]) * 1.0
    #des_angvel = jnp.tanh(logits["des_com_angvel"]) * 1.0
    des_angvel = logits["des_com_angvel"] * 0.20
    des_angvel_mag = jnp.clip(jnp.linalg.norm(des_angvel), 0.0, 4.0)
    des_angvel = des_angvel * (des_angvel_mag / (1e-6 + jnp.linalg.norm(des_angvel)))
    #des_com_vel = jnp.tanh(logits["des_com_vel"]) * 0.7
    des_com_vel = logits["des_com_vel"] * 0.05
    des_com_vel_mag = jnp.clip(jnp.linalg.norm(des_com_vel), 0.0, 2.0)
    des_com_vel = des_com_vel * (des_com_vel_mag / (1e-6 + jnp.linalg.norm(des_com_vel)))
    w = logits["w"]

    torque_logit = jnp.tanh(logits["torque"])
    tau_limits = ids["tau_limits"]
    vel_limit = jnp.ones_like(tau_limits) * 10.0
    qvel = data.qvel[ids["joint_vel_ids"]][6:]
    tau_naive = tau_limits * torque_logit
    spd_fac = jnp.clip(jnp.abs(qvel), 0.0, vel_limit) / vel_limit
    sign = jnp.where(qvel * torque_logit >= 0, 1.0, 0.0)
    tau = tau_naive * (1.0 - spd_fac * sign)


    d_gain_lin = jnp.tanh(logits["d_gain"][0]) * 3.0 + 4.0
    #d_gain_lin = jnp.tanh(logits["d_gain"][0]) * 6.0 + 7.0
    d_gain_angvel = jnp.tanh(logits["d_gain"][1]) * 0.05 + 0.07

    outputs = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_angvel,
        "w": w,
        "torque": tau,
        "d_gain_lin": d_gain_lin,
        "d_gain_angvel": d_gain_angvel
    }
    return outputs

def highlvlPD(data, com_vel, des_com_vel, des_angvel, lin_gain, ang_gain, ids):
    qvel = data.qvel[ids["joint_vel_ids"]]
    #com_vel = qvel[0:3]

    world_com_vel = lmath.rotate_des_com_vel(des_com_vel, data)
    
    c_lin_p_gain = lin_gain
    com_acc = c_lin_p_gain * (world_com_vel - com_vel)
    
    c_ang_p_gain = ang_gain
    com_angacc = c_ang_p_gain * (des_angvel - qvel[3:6])

    com_accs = jnp.concatenate([com_acc, com_angacc], axis = 0)

    return com_accs, world_com_vel

def step(model, data, act, ids, is_mjx = False, 
         debug = False, filt_state = None):
    output = ctrl2components(data, act, ids)
    des_pos = output["des_pos"]
    des_com_vel = output["des_com_vel"]
    des_angvel = output["des_com_angvel"]
    w = output["w"]
    tau = output["torque"]
    d_lin_gain = output["d_gain_lin"]
    d_ang_gain = output["d_gain_angvel"]

    p_weight = ids["p_gains"]
    d_weight = ids["d_gains"]

    qpos = data.qpos[ids["joint_pos_ids"]][7:]
    qvel = data.qvel[ids["joint_vel_ids"]][6:]

    jacs, eefpos, com_pos, com_vel, h = lmodel.jac_only_kin_values(model, data, ids, is_mjx = is_mjx)
    #com_vel = data.qvel[ids["joint_vel_ids"]][0:3]
    
    com_accs, world_com_vel = highlvlPD(data, com_vel, des_com_vel, des_angvel,
                                        d_lin_gain, d_ang_gain, ids)
    
    com_accs = com_accs
    
    #s = jnp.where(nn.sigmoid(w) > 0.5, 1.0, 0.0)
    u_ff, f, norm_dict = ft_ref(
        eefpos, com_pos, jacs, tau, com_accs, w, ids, debug, barrier = True
    )

    nle_ff = h[6:]
    u_ff = u_ff + nle_ff

    # Lower pd gains based on ff torque

    torque_fac = jnp.clip(jnp.abs(u_ff) / (ids["tau_limits"] + 1e-6), 0.0, 1.0)

    #p_weight = p_weight * (1.0 - torque_fac * 0.5)

    pd_tau = p_weight * (des_pos - qpos)


    u_ff = jnp.nan_to_num(u_ff, posinf = 0.0, neginf = 0.0, nan = 0.0)
    u_ff = jnp.clip(u_ff, -ids["tau_limits"] * 1.0, ids["tau_limits"] * 1.0)
    
    u = u_ff + pd_tau

    #u_final = u * (pd_weight) + pd_tau * (1.0 - pd_weight)

    tau_limits = ids["tau_limits"]

    if filt_state is not None:
        alpha = 0.7
        u_filt = alpha * filt_state["prev_u"] + (1 - alpha) * u_ff
        filt_state["prev_u"] = u_filt
        u_final = jnp.clip(u, -tau_limits, tau_limits)
    else:
        u_final = jnp.clip(u, -tau_limits, tau_limits)
    if debug:
        debug_info = {
            "pd_tau": pd_tau,
            "u": u_ff,
            "u_final": u_final,
            "f": f,
            "com_ref": com_accs,
            "des_com_vel": world_com_vel,
            "des_angvel": des_angvel,
            "real_com_vel": com_vel,
            "real_angvel": data.qvel[3:6],
            "qp_errors": norm_dict["errors"],
            "des_pos": des_pos,
            "tau": tau,
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
    w = jnp.array([10., 10., -5., -5.])
    frc = jnp.zeros([ids["ctrl_num"]])
    ff_gains = jnp.zeros([2])
    act = jnp.concatenate([des_pos, des_com_vel, des_com_angvel, w, frc, ff_gains], axis = 0)
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
    #fac = jnp.clip(t / tmax, 0.0, 1.0)
    fac = 1.0
    des_pos = des_pos * (1.0 - fac) + target_pose * fac
    
    des_com_vel = jnp.zeros([3])
    des_com_angvel = jnp.zeros([3])
    #w = jnp.ones([ids["eef_num"]]) * 4.0
    w = jnp.array([10., -5., -5., -5.])
    #des_pos_pd = jnp.zeros([ids["ctrl_num"]])

    frc = jnp.zeros([ids["ctrl_num"]])

    ff_gains = jnp.zeros([2])
    act = jnp.concatenate([des_pos, des_com_vel, des_com_angvel, w, frc, ff_gains], axis = 0)
    
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