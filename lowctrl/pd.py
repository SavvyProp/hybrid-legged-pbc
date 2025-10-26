import jax.numpy as jnp
import jax

def logit2limit(logit, ids, traj_center = None):
    joint20_limits = ids["jnt_limits"]
    #center = jnp.mean(joint20_limits, axis=1)
    center = ids["default_qpos"][7:]
    scaling = jnp.array([
        1.5,
        1.0,
        2.0, 1.8, 2.4, 2.4,
        2.0, 1.8, 2.4, 2.4,
        1.4, 
        1.5, 1.4, 1.0, 2.0, 1.0, 1.0,
        1.5, 1.4, 1.0, 2.0, 1.0, 1.0
    ])
    #des_pos = center + scaling * jnp.tanh(logit)
    #return des_pos
    if traj_center is not None:
        #des_pos = traj_center + 1.0 * logit#1.0 * jnp.tanh(logit)
        des_pos = center + scaling * logit
    else:
        des_pos = center + scaling * logit
        #des_pos = center + 1.0 * logit#1.0 * jnp.tanh(logit)
    return des_pos

def step(mjx_model, state, act, ids, traj_center = None):
    nn_p_logit = act[:ids["ctrl_num"]]
    des_pos = logit2limit(nn_p_logit, ids, traj_center = traj_center)

    kp = jnp.array(ids["p_gains"])
    kd = jnp.array(ids["d_gains"])

    qc = state.qpos[ids["joint_pos_ids"]][7:]
    qd = state.qvel[ids["joint_vel_ids"]][6:]

    u_nn_p = (des_pos - qc) * kp

    u = u_nn_p

    tau_limits = ids["tau_limits"]
    u = jnp.clip(u, -tau_limits, tau_limits)
    return u