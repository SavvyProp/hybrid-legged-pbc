import jax.numpy as jnp
import jax

def logit2limit(logit, ids):
    joint20_limits = ids["jnt_limits"]
    #center = jnp.mean(joint20_limits, axis=1)
    center = ids["default_qpos"][7:]
    upper_limit = joint20_limits[:, 1]
    lower_limit = joint20_limits[:, 0]
    scale = (upper_limit - lower_limit) / 2.0
    ave = (upper_limit + lower_limit) / 2.0
    # shift is - lower_limit
    phase_shift = (center - ave) / scale
    des_pos = scale * jnp.tanh(logit * 1.0 / scale + phase_shift) + ave
    return des_pos

def step(mjx_model, state, act, ids):
    nn_p_logit = act[:ids["ctrl_num"]]
    des_pos = logit2limit(nn_p_logit, ids)

    kp = jnp.array(ids["p_gains"])
    kd = jnp.array(ids["d_gains"])

    qc = state.qpos[ids["joint_pos_ids"]][7:]
    qd = state.qvel[ids["joint_vel_ids"]][6:]

    u_nn_p = (des_pos - qc) * kp

    u = u_nn_p

    tau_limits = ids["tau_limits"]
    u = jnp.clip(u, -tau_limits, tau_limits)
    return u