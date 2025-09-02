import jax.numpy as jnp

def logit2limit(logit, ids):
    #joint20_limits = ids["jnt_limits"]
    #center = jnp.mean(joint20_limits, axis=1)
    #d_top = joint20_limits[:, 1] - center
    #tanh_mag = jnp.abs(d_top)
    #return jnp.tanh(logit) * tanh_mag + center
    center = ids["default_qpos"][7:]
    scale = 0.5
    return jnp.tanh(logit) * scale + center

def logit2vel(logit):
    max_vel = 10.0
    return jnp.tanh(logit) * max_vel

def step(mjx_model, state, act, ids):
    nn_p_logit = act[:ids["ctrl_num"]]
    nn_d_logit = act[ids["ctrl_num"]:2*ids["ctrl_num"]]
    des_pos = logit2limit(nn_p_logit, ids)
    des_vel = logit2vel(nn_d_logit)

    kp = jnp.array(ids["p_gains"])
    kd = jnp.array(ids["d_gains"])

    qc = state.qpos[ids["joint_pos_ids"]][7:]
    qd = state.qvel[ids["joint_vel_ids"]][6:]

    u_nn_p = (des_pos - qc) * kp
    u_nn_d = (des_vel - qd) * kd * 0.1

    u = u_nn_p + u_nn_d
    return u, state