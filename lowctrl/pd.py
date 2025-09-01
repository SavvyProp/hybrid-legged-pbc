import jax.numpy as jnp

def logit2limit(logit, ids):
    joint20_limits = ids["jnt_limits"]
    center = jnp.mean(joint20_limits, axis=1)
    d_top = joint20_limits[:, 1] - center
    tanh_mag = jnp.abs(d_top)
    return jnp.tanh(logit) * tanh_mag + center

def logit2vel(logit):
    max_vel = 10.0
    return jnp.tanh(logit) * max_vel