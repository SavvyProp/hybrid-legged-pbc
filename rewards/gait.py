import jax.numpy as jnp


# Functions to manage phase and gait (desired contacts, height etc)
# To be consistent with the legged env, phase is represented by
# a continuous value between 0 and 1.

def create_stance_mask(phase: jnp.ndarray):
    """
    JAX version of the stance-mask helper.

    Parameters
    ----------
    phase : jnp.ndarray           # shape (B,), values in [0, 1)
        Scalar gait phase(s).

    Returns
    -------
    stance_mask : jnp.ndarray     # shape (B, 2), {0,1} ints
    mask_2      : jnp.ndarray     # shape (B, 2), {0,1} ints
    """
    # 1. Periodic signal duplicated in two columns
    sin_pos = jnp.sin(2.0 * jnp.pi * phase)          # (B,)
    sin_pos = jnp.expand_dims(sin_pos, -1)           # (B,1)
    sin_pos = jnp.repeat(sin_pos, 2, axis=1)         # (B,2)

    # 2. Alternating stance masks for the two legs
    leg0 = (sin_pos[:, 0] >= 0).astype(jnp.int32)    # (B,)
    leg1 = 1 - leg0
    stance_mask = jnp.stack([leg0, leg1], axis=1)    # (B,2)

    # 3. Double-support band around the sine zero crossings
    transition = (jnp.abs(sin_pos) < 0.1)            # boolean, (B,2)
    stance_mask = jnp.where(transition, 1, stance_mask)

    # 4. Complementary mask (mostly swing), with the same transition band
    mask_2 = 1 - stance_mask
    mask_2 = jnp.where(transition, 1, mask_2)

    return stance_mask, mask_2

def height_target(t):
    a5, a4, a3, a2, a1, a0 = jnp.array([9.6, 12.0, -18.8, 5.0, 0.1, 0.0]) * 0.6
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0