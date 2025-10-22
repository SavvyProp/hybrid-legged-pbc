import jax
import jax.numpy as jnp


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    """Normalizes wxyz quaternion(s)."""
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def quat_conj(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of wxyz quaternion(s)."""
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of wxyz quaternions, supports broadcasting on leading dims."""
    w1, x1, y1, z1 = jnp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(q2, 4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.concatenate([w, x, y, z], axis=-1)


def quat_to_mat(q: jnp.ndarray) -> jnp.ndarray:
    """Converts wxyz quaternion(s) to rotation matrix/matrices of shape (..., 3, 3)."""
    q = quat_normalize(q)
    w, x, y, z = jnp.split(q, 4, axis=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    r00 = ww + xx - yy - zz
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = ww - xx - yy + zz

    row0 = jnp.concatenate([r00, r01, r02], axis=-1)
    row1 = jnp.concatenate([r10, r11, r12], axis=-1)
    row2 = jnp.concatenate([r20, r21, r22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def rot_error_matrix(q_target: jnp.ndarray, q_current: jnp.ndarray) -> jnp.ndarray:
    """
    Returns rotation error matrix R_err that maps vectors from the current frame to the target frame.
    Inputs are wxyz quaternions; supports broadcasting on leading dimensions.
    R_err = R(q_target) @ R(q_current)^T = R(q_target * conj(q_current)).
    """
    q_t = quat_normalize(q_target)
    q_c = quat_normalize(q_current)
    q_err = quat_mul(q_t, quat_conj(q_c))
    return quat_to_mat(q_err)
