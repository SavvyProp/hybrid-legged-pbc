from mujoco import mjx
import jax.numpy as jnp
import jax

def quat_mul(a, b):
    """Hamilton product of two quaternions, both shape (..., 4)."""
    w0, x0, y0, z0 = jnp.split(a, 4, -1)
    w1, x1, y1, z1 = jnp.split(b, 4, -1)
    return jnp.concatenate((
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ), -1)


def quat_from_omega(omega):
    """Embed a 3-vector angular velocity in ℍ as (0, ω)."""
    return jnp.concatenate((jnp.zeros_like(omega[...,:1]), omega), axis=-1)


def qposdot_from_qvel(model: mjx.Model, qpos, qvel):
    """
    Pure-JAX re-implementation of mj_differentiatePos.

    Returns an (nq,) array whose layout matches qpos,
    even when the model contains quaternion joints.
    """
    qdot = jnp.zeros_like(qpos)

    for j in range(model.njnt):
        qadr  = model.jnt_qposadr[j]
        dadr  = model.jnt_dofadr[j]
        jtype = model.jnt_type[j]

        if jtype == mjx.JointType.FREE:
            # 3 translational dofs
            qdot = qdot.at[qadr:qadr+3].set(qvel[dadr:dadr+3])
            # quaternion derivative: ½ * [0, ω] ⊗ quat
            quat   = qpos[qadr+3:qadr+7]
            omega  = qvel[dadr+3:dadr+6]
            qdot_q = 0.5 * quat_mul(quat_from_omega(omega), quat)
            qdot = qdot.at[qadr+3:qadr+7].set(qdot_q)

        elif jtype == mjx.JointType.BALL:          # 4-d quaternion, 3-d vel
            quat  = qpos[qadr:qadr+4]
            omega = qvel[dadr:dadr+3]
            qdot_q = 0.5 * quat_mul(quat_from_omega(omega), quat)
            qdot = qdot.at[qadr:qadr+4].set(qdot_q)

        else:                                     # hinge or slide (1-dof)
            qdot = qdot.at[qadr].set(qvel[dadr])

    return qdot

@jax.jit
def skew(a: jnp.ndarray) -> jnp.ndarray:
    """Return the 3x3 skew-symmetric matrix of a vector a."""
    ax, ay, az = a
    return jnp.array([
        [0,   -az,  ay],
        [az,   0,  -ax],
        [-ay,  ax,   0]
    ])

def com_pos(mjx_model, mjx_data):
    total_mass = jnp.sum(mjx_model.body_mass)
    com = jnp.sum(mjx_model.body_mass[:, None] * mjx_data.xipos, # (nbody,1)*(nbody,3)
             axis=0) / total_mass  
    #com = mjx_data.subtree_com[0]
    return com

def vec2diags(v, ids):
    # v is a vector of length N
    # returns a matrix of shape (6N, 6N) with the elements of v repeated on the diagonals
    n = ids["eef_num"]
    D = jnp.zeros((n * 6, n * 6))
    rows = jnp.arange(n) * 6
    cols = rows
    D = D.at[rows[:, None] + jnp.arange(6)[None, :], cols[:, None] + jnp.arange(6)[None, :]].set(v[:, None])
    return D

def torqueCost(torque_mul, ids):
    flat_cost = jnp.ones(ids["eef_num"] * 6)
    for i in range(ids["eef_num"]):
        flat_cost = flat_cost.at[i * 6 + 3:i * 6 + 6].set(torque_mul)
    cost = jnp.eye(ids["eef_num"] * 6)
    cost = cost * flat_cost[None, :]
    return cost


def _qnormalize(q, eps=1e-12):
    return q / jnp.clip(jnp.linalg.norm(q, axis=-1, keepdims=True), eps, jnp.inf)

def _qconj(q):
    w, xyz = q[..., :1], q[..., 1:]
    return jnp.concatenate([w, -xyz], axis=-1)

def _qmul(p, q):
    # Hamilton product, both [...,4] in [w,x,y,z]
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = pw*qw - px*qx - py*qy - pz*qz
    x = pw*qx + px*qw + py*qz - pz*qy
    y = pw*qy - px*qz + py*qw + pz*qx
    z = pw*qz + px*qy - py*qx + pz*qw
    return jnp.stack([w, x, y, z], axis=-1)

def _quat_to_rotvec(q, eps=1e-12):
    """
    Map a unit quaternion q=[w, x, y, z] to a rotation vector r in R^3.
    Ensures shortest path by flipping sign if w < 0 (q ~ -q).
    """
    # Use the equivalent quaternion with nonnegative scalar to keep angle in [-pi, pi]
    q = jnp.where(q[..., :1] < 0, -q, q)

    w = jnp.clip(q[..., 0:1], -1.0, 1.0)
    v = q[..., 1:]
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)

    # Angle = 2 * atan2(||v||, w)
    angle = 2.0 * jnp.arctan2(v_norm, w)

    # For small angles, use first-order: angle * v/||v|| ≈ 2*v
    small = v_norm < 1e-8
    axis = jnp.where(small, v, v / jnp.clip(v_norm, eps, jnp.inf))
    rotvec = jnp.where(small, 2.0 * v, angle * axis)
    return rotvec

def angular_displacement_from_A_to_B(A, B):
    """
    Compute the minimal rotation vector r that rotates quaternion A into quaternion B.
    A, B: [..., 4] quaternions in [w,x,y,z]. They need not be perfectly normalized.
    Returns: r with shape [..., 3], where ||r|| is the angle in radians.
    """
    A = _qnormalize(A)
    B = _qnormalize(B)
    # delta q such that B = dq ⊗ A  =>  dq = B ⊗ conj(A)
    dq = _qmul(B, _qconj(A))
    return _quat_to_rotvec(_qnormalize(dq))

def quat_wxyz_to_R(q):
    """Return 3x3 rotation matrix from MuJoCo-style quaternion q=(w,x,y,z).
    Supports shapes (4,) or (...,4) and returns (...,3,3) or (3,3).
    """
    q = jnp.asarray(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # normalize to be safe
    n = jnp.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    r00 = 1.0 - 2.0*(yy + zz)
    r01 = 2.0*(xy - wz)
    r02 = 2.0*(xz + wy)
    r10 = 2.0*(xy + wz)
    r11 = 1.0 - 2.0*(xx + zz)
    r12 = 2.0*(yz - wx)
    r20 = 2.0*(xz - wy)
    r21 = 2.0*(yz + wx)
    r22 = 1.0 - 2.0*(xx + yy)
    R = jnp.stack([
        jnp.stack([r00, r01, r02], axis=-1),
        jnp.stack([r10, r11, r12], axis=-1),
        jnp.stack([r20, r21, r22], axis=-1),
    ], axis=-2)
    return R


def rotate_des_com_vel_from_q(des_com_vel_base, base_quat_wxyz):
    """Rotate COM velocity from base frame to world frame using a quaternion.
    des_com_vel_base: (...,3)
    base_quat_wxyz: (...,4) MuJoCo w,x,y,z orientation of base in world
    Returns (...,3) in world frame.
    """
    R = quat_wxyz_to_R(base_quat_wxyz)  # (...,3,3)
    v = jnp.asarray(des_com_vel_base)
    return jnp.einsum('...ij,...j->...i', R, v)


def rotate_des_com_vel(des_com_vel_base, data):
    """Rotate COM velocity from base frame to world using data.qpos[3:7].
    Works with MuJoCo MjData or MJX data objects that expose .qpos.
    """
    q = data.qpos[3:7]
    return rotate_des_com_vel_from_q(des_com_vel_base, q)