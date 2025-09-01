from mujoco import mjx
import jax.numpy as jnp

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