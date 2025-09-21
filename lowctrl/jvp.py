import numpy as np
import mujoco
from lowctrl import math as lmath
import jax.numpy as jnp
import mujoco.mjx as mjx
import jax


def get_djp(mjx_model, mjx_data, ids):
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel
    qdot_qpos = lmath.qposdot_from_qvel(mjx_model, qpos, qvel)

    orienlist = []
    for eef_name in ids["eef"].keys():
        orienmat = mjx_data.site_xmat[ids["eef"][eef_name]["site_id"]]
        orienlist.append(orienmat)

    def get_vel(qpos1):
        def get_fik(qpos2):
            d = mjx_data.replace(qpos=qpos2, 
                                        qvel = jnp.zeros_like(mjx_data.qvel))
            d = mjx.kinematics(mjx_model, d)
            poslist = []
            orienlist = []
            for eef_name in ids["eef"].keys():
                site_id = ids["eef"][eef_name]["site_id"]
                point = d.site_xpos[site_id]
                orien = d.site_xmat[site_id]
                poslist.append(point)
                orienlist.append(orien)
            com_pos = lmath.com_pos(mjx_model, d)
            #com_pos = jnp.zeros([3,])
            return com_pos, poslist, orienlist
        _, (com_pos, poslist, angmatlist) = jax.jvp(get_fik, (qpos1, ), (qdot_qpos, ))

        def rotmat2angvel(R, dR):
            K = R.T @ dR
            K = 0.5 * (K - K.T)
            omega = jnp.array([
                K[2, 1], K[0, 2], K[1, 0]
            ])
            return omega
        angvellist = []
        for c in range(ids["eef_num"]):
            angvel = rotmat2angvel(orienlist[c], angmatlist[c])
            angvellist.append(angvel)
        
        return com_pos, poslist, angvellist
    
    _, (com_acc, acclist, angacclist) = jax.jvp(get_vel, (qpos, ), (qdot_qpos, ))

    combined_list = []
    for c in range(ids["eef_num"]):
        combined_list += [acclist[c].flatten(), angacclist[c].flatten()]

    return com_acc, jnp.concatenate(combined_list, axis = 0)


def _clone_state(model: mujoco.MjModel, src: mujoco.MjData) -> mujoco.MjData:
    """Make a new MjData and copy the relevant state fields from src."""
    dst = mujoco.MjData(model)
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.qacc[:] = src.qacc
    dst.act[:]  = src.act
    if model.nmocap > 0:
        dst.mocap_pos[:]  = src.mocap_pos
        dst.mocap_quat[:] = src.mocap_quat
    return dst


def _site_spatial_vel(model: mujoco.MjModel, data: mujoco.MjData, site_id: int):
    """Return 6D world-frame spatial velocity [v; w] = [Jp; Jr] @ qvel."""
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site_id)
    return np.hstack([jp @ data.qvel, jr @ data.qvel])


def site_jdot_qdot_fd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    h: float = 1e-6,
    scheme: str = "central",  # "central" (O(h^2)) or "forward" (O(h))
):
    """
    Finite-difference estimate of Jdot(q, qdot) * qdot for a site (world frame).

    Returns:
        6-vector [a_lin_world; a_ang_world]
    """
    mujoco.mj_forward(model, data)  # ensure kinematics are up to date
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    if scheme == "central":
        d_minus = _clone_state(model, data)
        mujoco.mj_integratePos(model, d_minus.qpos, d_minus.qvel, -h)
        mujoco.mj_forward(model, d_minus)
        xdot_minus = _site_spatial_vel(model, d_minus, site_id)

        d_plus = _clone_state(model, data)
        mujoco.mj_integratePos(model, d_plus.qpos, d_plus.qvel, +h)
        mujoco.mj_forward(model, d_plus)
        xdot_plus = _site_spatial_vel(model, d_plus, site_id)

        return (xdot_plus - xdot_minus) / (2.0 * h)

    elif scheme == "forward":
        xdot_0 = _site_spatial_vel(model, data, site_id)

        d_plus = _clone_state(model, data)
        mujoco.mj_integratePos(model, d_plus.qpos, d_plus.qvel, +h)
        mujoco.mj_forward(model, d_plus)
        xdot_plus = _site_spatial_vel(model, d_plus, site_id)

        return (xdot_plus - xdot_0) / h

    else:
        raise ValueError("scheme must be 'central' or 'forward'")

def jdot_q_for_model_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Returns world-frame (3,) vector equal to dot(J_com) @ qvel for the whole model COM.
    Assumes `data.qacc` is set (e.g., after mj_step, mj_forward, or mj_inverse).
    """

    # 1) Ensure derived quantities are consistent with (q, qvel, qacc)
    mujoco.mj_forward(model, data)

    # 2) COM Jacobian of the entire model (subtree rooted at worldbody=0)
    #    Only the translational part (jacp) is needed.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))  # unused; COM rotational jacobian is effectively zero
    mujoco.mj_jacSubtreeCom(model, data, jacp, 0)  # bodyid=0 => whole model

    Jqacc = jacp @ data.qacc  # (3,)

    # 3) Whole-model COM linear acceleration (world frame)
    #    Use per-body COM accelerations from data.cacc, rotated to world, then mass-average.
    cacc = data.cacc.reshape(model.nbody, 6)      # [ang(0:3), lin(3:6)] in BODY frame at body COM
    R = data.xmat.reshape(model.nbody, 3, 3)      # body->world rotation
    a_lin_world = np.einsum('bij,bj->bi', R, cacc[:, 3:6])  # (nbody,3)

    masses = model.body_mass                      # (nbody,)
    # body 0 is the worldbody (mass=0), skip it in the average
    M = np.sum(masses[1:])
    a_com = (a_lin_world[1:] * masses[1:, None]).sum(axis=0) / M  # (3,)

    # 4) dot(J) @ qvel = a_com - J @ qacc
    print("data_qacc ", data.qacc)
    print("a_com ", a_com)
    print("Jqacc ", Jqacc)
    return a_com - Jqacc, jacp

import mujoco as mj

def com_jacobian(model: mj.MjModel, data: mj.MjData, body_id: int = 0) -> np.ndarray:
    """
    Return the linear CoM Jacobian (3 x nv) of a subtree.
    body_id=0 => whole-model CoM.
    """
    Jp = np.zeros((3, model.nv))
    # angular part not needed; we pass None for Ja
    mj.mj_jacSubtreeCom(model, data, Jp, body_id)  # fills 3*nv linear Jacobian (row-major)
    return Jp.reshape(3, model.nv)

def com_jacobian_dot_fd(model: mj.MjModel, data: mj.MjData, dt: float = 1e-6, body_id: int = 0) -> np.ndarray:
    """
    Finite-difference time derivative of the CoM Jacobian at (q, qdot) along the current velocity.
    Does NOT mutate the caller's data.
    """
    # J at t
    J0 = com_jacobian(model, data, body_id)

    # Create a scratch MjData and advance q by qdot*dt (with proper integration)
    d2 = mj.MjData(model)
    d2.qpos[:] = data.qpos
    d2.qvel[:] = data.qvel
    mj.mj_integratePos(model, d2.qpos, d2.qvel, dt)
    mj.mj_forward(model, d2)  # update kinematics at q(t+dt)

    # J at t+dt
    J1 = com_jacobian(model, d2, body_id)

    return (J1 - J0) / dt

def model_com2(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Returns the world-frame position of the model's center of mass.
    """
    jac = com_jacobian(model, data, body_id=0)  # (3, nv)
    jdot = com_jacobian_dot_fd(model, data, dt=1e-6, body_id=0)  # (3, nv)
    jvp = jdot @ data.qvel  # (3,)
    return jac, jvp
