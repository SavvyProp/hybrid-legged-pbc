# This script calculates an initial pose by rotating 
import numpy as np
import mujoco
import time
import mujoco.viewer
import os
import jax
import jax.numpy as jnp
import mujoco.mjx as mjx

def initial_pose(mj_model, base_qpos, ids):
    """
    Fix initial pose by:
    1) Rotating the base quaternion by the minimal angle so that the left and right
       foot sites have the same z value.
    2) Adjusting the right ankle pitch and roll so both feet are level (match the left foot's roll/pitch).

    Args:
      mj_model: mujoco.MjModel
      base_qpos: np.ndarray (nq,), free joint first 7 as [px, py, pz, qw, qx, qy, qz]
      ids: dict with site ids, e.g., ids["left_foot"]["site_id"], ids["right_foot"]["site_id"]

    Returns:
      np.ndarray: corrected qpos
    """
    qpos = np.array(base_qpos, dtype=np.float64)

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def quat_normalize(q):
        return q / (np.linalg.norm(q) + 1e-12)

    def axis_angle_to_quat(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = axis / n
        half = 0.5 * angle
        s = np.sin(half)
        return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])

    # 1) Minimal base rotation to equalize foot z
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)

    l_sid = ids["eef"]["left_foot"]["site_id"]
    r_sid = ids["eef"]["right_foot"]["site_id"]
    lp = data.site_xpos[l_sid].copy()
    rp = data.site_xpos[r_sid].copy()

    d = rp - lp
    d_xy = d.copy(); d_xy[2] = 0.0
    dz = d[2]

    if np.abs(dz) > 1e-9 and np.linalg.norm(d_xy) > 1e-9:
        axis = np.cross(d, d_xy)
        # rotate d towards its horizontal projection by angle theta to kill z
        theta = np.arctan2(dz, np.linalg.norm(d_xy))
        q_rot = axis_angle_to_quat(axis, -theta)
        # apply on base orientation (wxyz is [qw,qx,qy,qz])
        q_base = qpos[3:7]
        q_new = quat_normalize(quat_mul(q_rot, q_base))
        qpos[3:7] = q_new
        # re-forward
        data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, data)

    # 2) Flatten both feet using site frames so their local z-axes align with world +Z
    
    def mat_to_roll_pitch(R):
        # ZYX: yaw->pitch->roll; returns roll(x), pitch(y) relative to world
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        return roll, pitch

    Rl_site = data.site_xmat[l_sid].reshape(3, 3)
    Rr_site = data.site_xmat[r_sid].reshape(3, 3)
    roll_l, pitch_l = mat_to_roll_pitch(Rl_site)
    roll_r, pitch_r = mat_to_roll_pitch(Rr_site)

    # Joint IDs for ankles (adjust names if needed)
    jl_roll  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'Left_Ankle_Roll')
    jl_pitch = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'Left_Ankle_Pitch')
    jr_roll  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'Right_Ankle_Roll')
    jr_pitch = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'Right_Ankle_Pitch')

    qa_l_roll  = mj_model.jnt_qposadr[jl_roll]
    qa_l_pitch = mj_model.jnt_qposadr[jl_pitch]
    qa_r_roll  = mj_model.jnt_qposadr[jr_roll]
    qa_r_pitch = mj_model.jnt_qposadr[jr_pitch]

    # Subtract measured roll/pitch to make both feet flat
    qpos[qa_l_roll]  = qpos[qa_l_roll]  - roll_l
    qpos[qa_l_pitch] = qpos[qa_l_pitch] - pitch_l
    qpos[qa_r_roll]  = qpos[qa_r_roll]  - roll_r
    qpos[qa_r_pitch] = qpos[qa_r_pitch] - pitch_r

    # Clamp ankles to joint limits
    for jid in (jl_roll, jl_pitch, jr_roll, jr_pitch):
        qa = mj_model.jnt_qposadr[jid]
        lo, hi = mj_model.jnt_range[jid]
        if lo < hi:  # valid range
            qpos[qa] = np.clip(qpos[qa], lo, hi)

    # Final forward to validate
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)

    qpos[2] += 0.025

    return qpos
