# This script takes the joint position trajectory and 
# loops through it in cpu mujoco and computes body position
# Remaps to the 50 hz
# Saves all body position and quat (offset to base) to csv
# Load and save to motions

import numpy as np
import mujoco
import time
import mujoco.viewer
import os
import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
from motion_retarget.initial_pose import initial_pose
from models.booster_t1_pgnd.booster_ids import ids

def resample_motion(traj):
    if isinstance(traj, str):
        traj = np.load(f"gmr_motions/{traj}", allow_pickle=True)
    fps = float(np.array(traj['fps']).item() if np.size(traj['fps']) == 1 else traj['fps'])
    root_pos = np.array(traj['root_pos'])  # (N, 3)
    root_quat_xyzw = np.array(traj['root_rot'])  # (N, 4) assumed x, y, z, w
    root_quat = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], axis=1)  # convert to w, x, y, z
    joint_pos = np.array(traj['dof_pos'])   # (N, D)

    mj_model = mujoco.MjModel.from_xml_path('models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml')
    data = mujoco.MjData(mj_model)
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'Trunk')

    # Build original and target timelines
    orig_dt = 1.0 / fps
    N = root_pos.shape[0]
    t_orig = np.arange(N, dtype=np.float64) * orig_dt
    target_hz = 50.0
    target_dt = 1.0 / target_hz
    t_new = np.arange(0.0, t_orig[-1] + 1e-9, target_dt, dtype=np.float64)

    # Helpers
    def linear_interp(arr, t_src, t_tgt):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return np.interp(t_tgt, t_src, arr)
        out = np.empty((t_tgt.shape[0], arr.shape[1]), dtype=np.float64)
        for d in range(arr.shape[1]):
            out[:, d] = np.interp(t_tgt, t_src, arr[:, d])
        return out

    def slerp(q0, q1, a):
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            q = q0 + a * (q1 - q0)
            return q / np.linalg.norm(q)
        theta0 = np.arccos(dot)
        sin_theta0 = np.sin(theta0)
        theta = theta0 * a
        s0 = np.sin(theta0 - theta) / sin_theta0
        s1 = np.sin(theta) / sin_theta0
        q = s0 * q0 + s1 * q1
        return q / np.linalg.norm(q)

    # Interpolate root position and joints linearly
    root_pos_i = linear_interp(root_pos, t_orig, t_new)
    joint_pos_i = linear_interp(joint_pos, t_orig, t_new)

    # Interpolate quaternions with slerp per-sample
    def interp_quat_series(q_series, t_src, t_tgt):
        out = np.empty((t_tgt.shape[0], 4), dtype=np.float64)
        for i, tt in enumerate(t_tgt):
            idx = np.searchsorted(t_src, tt, side='right') - 1
            idx = np.clip(idx, 0, len(t_src) - 2)
            t0, t1 = t_src[idx], t_src[idx + 1]
            a = 0.0 if t1 == t0 else (tt - t0) / (t1 - t0)
            out[i] = slerp(q_series[idx], q_series[idx + 1], a)
        return out

    root_quat_i = interp_quat_series(root_quat, t_orig, t_new)

    # Assemble qpos frames
    nq = mj_model.nq
    qpos_frames = np.zeros((t_new.shape[0], nq), dtype=np.float64)

    # Determine if joint_pos already contains full qpos
    if joint_pos_i.shape[1] == nq:
        qpos_frames[:] = joint_pos_i
    else:
        base_dofs = 7 if nq - joint_pos_i.shape[1] == 7 else max(0, nq - joint_pos_i.shape[1])
        if base_dofs != 7:
            # Fallback: try to place joints at the end
            base_dofs = min(7, nq)
        qpos_frames[:, :3] = root_pos_i
        qpos_frames[:, 3:7] = root_quat_i
        qpos_frames[:, base_dofs:base_dofs + joint_pos_i.shape[1]] = joint_pos_i

    # Compute full qvel (base + joints) via finite difference at 50 Hz
    qvel_frames = compute_qvel_sequence_from_qpos(mj_model, qpos_frames, target_dt)

    body_poses_seq = []
    body_vels_seq = []
    for i in range(qpos_frames.shape[0]):
        data.qpos[:] = qpos_frames[i]
        data.qvel[:] = qvel_frames[i]
        mujoco.mj_forward(mj_model, data)
        rel_arr = body_poses_in_base(mj_model, data, base_id)
        vel_arr = body_vels_in_base(mj_model, data)
        body_poses_seq.append(rel_arr)
        body_vels_seq.append(vel_arr)

    # Save combined body poses, body velocities, qpos, and qvel to a single CSV
    if body_poses_seq:
        body_poses = np.stack(body_poses_seq, axis=0)  # (T, n_bodies-1, 7)
        body_vels = np.stack(body_vels_seq, axis=0)    # (T, n_bodies-1, 6) world frame
        combined = np.concatenate([
            body_poses.reshape(body_poses.shape[0], -1),
            body_vels.reshape(body_vels.shape[0], -1),
            qpos_frames,
            qvel_frames
        ], axis=1)
        print(body_poses.shape, body_vels.shape)
    else:
        combined = np.concatenate([qpos_frames, qvel_frames], axis=1)

    initial_pose_ = initial_pose(mj_model, qpos_frames[0], ids=ids)
    np.savetxt('motions/CMU_02_05.csv', combined, delimiter=',')
    np.savetxt('motions/CMU_02_05_initial_pose.csv', initial_pose_, delimiter=',')

    # Visualize at 50 Hz in viewer
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        start = time.time()
        for i in range(qpos_frames.shape[0]):
            if not viewer.is_running():
                break
            data.qpos[:] = qpos_frames[i]
            data.qvel[:] = qvel_frames[i]
            mujoco.mj_forward(mj_model, data)
            viewer.sync()
            # maintain 50 Hz pacing
            next_t = start + (i + 1) * target_dt
            now = time.time()
            if next_t > now:
                time.sleep(next_t - now)

def body_poses_in_base(mj_model, data, base_body_id, save_path=None):
    """
    Compute position and orientation quaternion for every body (excluding the base) in the base frame.
    Returns an array of shape (num_bodies_excl_base, 7): [px, py, pz, qw, qx, qy, qz] per body.
    Optionally saves the array to save_path (np.save).
    """
    # base_body_id is provided by caller

    # Ensure kinematics are up-to-date
    mujoco.mj_forward(mj_model, data)

    # Global poses
    xpos = data.xpos.copy()   # (nbody, 3)
    xquat = data.xquat.copy() # (nbody, 4) wxyz

    # Helpers
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def quat_to_mat(q):
        w, x, y, z = q
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        return np.array([
            [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
            [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
        ])

    pb = xpos[base_body_id]
    qb = xquat[base_body_id]
    RbT = quat_to_mat(quat_conj(qb))  # R_b^T

    rel_list = []
    for bid in range(mj_model.nbody):
        if bid == 0 or bid == base_body_id:
            continue
        pb_i = xpos[bid]
        qb_i = xquat[bid]
        # relative position: R_b^T * (p_i - p_b)
        rel_p = RbT @ (pb_i - pb)
        # relative orientation: q_rel = inv(qb) * qi
        rel_q = quat_mul(quat_conj(qb), qb_i)
        # normalize for safety
        rel_q = rel_q / np.linalg.norm(rel_q)
        rel_list.append(np.concatenate([rel_p, rel_q]))

    rel_arr = np.stack(rel_list, axis=0) if rel_list else np.zeros((0, 7))

    if save_path is not None:
        np.save(save_path, rel_arr)

    return rel_arr

def body_vels_in_base(mj_model, data):
    """
    Compute each body's linear and angular velocity in the WORLD frame (exclude world and base).
    Returns (n_bodies-2, 6) with [vx, vy, vz, wx, wy, wz].
    """
    mujoco.mj_forward(mj_model, data)
    return data.cvel

def body_poses_in_base_mjx(mjx_model, mjx_data, base_body_id):
    """
    MJX (JAX) version: compute each body's [pos, quat] in base frame (exclude world and base).
    Returns (n_bodies-2, 7) with [px, py, pz, qw, qx, qy, qz].
    """

    xpos = mjx_data.xpos      # (nbody, 3)
    xquat = mjx_data.xquat    # (nbody, 4) wxyz

    def quat_conj(q):
        return jnp.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return jnp.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def quat_to_mat(q):
        w, x, y, z = q
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        return jnp.array([
            [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
            [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
        ])

    pb = xpos[base_body_id]
    qb = xquat[base_body_id]
    RbT = quat_to_mat(quat_conj(qb))

    nbody = mjx_model.nbody
    all_ids = jnp.arange(nbody)
    mask = (all_ids != 0) & (all_ids != base_body_id)
    idx = jnp.where(mask, size=nbody-2, fill_value=0)[0]

    def compute_for_id(bid):
        pi = xpos[bid]
        qi = xquat[bid]
        rel_p = RbT @ (pi - pb)
        rel_q = quat_mul(quat_conj(qb), qi)
        rel_q = rel_q / jnp.linalg.norm(rel_q)
        return jnp.concatenate([rel_p, rel_q])

    poses_all = jax.vmap(compute_for_id)(idx)
    return poses_all


def quat_error_magnitude(qt, qc):
    """
    Quaternion error magnitude (geodesic angle, radians) between two wxyz quaternions.
    Works with single quaternions (4,) or batched (N, 4). Returns scalar for single, (N,) for batch.
    """
    qt = jnp.asarray(qt, dtype=jnp.float64)
    qc = jnp.asarray(qc, dtype=jnp.float64)
    single = False
    if qt.ndim == 1:
        qt = qt[None, :]
        qc = qc[None, :]
        single = True
    qt = qt / (jnp.linalg.norm(qt, axis=-1, keepdims=True) + 1e-12)
    qc = qc / (jnp.linalg.norm(qc, axis=-1, keepdims=True) + 1e-12)
    dot = jnp.sum(qt * qc, axis=-1)
    dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
    ang = 2.0 * jnp.arccos(dot)
    return ang[0] if single else ang

def compute_qvel_from_qpos_fd(mj_model, qpos_prev, qpos_curr, dt):
    """
    Compute full generalized velocity qvel (base + joints) from two qpos samples using finite differences.
    Handles free (7 qpos -> 6 qvel), ball (4 -> 3), hinge/slide (1 -> 1).
    Quaternions are wxyz.
    """
    nv = mj_model.nv
    qvel = np.zeros((nv,), dtype=np.float64)

    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def quat_delta_to_omega(q_prev, q_curr, dt):
        # Ensure continuity
        if np.dot(q_prev, q_curr) < 0.0:
            q_curr = -q_curr
        dq = quat_mul(q_curr, quat_conj(q_prev))
        dq = dq / (np.linalg.norm(dq) + 1e-12)
        w = np.clip(dq[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(max(1e-12, 1.0 - w*w))
        axis = dq[1:4] / s
        return axis * (angle / dt)

    for j in range(mj_model.njnt):
        jtype = mj_model.jnt_type[j]
        qa = mj_model.jnt_qposadr[j]
        da = mj_model.jnt_dofadr[j]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            # translational velocity
            qvel[da:da+3] = (qpos_curr[qa:qa+3] - qpos_prev[qa:qa+3]) / dt
            # angular velocity from quaternion delta
            w = quat_delta_to_omega(qpos_prev[qa+3:qa+7], qpos_curr[qa+3:qa+7], dt)
            qvel[da+3:da+6] = w
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            w = quat_delta_to_omega(qpos_prev[qa:qa+4], qpos_curr[qa:qa+4], dt)
            qvel[da:da+3] = w
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE or jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            qvel[da] = (qpos_curr[qa] - qpos_prev[qa]) / dt
        else:
            # Unsupported/unknown
            pass

    return qvel


def compute_qvel_sequence_from_qpos(mj_model, qpos_frames, dt):
    """
    Vectorized wrapper: compute qvel for each frame in a sequence (first frame zeros).
    """
    T = qpos_frames.shape[0]
    nv = mj_model.nv
    qvel_frames = np.zeros((T, nv), dtype=np.float64)
    for t in range(1, T):
        qvel_frames[t] = compute_qvel_from_qpos_fd(mj_model, qpos_frames[t-1], qpos_frames[t], dt)
    return qvel_frames