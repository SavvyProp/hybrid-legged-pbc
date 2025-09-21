import time
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from lowctrl import maqp as maqp_mod
from lowctrl import model as lmodel
from models.booster_t1_pgnd import booster_ids as bids


def _block_until_ready(x):
    leaves, _ = jax.tree_util.tree_flatten(x)
    if leaves:
        _ = jax.block_until_ready(leaves[0])


def _time_jitted(name: str, fn: Callable, args: Tuple, reps: int = 50) -> float:
    jf = jax.jit(fn)
    out = jf(*args)
    _block_until_ready(out)
    for _ in range(5):
        res = jf(*args)
    t0 = time.perf_counter()
    res = None
    for _ in range(reps):
        res = jf(*args)
    _block_until_ready(res)
    t = time.perf_counter() - t0
    per_call_ms = (t / reps) * 1000.0
    print(f"{name:28s}: {per_call_ms:8.3f} ms")
    return per_call_ms


def _highlvlPD_like(qpos, qvel, des_pos, des_com_vel, des_angvel, ids):
    """Pure-array version of highlvlPD for JIT timing."""
    qpos_j = qpos[ids["joint_pos_ids"]]
    qvel_j = qvel[ids["joint_vel_ids"]]
    jp_gain = 200.0
    jd_gain = 20.0
    qacc = jp_gain * (des_pos - qpos_j[7:]) - jd_gain * qvel_j[6:]
    c_lin_p_gain = 50.0
    com_acc = c_lin_p_gain * (des_com_vel - qvel[0:3])
    c_ang_p_gain = 10.0
    com_angacc = c_ang_p_gain * (des_angvel - qvel[3:6])
    com_accs = jnp.concatenate([com_acc, com_angacc], axis=0)
    return qacc, com_accs


def benchmark_maqp_components(reps: int = 50) -> Dict[str, float]:
    """JIT-compile and time the main component functions used inside maqp.maqp.

    Returns a dict of per-call runtimes in milliseconds.
    """
    # Build model/data and extract kinematics needed by maqp
    model = mujoco.MjModel.from_xml_path('models/booster_t1/flat_scene.xml')
    data = mujoco.MjData(model)
    init_qpos = model.keyframe('home').qpos
    data.qpos = init_qpos
    mujoco.mj_step(model, data)  # populate derived quantities

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    ids = bids.ids

    # Kinematics pack (all arrays are device arrays via MJX getters)
    m, h, jacs, jvp, jac_com, com_jvp, eefpos, com_pos = lmodel.get_kin_values(
        mjx_model, mjx_data, ids, is_mjx=True
    )

    # Typical inputs used by maqp
    eef_num = ids["eef_num"]
    ctrl_num = ids["ctrl_num"]
    uc_size = ctrl_num + 6
    F_size = eef_num * 6
    mat_height = 6 + F_size + uc_size

    w = jnp.zeros((eef_num,), dtype=m.dtype)
    s = jax.nn.sigmoid(w)
    a_stc = jnp.zeros((6 * eef_num,), dtype=m.dtype)
    qc_weight = jnp.ones((ctrl_num,), dtype=m.dtype)
    qc_ref = jnp.zeros((ctrl_num,), dtype=m.dtype)
    com_ref = jnp.zeros((6,), dtype=m.dtype)

    # Selection matrices
    select = {
        "q_ddot_com": jnp.block([jnp.eye(6, dtype=m.dtype), jnp.zeros((6, mat_height - 6), dtype=m.dtype)]),
        "F": jnp.block([
            jnp.zeros((F_size, 6), dtype=m.dtype),
            jnp.eye(F_size, dtype=m.dtype),
            jnp.zeros((F_size, mat_height - 6 - F_size), dtype=m.dtype),
        ]),
        "q_ddot_uc": jnp.block([
            jnp.zeros((uc_size, mat_height - uc_size), dtype=m.dtype),
            jnp.eye(uc_size, dtype=m.dtype),
        ]),
    }

    # Precompute centroidal mapping
    a, g = maqp_mod.make_centroidal_a(m, eefpos, com_pos, ids)

    times: Dict[str, float] = {}

    # --- Step() pipeline pieces ---
    act = maqp_mod.default_act(ids)
    times["ctrl2components"] = _time_jitted(
        "ctrl2components",
        lambda A: maqp_mod.ctrl2components(A, ids),
        (act,), reps,
    )
    des_pos, des_com_vel, des_angvel, w, qc_weight = maqp_mod.ctrl2components(act, ids)
    times["highlvlPD"] = _time_jitted(
        "highlvlPD",
        lambda qpos, qvel, dp, dv, da: _highlvlPD_like(qpos, qvel, dp, dv, da, ids),
        (mjx_data.qpos, mjx_data.qvel, des_pos, des_com_vel, des_angvel), reps,
    )

    times["get_kin_values"] = _time_jitted(
        "get_kin_values",
        lambda M, D: lmodel.get_kin_values(M, D, ids, True),
        (mjx_model, mjx_data), reps,
    )

    # Prepare inputs for full maqp timing
    qacc_c, com_accs = _highlvlPD_like(mjx_data.qpos, mjx_data.qvel, des_pos, des_com_vel, des_angvel, ids)
    a_stc = jnp.zeros(6 * eef_num, dtype=m.dtype)

    times["maqp_full"] = _time_jitted(
        "maqp_full",
        lambda M, H, W, A, E, C, J, Jv, CJ, CJv, CR, QR, QW: maqp_mod.maqp(M, H, W, A, E, C, J, Jv, CJ, CJv, CR, QR, QW, ids, True, False),
        (m, h, w, a_stc, eefpos, com_pos, jacs, jvp, jac_com, com_jvp, com_accs, qacc_c, qc_weight), reps,
    )

    times["step_mjx"] = _time_jitted(
        "step (is_mjx=True)",
        lambda M, D, A: maqp_mod.step(M, D, A, ids, True),
        (mjx_model, mjx_data, act), reps,
    )

    # Assembly terms
    times["make_centroidal_a"] = _time_jitted(
        "make_centroidal_a",
        lambda M, E, C: maqp_mod.make_centroidal_a(M, E, C, ids),
        (m, eefpos, com_pos), reps,
    )
    times["centroidal_acc_q"] = _time_jitted(
        "centroidal_acc_q", maqp_mod.centroidal_acc_q, (com_ref,), reps
    )
    times["centroidal_cons_q"] = _time_jitted(
        "centroidal_cons_q", maqp_mod.centroidal_cons_q, (a, g), reps
    )
    times["f_mag_q"] = _time_jitted(
        "f_mag_q", lambda W: maqp_mod.f_mag_q(W, ids), (w,), reps
    )
    times["q_ddot_c_q"] = _time_jitted(
        "q_ddot_c_q", lambda QW, QR: maqp_mod.q_ddot_c_q(QW, QR, ids), (qc_weight, qc_ref), reps
    )
    times["qu_mag_q"] = _time_jitted(
        "qu_mag_q", lambda: maqp_mod.qu_mag_q(), tuple(), reps
    )
    times["eefs_acc_q"] = _time_jitted(
        "eefs_acc_q", lambda S, J, Jv, A: maqp_mod.eefs_acc_q(S, J, Jv, A, ids), (s, jacs, jvp, a_stc), reps
    )

    # Constraints
    times["centroidal_qacc_cons"] = _time_jitted(
        "centroidal_qacc_cons", lambda Sel, A_: maqp_mod.centroidal_qacc_cons(Sel, A_), (select, a), reps
    )
    times["centroidal_quc_cons"] = _time_jitted(
        "centroidal_quc_cons", lambda Sel, CJVP, CJAC: maqp_mod.centroidal_quc_cons(Sel, CJVP, CJAC), (select, com_jvp, jac_com), reps
    )
    times["fullbody_u_cons"] = _time_jitted(
        "fullbody_u_cons", lambda Sel, M, H, J: maqp_mod.fullbody_u_cons(Sel, M, H, J), (select, m, h, jacs), reps
    )

    # Build a representative QP (as in maqp) to time the solver
    weights = jnp.array([10000.0, 1000.0, 1.0, 0.1, 0.01, 0.01], dtype=m.dtype)

    qp_q = jnp.zeros((mat_height, mat_height), dtype=m.dtype)
    qp_c = jnp.zeros((mat_height,), dtype=m.dtype)

    big_q_com, small_q_com = maqp_mod.centroidal_acc_q(com_ref)
    qp_q = qp_q.at[:6, :6].add(big_q_com * weights[0])
    qp_c = qp_c.at[:6].add(small_q_com * weights[0])

    big_q_cent, small_q_cent = maqp_mod.centroidal_cons_q(a, g)
    qp_q = qp_q.at[: (6 + F_size), : (6 + F_size)].add(big_q_cent * weights[1])
    qp_c = qp_c.at[: (6 + F_size)].add(small_q_cent * weights[1])

    big_q_f, small_q_f = maqp_mod.f_mag_q(w, ids)
    qp_q = qp_q.at[6 : 6 + F_size, 6 : 6 + F_size].add(big_q_f * weights[2])
    qp_c = qp_c.at[6 : 6 + F_size].add(small_q_f * weights[2])

    big_q_qc, small_q_qc = maqp_mod.q_ddot_c_q(qc_weight, qc_ref, ids)
    qp_q = qp_q.at[12 + F_size : 12 + F_size + ctrl_num, 12 + F_size : 12 + F_size + ctrl_num].add(big_q_qc * weights[3])
    qp_c = qp_c.at[12 + F_size : 12 + F_size + ctrl_num].add(small_q_qc * weights[3])

    big_q_qu, small_q_qu = maqp_mod.qu_mag_q()
    qp_q = qp_q.at[6 + F_size : 12 + F_size, 6 + F_size : 12 + F_size].add(big_q_qu * weights[4])
    qp_c = qp_c.at[6 + F_size : 12 + F_size].add(small_q_qu * weights[4])

    big_q_acc, small_q_acc = maqp_mod.eefs_acc_q(s, jacs, jvp, a_stc, ids)
    qp_q = qp_q.at[6 + F_size :, 6 + F_size :].add(big_q_acc * weights[5])
    qp_c = qp_c.at[6 + F_size :].add(small_q_acc * weights[5])

    cons_lhs_list = []
    cons_rhs_list = []

    c_lhs2, c_rhs2 = maqp_mod.centroidal_quc_cons(select, com_jvp, jac_com)
    cons_lhs_list.append(c_lhs2)
    cons_rhs_list.append(c_rhs2)

    fb_lhs, fb_rhs = maqp_mod.fullbody_u_cons(select, m, h, jacs)
    cons_lhs_list.append(fb_lhs)
    cons_rhs_list.append(fb_rhs)

    cons_lhs = jnp.vstack(cons_lhs_list)
    cons_rhs = jnp.concatenate(cons_rhs_list, axis=0)

    times["schur_solve"] = _time_jitted(
        "schur_solve", maqp_mod.schur_solve, (qp_q, qp_c, cons_lhs, cons_rhs), reps
    )

    return times


def benchmark_get_kin_values_components(mjx_model, mjx_data, ids, reps: int = 50) -> Dict[str, float]:
    """Benchmark likely component helpers used by get_kin_values (is_mjx=True).
    Tries multiple canonical function names in lowctrl.model and times those found.
    """
    candidates = [
        # Mass matrix / dynamics terms
        ("get_mass_matrix", (mjx_model, mjx_data, ids, True)),
        ("mass_matrix", (mjx_model, mjx_data, ids, True)),
        ("get_M", (mjx_model, mjx_data, ids, True)),
        ("get_bias", (mjx_model, mjx_data, ids, True)),
        ("bias", (mjx_model, mjx_data, ids, True)),
        ("get_h", (mjx_model, mjx_data, ids, True)),
        # End-effector Jacobians / JVP
        ("get_eef_jacobians", (mjx_model, mjx_data, ids, True)),
        ("get_jacobians", (mjx_model, mjx_data, ids, True)),
        ("get_contact_jacobians", (mjx_model, mjx_data, ids, True)),
        ("get_eef_jvp", (mjx_model, mjx_data, ids, True)),
        ("get_jvp", (mjx_model, mjx_data, ids, True)),
        ("get_jacobian_dot_q", (mjx_model, mjx_data, ids, True)),
        # COM Jacobian / JVP / position
        ("get_com_jacobian", (mjx_model, mjx_data, ids, True)),
        ("get_com_jvp", (mjx_model, mjx_data, ids, True)),
        ("get_com_pos", (mjx_model, mjx_data, ids, True)),
        ("get_com_position", (mjx_model, mjx_data, ids, True)),
        # End-effector positions
        ("get_eef_positions", (mjx_model, mjx_data, ids, True)),
    ]

    times: Dict[str, float] = {}
    for fname, args in candidates:
        fn = getattr(lmodel, fname, None)
        if callable(fn):
            try:
                times[fname] = _time_jitted(fname, fn, args, reps)
            except Exception as e:
                print(f"Skip {fname}: {e}")
    if not times:
        print("No component helpers discovered in lowctrl.model; only get_kin_values available.")
    return times


if __name__ == "__main__":
    print("Benchmarking maqp component functions (JIT, per-call):")
    results = benchmark_maqp_components(reps=50)
    # Print a simple sorted summary
    print("\nSorted results (slowest first):")
    for k, v in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:28s}: {v:8.3f} ms")

    # Benchmark get_kin_values components (MJX)
    print("\nBenchmarking get_kin_values components (is_mjx=True):")
    model = mujoco.MjModel.from_xml_path('models/booster_t1/flat_scene.xml')
    data = mujoco.MjData(model)
    data.qpos = model.keyframe('home').qpos
    mujoco.mj_step(model, data)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    comp_results = benchmark_get_kin_values_components(mjx_model, mjx_data, bids.ids, reps=50)
    print("\nSorted results (slowest first):")
    for k, v in sorted(comp_results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:28s}: {v:8.3f} ms")
