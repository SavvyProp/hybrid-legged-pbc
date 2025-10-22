import os
import numpy as np
import matplotlib.pyplot as plt

import models.booster_t1_pgnd.booster_ids as bids

def load_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.loadtxt(path, delimiter=",")


def plot_cols_11_16_overlaid(pd_tau: np.ndarray, u: np.ndarray, *, 
                             legend,
                             ylabel: str = "torque",
                             idxs: np.ndarray = np.arange(11,17),
                             suptitle: str = "u & pd_tau cols 11–16", save_path: str | None = None):
    """Plot 6 subplots (columns 11..16, 1-based) overlaying pd_tau and u.
    pd_tau, u: (n, 23) arrays. Returns (fig, axes).
    """
    assert pd_tau.shape[1] >= 16 and u.shape[1] >= 16, "Expect (n,23) inputs"
    #idxs = np.arange(10, 16)  # 0-based indices for columns 11..16
    #idxs = np.arange(11, 17)
    n = pd_tau.shape[0]
    t = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()
    for i, col in enumerate(idxs):
        ax = axes[i]
        ax.plot(t, pd_tau[:, col], label=legend[0], linewidth=1.2)
        ax.plot(t, u[:, col], label=legend[1], linewidth=1.0)
        #ax.set_title(f'col {col+1}')
        ax.set_title(f"{bids.joint_names[col]} {ylabel}")
        ax.grid(True, linestyle='--', alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel(ylabel)
        if i // 3 == 1:
            ax.set_xlabel('timestep')
        if i == 0:
            ax.legend(loc='best')
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_f_first12(f: np.ndarray, *, suptitle: str = "f[:, 0:12] over time", save_path: str | None = None):
    """Plot 12 subplots for columns 0..11 of f (shape: n x >=12).
    Returns (fig, axes).
    """
    assert f.ndim == 2 and f.shape[1] >= 12, "f must be (n, >=12)"
    n = f.shape[0]
    t = np.arange(n)

    name_list = ["Left Foot Fx", "Left Foot Fy", "Left Foot Fz",
                 "Left Foot Taux", "Left Foot Tauy", "Left Foot Tauz",
                 "Right Foot Fx", "Right Foot Fy", "Right Foot Fz",
                 "Right Foot Taux", "Right Foot Tauy", "Right Foot Tauz"]

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    for i in range(12):
        ax = axes[i]
        ax.plot(t, f[:, i], linewidth=1.1)
        ax.set_title(f"{name_list[i]}")
        ax.grid(True, linestyle='--', alpha=0.3)
        if name_list[i][-2] == "F":
            ax.set_ylabel('force (N)')
        else:
            ax.set_ylabel('torque (Nm)')
        if i >= 8:
            ax.set_xlabel('timestep')
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_qddot_com_vs_ref(q_ddot_com: np.ndarray, com_ref: np.ndarray, *, suptitle: str = "q_ddot_com vs com_ref (xyz)", save_path: str | None = None, idx_range: tuple[int, int] | None = None):
    """Plot 3 subplots overlaying xyz components of q_ddot_com and com_ref.
    Accepts (n,6) or (n,3) arrays; only the first 3 components are plotted.
    """
    q = np.asarray(q_ddot_com)
    r = np.asarray(com_ref)
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("q_ddot_com and com_ref must be 2D arrays")
    if q.shape[1] >= 3:
        q = q[:, :3]
    else:
        raise ValueError("q_ddot_com must have at least 3 columns")
    if r.shape[1] >= 3:
        r = r[:, :3]
    else:
        raise ValueError("com_ref must have at least 3 columns")

    n = min(q.shape[0], r.shape[0])
    lo, hi = (0, n) if idx_range is None else (max(0, int(idx_range[0])), min(n, int(idx_range[1])))
    q = q[lo:hi]
    r = r[lo:hi]
    t = np.arange(lo, lo + q.shape[0])

    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i in range(3):
        ax = axes[i]
        ax.plot(t, q[:, i], label="q_ddot_com", linewidth=1.2)
        ax.plot(t, r[:, i], label="com_ref", linewidth=1.0)
        ax.set_title(labels[i])
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel("acc (m/s^2)")
        ax.set_xlabel("timestep")
        if i == 0:
            ax.legend(loc='best')
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_qc_weight_cols_11_16(qc_weight: np.ndarray, *, suptitle: str = "qc_weight cols 11–16", save_path: str | None = None):
    """Plot 6 subplots for qc_weight columns 11..16 (1-based indexing).
    qc_weight: (n, >=16) array. Returns (fig, axes).
    """
    assert qc_weight.ndim == 2 and qc_weight.shape[1] >= 16, "qc_weight must have at least 16 columns"
    idxs = np.arange(11, 17)  # 0-based indices for columns 11..16
    n = qc_weight.shape[0]
    t = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()
    for i, col in enumerate(idxs):
        ax = axes[i]
        ax.plot(t, qc_weight[:, col], linewidth=1.2)
        title = f"col {col+1} qc_weight"
        try:
            title = f"{bids.joint_names[col]} qc_w"  # use joint name if available
        except Exception:
            pass
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel('weight')
        if i // 3 == 1:
            ax.set_xlabel('timestep')
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_norms(norm_big_q: np.ndarray, norm_small_q: np.ndarray, names: list[str], *,
               suptitle: str = "norm_big_q & norm_small_q components", layout: str = "2x7",
               save_path: str | None = None):
    """Plot 14 subplots: first 7 from norm_big_q columns, next 7 from norm_small_q.
    norm_big_q, norm_small_q: (n,7) arrays; names: list of 7 labels for columns.
    layout: "2x7" (2 rows, 7 cols) or "7x2".
    """
    norm_big_q = np.asarray(norm_big_q)
    norm_small_q = np.asarray(norm_small_q)
    if norm_big_q.ndim != 2 or norm_small_q.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")
    if norm_big_q.shape[1] != 7 or norm_small_q.shape[1] != 7:
        raise ValueError("Expect exactly 7 columns in each input")
    if len(names) != 7:
        raise ValueError("names list must have length 7")
    n = min(norm_big_q.shape[0], norm_small_q.shape[0])
    t = np.arange(n)

    if layout == "2x7":
        fig, axes = plt.subplots(2, 7, figsize=(18, 5), sharex=True)
        big_axes = axes[0]
        small_axes = axes[1]
    elif layout == "7x2":
        fig, axes = plt.subplots(7, 2, figsize=(10, 16), sharex=True)
        big_axes = axes[:, 0]
        small_axes = axes[:, 1]
    else:
        raise ValueError("layout must be '2x7' or '7x2'")

    # Plot big_q components
    for i in range(7):
        ax = big_axes[i]
        ax.plot(t, norm_big_q[:n, i], linewidth=1.1)
        ax.set_title(f"big {names[i]}")
        ax.grid(True, linestyle='--', alpha=0.3)
        if layout == "2x7" and i == 0:
            ax.set_ylabel('norm_big_q')
        if layout == "7x2":
            ax.set_ylabel(names[i])

    # Plot small_q components
    for i in range(7):
        ax = small_axes[i]
        ax.plot(t, norm_small_q[:n, i], linewidth=1.1, color='tab:orange')
        ax.set_title(f"small {names[i]}")
        ax.grid(True, linestyle='--', alpha=0.3)
        if layout == "2x7" and i == 0:
            ax.set_ylabel('norm_small_q')
        if layout == "7x2":
            ax.set_ylabel(names[i])
        ax.set_xlabel('timestep')

    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_qp_errors(qp_errors: np.ndarray, names: list[str], *, suptitle: str = "QP error components", save_path: str | None = None):
    """Plot 7 subplots (1 row) for qp_errors columns.
    qp_errors: (n,7) array; names: list of 7 labels.
    """
    qp_errors = np.asarray(qp_errors)
    if qp_errors.ndim != 2 or qp_errors.shape[1] != 7:
        raise ValueError("qp_errors must have shape (n,7)")
    if len(names) != 7:
        raise ValueError("names must have length 7")
    n = qp_errors.shape[0]
    t = np.arange(n)
    fig, axes = plt.subplots(1, 7, figsize=(18, 3.2), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i in range(7):
        ax = axes[i]
        ax.plot(t, qp_errors[:, i], linewidth=1.1)
        ax.set_title(names[i])
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel('error')
        ax.set_xlabel('t')
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


def main():
    pd_tau_path = os.path.join('data', 'pd_tau.csv')
    u_path = os.path.join('data', 'u.csv')

    pd_tau = load_matrix(pd_tau_path)
    u = load_matrix(u_path)
    f = load_matrix(os.path.join('data', 'f.csv'))
    q_ddot_com = load_matrix(os.path.join('data', 'q_ddot_com.csv'))
    com_ref = load_matrix(os.path.join('data', 'com_ref.csv'))
    real_com_vel = load_matrix(os.path.join('data', 'real_com_vel.csv'))
    des_com_vel = load_matrix(os.path.join('data', 'des_com_vel.csv'))
    des_com_angvel = load_matrix(os.path.join('data', 'des_angvel.csv'))
    real_angvel = load_matrix(os.path.join('data', 'real_angvel.csv'))

    real_pos = load_matrix(os.path.join('data', 'real_pos.csv'))
    des_pos = load_matrix(os.path.join('data', 'des_pos.csv'))

    u_final = load_matrix(os.path.join('data', 'u_final.csv'))

    f_ref = load_matrix(os.path.join('data', 'f_ref.csv'))
    tau = load_matrix(os.path.join('data', 'tau.csv'))

    range_lower = 300
    range_upper = 500

    # Select plotting range
    n = des_com_vel.shape[0]
    lo = int(np.clip(range_lower, 0, max(n - 1, 0)))
    hi = int(np.clip(range_upper, lo + 1, n))
    t = np.arange(lo, hi)

    # Slice data
    dcv = des_com_vel[lo:hi, :]
    rcv = real_com_vel[lo:hi, :]
    dav = des_com_angvel[lo:hi, :]
    rav = real_angvel[lo:hi, :]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    labels = ['x', 'y', 'z']

    # Linear COM velocity subplots (row 0)
    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, dcv[:, i], label='desired', linewidth=1.5)
        ax.plot(t, rcv[:, i], label='real', linewidth=1.0)
        ax.set_title(f'COM vel {labels[i]}')
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel('m/s')
        if i == 2:
            ax.legend(loc='best')

    # Angular velocity subplots (row 1)
    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, dav[:, i], label='desired', linewidth=1.5)
        ax.plot(t, rav[:, i], label='real', linewidth=1.0)
        ax.set_title(f'COM ang vel {labels[i]}')
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel('rad/s')
        if i == 2:
            ax.legend(loc='best')

    axes[1, 1].set_xlabel('timestep')

    fig.tight_layout()
    out_path = os.path.join('data', 'com_vel_and_angvel_xyz.png')
    fig.savefig(out_path, dpi=150)

    # New: focused plot for cols 11..16 (1-based)
    plot_cols_11_16_overlaid(u_final[range_lower:range_upper, :], 
                             u[range_lower:range_upper, :], 
                             legend=['u_ff + u_pd', 'u_ff'],
                             ylabel='torque (Nm)',
                             save_path=os.path.join('data', 'u_pd_tau_cols_11_16.png'))
    
    plot_cols_11_16_overlaid(tau[range_lower:range_upper, :], 
                             u[range_lower:range_upper, :], 
                             legend=['u_ref', 'u_ff'],
                             ylabel='torque (Nm)',
                             save_path=os.path.join('data', 'u_ff_tau_cols_11_16.png'))
    
    plot_cols_11_16_overlaid(des_pos[range_lower:range_upper, :], 
                             real_pos[range_lower:range_upper, :], 
                             legend=['des_pos', 'real_pos'],
                             ylabel='position (rad)',
                             suptitle='des_pos & real_pos cols 11–16',
                             save_path=os.path.join('data', 'des_pos_vs_real_pos.png'))

    # New: plot for f[:, 0:12]
    plot_f_first12(f[range_lower:range_upper, :], save_path=os.path.join('data', 'f_first12.png'))

    # New: plot for q_ddot_com vs com_ref
    plot_qddot_com_vs_ref(q_ddot_com[range_lower:range_upper, :], 
                          com_ref[range_lower:range_upper, :], save_path=os.path.join('data', 'q_ddot_com_vs_com_ref.png'))

    

    # New: plot for qp_errors

    plt.show()

if __name__ == '__main__':
    main()