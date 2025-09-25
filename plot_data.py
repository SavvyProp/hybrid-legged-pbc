import os
import numpy as np
import matplotlib.pyplot as plt

import models.booster_t1_pgnd.booster_ids as bids

def load_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.loadtxt(path, delimiter=",")


def plot_cols_11_16_overlaid(pd_tau: np.ndarray, u: np.ndarray, *, suptitle: str = "u & pd_tau cols 11–16", save_path: str | None = None):
    """Plot 6 subplots (columns 11..16, 1-based) overlaying pd_tau and u.
    pd_tau, u: (n, 23) arrays. Returns (fig, axes).
    """
    assert pd_tau.shape[1] >= 16 and u.shape[1] >= 16, "Expect (n,23) inputs"
    #idxs = np.arange(10, 16)  # 0-based indices for columns 11..16
    idxs = np.arange(11, 17)
    n = pd_tau.shape[0]
    t = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()
    for i, col in enumerate(idxs):
        ax = axes[i]
        ax.plot(t, pd_tau[:, col], label='pd_tau', linewidth=1.2)
        ax.plot(t, u[:, col], label='u', linewidth=1.0)
        #ax.set_title(f'col {col+1}')
        ax.set_title(f"{bids.joint_names[col]} torque")
        ax.grid(True, linestyle='--', alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel('torque')
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

    range_lower = 100
    range_upper = 200

    # Select plotting range
    range_lower = 100
    range_upper = 200
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
    plot_cols_11_16_overlaid(pd_tau[range_lower:range_upper, :], 
                             u[range_lower:range_upper, :], save_path=os.path.join('data', 'u_pd_tau_cols_11_16.png'))

    # New: plot for f[:, 0:12]
    plot_f_first12(f[range_lower:range_upper, :], save_path=os.path.join('data', 'f_first12.png'))

    plt.show()

if __name__ == '__main__':
    main()
