import numpy as np
import os
from models.booster_t1_pgnd import booster_ids as bids
import matplotlib.pyplot as plt

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

u = load_matrix("data/u.csv")
a = 0.8
u_filt = np.zeros_like(u)
for c in range(u.shape[0]):
    if c == 0:
        u_filt[c, :] = u[c, :]
    else:
        u_filt[c, :] = a * u_filt[c-1, :] + (1 - a) * u[c, :]

range_lower = 100
range_upper = 200

plot_cols_11_16_overlaid(u[range_lower:range_upper, :], 
                             u_filt[range_lower:range_upper, :], 
                             legend=['u', 'u_filt'],
                             ylabel='torque (Nm)',
                             save_path=os.path.join('data', 'u_pd_tau_cols_11_16.png'))
plt.show()