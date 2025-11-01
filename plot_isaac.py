import numpy as np
import matplotlib.pyplot as plt

ft_data = np.load("isaac_data/CMU_17/ft_eval_data.npz")
pd_data = np.load("isaac_data/CMU_17/pd_eval_data.npz")
ft_debug_data = np.load("isaac_data/ft_debug_data.npz")

frc_range = ft_data["forces"]

def get_num_alive(data):
    terminated = data["terminated"]
    did_term = np.any(terminated, axis = -1)
    num_alive = np.sum(~did_term, axis=1)
    return num_alive

# find the 

def get_ave_vel_error(data, frc_idx):
    impulse_time = np.round(np.sin(np.arange(36) / 6 ) * 75 + 100)
    window = 25
    lower = impulse_time - window
    upper = impulse_time + window
    vels = data["root_vel_error"]
    terminated = data["terminated"]
    did_term = np.any(terminated, axis = -1)
    ave_vels = np.zeros((window * 2,))
    num_samps = 0
    for c in range(36):
        if not did_term[frc_idx, c]:
            num_samps += 1
            ave_vels += vels[frc_idx, c, round(lower[c]):round(upper[c])]
    ave_vels /= num_samps
    return ave_vels

def plot_ft_debug_data(data):
    force = data["f"]
    for c in range(12):
        plt.plot(force[:, 0, c + 12], label=f"Comp {c}")
        plt.show()
    #torque = data["ff_tau"]
    #for c in range(torque.shape[2]):
    #    plt.plot(torque[:, 0, c], label=f"Torque Comp {c}")

plot_ft_debug_data(ft_debug_data)

ft_num_alive = get_num_alive(ft_data)
pd_num_alive = get_num_alive(pd_data)
plt.plot(frc_range, ft_num_alive / 36, label="FT")
plt.plot(frc_range, pd_num_alive / 36, label="PD")
plt.xlabel("Impulse Force Magnitude (N)")
plt.ylabel("Proporition of Trials Alive")
plt.title("Trial Survival vs Impulse Force Magnitude")
plt.legend()
plt.grid()
plt.show()

def plot_all_velocity_errors(ft_data, pd_data, frc_range, num_indices=None):
    if num_indices is None:
        num_indices = len(frc_range)
    n = num_indices
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for c in range(n):
        r = c // cols
        ax = axes[r, c % cols]
        ft_ave_vel = get_ave_vel_error(ft_data, frc_idx=c)
        pd_ave_vel = get_ave_vel_error(pd_data, frc_idx=c)
        x = np.arange(len(ft_ave_vel)) - len(ft_ave_vel) // 2
        x = x * 0.02  # assuming 50 Hz, convert to seconds
        ax.plot(x, ft_ave_vel, label="FT")
        ax.plot(x, pd_ave_vel, label="PD")
        ax.set_xlabel("Time around Impulse (s)")
        ax.set_title(f"Force: {frc_range[c]:.1f} N")
        ax.grid(True)
        if c == 0:
            ax.legend()

    # Hide any unused subplots
    total_axes = rows * cols
    for k in range(n, total_axes):
        r = k // cols
        axes[r, k % cols].axis('off')

    fig.suptitle("Averaged Velocity Error around Impulse Time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_all_velocity_errors(ft_data, pd_data, frc_range, num_indices=8)