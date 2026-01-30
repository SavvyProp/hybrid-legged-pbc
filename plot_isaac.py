import numpy as np
import matplotlib.pyplot as plt

joints = ['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll']
view_torques = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll"]

#ft_data = np.load("isaac_data/CMU_38/ft_eval_data.npz")
#pd_data = np.load("isaac_data/CMU_38/pd_eval_data.npz")
ft_spd_data = np.load("isaac_data/CMU_38/ft_spd_eval_data_nofrc.npz")
pd_spd_data = np.load("isaac_data/CMU_38/pd_spd_eval_data_nofrc.npz")
ft_debug_data = np.load("isaac_data/ft_debug_data.npz")

#frc_range = ft_data["forces"]

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
    label_lin = ["x", "y", "z"]
    length = 200
    x = np.arange(length) * 0.02
    for c in range(3):
        plt.plot(x, force[:length, 0, c + 12], label=f"Force {label_lin[c]}")
    plt.ylabel("Force (N)")
    plt.grid()
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
    label_lin = ["x", "y", "z"]
    x = np.arange(length) * 0.02
    for c in range(3):
        plt.plot(x, force[:length, 0, c + 15], label=f"Torque {label_lin[c]}")
    plt.ylabel("Torque (Nm)")
    plt.grid()
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
    return

def extract_torque_columns(torque: np.ndarray, joints_spec, joint_names):
    """
    Select 6 torque columns by joint names or indices.
    Returns (N, 6) array, resolved labels, and indices.
    """
    if torque.ndim != 2:
        raise ValueError("torque must be 2D (N, 23)")
    if len(joints_spec) != 6:
        raise ValueError("joints_spec must have length 6")
    indices, labels = [], []
    for item in joints_spec:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            indices.append(idx)
            labels.append(joint_names[idx] if 0 <= idx < len(joint_names) else f"Joint {idx}")
        else:
            name = str(item)
            if name not in joint_names:
                raise ValueError(f"Unknown joint name: {name}")
            idx = joint_names.index(name)
            indices.append(idx)
            labels.append(joint_names[idx])
    selected = torque[:, indices]
    return selected, labels, indices

def plot_torques(torque: np.ndarray, joints_in_view=None, dt: float = 0.02, joint_labels=None, title: str | None = None):
    """
    Plot 6 selected joint torque columns as 6 stacked subplots.
    torque: (N, 23)
    joints_in_view: 6 joint names or indices; if None, use module-level view_torques
    dt: sample period in seconds (default 0.02s => 50 Hz)
    joint_labels: optional list of 6 labels overriding default labels
    title: optional figure title
    """
    if joints_in_view is None:
        joints_spec = view_torques  # module-level default (names)
    else:
        joints_spec = joints_in_view

    sel, default_labels, indices = extract_torque_columns(torque, joints_spec, joints)
    labels = joint_labels if (joint_labels is not None and len(joint_labels) == 6) else default_labels

    time = np.arange(sel.shape[0]) * dt
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axes[i].plot(time, sel[:, i])
        axes[i].set_ylabel(labels[i] + " Torque (Nm)")
        axes[i].grid(True)
    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
#plot_ft_debug_data(ft_debug_data)
#plot_torques(ft_debug_data["ff_tau"][:, 0, :], title="Feedforward Torques from FT Controller")
#plot_ft_debug_data(ft_debug_data)

#ft_num_alive = get_num_alive(ft_data)
#pd_num_alive = get_num_alive(pd_data)
#plt.plot(frc_range, ft_num_alive / 36, label="FT")
#plt.plot(frc_range, pd_num_alive / 36, label="PD")
#plt.xlabel("Impulse Force Magnitude (N)")
#plt.ylabel("Proporition of Trials Alive")
#plt.title("Trial Survival vs Impulse Force Magnitude")
#plt.legend()
#plt.grid()
#plt.show()

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
        ax.set_ylabel("Velocity Error (m/s)")
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


body_names = ["Trunk", "Left Hand", "Right Hand", "Left Foot", "Right Foot"]

def plot_single_errors(ft_spd, pd_spd, name = "Linear Velocity (m/s)"):
    # Compute average across trials (axis=1), resulting in arrays of shape (N, 5)
    ft_ave_spd = np.mean(ft_spd, axis = 1)[0:400, :, :]
    pd_ave_spd = np.mean(pd_spd, axis = 1)[0:400, :, :]
    # Time axis in seconds assuming 50 Hz (dt = 0.02s)
    time = np.arange(ft_ave_spd.shape[0]) * 0.02
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    for i in range(5):
        ft_xy = np.linalg.norm(ft_ave_spd[:, i, 0:], axis=1)
        pd_xy = np.linalg.norm(pd_ave_spd[:, i, 0:], axis=1)
        axes[i].plot(time, ft_xy, label=f"FT {0}")
        axes[i].plot(time, pd_xy, label=f"PD {0}")
        axes[i].set_ylabel(name)
        axes[i].set_title(body_names[i])
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()


def plot_rootpos_err(ft_root_pos, pd_root_pos):
    # Compute average across trials (axis=1), resulting in arrays of shape (N, 3)
    ft_ave_pos = np.mean(ft_root_pos, axis = 1)
    pd_ave_pos = np.mean(pd_root_pos, axis = 1)
    # Time axis in seconds assuming 50 Hz (dt = 0.02s)
    time = np.arange(ft_ave_pos.shape[0]) * 0.02
    plt.plot(time, ft_ave_pos, label="FT pos err")
    plt.plot(time, pd_ave_pos, label="PD pos err")
    plt.xlabel("Time (s)")
    plt.ylabel("Root Position Error (m)")
    plt.title("Root Position Error Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

def table_data(ft_spd_data, pd_spd_data):
    ave_ft_vel = np.linalg.norm(ft_spd_data["root_vel_error"][..., 0:2], axis = -1)
    ave_pd_vel = np.linalg.norm(pd_spd_data["root_vel_error"][..., 0:2], axis = -1)
    ave_ft_spd = np.mean(ave_ft_vel, axis = (0, 1))
    ave_pd_spd = np.mean(ave_pd_vel, axis = (0, 1))
    print(ave_ft_spd, ave_pd_spd)
    trunk_err = (ave_pd_spd - ave_ft_spd) / ave_pd_spd
    print("Trunk Speed Error (%): ", trunk_err * 100)

    ave_ft_vel = np.linalg.norm(ft_spd_data["root_angvel_error"][..., 0:2], axis = -1)
    ave_pd_vel = np.linalg.norm(pd_spd_data["root_angvel_error"][..., 0:2], axis = -1)
    ave_ft_spd = np.mean(ave_ft_vel, axis = (0, 1))
    ave_pd_spd = np.mean(ave_pd_vel, axis = (0, 1))
    print(ave_ft_spd, ave_pd_spd)
    trunk_err = (ave_pd_spd - ave_ft_spd) / ave_pd_spd
    print("Trunk Angvel Error (%): ", trunk_err * 100)

def table_speed_error_stats(ft_npz=ft_spd_data, pd_npz=pd_spd_data, key: str = "root_vel_error"):
    """
    Print average speed error ± std for FT and PD, plus percent difference vs PD.
    Expects arrays with last dimension = 5 (bodies): time/envs/... x 5.
    key examples: "root_vel_error", "root_angvel_error" in the NPZ files.
    """
    if key not in ft_npz or key not in pd_npz:
        raise KeyError(f"Key '{key}' not found in provided NPZs")
    ft = ft_npz[key]
    pd = pd_npz[key]

    ft_flat = np.linalg.norm(ft[..., :], axis = -1)
    pd_flat = np.linalg.norm(pd[..., :], axis = -1)

    ft_flat = ft_flat.reshape(-1, ft_flat.shape[-1])  # (time*envs, 5)
    pd_flat = pd_flat.reshape(-1, pd_flat.shape[-1])  # (time*envs, 5)
    ft_mean = ft_flat.mean(axis=0)
    pd_mean = pd_flat.mean(axis=0)
    ft_std = ft_flat.std(axis=0) / np.sqrt(64)
    pd_std = pd_flat.std(axis=0) / np.sqrt(64)

    eps = 1e-8
    pct_diff = (ft_mean - pd_mean) / (np.where(np.abs(pd_mean) < eps, eps, pd_mean)) * 100.0

    header = (
        f"{'Body':<14} | {'FT Mean±Std':<22} | {'PD Mean±Std':<22} | {'% Diff vs PD':>12}\n" +
        "-" * 80
    )
    print(header)
    for i, name in enumerate(body_names):
        ft_str = f"{ft_mean[i]:.4f} ± {ft_std[i]:.4f}"
        pd_str = f"{pd_mean[i]:.4f} ± {pd_std[i]:.4f}"
        diff_str = f"{pct_diff[i]:.2f}%"
        print(f"{name:<14} | {ft_str:<22} | {pd_str:<22} | {diff_str:>12}")

    ft_all_mean = ft_mean.mean()
    pd_all_mean = pd_mean.mean()
    ft_all_std = ft_std.mean()
    pd_all_std = pd_std.mean()
    overall_pct = (ft_all_mean - pd_all_mean) / (pd_all_mean if abs(pd_all_mean) > eps else eps) * 100.0
    print("-" * 80)
    print(f"{'Overall':<14} | {ft_all_mean:.4f} ± {ft_all_std:.4f}   {'':<8}| {pd_all_mean:.4f} ± {pd_all_std:.4f}   {'':<8}| {overall_pct:>11.2f}%")

def plot_vel_fft(ft_vel, pd_vel):
    data_slice = ft_vel[:, 0, 2]
    X = np.fft.fft(data_slice)
    f = np.fft.fftfreq(data_slice.shape[0], 1/500)
    X2 = np.fft.fft(pd_vel[:, 0, 2])

    # Only take the positive frequencies
    mask = f >= 0
    plt.plot(f[mask], np.abs(X[mask]) * 2 / data_slice.shape[0])  # Normalize amplitude
    plt.plot(f[mask], np.abs(X2[mask]) * 2 / data_slice.shape[0])  # Normalize amplitude
    plt.title("Frequency Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

print("Speed Error Statistics (m/s):")
table_speed_error_stats()
print("Angular Velocity Error Statistics (rad/s):")
table_speed_error_stats(key="root_angvel_error")
#table_data(ft_spd_data, pd_spd_data)
#plot_single_errors(ft_spd_data["root_vel_error"], pd_spd_data["root_vel_error"], name="Linear Velocity Error (m/s)")
#plot_single_errors(ft_spd_data["root_angvel_error"], pd_spd_data["root_angvel_error"], name="Angular Velocity Error (rad/s)")
#plot_rootpos_err(ft_spd_data["root_orient_error"], pd_spd_data["root_orient_error"])
#plot_vel_fft(ft_spd_data["substep_root_vel"], pd_spd_data["substep_root_vel"])
#plot_all_velocity_errors(ft_data, pd_data, frc_range, num_indices=8)