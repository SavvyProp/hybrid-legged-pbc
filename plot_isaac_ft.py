import numpy as np
import matplotlib.pyplot as plt

joints = ['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll']
view_torques = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll"]

#ft_data = np.load("isaac_data/CMU_38/ft_eval_data.npz")
#pd_data = np.load("isaac_data/CMU_38/pd_eval_data.npz")
ft_spd_data = np.load("isaac_data/CMU_17/ft_spd_eval_data_fixed_start.npz")
pd_spd_data = np.load("isaac_data/CMU_17/pd_spd_eval_data_fixed_start.npz")
ft_debug_data = np.load("isaac_data/CMU_17/ft_debug_data.npz")

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
        plt.plot(x, force[:length, 0, c + 18], label=f"Force {label_lin[c]}")
    plt.ylabel("Force (N)")
    plt.grid()
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
    label_lin = ["x", "y", "z"]
    x = np.arange(length) * 0.02
    for c in range(3):
        plt.plot(x, force[:length, 0, c + 21], label=f"Torque {label_lin[c]}")
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

def plot_torques(torque_output: np.ndarray, torque_ref: np.ndarray | None = None, applied_torque: np.ndarray | None = None, env: int = 0, joints_in_view=None, dt: float = 0.02, joint_labels=None, title: str | None = None, start_s: float = 0.0, end_s: float | None = None):
    """
    Plot 6 selected joint torque columns in a 3x2 grid over a selected time window.
    Overlays torque_output (ff_tau), optional torque_ref (tau_ref), and optional applied_torque on each subplot.

    Inputs can be either:
      - 3D arrays (T, E, D): use 'env' to select environment
      - 2D arrays (T, D): 'env' is ignored
    """
    # Normalize to 2D (T, D)
    def to_2d(arr: np.ndarray | None, name: str) -> np.ndarray | None:
        if arr is None:
            return None
        if arr.ndim == 3:
            T, E, D = arr.shape
            if not (0 <= env < E):
                raise ValueError(f"env must be in [0, {E-1}] for {name}")
            return arr[:, env, :]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"{name} must be 2D (T,D) or 3D (T,E,D)")

    out_2d = to_2d(torque_output, "torque_output")
    ref_2d = to_2d(torque_ref, "torque_ref")
    app_2d = to_2d(applied_torque, "applied_torque")

    # Select joints (columns)
    joints_spec = view_torques if joints_in_view is None else joints_in_view
    sel_out, default_labels, _ = extract_torque_columns(out_2d, joints_spec, joints)
    sel_ref = extract_torque_columns(ref_2d, joints_spec, joints)[0] if ref_2d is not None else None
    sel_app = extract_torque_columns(app_2d, joints_spec, joints)[0] if app_2d is not None else None
    labels = joint_labels if (joint_labels is not None and len(joint_labels) == 6) else default_labels

    # Time window
    T = sel_out.shape[0]
    start_idx = max(0, int(round(start_s / dt)))
    end_idx = T if end_s is None else min(T, int(round(end_s / dt)))
    if start_idx >= end_idx:
        raise ValueError("Invalid start/end times after conversion to indices")
    time = np.arange(start_idx, end_idx) * dt

    # Plot 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for i in range(6):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        ax.plot(time, sel_out[start_idx:end_idx, i], label="ff_tau")
        if sel_ref is not None:
            ax.plot(time, sel_ref[start_idx:end_idx, i], label="tau_ref", linestyle="--")
        if sel_app is not None:
            ax.plot(time, sel_app[start_idx:end_idx, i], label="applied_torque", linestyle=":")
        ax.set_title(labels[i])
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True)
        if i == 0:
            ax.legend()
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def plot_all_wrenches(data, env: int = 0, start_s: float = 0.0, end_s: float | None = None, dt: float = 0.02, show_legend: bool = True):
    """
    Plot all 4 wrenches (Left Hand, Right Hand, Left Foot, Right Foot) from data["f"].
    data["f"] shape: (T, E, 24) where 24 = 4 wrenches * 6 (Fx,Fy,Fz,Tx,Ty,Tz).
    - env: environment index along the second dimension.
    - start_s, end_s: start and end times in seconds (converted to indices via dt).
    - dt: sample period in seconds.
    - show_legend: whether to display legends for component overlays.
    Creates 8 subplots: 4 for forces (Fx,Fy,Fz overlaid) and 4 for torques (Tx,Ty,Tz overlaid).
    """
    f = data["f"]  # (T, E, 24)
    if f.ndim != 3 or f.shape[2] != 24:
        raise ValueError("data['f'] must be of shape (T, E, 24)")
    if not (0 <= env < f.shape[1]):
        raise ValueError(f"env must be in [0, {f.shape[1]-1}]")

    T = f.shape[0]
    start_idx = max(0, int(round(start_s / dt)))
    end_idx = T if end_s is None else min(T, int(round(end_s / dt)))
    if start_idx >= end_idx:
        raise ValueError("Invalid start/end times after conversion to indices")

    time = np.arange(start_idx, end_idx) * dt
    wrench_names = ["Left Hand", "Right Hand", "Left Foot", "Right Foot"]
    comp_labels = ["x", "y", "z"]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

    for i in range(4):
        base = i * 6  # indices: base+0..2 => force, base+3..5 => torque
        # Forces subplot
        ax_f = axes[i, 0]
        for c in range(3):
            ax_f.plot(time, f[start_idx:end_idx, env, base + c], label=comp_labels[c])
        ax_f.set_title(f"{wrench_names[i]} Force")
        ax_f.set_ylabel("Force (N)")
        ax_f.grid(True)

        # Torques subplot
        ax_t = axes[i, 1]
        for c in range(3):
            ax_t.plot(time, f[start_idx:end_idx, env, base + 3 + c], label=comp_labels[c])
            ax_t.set_title(f"{wrench_names[i]} Torque")
            ax_t.set_ylabel("Torque (Nm)")
            ax_t.grid(True)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    if show_legend:
        axes[0, 0].legend(title="Components")
        axes[0, 1].legend(title="Components")

    fig.tight_layout()
    plt.show()


def plot_vels(ft_debug_data, env, start_s: float = 1.0, end_s: float | None = 4.0, dt: float = 0.02):
    """
    Plot velocities in a 3x2 grid:
      Left column (rows x,y,z): real vs desired linear velocity components.
      Right column (rows x,y,z): real vs desired angular velocity components.

    Expects arrays of shape (T, E, 3) in ft_debug_data with keys:
      real_vel, des_com_vel, real_angvel, des_com_angvel
    """
    real_vel = ft_debug_data["real_vel"]
    des_vel = ft_debug_data["des_com_vel"]
    real_ang = ft_debug_data["real_angvel"]
    des_ang = ft_debug_data["des_com_angvel"]
    # validate shapes
    for name, arr in {"real_vel": real_vel, "des_com_vel": des_vel, "real_angvel": real_ang, "des_com_angvel": des_ang}.items():
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (T, E, 3)")
    T = real_vel.shape[0]
    if not (0 <= env < real_vel.shape[1]):
        raise ValueError(f"env must be in [0, {real_vel.shape[1]-1}]")
    start_idx = max(0, int(round(start_s / dt)))
    end_idx = T if end_s is None else min(T, int(round(end_s / dt)))
    if start_idx >= end_idx:
        raise ValueError("Invalid start/end times after conversion to indices")
    time = np.arange(start_idx, end_idx) * dt
    axis_labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for i in range(3):
        # Linear
        ax_lin = axes[i, 0]
        ax_lin.plot(time, real_vel[start_idx:end_idx, env, i], label="real")
        ax_lin.plot(time, des_vel[start_idx:end_idx, env, i], label="desired", linestyle="--")
        ax_lin.set_ylabel(f"{axis_labels[i]} lin (m/s)")
        ax_lin.set_title(f"Linear v{axis_labels[i]}")
        ax_lin.grid(True)
        # Angular
        ax_ang = axes[i, 1]
        ax_ang.plot(time, real_ang[start_idx:end_idx, env, i], label="real")
        ax_ang.plot(time, des_ang[start_idx:end_idx, env, i], label="desired", linestyle="--")
        ax_ang.set_ylabel(f"{axis_labels[i]} ang (rad/s)")
        ax_ang.set_title(f"Angular ω{axis_labels[i]}")
        ax_ang.grid(True)
        if i == 0:
            ax_lin.legend()
            ax_ang.legend()
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()

def plot_contact_prob(contact_w: np.ndarray, contact_ref: np.ndarray | None = None, env: int = 0, start_s: float = 0.0, end_s: float | None = None, dt: float = 0.02):
    """Plot contact probabilities (contact_w overlaid with optional contact_ref) for one environment.
    Arrays shape: (T, E, 4) => [LH, RH, LF, RF]. Creates 4 stacked subplots.
    """
    if contact_w.ndim != 3 or contact_w.shape[2] != 4:
        raise ValueError("contact_w must have shape (T, E, 4)")
    print(contact_ref.shape)
    if contact_ref is not None:
        if contact_ref.ndim != 3 or contact_ref.shape[2] != 4:
            raise ValueError("contact_ref must have shape (T, E, 4)")
        if contact_ref.shape[:2] != contact_w.shape[:2]:
            raise ValueError("contact_ref must match first two dims of contact_w")

    T = contact_w.shape[0]
    if not (0 <= env < contact_w.shape[1]):
        raise ValueError(f"env must be in [0, {contact_w.shape[1]-1}]")

    start_idx = max(0, int(round(start_s / dt)))
    end_idx = T if end_s is None else min(T, int(round(end_s / dt)))
    if start_idx >= end_idx:
        raise ValueError("Invalid start/end times after conversion to indices")

    time = np.arange(start_idx, end_idx) * dt
    labels = ["Left Hand", "Right Hand", "Left Foot", "Right Foot"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i in range(4):
        ax = axes[i]
        ax.plot(time, contact_w[start_idx:end_idx, env, i], label="contact")
        if contact_ref is not None:
            ax.plot(time, contact_ref[start_idx:end_idx, env, i], label="true contact", linestyle="--")
        ax.set_ylabel(labels[i])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.set_title(f"Contact Probability - {labels[i]}")
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()

plot_vels(ft_debug_data, env=0, start_s=1.0, end_s=5.0, dt=0.02)
plot_all_wrenches(ft_debug_data, env=0, start_s=1.0, end_s=5.0, dt=0.02, show_legend=True)
#plot_ft_debug_data(ft_debug_data)
print(ft_debug_data["applied_torque"].shape, ft_debug_data["contact_mask"].shape)
plot_contact_prob(ft_debug_data["contact_w"],
                  ft_debug_data["contact_mask"], env=0, start_s=0.0, end_s=5.0, dt=0.02)
plot_torques(ft_debug_data["ff_tau"], torque_ref=ft_debug_data["tau_ref"],
             applied_torque=ft_debug_data["applied_torque"], env=0, title="Feedforward Torques from FT Controller", start_s=0.0, end_s=5.0)
#plot_ft_debug_data(ft_debug_data)
# plot_contact_prob(ft_debug_data["contact_prob"], env=0, start_s=0.0, end_s=4.0, dt=0.02)
