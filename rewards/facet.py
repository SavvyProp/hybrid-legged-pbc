import jax.numpy as jnp
import jax

# Create an initial force trajectory of a list of forces and their duration
# and circle round

class ForceTrajectory:
    def __init__(self):

        self.samples = 20
        self.durations_min = 0.40
        self.durations_max = 0.80

        self.forces_min = jnp.array([0., 0., 0.])
        self.forces_max = jnp.array([20., 20., 5.])

    def sample_force_traj(self, key):

        key_force, key_sign, key_dur, key2 = jax.random.split(key, 4)

        forces = jax.random.uniform(key_force, 
                                    shape = (self.samples, 3),
                                    minval = 0.0,
                                    maxval = 1.0)
        forces = self.forces_min + forces * (self.forces_max - self.forces_min)
        sign = jax.random.bernoulli(key_sign, p=0.5, shape=(self.samples, 3))
        forces = jnp.where(sign, forces, -forces)

        durations = jax.random.uniform(key_dur,
                                       shape = (self.samples,),
                                       minval = self.durations_min,
                                       maxval = self.durations_max)
        # End timestamps (inclusive end of each segment)
        timestamps = jnp.cumsum(durations)
        total_time = timestamps[-1]
        start_times = jnp.concatenate([jnp.array([0.0], dtype=durations.dtype), timestamps[:-1]])

        return {
            "forces": forces,          # (N,3)
            "durations": durations,    # (N,)
            "start_times": start_times,  # (N,)
            "timestamps": timestamps,  # (N,) end times
            "total_time": total_time   # scalar
        }, key2
    
    def get_force_at_time(self, traj, t):
        """Return force(s) at time t (scalar or batch) with periodic wrap.
        t: float or (...,) array of times.
        Returns force with shape (..., 3).
        Parallelizable via vmap rules (pure JAX ops).
        """
        forces = traj["forces"]            # (N,3)
        end_ts = traj["timestamps"]        # (N,)
        total = traj["total_time"]         # scalar
        # Normalize times into one period
        t_mod = jnp.mod(t, total)
        # Flatten t for searchsorted (works for scalar or any shape)
        t_flat = t_mod.reshape(-1)
        # searchsorted gives insertion index where end_ts[idx-1] <= t < end_ts[idx]
        idx = jnp.searchsorted(end_ts, t_flat, side='right')
        idx = jnp.clip(idx, 0, forces.shape[0]-1)
        sel = forces[idx]  # (num_t,3)
        # Reshape back to original t shape + (3,)
        return sel.reshape(t_mod.shape + (3,))
    
# Command is given in global position, write a function to 
# use in _get_obs to move and rotate into the local frame of base

# Command is [des pos global (3,), K_p, K_d, m]

def get_ddot_x_ref(cmds, x_real, x_dot_real, f_ext):
    des_pos = cmds[:3]
    K_p = cmds[3]
    K_d = cmds[4]
    m = cmds[5]

    pos_err = des_pos - x_real
    vel_err = - x_dot_real

    f_spring = K_p * pos_err + K_d * vel_err

    f_net = f_spring + f_ext
    return f_net / m

def global_to_local(cmds, data, ids):
    base_body_ids = ids["base_id"]
    des_pos = cmds[:3]
    local_pos = data.xpos[base_body_ids]
    rotmat = data.xmat[base_body_ids].reshape(3, 3)

    forward_vec = data.site_xmat[ids["imu_id"]].T @ jnp.array([1., 0, 0])

    local_pos = des_pos - local_pos

    local_pos_rot = rotmat.T @ local_pos

    return jnp.concatenate([local_pos, des_pos, forward_vec, cmds[3:]], axis = 0)

# Track a rolling history of n timesteps of
# real vel, real pos, and des acc
# at each timestep do the integrals for velocity and rolls
# Order: roll, push true values, integrate accs, then integrate vels

class ReferenceTrajectory:
    def __init__(self):
        self.windows = jnp.array([8, 16, 32])
        self.hist_length = 48

    def make_rolling_history(self):
        return jnp.zeros([self.hist_length, 9 + self.windows.shape[0] * 6])
    
    def update_rolling_history(self, dt, hist, 
                               real_pos, real_vel, des_acc,
                               ):
        hist = jnp.roll(hist, shift = 1, axis = 0)
        hist = hist.at[0, :3].set(real_pos)
        hist = hist.at[0, 3:6].set(real_vel)
        hist = hist.at[0, 6:9].set(des_acc)

        windows = [8, 16, 32]

        for i in range(len(windows)):
            w = windows[i]
            sum_range = jnp.arange(w - 1)
            delta_v = jnp.sum(hist[sum_range, 6:9], axis = 0) * dt
            x_dot_ref = delta_v + hist[sum_range[-1], 3:6]
            hist = hist.at[0, 9 + 3*i:12 + 3*i].set(x_dot_ref)

            delta_x = jnp.sum(hist[sum_range, 9 + 3*i:12 + 3*i], axis = 0) * dt
            x_ref = delta_x + hist[sum_range[-1], 0:3]
            hist = hist.at[0, 9 + 3 * self.windows.shape[0] + 3*i:
                           12 + 3 * self.windows.shape[0] + 3*i].set(x_ref)
            
        return hist