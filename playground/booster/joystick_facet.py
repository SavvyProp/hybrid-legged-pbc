
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
#from mujoco_playground._src.locomotion.t1 import base as t1_base
from mujoco_playground._src.locomotion.t1 import t1_constants as consts
from playground.booster import joystick

from rewards.mjx_col import get_contacts
from rewards import facet

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      episode_length=1000,
      action_repeat=1,
      action_scale=1.0,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin=1.0,
              tracking_ang_vel=0.5,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy related rewards.
              torques=0.0,
              action_rate=-0.001,
              energy=0.0,
              dof_acc=-5e-7,
              dof_vel=-5e-5,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              stand_still=0.0,
              alive=0.25,
              termination=-100.0,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.1,
              dof_pos_limits=-1.0,
              pose=-1.0,
              feet_distance=-1.0,
              collision=-1.0,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.12,
          base_height_target=0.665,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 1.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.8, 0.8],
      ang_vel_yaw=[-1.0, 1.0],
      impl="jax",
      nconmax=8 * 8192,
      njmax=80,
  )


class Joystick(joystick.Joystick):
  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if task.startswith("rough"):
      config.nconmax = 100 * 8192
      config.njmax = 500
    super().__init__(
      task = task,
      config = config,
      config_overrides=config_overrides
    )
    self.force_traj_gen = facet.ForceTrajectory()
    self.reference_traj = facet.ReferenceTrajectory()

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (23,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = self.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.25, 1.75)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    metacmd = self.sample_metacommand(cmd_rng)
    cmd = self.sample_command(metacmd)

    # Sample push interval.

    force_traj, rng = self.force_traj_gen.sample_force_traj(rng)
    kin_hist = self.reference_traj.make_rolling_history()

    info = {
        "rng": rng,
        "step": 0,
        "metacommand": metacmd,
        "command": cmd,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": jp.zeros(self.action_size),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "filtered_linvel": jp.zeros(3),
        "filtered_angvel": jp.zeros(3),
        "force_traj": force_traj,
        "kin_hist": kin_hist,
        "time": 0.0,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    #contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
    contact = get_contacts(data.contact, self.ids)

    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
  
  def sample_metacommand(self, rng: jax.Array, data = None) -> jax.Array:
    # sample a meta command which comes in the form of (glob pos), (current pos offset), (use offset), (kp, kd, m), angvel_yaw
    base_body_id = self.ids["base_id"]
    if data is None:
      base_pos = jp.zeros(3)
    else:
      base_pos = data.xpos[base_body_id]
    rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8, rng9 = jax.random.split(rng, 9)

    t_exp = 5.0

    lin_pos_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_pos_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    des_pos = base_pos[:2] + jp.array([lin_pos_x, lin_pos_y]) * t_exp


    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    k_p = jax.random.uniform(rng4,
                             minval=5.0,
                             maxval=25.0)
    k_d = 1.9 * jp.sqrt(k_p)
    m = jax.random.uniform(rng7,
                           minval=0.8,
                           maxval=1.2) * self.ids["mass"]


    lin_vel_x = jax.random.uniform(
        rng5, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng6, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    des_pos_offset = jp.array([lin_vel_x, lin_vel_y]) * k_d / k_p

    # prob of pos command, vel command, or both zero

    choice = jax.random.choice(rng8, 3, p=jp.array([0.4, 0.4, 0.2]))

    metapos_pos = jp.hstack([des_pos, jp.array([0.0, 0.0, 0.0])])

    metapos_vel = jp.hstack([jp.array([0.0, 0.0]), des_pos_offset, jp.array([1.0])])

    metapos_zero = jp.hstack([jp.array([0.0, 0.0, 0.0, 0.0, 1.0])])

    metapos = jp.where(choice == 0, metapos_pos, jp.where(choice == 1, metapos_vel, metapos_zero))

    metacmd = jp.hstack([metapos, jp.array([k_p, k_d, m, ang_vel_yaw])])

    return metacmd

  
  def sample_command(self, metacmd, data = None) -> jax.Array:
    base_body_id = self.ids["base_id"]
    if data is None:
      base_pos = jp.zeros(3)
    else:
      base_pos = data.xpos[base_body_id]
    
    z = 0.665
    des_pos = metacmd[:2]
    des_pos_offset = metacmd[2:4]
    use_offset = metacmd[4]

    set_pos = des_pos + des_pos_offset + use_offset * base_pos[:2]

    k_p = metacmd[5]
    k_d = metacmd[6]
    m = metacmd[7]
    ang_vel_yaw = metacmd[8]

    cmd = jp.hstack([set_pos, jp.array([z]), jp.array([k_p, k_d, m, ang_vel_yaw])])
    
    return cmd
  
  def apply_pushes(self, data: mjx.Data, info: dict[str, Any]):
    lin_force = self.force_traj_gen.get_force_at_time(
        info["force_traj"], info["time"])
    wrench = jp.hstack([lin_force, jp.zeros(3)])
    xfrc = data.xfrc_applied.at[self.ids["base_id"]].set(wrench)
    data = data.replace(xfrc_applied=xfrc)
    return data, lin_force
  
  def update_kin_hist(self, data: mjx.Data, info: dict[str, Any], f_ext):
    x_pos = data.xpos[self.ids["base_id"]]
    x_vel = data.qvel[:3]
    acc = facet.get_ddot_x_ref(
        info["command"], x_pos, x_vel, f_ext
    )
    kin_hist = self.reference_traj.update_rolling_history(
        self.dt, info["kin_hist"], x_pos, x_vel, acc
    )
    info["kin_hist"] = kin_hist
    return info

  
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )

    # Code to do force pushes

    data, lin_force = self.apply_pushes(state.data, state.info)
    state = state.replace(data=data)
    
    motor_targets = action #self._default_pose + action * self._config.action_scale
    data = joystick.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps, self.ids
    )
    # Code to update kin hist

    self.update_kin_hist(state.data, state.info, lin_force)

    state.info["motor_targets"] = motor_targets

    linvel = self.get_local_linvel(data)
    state.info["filtered_linvel"] = (
        linvel * 1.0 + state.info["filtered_linvel"] * 0.0
    )
    angvel = self.get_gyro(data)
    state.info["filtered_angvel"] = (
        angvel * 1.0 + state.info["filtered_angvel"] * 0.0
    )

    contact = get_contacts(data.contact, self.ids)

    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["time"] += self.dt
    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["phase"] = jp.where(
        jp.linalg.norm(state.info["command"]) > 0.01,
        state.info["phase"],
        jp.ones(2) * jp.pi,
    )
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["metacommand"] = jp.where(
        state.info["step"] > 500,
        self.sample_metacommand(cmd_rng, data = data),
        state.info["metacommand"],
    )
    state.info["command"] = self.sample_command(state.info["metacommand"], data = data)
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  
  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    cmd = facet.global_to_local(info["command"], data, self.ids)

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        cmd,  # 7
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        phase,
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }
  
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking rewards.
        "tracking_lin": self._reward_tracking(
            info
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], info["filtered_angvel"]
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(info["filtered_linvel"]),
        "ang_vel_xy": self._cost_ang_vel_xy(info["filtered_angvel"]),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data, info),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "dof_vel": self._cost_dof_vel(data.qvel[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": self._cost_collision(data),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
        "feet_distance": self._cost_feet_distance(data, info),
    }
  
  def _reward_tracking(
      self, info: dict[str, Any]
  ) -> jax.Array:
    xpos = info["kin_hist"][0, :3]
    xvel = info["kin_hist"][0, 3:6]
    windows = self.reference_traj.windows
    rew_sum = 0.0
    for i in range(windows.shape[0]):
      x_dot_ref = info["kin_hist"][0, 9 + 3*i:12 + 3*i]
      x_ref = info["kin_hist"][0, 9 + 3 * windows.shape[0] + 3*i:
                           12 + 3 * windows.shape[0] + 3*i]
      pos_mag = jp.sum(jp.square(xpos - x_ref))
      vel_mag = jp.sum(jp.square(xvel - x_dot_ref))
      pos_err_rew = jp.exp(
        -pos_mag / 0.25
      )
      vel_err_rew = jp.exp(
        -vel_mag / 0.25
      ) - 0.5 * vel_mag
      rew_sum += pos_err_rew + vel_err_rew * 2.0
    return rew_sum / windows.shape[0]
  
  def halt_cmd(self, cmd):
    return 0.0