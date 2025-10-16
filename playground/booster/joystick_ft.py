# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Booster T1."""

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
from rewards import rewards
from lowctrl.ft_ref import ctrl2logits, default_act
from lowctrl import ft_ref
from rewards.mjx_col import get_contacts, get_forces
from flax import linen as nn

def phys_step(
    model: mjx.Model,
    data: mjx.Data,
    filt_state,
    action: jax.Array,
    n_substeps: int = 1,
) -> tuple[mjx.Data, Any]:
  """Advance physics n_substeps times applying MAQP control each substep."""
  def single_step(carry, _):
    data, filt = carry
    ctrl, filt = ft_ref.step(model, data, action, bids.ids,
                           is_mjx=True,
                           filt_state=filt)
    data = data.replace(ctrl=ctrl)
    data = mjx.step(model, data)
    return (data, filt), None

  (data, filt_state), _ = jax.lax.scan(single_step,
                                       (data, filt_state),
                                       None,
                                       length=n_substeps)
  return data, filt_state

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      episode_length=500,
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
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy related rewards.
              torques=0.0,
              action_rate=-0.002,
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
              pbc_w=-1.0,
              maqp_cons=0.50,
              vel_def=1.0,
              vel_action_rate = -0.001,
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

from models.booster_t1_pgnd import booster_ids as bids

class Joystick(joystick.Joystick):
  """Track a joystick command."""

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
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)
    debug_dict = ft_ref.step(self._mjx_model, 
                           data, ft_ref.default_act(self.ids), self.ids, 
                           is_mjx=True, debug=True)
    for key in debug_dict:
      debug_dict[key] = jp.zeros_like(debug_dict[key])

    force_traj, rng = self.force_traj_gen.sample_force_traj(rng)
    force_lin = jp.zeros(3)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_u_act": jp.zeros(self.ids["ctrl_num"]),
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        #"motor_targets": jp.zeros(self.action_size),
        "motor_targets": default_act(self.ids),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "filtered_linvel": jp.zeros(3),
        "filtered_angvel": jp.zeros(3),
        "force_traj": force_traj,
        "force_lin": force_lin,
        "filt_state": ft_ref.make_filt_state(self.ids),
        "ft_dict": debug_dict,
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

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    data, lin_force = self.apply_pushes(state.data, state.info)
    state.info["force_lin"] = lin_force
    state = state.replace(data=data)
    # state = self._reset_if_outside_bounds(state)

    motor_targets = action #self._default_pose + action * self._config.action_scale
    data, filt_state = phys_step(
        self.mjx_model, state.data, state.info["filt_state"], motor_targets, self.n_substeps
    )

    state.info["filt_state"] = filt_state

    state.info["motor_targets"] = motor_targets

    linvel = self.get_local_linvel(data)
    state.info["filtered_linvel"] = (
        linvel * 1.0 + state.info["filtered_linvel"] * 0.0
    )
    angvel = self.get_gyro(data)
    state.info["filtered_angvel"] = (
        angvel * 1.0 + state.info["filtered_angvel"] * 0.0
    )

    #contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
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

    state.info["ft_dict"] = ft_ref.step(self._mjx_model, 
                           data, action, self.ids, 
                           is_mjx=True, debug=True)
    
    state.info["time"] += self.dt
    state.info["step"] += 1
    state.info["push_step"] += 1
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
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
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

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        info["ft_dict"]["f"],
        info["ft_dict"]["u"],
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
        info["force_lin"], # 3
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
    components = ft_ref.ctrl2components(data, action, self.ids)
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], info["filtered_linvel"]
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
        "pbc_w": self._cost_pbc_w(action, contact),
        "maqp_cons": self._reward_maqp_cons(data, info, action),
        "vel_def": self._reward_des_vel(components, info["command"]),
        "vel_action_rate": self._cost_vel_action_rate(action, info["last_act"]),
    }
    
  def _cost_vel_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    vel_act = ctrl2logits(act, self.ids)["des_com_vel"]
    angvel_act = ctrl2logits(act, self.ids)["des_com_angvel"]
    vel_last_act = ctrl2logits(last_act, self.ids)["des_com_vel"]
    angvel_last_act = ctrl2logits(last_act, self.ids)["des_com_angvel"]

    c1 = jp.sum(jp.square(vel_act - 
                          vel_last_act))
    c2 = jp.sum(jp.square(angvel_act - 
                          angvel_last_act))
    return c1 + c2
  
  def _reward_des_vel(self, components, lin_vel):
    des_vel_mag = jp.linalg.norm(components["des_com_vel"])
    des_angvel_mag = jp.linalg.norm(components["des_com_angvel"])
    des_vel_cap = 1.5
    des_angvel_cap = 3.0
    des_vel_rew = jp.clip(des_vel_mag - des_vel_cap,
                           min = 0.0, max = None)
    des_angvel_rew = jp.clip(des_angvel_mag - des_angvel_cap,
                           min = 0.0, max = None)
    rew_vel_lim = jp.exp(-(des_vel_rew + des_angvel_rew * 0.50))

    # vel tracking reward

    lin_vel_error = jp.sum(jp.square(lin_vel[:2] - components["des_com_vel"][:2]))
    linvel_rew = jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    return rew_vel_lim * 0.1 + linvel_rew
  
  def _reward_maqp_cons(self, data, info, action):
    debug_dict = info["ft_dict"]
    f = debug_dict["f"]
    l_true, r_true = get_forces(data, self.ids)
    #f = get_frc_pbc(self._mjx_model, data, action)
    #f = jp.zeros([24]) # Placeholder
    lf = f[0:3]
    rf = f[6:9]
    fac = 20000
    left_frc_error = jp.sum(jp.square(lf - l_true)) / fac
    right_frc_error = jp.sum(jp.square(rf - r_true)) / fac
    frc_error = left_frc_error + right_frc_error
    frc_rew = jp.exp(-frc_error)

    u = debug_dict["u"]
    tau_limits = self.ids["tau_limits"]
    torque_sum = jp.sum(jp.clip(jp.abs(tau_limits) - jp.abs(u), 
                                None, 0.0))
    torque_lim_rew = jp.exp(torque_sum / 50.0)

    # foot torque penalty method

    lt = jp.linalg.norm(f[3:6])
    rt = jp.linalg.norm(f[9:12])

    def foot_torque_penalty(tau):
      t2 = jp.clip(tau - 6.0, 0.0, None)
      return jp.exp(-t2 / 10.0)
    
    lt_rew = foot_torque_penalty(lt)
    rt_rew = foot_torque_penalty(rt)
    foot_torque_rew = (lt_rew + rt_rew) / 2.0

    # maqp torque rate penalty

    #u_action_rate = jp.sum(jp.square(u - info["last_u_act"]))
    #u_action_rate *= -0.000005
    #u_action_rate = jp.clip(u_action_rate, -0.30, 0.0)
    info["last_u_act"] = u

    total_rew = (torque_lim_rew * 0.30 + 
                 frc_rew * 0.10 + 
                 foot_torque_rew * 0.30)
                 #u_action_rate * 1.0)

    rew = jp.nan_to_num(total_rew, nan=-1.0, posinf=-1.0, neginf=-1.0)

    return rew
  
  # Tracking rewards.

  def _cost_pbc_w(self, action, contact):
    logits = ctrl2logits(action, bids.ids)
    return rewards.reward_pbc_w_leg_only(logits["w"], contact)
  
  @property
  def action_size(self) -> int:
    return ft_ref.default_act(self.ids).shape[0]