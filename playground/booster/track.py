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
from playground.booster import base_pd as t1_base
from rewards.mjx_col import get_contact_dict
from rewards import facet
from lowctrl import pd
from motion_retarget import motion_retarget
from rewards import rot

def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int,
    ids: Dict[str, Any],
    traj_center: Optional[jax.Array] = None,
) -> mjx.Data:
  def single_step(data, _):
    ctrl = pd.step(model, data, action, ids, traj_center = traj_center)
    data = data.replace(ctrl = ctrl)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]

ppo_params = config_dict.create(
      num_timesteps=30_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=500,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      gae_lambda=0.95,
      learning_rate=1e-4,
      entropy_cost=0.001,
      num_envs=8192,
      batch_size=256,
      clipping_epsilon=0.2,
      max_grad_norm=1.0,
      num_resets_per_eval=1,
  )

ppo_params.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(1024, 512, 512, 256, 256),
        value_hidden_layer_sizes=(1024, 512, 512, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type = "normal",
        noise_std_type = "scalar",
        state_dependent_std = False
    )

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
              # base pose
              base_pos = 0.5,
              base_quat = 0.5,
              # body pos
              body_pos = 1.0,
              body_orien = 1.0,
              # body linvel
              body_linvel = 1.0,
              body_angvel = 1.0,
              # action rate
              action_rate = -1e-1,
              # Termination
              termination=-10.0,
              dof_pos_limits=-1.0,
          ),
          pos_sigma=0.3,
          ang_sigma=0.4,
          linvel_sigma=1.0,
          angvel_sigma=3.14
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

def parse_motion(task, num_bodies, num_joints):
  init_pose = np.genfromtxt(f"motions/{task}_initial_pose.csv", delimiter=",")
  task_array = np.genfromtxt(f"motions/{task}.csv", delimiter=",")
  body_poses = task_array[:, :num_bodies * 7].reshape(-1, num_bodies, 7)
  body_vels = task_array[:, num_bodies * 7: 
                         num_bodies * 7 + (num_bodies + 2 ) * 6].reshape(-1, num_bodies + 2, 6)
  qpos = task_array[:, num_bodies * 7 + (num_bodies + 2 ) * 6:-num_joints - 6]
  qvel = task_array[:, -num_joints - 6:]
  body_poses = jp.array(body_poses)
  body_vels = jp.array(body_vels)
  qpos = jp.array(qpos)
  qvel = jp.array(qvel)
  init_pose = jp.array(init_pose) 
  return body_poses, body_vels, qpos, qvel, init_pose
  

class Track(t1_base.T1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "CMU_02_05",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
    )
    self.task = task
    self.force_traj_gen = facet.ForceTrajectory()
    self._post_init()

  def override_config(self, config):
    config.episode_length = self.traj_length
    return config

  def _post_init(self) -> None:
    self.body_poses, self.body_vels, self.qpos_traj, self.qvel_traj, self.init_pose = parse_motion(self.task, 
                                                                   self.ids["num_bodies"],
                                                                   self.ids["ctrl_num"])
    self.traj_length = self.qpos_traj.shape[0]
    #self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    # Take self._init_q from first frame of qpos_traj
    self._init_q = self.init_pose
    
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    hip_indices = []
    hip_joint_names = ["Hip_Roll", "Hip_Yaw"]
    for side in ["Left", "Right"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7
        )
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    for side in ["Left", "Right"]:
      knee_indices.append(
          self._mj_model.joint(f"{side}_Knee_Pitch").qposadr - 7
      )
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array([
        1.0, 1.0,  # Head.
        0.1, 1.0, 1.0, 1.0,  # Left arm.
        0.1, 1.0, 1.0, 1.0,  # Right arm.
        1.0,  # Waist.
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Left leg.
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Right leg.
    ])
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._floor_geom_id = self._mj_model.geom("floor").id

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

  def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
    qpos = state.data.qpos
    new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
    new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
    qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
    state = state.replace(data=state.data.replace(qpos=qpos))
    return state

  def reset(self, rng: jax.Array) -> mjx_env.State:

    rng, key = jax.random.split(rng)
    jitter = jax.random.uniform(key, (23,), minval=-0.1, maxval=0.1)

    qpos = self._init_q
    qpos = qpos.at[7:].add(jitter)
    qvel = jp.zeros(self.mjx_model.nv)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2)
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

    force_traj, rng = self.force_traj_gen.sample_force_traj(rng)
    force_traj["forces"] *= 0.00
    force_lin = jp.zeros(3)

    info = {
        "rng": rng,
        "step": 0,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": jp.zeros(self.action_size),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Push related.
        "force_traj": force_traj,
        "force_lin": force_lin,
        "time": 0.0,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())


    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
  
  def apply_pushes(self, data: mjx.Data, info: dict[str, Any]):
    lin_force = self.force_traj_gen.get_force_at_time(
        info["force_traj"], info["time"])
    wrench = jp.hstack([lin_force, jp.zeros(3)])
    xfrc = data.xfrc_applied.at[self.ids["base_id"]].set(wrench)
    data = data.replace(xfrc_applied=xfrc)
    return data, lin_force

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    
    data, lin_force = self.apply_pushes(state.data, state.info)
    state.info["force_lin"] = lin_force
    state = state.replace(data=data)

    # state = self._reset_if_outside_bounds(state)

    traj_center = self.qpos_traj[state.info["step"], 7:]

    motor_targets = action #self._default_pose + action * self._config.action_scale
    data = step(
        self.mjx_model, state.data, motor_targets, self.n_substeps, self.ids, traj_center = traj_center
    )
    state.info["motor_targets"] = motor_targets

    contacts = get_contact_dict(data.contact, self.ids)
    #contacts = {
    #  "trunk": 0,
    #  "head": 0
    #}

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data, contacts)

    body_poses = motion_retarget.body_poses_in_base_mjx(self._mjx_model,
                                                        data,
                                                        self.ids["base_id"])

    rewards = self._get_reward(
        data, action, state.info, state.metrics, body_poses, done
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["time"] += self.dt
    state.info["step"] += 1
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
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


    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    ref_pos = self.qpos_traj[info["step"], 7:]
    ref_vel = self.qvel_traj[info["step"], :]

    # Pose track error.
    current_pose = data.qpos[:7]
    ref_pose = self.qpos_traj[info["step"], :7]

    pos_error = ref_pose[:3] - current_pose[:3]
    orien_mat = rot.rot_error_matrix(ref_pose[3:7], current_pose[3:7])

    pose_track_error = jp.hstack([
        pos_error,
        orien_mat[:, :2].flatten()
    ])
    
    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        pose_track_error,
        ref_pos,
        ref_vel,
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
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
        pose_track_error,
        ref_pos,
        ref_vel,
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
      body_poses: jax.Array,
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "termination": self._cost_termination(done),
        "base_pos": self._reward_base_pos(data, info),
        "base_quat": self._reward_base_quat(data, info),
        "body_pos": self._reward_body_pos(body_poses, info),
        "body_orien": self._reward_body_orien(body_poses, info),
        "body_linvel": self._reward_body_linvel(data, info),
        "body_angvel": self._reward_body_angvel(data, info),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }
  
  def _get_termination(self, data, contacts) -> jax.Array:
    current_base_pos = data.qpos[:3]
    ref_base_pos = self.qpos_traj[0, :3]
    dist = jp.linalg.norm(current_base_pos - ref_base_pos)
    out_of_bounds = dist > 0.3
    contact_termination = contacts["trunk"] | contacts["head"]
    return (
        out_of_bounds | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )
  
  def _reward_base_pos(self, data, info):
    reference_base_pos = self.qpos_traj[info["step"], :3]
    current_base_pos = data.qpos[:3]
    err =jp.sum(jp.square(reference_base_pos - current_base_pos))
    rew = jp.exp(-err / (self._config.reward_config.pos_sigma ** 2))
    return rew
  
  def _reward_base_quat(self, data, info):
    reference_base_quat = self.qpos_traj[info["step"], 3:7]
    current_base_quat = data.qpos[3:7]
    err = motion_retarget.quat_error_magnitude(
      reference_base_quat, current_base_quat
    ) ** 2
    rew = jp.exp(-err / (self._config.reward_config.ang_sigma ** 2))
    return rew
  
  def _reward_body_pos(self, body_poses, info):
    reference_body_pose = self.body_poses[info["step"], :, :3]
    current_body_pos = body_poses[:, :3]
    err = jp.sum(jp.square(reference_body_pose - current_body_pos), axis = -1)
    mean_err = jp.mean(err)
    rew = jp.exp(-mean_err / (self._config.reward_config.pos_sigma **2))
    return rew
  
  def _reward_body_orien(self, body_poses, info):
    reference_body_quat = self.body_poses[info["step"], :, 3:7]
    current_body_quat = body_poses[:, 3:7]
    err = motion_retarget.quat_error_magnitude(
      reference_body_quat, current_body_quat)
    mean_err = jp.mean(err ** 2)
    rew = jp.exp(-mean_err / (self._config.reward_config.ang_sigma **2))
    return rew
  
  def _reward_body_linvel(self, data, info):
    reference_body_vel = self.body_vels[info["step"], :, :3]
    body_linvel = data.cvel[:, :3]
    err = jp.sum(jp.square(reference_body_vel - body_linvel), axis =-1)
    mean_err = jp.mean(err)
    rew = jp.exp(-mean_err / (self._config.reward_config.linvel_sigma **2))
    return rew
  
  def _reward_body_angvel(self, data, info):
    reference_body_angvel = self.body_vels[info["step"], :, 3:6]
    body_angvel = data.cvel[:, 3:6]
    err = jp.sum(jp.square(reference_body_angvel - body_angvel), axis =-1)
    mean_err = jp.mean(err)
    rew = jp.exp(-mean_err / (self._config.reward_config.angvel_sigma **2))
    return rew

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.sum(jp.square(act - last_act))
    return c1
  
  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done
  
  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)