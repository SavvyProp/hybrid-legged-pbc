
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
from playground.booster import track
from rewards import rewards
from lowctrl.ft_ref import ctrl2logits, default_act
from lowctrl import ft_ref
from rewards.mjx_col import get_contacts, get_forces
from flax import linen as nn
from models.booster_t1_pgnd import booster_ids as bids
from rewards.mjx_col import get_contact_dict
from motion_retarget import motion_retarget

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
              pbc_w=-1.0,
              maqp_cons=0.50,
              vel_def=1.0,
              vel_action_rate = -0.001,
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

class Track(track):
  def __init__(self, 
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,):
    super().__init__(config, config_overrides)

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    state_obs = super()._get_obs(data, info)
    f = info["ft_dict"]["f"],
    u = info["ft_dict"]["u"],
    state = state_obs["state"]
    priviledged_state = state_obs["privileged_state"]
    state = jp.hstack([state, f, u])
    priviledged_state = jp.hstack([priviledged_state, f, u])
    return {
        "state": state,
        "privileged_state": priviledged_state,
    }

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.3, maxval=0.3)
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
    force_traj["forces"] *= 0.05
    force_lin = jp.zeros(3)

    debug_dict = ft_ref.step(self._mjx_model, 
                           data, ft_ref.default_act(self.ids), self.ids, 
                           is_mjx=True, debug=True)
    for key in debug_dict:
      debug_dict[key] = jp.zeros_like(debug_dict[key])
    
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
        "ft_dict": debug_dict
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())


    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
  
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    
    data, lin_force = self.apply_pushes(state.data, state.info)
    state.info["force_lin"] = lin_force
    state = state.replace(data=data)
    
    # state = self._reset_if_outside_bounds(state)

    motor_targets = action #self._default_pose + action * self._config.action_scale
    data = phys_step(
        self.mjx_model, state.data, motor_targets, self.n_substeps, self.ids
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
    
    state.info["ft_dict"] = ft_ref.step(self._mjx_model, 
                           data, action, self.ids, 
                           is_mjx=True, debug=True)
    
    state.info["time"] += self.dt
    state.info["step"] += 1
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state