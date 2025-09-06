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
"""Base classes for T1."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.t1 import t1_constants as consts

from models.booster_t1_pgnd import booster_ids as bids
from lowctrl import eefpbc

def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  def single_step(data, _):
    ctrl = eefpbc.step(model, data, action, bids.ids)
    #print(ctrl)
    #print(data.ctrl)
    data = data.replace(ctrl = ctrl)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]

def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
  """Initialize MJX Data."""
  data = mjx.make_data(
      model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
  )
  if qpos is not None:
    data = data.replace(qpos=qpos)
  if qvel is not None:
    data = data.replace(qvel=qvel)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)
  if act is not None:
    data = data.replace(act=act)
  if mocap_pos is not None:
    data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
  if mocap_quat is not None:
    data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
  return data

class T1Env(mjx_env.MjxEnv):
  """Base class for T1 environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._mj_model = mujoco.MjModel.from_xml_path(
            "models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml"
        )
    self._mj_model.opt.timestep = self.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path
    self.ids = bids.ids

  # Sensor readings.

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GRAVITY_SENSOR}"
    )

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}"
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}"
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}"
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}"
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(self.mj_model, data, f"{consts.GYRO_SENSOR}")

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    jnt_num = self.ids["ctrl_num"]
    eef_num = self.ids["eef_num"]
    return jnt_num + eef_num * 4 + 2

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
  
  def step(self, state, action):
    new_data = step(self.mjx_model, state, action, self.n_substeps)
    return new_data
  
  def make_data(self, mj_model, **kwargs):
    return make_data(mj_model, **kwargs)
