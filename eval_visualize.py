from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
import jax
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax.numpy as jnp
from brax.training.acme import running_statistics
#from playground.booster import joystick
from playground.booster import joystick_ft as joystick
from playground.booster.config import ppo_params
from models.booster_t1_pgnd.booster_ids import ids
from rewards.mjx_col import get_forces
from lowctrl import ft_ref
from eval import impulse_frc_viz

mj_model = mujoco.MjModel.from_xml_path('models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml')
data = mujoco.MjData(mj_model)
init_qpos = mj_model.keyframe('home').qpos
data.qpos = init_qpos

data_list = impulse_frc_viz.make_rollout("ftk")

viewer = mujoco.viewer.launch_passive(mj_model, data)
import time
while True:
    for c1 in range(len(data_list)):
        #print("=========================")
        #print(ctrl_list[c1])
        #print(nn_p_list[c1])
        #print(obs_list[c1])
        pipeline_state = data_list[c1]
        #print(state.info["phase"])
        time.sleep(0.02)
        mjx.get_data_into(data, mj_model, pipeline_state)
        viewer.sync()