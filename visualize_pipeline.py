from datetime import datetime
import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from matplotlib import pyplot as plt
from envs.booster_flatwalk_pd import FlatwalkEnv, metrics_dict
import os
import jax
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
#from playground.booster import joystick
from playground.booster import joystick_pbc as joystick
from playground.booster.config import ppo_params
from lowctrl.eefpbc import ctrl2components

env = joystick.Joystick()

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))

def makeIFN():
    from brax.training.agents.ppo import networks as ppo_networks
    import functools
    import networks.mlp as mlp
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )
    # normalize = running_statistics.normalize
    #normalize = lambda x, y: x
    normalize = running_statistics.normalize
    obs_size = env.observation_size
    ppo_network = network_factory(
        obs_size, env.action_size, preprocess_observations_fn=normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return make_inference_fn

def debug_eefpbc(act):
    from models.booster_t1_pgnd.booster_ids import ids
    (des_pos, gnd_acc, 
     qp_weights, tau_mix, 
     w, oriens, 
     base_acc, select) = ctrl2components(act, ids)
    print(w, tau_mix)

dir = "training/test_6"

model_path = dir + "/walk_policy"
saved_params = model.load_params(model_path)

# print out stats to catch any NaNs/Infs early

inference_fn = makeIFN()(saved_params)
jit_inference_fn = jax.jit(inference_fn)

rng = jax.random.PRNGKey(0)
mj_model = mujoco.MjModel.from_xml_path('models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml')
data = mujoco.MjData(mj_model)
init_qpos = mj_model.keyframe('home').qpos
data.qpos = init_qpos
print("Precomputing rollout")
pipeline_state_list = []
ctrl_list = []
obs_list = []
nn_p_list = []
states = []

for c in range(1000):
    act_rng, rng = jax.random.split(rng)
    obs_list += [state.obs]
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    debug_eefpbc(ctrl)
    #raw_action = ctrl[2 * HIDDEN_SIZE * DEPTH:]
    #nn_p, nn_d = raw_pd(raw_action)
    state = jit_step(state, ctrl)
    pipeline_state = state.data
    #nn_p_list += [nn_p]
    #nn_p_list += [nn_p]
    ctrl_list += [ctrl]
    states += [state]
    pipeline_state_list += [pipeline_state]


print("Rollout precomputed")

viewer = mujoco.viewer.launch_passive(mj_model, data)
import time
while True:
    for c1 in range(1000):
        print("=========================")
        #print(ctrl_list[c1])
        #print(nn_p_list[c1])
        #print(obs_list[c1])
        pipeline_state = pipeline_state_list[c1]
        state = states[c1]
        #print(state.info["phase"])
        #print(state.metrics)
        time.sleep(0.05)
        mjx.get_data_into(data, mj_model, pipeline_state)
        viewer.sync()
viewer.close()