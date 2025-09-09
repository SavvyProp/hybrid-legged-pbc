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
from lowctrl.eefpbc import ctrl2components, test_pbcs
from models.booster_t1_pgnd.booster_ids import ids
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
    (des_pos, 
     qp_weights, 
     w, oriens, 
    ) = ctrl2components(act, ids)
    print(w)

dir = "training/test_pbc_8"

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

base_tau = np.zeros([1000, 23])
inv_tau = np.zeros([1000, 23])
pinv_tau = np.zeros([1000, 23])

for c in range(1000):
    act_rng, rng = jax.random.split(rng)
    obs_list += [state.obs]
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    base, inv, pinv = test_pbcs(env._mjx_model, state.data, ctrl)
    base_tau[c, :] = np.array(base)
    inv_tau[c, :] = np.array(inv)
    pinv_tau[c, :] = np.array(pinv)
    state = jit_step(state, ctrl)
    pipeline_state = state.data
    ctrl_list += [ctrl]
    states += [state]
    pipeline_state_list += [pipeline_state]

np.savetxt("data/base_tau.csv", base_tau, delimiter=",")
np.savetxt("data/inv_tau.csv", inv_tau, delimiter=",")
np.savetxt("data/pinv_tau.csv", pinv_tau, delimiter=",")


print("Rollout precomputed")
