from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
import jax
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax.numpy as jnp
from brax.training.acme import running_statistics
from playground.booster import joystick as joystick_pd
from playground.booster import joystick_ft as joystick
from playground.booster.config import ppo_params
from models.booster_t1_pgnd.booster_ids import ids
from rewards.mjx_col import get_forces
from lowctrl import ft_ref

ctrl_type = "pd"

if ctrl_type == "pd":
    env = joystick_pd.Joystick()
else:
    env = joystick.Joystick()

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
# JIT maqp.step by closing over non-array args (model, ids, flags)
@jax.jit
def jit_ft_step(mjx_state, act):
    return ft_ref.step(env._mjx_model, mjx_state, act, ids, is_mjx=True, debug=True)

def makeIFN():
    from brax.training.agents.ppo import networks as ppo_networks
    import functools
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

#jit_debug_step = jax.jit(eefpbc.debug_step)
from models.booster_t1_pgnd.booster_ids import ids
import numpy as np
array_dict = {}

rew = 0.0
from playground.booster.config import baseline_reward_names
def metrics_count(rew, metrics):
    for rname in baseline_reward_names:
        name = "reward/" + rname
        rew += metrics[name]
    return rew

if ctrl_type == "pd":
    dir = "training/pd_1"
else:
    dir = "training/t_7"

model_path = dir + "/walk_policy"
saved_params = model.load_params(model_path)

# print out stats to catch any NaNs/Infs early

inference_fn = makeIFN()(saved_params)
jit_inference_fn = jax.jit(inference_fn)


def test_force(force):
    rng = jax.random.PRNGKey(0)
    state = jit_reset(jax.random.PRNGKey(0))
    mj_model = mujoco.MjModel.from_xml_path('models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml')
    data = mujoco.MjData(mj_model)
    init_qpos = mj_model.keyframe('home').qpos
    data.qpos = init_qpos
    rew = 0.0
    def apply_wrench_to_body(state, body_id: int, wrench_6: jnp.ndarray):
        """Return a new state with xfrc_applied[body_id] = wrench_6 (Fx,Fy,Fz,Tx,Ty,Tz) in world frame."""
        d = state.data
        xfrc = d.xfrc_applied.at[body_id].set(wrench_6)
        d = d.replace(xfrc_applied=xfrc)
        return state.replace(data=d)

    for c in range(1000):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        wrench = jnp.array([force, 0., 0., 0., 0., 0.])
        zero_wrench = jnp.array([0., 0., 0., 0., 0., 0.])
        if c >= 500 and c <= 501:
            state = apply_wrench_to_body(state, 1, wrench)
        else:
            state = apply_wrench_to_body(state, 1, zero_wrench)
        state = jit_step(state, ctrl)
        rew = metrics_count(rew, state.metrics)

    for key in array_dict:
        np.savetxt(f"data/{key}.csv", array_dict[key].reshape(1000, -1), delimiter=",")
    print("Rollout precomputed")
    print("Total reward:", rew)
    return rew

arr = []
for c in range(20):
    print("Testing force:", c * 30)
    rew_a = test_force(c * 30)
    arr += [rew_a]

if ctrl_type == "pd":
    np.savetxt("data/pd_impulse_rew.csv", np.array(arr).reshape(-1, 1), delimiter=",")
else:
    np.savetxt("data/impulse_rew.csv", np.array(arr).reshape(-1, 1), delimiter=",")