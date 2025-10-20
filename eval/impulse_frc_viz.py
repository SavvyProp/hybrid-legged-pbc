from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
#from playground.booster import joystick
#from playground.booster import joystick_ft as joystick
from playground.booster.config import ppo_params
from models.booster_t1_pgnd.booster_ids import ids
from lowctrl import ft_ref

def make_rollout(type):
    if type[0:2] == "pd":
        from playground.booster import joystick
        env = joystick.Joystick()
    else:
        from playground.booster import joystick_ft as joystick
        env = joystick.Joystick()

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    #state = jit_reset(jax.random.PRNGKey(0))


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
    
    if type[0:2] == "pd":
        model_path = "training/pd_3/walk_policy"
    else:
        model_path = "training/ftk3/walk_policy"

    saved_params = model.load_params(model_path)

    inference_fn = makeIFN()(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    def apply_frc(state, frc):
        force_traj = state.info["force_traj"]
        height = force_traj["forces"].shape[0]
        new_forces = jnp.tile(frc[None, :], (height, 1))
        new_force_traj = {**force_traj, "forces": new_forces}
        new_info = {**state.info, "force_traj": new_force_traj}
        return state.replace(info=new_info)
    
    ctrl_list = []
    obs_list = []
    data_list = []
    states = []
    
    state = jit_reset(jax.random.PRNGKey(69))

    data = state.data.replace(qpos = env.mj_model.keyframe('home').qpos)
    state = state.replace(data = data)

    command = jnp.array([0.5, 0.0, -0.1])

    rng = jax.random.PRNGKey(169)
    for c in range(500):
        state.info["command"] = command
        act_rng, rng = jax.random.split(rng)
        obs_list += [state.obs]
        ctrl, _ = jit_inference_fn(state.obs, act_rng)

        if c == 150:
            frc = jnp.array([0., 1200., 0.])
        else:
            frc = jnp.array([0., 0., 0.])

        state = apply_frc(state, frc)

        state = jit_step(state, ctrl)
        data_c = state.data
        ctrl_list += [ctrl]
        states += [state]
        data_list += [data_c]

    return data_list