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

def test_force(type):
    if type == "pd":
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
    
    dir = "training/ft_ext"

    
    if type == "pd":
        model_path = "training/pd_3/walk_policy"
    else:
        model_path = "training/ft_ext/walk_policy"

    saved_params = model.load_params(model_path)

    inference_fn = makeIFN()(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    def get_alive_duration(metrics):
        if jnp.abs(metrics["reward/termination"]) < 0.01:
            return 1
        else:
            return 0
        
    def apply_frc(state, frc):
        force_traj = state.info["force_traj"]["forces"]
        height = force_traj.shape[0]
        new_force_traj = jnp.tile(frc[None, :], (height, 1))
        state.info["force_traj"]["forces"] = new_force_traj
        return state

        
    def sim_loop(state, frc, rng):
        ctrl_list = []
        obs_list = []
        pipeline_state_list = []
        states = []
        alive = []
        
        for c in range(200):
            act_rng, rng = jax.random.split(rng)
            obs_list += [state.obs]
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = apply_frc(state, jnp.zeros([3]))
            if c == 120 or c == 121:
                state = apply_frc(state, frc)

            state = jit_step(state, ctrl)
            pipeline_state = state.data
            alive += [get_alive_duration(state.metrics)]
            ctrl_list += [ctrl]
            states += [state]
            pipeline_state_list += [pipeline_state]
        return alive
    
    success_rate = np.zeros([16])

    # Force range from 200 N to 1000 N in steps of 50 N
    frc_x = np.arange(200., 1000., 50.)
    for i in range(16):
        for c in range(16):
            state = jit_reset(jax.random.PRNGKey(c))
            frc = jnp.array([frc_x[i], 0., 0.])
            alive = sim_loop(state, frc, jax.random.PRNGKey(64 + c))
            success_rate[i] += alive[-1]

            print(f"Force {frc_x[i]} N, Trial {c}, Alive: {alive[-1]}")

    success_rate = success_rate / 16.0

    save = np.concatenate([frc_x[:, None], success_rate[:, None]], axis=1)

    np.savetxt(f"data/eval/{type}_impulse_frc_success_rate.csv", save, delimiter=",")