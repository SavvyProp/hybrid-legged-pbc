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
        model_path = f"training/{type}/walk_policy"

    saved_params = model.load_params(model_path)

    inference_fn = makeIFN()(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    def get_alive_duration(metrics):
        return jnp.where(jnp.abs(metrics["reward/termination"]) < 0.01, 1, 0)
        
    def batched_sim_loop(states, rngs):
        def single_sim_loop(state, rng):
            def apply_frc_and_step(carry, c):
                state, rng = carry
                act_rng, rng = jax.random.split(rng)
                
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                
                # Apply force directly in the loop
                frc = jnp.array([c * 0.1, 0., 0.])
                force_traj = state.info["force_traj"]["forces"]
                height = force_traj.shape[0]
                new_force_traj = jnp.tile(frc[None, :], (height, 1))
                state.info["force_traj"]["forces"] = new_force_traj
                state = jit_step(state, ctrl)
                alive = get_alive_duration(state.metrics)
                
                return (state, rng), alive
            
            (_, _), alive_array = jax.lax.scan(apply_frc_and_step, (state, rng), jnp.arange(1000))
            return alive_array
        
        return jax.vmap(single_sim_loop)(states, rngs)
    
    # Batch reset and simulation
    batch_size = 32
    reset_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    sim_keys = jax.random.split(jax.random.PRNGKey(64), batch_size)
    
    initial_states = jax.vmap(jit_reset)(reset_keys)
    all_alive = batched_sim_loop(initial_states, sim_keys)
    
    # Sum across all batches
    num_alive = jnp.sum(all_alive, axis=0)

    np.savetxt(f"data/eval/{type}_constant_frc_alive.csv", np.array(num_alive).reshape(-1, 1), delimiter=",")

