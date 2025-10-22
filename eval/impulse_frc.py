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
        model_path = f"training/{type}/walk_policy"

    saved_params = model.load_params(model_path)

    inference_fn = makeIFN()(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    def get_alive_duration(metrics):
        return jnp.where(jnp.abs(metrics["reward/termination"]) < 0.01, jnp.int32(1), jnp.int32(0))
        
    def apply_frc(state, frc):
        force_traj = state.info["force_traj"]
        height = force_traj["forces"].shape[0]
        new_forces = jnp.tile(frc[None, :], (height, 1))
        new_force_traj = {**force_traj, "forces": new_forces}
        new_info = {**state.info, "force_traj": new_force_traj}
        return state.replace(info=new_info)

    # Vectorized helpers
    v_reset = jax.vmap(jax.jit(env.reset))
    v_infer = jax.vmap(jit_inference_fn, in_axes=(0, 0))
    v_step = jax.vmap(jax.jit(env.step), in_axes=(0, 0))
    v_apply_frc = jax.vmap(apply_frc, in_axes=(0, 0))
    v_alive = jax.vmap(get_alive_duration)
        
    def sim_loop(state, frc, rng):
        # choose random start step in [100, 149] and apply for 2 steps
        force_rng, rng = jax.random.split(rng)
        fstart = int(jax.random.randint(force_rng, (), 100, 150))
        ctrl_list = []
        obs_list = []
        pipeline_state_list = []
        states = []
        alive = []
        
        for c in range(200):
            act_rng, rng = jax.random.split(rng)
            obs_list += [state.obs]
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            # apply external force for exactly two steps: fstart and fstart+1
            if c == fstart:
                state = apply_frc(state, frc)
            else:
                state = apply_frc(state, jnp.zeros([3]))

            state = jit_step(state, ctrl)
            pipeline_state = state.data
            alive += [get_alive_duration(state.metrics)]
            ctrl_list += [ctrl]
            states += [state]
            pipeline_state_list += [pipeline_state]
        return alive

    @jax.jit
    def sim_loop_batched(states, rngs, frc):
        # per-env random start steps in [100, 149]
        keys2 = jax.vmap(lambda k: jax.random.split(k, 2))(rngs)
        force_keys = keys2[:, 0]
        rngs = keys2[:, 1]
        fstart = jax.vmap(lambda k: jax.random.randint(k, (), 100, 150))(force_keys)  # (num_envs,)

        def body(carry, step_idx):
            states, rngs = carry
            sub = jax.vmap(lambda k: jax.random.split(k, 2))(rngs)
            act_rng = sub[:, 0]
            rngs = sub[:, 1]

            # policy
            ctrl, _ = v_infer(states.obs, act_rng)

            # per-env force for exactly two steps: fstart and fstart+1
            apply_mask = (step_idx == fstart)  # (num_envs,)
            frc_batch = frc[None, :] * apply_mask.astype(frc.dtype)[:, None]
            states = v_apply_frc(states, frc_batch)

            # step envs
            states = v_step(states, ctrl)

            # alive metric per env
            alive = v_alive(states.metrics).astype(jnp.int32)
            return (states, rngs), alive

        (_, _), alive_hist = jax.lax.scan(body, (states, rngs), jnp.arange(300))
        return alive_hist  # (T, num_envs)
    
    success_rate = np.zeros([16])

    # Force range from 200 N to 1000 N in steps of 50 N
    frc_x = np.arange(400., 2000., 100.)
    num_envs = 32

    # Prepare batched initial states and rngs (same resets per force level)
    reset_keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    states0 = v_reset(reset_keys)
    base_keys0 = jax.random.split(jax.random.PRNGKey(64), num_envs)

    for i in range(16):
        frc = jnp.array([frc_x[i], 0., 0.])
        alive_hist = sim_loop_batched(states0, base_keys0, frc)
        alive_last = alive_hist[-1]  # (num_envs,) int32 in {0,1}
        success_rate[i] = float(jnp.mean(alive_last))
        print(f"Force {frc_x[i]:.1f} N: success rate {success_rate[i]:.3f}")

    save = np.stack([frc_x, success_rate], axis=1)

    from pathlib import Path
    out_path = Path(f"data/eval/{type}_impulse_frc_1dt_success_rate.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, save, delimiter=",", header="force,success_rate", comments="")