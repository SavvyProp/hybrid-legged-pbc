from brax import envs
from envs.booster_flatwalk_pd import FlatwalkEnv as FlatwalkEnvPD
from envs.booster_flatwalk_pbc import FlatwalkEnv as FlatwalkEnvPBC
import jax
import jax.numpy as jnp
import time
from pipelines.booster_eefpbc import default_act

def env_step_runtime(ENV, name, ctrl):
    envs.register_environment(name, ENV)
    env = envs.get_environment(name)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0))
    for c in range(100):
        st = time.perf_counter()
        state = jit_step(state, ctrl)
        et = time.perf_counter() - st
        print(f"Step {c}, Time taken: {et:.4f} seconds")

def env_step_runtime_batched(ENV, name, ctrl, batch_size=1000):
    """Batched runtime test: vmap + jit over batch dimension."""
    envs.register_environment(name, ENV)
    env = envs.get_environment(name)

    # Build batched initial state by vmapping reset over different keys
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size)
    batched_reset = jax.jit(jax.vmap(env.reset))
    state = batched_reset(keys)

    # Prepare vmapped step (state, action) -> new_state
    step_fn = jax.vmap(env.step, in_axes=(0, 0))
    jit_step_batched = jax.jit(step_fn)

    # Timing loop
    for c in range(100):
        st = time.perf_counter()
        state = jit_step_batched(state, ctrl)
        et = time.perf_counter() - st
        print(f"[BATCH {batch_size}] Step {c}, Time taken: {et:.4f} seconds")

print("PD")
ctrl_pd = jnp.zeros([46])
env_step_runtime(FlatwalkEnvPD, 'digit_flatwalk_pd', ctrl_pd)

# Batched PD
BATCH_SIZE = 1000
ctrl_pd_batch = jnp.zeros([BATCH_SIZE, 46])
print("PD Batched")
env_step_runtime_batched(FlatwalkEnvPD, 'digit_flatwalk_pd_batched', ctrl_pd_batch, batch_size=BATCH_SIZE)

print("PBC")

ctrl_pbc = default_act()
env_step_runtime(FlatwalkEnvPBC, 'digit_flatwalk_pbc', ctrl_pbc)

# Batched PBC
ctrl_pbc_batch = jnp.tile(ctrl_pbc[None, :], (BATCH_SIZE, 1))
print("PBC Batched")
env_step_runtime_batched(FlatwalkEnvPBC, 'digit_flatwalk_pbc_batched', ctrl_pbc_batch, batch_size=BATCH_SIZE)
