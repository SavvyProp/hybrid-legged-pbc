import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
from lowctrl.maqp import default_act
from lowctrl import maqp
from models.booster_t1_pgnd import booster_ids as bids
import jax
import time

model = mujoco.MjModel.from_xml_path('models/booster_t1/flat_scene.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

init_qpos = model.keyframe('home').qpos
data.qpos = init_qpos
mujoco.mj_step(model, data) # sim first step

init_com = data.subtree_com[0].copy()

act = default_act(bids.ids)
ctrl = jnp.zeros([23])


state = mjx.put_data(model, data)
t = 0
@jax.jit
def step_fn(mjx_model, mjx_state, init_com, t):
    #act = default_act(bids.ids)
    act = maqp.test_act_move_com(init_com, mjx_state, t, bids.ids)
    ctrl = maqp.step(mjx_model, mjx_state, act, bids.ids, is_mjx = True)
    data = mjx_state.replace(ctrl=ctrl)
    data = mjx.step(mjx_model, data)
    return data

# Add a benchmark utility to time step_fn and its vmapped version over a batch

def benchmark_step(mjx_model, mjx_state, init_com, *, batch_size: int = 1000, iters: int = 10, t0: int = 0):
    """Benchmark single-step and vmapped-step runtimes.
    Prints total and per-step/per-item timings with JIT warmup and device sync.
    """
    # Warmup single step compile
    out = step_fn(mjx_model, mjx_state, init_com, t0)
    _ = jax.block_until_ready(out.qpos)

    # Time single-step
    t_start = time.perf_counter()
    data_single = mjx_state
    for k in range(iters):
        data_single = step_fn(mjx_model, data_single, init_com, t0 + k)
    _ = jax.block_until_ready(data_single.qpos)
    t_single = time.perf_counter() - t_start

    # Prepare vmapped function (map over t only; model/state/init_com are broadcast)
    vmapped = jax.jit(jax.vmap(step_fn, in_axes=(None, None, None, 0)))

    # Warmup vmapped compile
    t_vec = jnp.arange(batch_size, dtype=jnp.int32) + t0
    out_b = vmapped(mjx_model, mjx_state, init_com, t_vec)
    _ = jax.block_until_ready(out_b.qpos)

    # Time vmapped call
    t_start = time.perf_counter()
    out_b = vmapped(mjx_model, mjx_state, init_com, t_vec)
    _ = jax.block_until_ready(out_b.qpos)
    t_vmapped = time.perf_counter() - t_start

    print("Single-step: total_s=", t_single, " per_step_s=", t_single / iters)
    print("Vmap batch:", batch_size, " total_s=", t_vmapped, " per_item_s=", t_vmapped / batch_size)


if __name__ == "__main__":
    benchmark_step(mjx_model, state, init_com, batch_size=1000, iters=10, t0=0)

