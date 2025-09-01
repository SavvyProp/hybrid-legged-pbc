import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
from pipelines.booster_eefpbc import multistep, init, default_act
import jax

model = mujoco.MjModel.from_xml_path('models/booster_t1/flat_scene.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

init_qpos = model.keyframe('home').qpos
data.qpos = init_qpos
mujoco.mj_step(model, data) # sim first step

act = default_act()
ctrl = jnp.zeros([23])


state = init(
    mjx_model,
    jnp.array(init_qpos),
    jnp.zeros(model.nv),
    jnp.zeros([0,]),
    ctrl,
)

@jax.jit
def step_fn(mjx_model, mjx_state):
    act = default_act()
    mjx_state = multistep(mjx_model, mjx_state, 1, act)
    #state = mjx.step(mjx_model, state)
    return mjx_state


viewer = mujoco.viewer.launch_passive(model, data)
for c in range(5000):
    print("step {}".format(c))
    #mujoco.mj_step(model, data)
    state = step_fn(mjx_model, state)
    #if (c % 100) == 0:
    #    np.savetxt("data/debug.csv", state.debug, delimiter=",")
    mjx.get_data_into(data, model, state)
    viewer.sync()
viewer.close()