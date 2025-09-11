import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
from pipelines.booster_eefpbc import multistep, init
from lowctrl.eefpbc import default_act
from lowctrl import eefpbc
from models.booster_t1_pgnd import booster_ids as bids
import jax

model = mujoco.MjModel.from_xml_path('models/booster_t1/flat_scene.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

init_qpos = model.keyframe('home').qpos
data.qpos = init_qpos
mujoco.mj_step(model, data) # sim first step

act = default_act(bids.ids)
ctrl = jnp.zeros([23])


state = init(
    mjx_model,
    jnp.array(init_qpos),
    jnp.zeros(model.nv),
    jnp.zeros([0,]),
    ctrl,
)

def step_fn(mjx_model, mjx_state):
    act = default_act(bids.ids)
    pos = bids.ids["default_qpos"][7:]
    ctrl = eefpbc.step(mjx_model, mjx_state, act, bids.ids, override_pos = pos)
    data = mjx_state.replace(ctrl=ctrl)
    data = mjx.step(mjx_model, data)
    return data


for c in range(5000):
    print("step {}".format(c))
    state = step_fn(mjx_model, state)
    