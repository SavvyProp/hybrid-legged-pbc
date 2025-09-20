import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
from lowctrl import maqp
from lowctrl import model as lmodel
from models.booster_t1_pgnd import booster_ids as bids
import jax

model = mujoco.MjModel.from_xml_path('models/booster_t1/flat_scene.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

init_qpos = model.keyframe('home').qpos
data.qpos = init_qpos
mujoco.mj_step(model, data) # sim first step

ctrl = jnp.zeros([23])

base_com = jnp.zeros([3])
init_com = data.subtree_com[0].copy()

t = 0
def step_fn(mj_model, mj_state, t):
    #act = maqp.default_act(bids.ids)
    act = maqp.test_act_move_com(init_com, data, t, bids.ids)
    pos = bids.ids["default_qpos"][7:]
    pos = pos.at[2].set(jnp.sin(t) * 0.4)
    pos = pos.at[6].set(jnp.sin(t) * 0.4)
    
    ctrl = maqp.step(mj_model, mj_state, act, bids.ids, is_mjx = False)
    ctrl2 = maqp.step_centroidal(mj_model, mj_state, act, bids.ids, is_mjx = False)
    print("ctrls", ctrl[11:], ctrl2[11:])
    return ctrl


viewer = mujoco.viewer.launch_passive(model, data)
for c in range(5000):
    print("step {}".format(c))
    ctrl = step_fn(model, data, t)
    data.ctrl = ctrl
    mujoco.mj_step(model, data)
    t += 0.001
    viewer.sync()

viewer.close()