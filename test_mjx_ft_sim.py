import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
from lowctrl.ft_ref import default_act
from lowctrl import ft_ref
from models.booster_t1_pgnd import booster_ids as bids
import jax

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
    act = ft_ref.default_act_lock_com(init_com, mjx_state, bids.ids)
    output = ft_ref.step(mjx_model, mjx_state, act, bids.ids, is_mjx = True, debug=True)
    data = mjx_state.replace(ctrl=output["u"])
    data = mjx.step(mjx_model, data)
    return data, output


viewer = mujoco.viewer.launch_passive(model, data)
for c in range(5000):
    print("step {}".format(c))
    #mujoco.mj_step(model, data)
    state, output = step_fn(mjx_model, state, init_com, t)
    print(output["des_com_vel"])
    #m_uc, h_uc = eefpbc.get_mh(mjx_model, state, bids.ids)
    #print(m_uc)
    #if (c % 100) == 0:
    #    np.savetxt("data/debug.csv", state.debug, delimiter=",")
    t += 0.001
    mjx.get_data_into(data, model, state)
    viewer.sync()

viewer.close()