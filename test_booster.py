import mujoco
import mujoco.viewer
from mujoco import mjx
import jax.numpy as jnp
from pipelines.booster_eefpbc import multistep, init, default_act
import jax
from rewards.mjx_col import get_contacts
from models.booster_t1_pgnd import booster_ids as bids

model = mujoco.MjModel.from_xml_path('models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

init_qpos = model.keyframe('home').qpos
data.qpos = init_qpos
mujoco.mj_step(model, data) # sim first step

viewer = mujoco.viewer.launch_passive(model, data)
state = mjx.make_data(mjx_model)
@jax.jit
def step_fn(mjx_model, state):
    #mjx_state = multistep(mjx_model, mjx_state, 1, act)
    state = mjx.step(mjx_model, state)
    return state

_right_foot_floor_found_sensor = [
        model.sensor(f"right_foot_{i}_floor_found").id
        for i in range(1, 5)
    ]

floor_geom_id = model.geom("floor").id
right_foot_geom_id = model.geom("right_foot_1").id
for c in range(1000):
    print("step {}".format(c))
    state = step_fn(mjx_model, state)
    mjx.get_data_into(data, model, state)
    contact = get_contacts(state.contact, bids.ids)
    print(contact, bids.ids["col"])
    #print(check_collision(data.contact, floor_geom_id, right_foot_geom_id))
    #print(data.sensordata)
    #print(data.contact)
    #print(state.sensordata)
    #mujoco.mj_step(model, data)
    viewer.sync()
viewer.close()