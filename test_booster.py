import mujoco
import mujoco.viewer
import numpy as np
import time

# Load initial pose (expects a single row or column). Flatten to 1D.
initial_pose = np.genfromtxt('motions/CMU_02_05_initial_pose.csv', delimiter=',')
initial_pose = np.asarray(initial_pose, dtype=float).ravel()
#initial_pose[2] += 0.025
# Load model and create data
mj_model = mujoco.MjModel.from_xml_path('models/booster_t1_pgnd/scene_mjx_feetonly_flat_terrain.xml')
data = mujoco.MjData(mj_model)

# Fit initial pose to model nq (pad/truncate as needed)
nq = mj_model.nq
q = np.zeros(nq, dtype=float)
count = min(nq, initial_pose.size)
q[:count] = initial_pose[:count]

# Set state and forward
data.qpos[:] = q
if mj_model.nv:
    data.qvel[:] = 0
mujoco.mj_forward(mj_model, data)

# Show in viewer
with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(1.0 / 60.0)