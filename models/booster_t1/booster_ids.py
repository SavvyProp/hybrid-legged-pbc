import numpy as np
import jax.numpy as jnp
import mujoco
import yaml

model = mujoco.MjModel.from_xml_path("models/booster_t1/flat_scene.xml")

joint_names = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

eefnames = [
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand"
]

def get_eef_ids(model, eef_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_name)
    body_id = model.site_bodyid[site_id]
    return site_id, body_id

eef_dict = {}
for eef_name in eefnames:
    site_id, body_id = get_eef_ids(model, eef_name)
    eef_dict[eef_name] = {
        "site_id": site_id,
        "body_id": body_id
    }

def get_jnt_pos_limits(model, jnt_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    return model.jnt_range[joint_id, :]


jnt_limits = jnp.zeros([0, 2])
for jnt_name in joint_names:
    limits = get_jnt_pos_limits(model, jnt_name)
    jnt_limits = jnp.concatenate([jnt_limits, jnp.expand_dims(limits, axis=0)], axis=0)

joint_pos_ids = [0, 1, 2, 3, 4, 5, 6]
joint_vel_ids = [0, 1, 2, 3, 4, 5]
for jnt_name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    qpos_idx = model.jnt_qposadr[joint_id]
    qvel_idx = model.jnt_dofadr[joint_id]
    joint_vel_ids.append(int(qvel_idx))
    joint_pos_ids.append(int(qpos_idx))

joint_pos_ids = jnp.array(joint_pos_ids)
joint_vel_ids = jnp.array(joint_vel_ids)


def load_gains():
    """
    Load gains from digit_gains.yaml and return organized gain arrays.
    
    Returns:
        h_gains: List from iik.h_gain
        eef_gains: List concatenating lleg_gain + rleg_gain + larm_gain + rarm_gain  
        pbc_p: List of p gains from pbc ordered by PD_KEYS_TOE_AG
        pbc_d: List of d gains from pbc ordered by PD_KEYS_TOE_AG
    """
    # Get the directory of this file
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #yaml_path = os.path.join(current_dir, 'config', 'digit_gains.yaml')
    yaml_path = "models/booster_t1/gains.yaml"
    
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract h_gains directly from iik
    joint_p = []
    joint_d = []

    for joint_name in joint_names:
        joint_p.append(config[joint_name][0])
        joint_d.append(config[joint_name][1])

    return joint_p, joint_d

p, d = load_gains()

eef_num = len(eefnames)

default_qpos = jnp.array(model.keyframe('home').qpos)[joint_pos_ids]

max_vel = jnp.ones([23]) * 10.0

ids = {
    "joint_pos_ids": joint_pos_ids,
    "joint_vel_ids": joint_vel_ids,
    "jnt_limits": jnt_limits,
    "p_gains": jnp.array(p),
    "d_gains": jnp.array(d),
    "eef": eef_dict,
    "eef_num": eef_num,
    "ctrl_num": len(joint_names),
    "default_qpos": default_qpos,
    "max_vel": max_vel
}