import jax.numpy as jnp
from mujoco import mjx
import jax
import lowctrl.math as lmath
import lowctrl.pd as lpd
from flax import linen as nn
import mujoco
import numpy as np
from mujoco.mjx._src import support
import lowctrl.jvp as jvp

def get_jac(mj_model, mj_data, site_name, ids, is_mjx = True):
    """
    Gets the jacobian of position and rotation for a site
    """
    site_id = ids["eef"][site_name]["site_id"]
    body_id = ids["eef"][site_name]["body_id"]
    point = mj_data.site_xpos[site_id]
    if is_mjx:
        jacp, jacr = mjx.jac(mj_model, mj_data, point, body_id)
        jacp = jacp.T
        jacr = jacr.T
    else:
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
    j = jnp.vstack((jacp, jacr))
    return j

def jac_stack(mjx_model, mjx_data, ids, is_mjx = True):
    """
    Stacks the jacobians of all end effectors vertically
    """
    jacs = []
    for eef_name in ids["eef"].keys():
        j = get_jac(mjx_model, mjx_data, eef_name, ids, is_mjx = is_mjx)
        jacs.append(j)
    return jnp.vstack(jacs)


def get_djp_mj(mj_model, mj_data, ids):
    """
    Finite difference estimate of Jdot(q, qdot) * qdot for all end effectors
    """
    eef_num = ids["eef_num"]
    jvp_list = []
    for eef_name in ids["eef"].keys():
        site_id = ids["eef"][eef_name]["site_id"]
        jvp_ = jvp.site_jdot_qdot_fd(mj_model, mj_data, eef_name, scheme="central")
        jvp_list.append(jvp_)
    return jnp.concatenate(jvp_list, axis = 0)

def jac_com(m, d):
    root  = m.body_rootid[1]                 # body 0 = world, 1 = robot base
    point = d.subtree_com[root]              # 3-vector centre of mass of that subtree

    jac_trans, _ = support.jac(m, d, point, root)   # (nv, 3) translational block
    J_com = jac_trans.T 
    return J_com

def com_pos_fn(qpos, mjx_model, mjx_data):
    d = mjx_data.replace(qpos=qpos)
    # Ensure kinematics are updated; if mjx has a pure forward kinematics util, use it.
    d = mjx.step(mjx_model, d)  # with zero ctrl/vel this updates kinematics
    root = mjx_model.body_rootid[1]
    return d.subtree_com[root]

def jac_com_mjx(model, data):
    J = jax.jacrev(lambda q: com_pos_fn(q, model, data))(data.qpos)
    return J

def get_mh(mj_model, mj_data, ids, is_mjx = True):
    """
    Gets the mass matrix and bias force for the 36 digit model
    """
    #m = mjx.full_m(mjx_model, mjx_data)
    if is_mjx:
        m = mjx.full_m(mj_model, mj_data)
    else:
        m = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, m, mj_data.qM)
    vel_ids = ids["joint_vel_ids"]
    m_36 = m[np.ix_(vel_ids, vel_ids)]
    h_36 = mj_data.qfrc_bias[vel_ids]
    return m_36, h_36

def get_eefpos(data, ids):
    poslist = []
    for eef_name in ids["eef"].keys():
        site_id = ids["eef"][eef_name]["site_id"]
        point = data.site_xpos[site_id]
        poslist.append(point[None, :])
    return jnp.concatenate(poslist, axis = 0)

def get_compos(data):
    return data.subtree_com[0]

def get_kin_values(model, data, ids, is_mjx = True):
    j_stack = jac_stack(model, data, ids, is_mjx = is_mjx)
    m, h = get_mh(model, data, ids, is_mjx = is_mjx)
    if is_mjx:
        com_jvp, jvp_ = jvp.get_djp(model, data, ids)
        #jac_com_ = jnp.ones([3, model.nv])
        jac_com_ = jac_com(model, data)
    else:
        jvp_ = get_djp_mj(model, data, ids)
        #com_jvp, jac_com_ = jvp.jdot_q_for_model_com(model, data)
        jac_com_, com_jvp = jvp.model_com2(model, data)
    eef_pos = get_eefpos(data, ids)
    com_pos = get_compos(data)
    return m, h, j_stack, jvp_, jac_com_, com_jvp, eef_pos, com_pos