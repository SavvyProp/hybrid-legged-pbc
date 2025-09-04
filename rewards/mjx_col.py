import jax.numpy as jnp
import mujoco.mjx as mjx
import jax
import mujoco
from typing import Any, Tuple, Union

def check_collision(contact, geom1, geom2):
   mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
   mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
   idx = jnp.where(mask, contact.dist, 1e4).argmin()
   dist = contact.dist[idx] * mask[idx]
   #normal = (dist < 0) * contact.frame[idx, 0, :3]
   return dist < 0

def get_contacts(contact, ids):
    left_foot = jnp.array([ 
        check_collision(contact, ids["col"]["floor"], id)
        for id in ids["col"]["left_foot"]])
    right_foot = jnp.array([ 
        check_collision(contact, ids["col"]["floor"], id)
        for id in ids["col"]["right_foot"]])
    
    contact = jnp.array([jnp.any(left_foot), jnp.any(right_foot)])
    return contact

def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
   """Get the distance and normal of the collision between two geoms."""
   mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
   mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
   idx = jnp.where(mask, contact.dist, 1e4).argmin()
   dist = contact.dist[idx] * mask[idx]
   normal = (dist < 0) * contact.frame[idx, 0, :3]
   return dist, normal

def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
   """Return True if the two geoms are colliding."""
   return get_collision_info(state.contact, geom1, geom2)[0] < 0

def feet_contact(state, floor_id, left_foot_id, right_foot_id):
    l = geoms_colliding(state, left_foot_id, floor_id)
    r = geoms_colliding(state, right_foot_id, floor_id)
    contact = jnp.array([l, r])
    return contact



def get_contact_forces(d):
    #assert (s.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL)  # Assert cone is PYRAMIDAL

    # mju_decodePyramid
    # 1: force: result
    # 2: pyramid: d.efc_force + contact.efc_address
    # 3: mu: contact.friction
    # 4: dim: contact.dim

    contact = d.contact
    cnt = d.ncon

    # Generate 2d array of efc_force indexed by efc_address containing the maximum
    # number of potential elements (10).
    # This enables us to operate on each contact force pyramid rowwise.
    efc_argmap = jnp.linspace(
        contact.efc_address,
        contact.efc_address + 9,
        10, dtype=jnp.int32
    ).T
    # OOB access clamps in jax, this is safe
    pyramid = d.efc_force[efc_argmap.reshape((efc_argmap.size))].reshape(efc_argmap.shape)

    # Calculate normal forces
    # force[0] = 0
    # for (int i=0; i < 2*(dim-1); i++) {
    #   force[0] += pyramid[i];
    # }
    index_matrix = jnp.repeat(jnp.arange(10)[None, :], cnt, axis=0)
    force_normal_mask = index_matrix < (2 * (contact.dim - 1)).reshape((cnt, 1))
    force_normal = jnp.sum(jnp.where(force_normal_mask, pyramid, 0), axis=1)

    # Calculate tangent forces
    # for (int i=0; i < dim-1; i++) {
    #   force[i+1] = (pyramid[2*i] - pyramid[2*i+1]) * mu[i];
    # }
    pyramid_indexes = jnp.arange(5) * 2
    force_tan_all = (pyramid[:, pyramid_indexes] - pyramid[:, pyramid_indexes + 1]) * contact.friction
    force_tan = jnp.where(pyramid_indexes < contact.dim.reshape((cnt, 1)), force_tan_all, 0)

    # Full force array
    forces = jnp.concatenate((force_normal.reshape((cnt, 1)), force_tan), axis=1)

    # Special case frictionless contacts
    # if (dim == 1) {
    #   force[0] = pyramid[0];
    #   return;
    # }
    frictionless_mask = contact.dim == 1
    frictionless_forces = jnp.concatenate((pyramid[:, 0:1], jnp.zeros((pyramid.shape[0], 5))), axis=1)
    return jnp.where(
        frictionless_mask.reshape((cnt, 1)),
        frictionless_forces,
        forces
    )


def get_feet_forces(state, floor_id, left_foot_id, right_foot_id):
    
    #forces = get_contact_forces(state)
    forces = get_contact_forces_global(state)
    
    # Identifiers for the floor, right foot, and left foot

    # Find contacts that involve both the floor and the respective foot
    # This assumes dx.contact.geom contains two entries per contact, one for each of the two contacting geometries
    right_bm = jnp.sum(jnp.abs(state.contact.geom - jnp.array([[floor_id, right_foot_id]])), axis = 1)
    right_bm2 = jnp.sum(jnp.abs(state.contact.geom - jnp.array([[right_foot_id, floor_id]])), axis=1)
    right_bm = jnp.where(right_bm == 0 , 1, 0)
    right_bm2 = jnp.where(right_bm2 == 0, 1, 0)

    right_bm = right_bm + right_bm2


    left_bm = jnp.sum(jnp.abs(state.contact.geom - jnp.array([[floor_id, left_foot_id]])), axis=1)
    left_bm2 = jnp.sum(jnp.abs(state.contact.geom - jnp.array([[left_foot_id, floor_id]])), axis=1)
    left_bm = jnp.where(left_bm == 0, 1, 0)
    left_bm2 = jnp.where(left_bm2 == 0, 1, 0)

    left_bm = left_bm + left_bm2

    # Sum forces for the identified contacts
    total_right_forces = jnp.sum(forces * right_bm[:, None], axis=0)
    total_left_forces = jnp.sum(forces * left_bm[:, None], axis=0)

    return total_left_forces, total_right_forces


# Functions to manage the feet airtime and contact time state
# Managed by 2 size jax arrs, previous contact and airtime

def update_feet_airtime(contact, airtime, contacttime, dt):
    """
    Update the airtime state based on the current contact state.
    """
    # If contact is 1 then airtime should reset to 0
    airtime = airtime + dt
    contacttime = contacttime + dt
    airtime = airtime * (1 - contact)
    contacttime = contacttime * contact
    return airtime, contacttime


def get_contact_forces_global(d):
    """
    Compute per-contact forces in the global frame.
    """
    # 1) get local forces [ncon × max_nd]
    local_forces = get_contact_forces(d)
    
    contact = d.contact

    # 2) stored frames are already [ncon × n_axes × 3]
    frames = contact.frame

    # 3) only keep the first n_axes local components
    n_axes = frames.shape[1]
    local = local_forces[:, :n_axes]  # [ncon × n_axes]

    # 4) project into world frame → [ncon × 3]
    global_forces = jnp.einsum('nc,nck->nk', local, frames)
    return global_forces