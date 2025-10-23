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

def get_forces(data, ids):

    left_force = jnp.zeros(3,)
    for id in ids["col"]["left_foot"]:
        left_force += contact_forces_between_geoms_world(data, ids["col"]["floor"], id)
    right_force = jnp.zeros(3,)
    for id in ids["col"]["right_foot"]:
        right_force += contact_forces_between_geoms_world(data, ids["col"]["floor"], id)

    return left_force, right_force

def get_contacts(contact, ids):
    left_foot = check_collision(contact, ids["col"]["floor"], 
                                id["col"]["left_foot"])
    right_foot = check_collision(contact, ids["col"]["floor"], 
                                 id["col"]["right_foot"])
    
    contact = jnp.array([left_foot, right_foot])
    return contact

def get_contact_dict(contact, ids):
    contact_dict = {}
    for key in ids["col"]:
        if key != "floor":
            contact_dict[key] = check_collision(contact, ids["col"]["floor"], 
                                                ids["col"][key])
    return contact_dict

# Rewrite get_contacts to return a dictionary of body_id, contact


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
    
def contact_force_world_for_pair(data: mjx.Data, floor_geom_id: int, foot_geom_id: int) -> jax.Array:
    """
    Returns the net contact force (3,) on the foot in world frame for contacts
    between the specified floor geom and foot geom.

    This mirrors MuJoCo's mj_contactForce logic using the constraint forces (efc_force)
    and the contact frame axes. Sign is chosen so the returned force is the force
    acting on the 'foot' geom.

    If no such contacts exist at this step, returns zeros.
    """
    contact = data.contact

    # mask contacts matching (floor, foot) in either order
    is_pair = ((contact.geom1 == floor_geom_id) & (contact.geom2 == foot_geom_id)) | \
              ((contact.geom2 == floor_geom_id) & (contact.geom1 == foot_geom_id))

    if contact.pos.shape[0] == 0:
        return jnp.zeros((3,), dtype=data.qpos.dtype)

    # sign: +1 if foot is geom2 (frame normal points from geom1->geom2), else -1
    sign = jnp.where(contact.geom2 == foot_geom_id, 1.0, -1.0).astype(data.qpos.dtype)
    sign = sign * is_pair.astype(data.qpos.dtype)

    # addresses into efc_force and per-contact dims
    adr = contact.efc_address
    dim = contact.dim

    # gather normal/tangent constraint forces (0 if not present or not the target pair)
    efc = data.efc_force  # shape (nefc,)
    fn = jnp.where(is_pair, efc[adr], 0.0)
    ft1 = jnp.where(is_pair & (dim >= 2), efc[adr + 1], 0.0)
    ft2 = jnp.where(is_pair & (dim >= 3), efc[adr + 2], 0.0)

    # contact frame axes in world frame: frame[k, :3] gives axis k (0:normal,1:t1,2:t2)
    # shape: (ncon, 3, 3) — rows are axes, columns are xyz
    axes = contact.frame[..., :3]

    # world force for each contact = sign * sum_k f_k * axis_k
    f_local = jnp.stack([fn, ft1, ft2], axis=1)                # (ncon, 3)
    f_world_each = sign[:, None] * jnp.sum(axes * f_local[:, :, None], axis=1)  # (ncon, 3)

    # sum over all matching contacts affecting this foot
    f_world = jnp.sum(f_world_each, axis=0)
    return f_world

# ...existing code...
def contact_forces_between_geoms_world(
    mjx_data,
    geom_a: int,
    geom_b: int,
    *,
    cone: str = "pyramidal",
    sum_result: bool = True,
    force_on: str = "b",
):
    """
    World-frame contact force(s) between geoms `geom_a` and `geom_b`, computed from
    mjx_data.contact[*] and mjx_data.efc_force.

    Returns:
        (3,) if sum_result=True.  (JIT-safe; avoids boolean-array compression.)
    Notes:
      - Uses contact.frame rows as axes; rotates local→world with frame.T.
      - By MuJoCo convention the contact normal (frame row 0) points from geom[0] → geom[1].
      - The force constructed below is the force applied to contact.geom[1]; we flip sign
        as needed so the returned vector always acts on the requested `force_on` body.
    """
    ncon = mjx_data.ncon
    contact = mjx_data.contact

    # Geom pairs (handle both newer 'geom' field and older 'geom1/geom2')
    if hasattr(contact, "geom"):
        g0, g1 = contact.geom[:ncon, 0], contact.geom[:ncon, 1]
    else:
        g0 = contact.geom1[:ncon]
        g1 = contact.geom2[:ncon]

    # Per-contact masks
    mask_ab = (g0 == geom_a) & (g1 == geom_b)
    mask_ba = (g0 == geom_b) & (g1 == geom_a)
    mask = mask_ab | mask_ba

    # Sign to make result act on requested body
    if force_on.lower() == "b":
        # Constructed force acts on geom[1]; + for (a,b), - for (b,a)
        sign = jnp.where(mask_ab, +1.0, jnp.where(mask_ba, -1.0, 0.0))
    else:  # force_on == "a"
        sign = jnp.where(mask_ab, -1.0, jnp.where(mask_ba, +1.0, 0.0))

    adr = contact.efc_address[:ncon]          # (ncon,)
    dim = contact.dim[:ncon]                  # (ncon,)
    efc = mjx_data.efc_force                  # (nefc,)
    R = contact.frame[:ncon].reshape(-1, 3, 3)  # rows=axes (ncon,3,3)

    # Local forces per contact (fn, ft1, ft2), zero for non-matches
    if cone.lower().startswith("ellip"):
        idx = adr[:, None] + jnp.arange(3)                     # (ncon,3)
        use = (jnp.arange(3)[None, :] < jnp.clip(dim, 0, 3)[:, None])
        f_local = jnp.where(use, efc[idx], 0.0)                # (ncon,3)
    elif cone.lower().startswith("pyr"):
        if not hasattr(contact, "friction"):
            raise ValueError(
                "contact.friction missing in this MJX build; pass cone='elliptic' "
                "or compute via CPU mj_contactForce."
            )
        mu_t = contact.friction[:ncon, 0]                      # (ncon,)
        idx4 = adr[:, None] + jnp.arange(4)                    # (ncon,4)
        use4 = (jnp.arange(4)[None, :] < jnp.clip(dim, 0, 4)[:, None])
        lam4 = jnp.where(use4, efc[idx4], 0.0)                 # (ncon,4)
        fn = lam4.sum(axis=1)
        ft1 = mu_t * (lam4[:, 0] - lam4[:, 1])
        ft2 = mu_t * (lam4[:, 2] - lam4[:, 3])
        f_local = jnp.stack([fn, ft1, ft2], axis=1)            # (ncon,3)
    else:
        raise ValueError("cone must be 'pyramidal' or 'elliptic'")

    # Zero out non-matching contacts
    f_local = f_local * mask[:, None]

    # Rotate to world: rows are axes ⇒ world = R.T @ local
    f_world_each = jnp.einsum('nij,nj->ni', jnp.swapaxes(R, 1, 2), f_local)  # (ncon,3)
    f_world_each = f_world_each * sign[:, None]                               # (ncon,3)

    # Sum over all matching contacts
    return f_world_each.sum(axis=0)