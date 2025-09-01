from typing import Any, Optional
from brax.base import Contact, Motion, System, Transform
import jax
from jax import numpy as jnp
from mujoco import mjx
from lowctrl import pd
from models.booster_t1.booster_ids import ids as bids

from brax.envs import PipelineEnv
from brax import base
from brax.mjx.base import State as mjx_State
from mujoco.mjx._src.types import Contact as MJXContact
from functools import partial
from pipelines.booster_eefpbc import State

def default_act():
	return jnp.zeros([46])

def _reformat_contact(sys: System, contact: MJXContact) -> Contact:
	"""Reformats the mjx.Contact into a brax.base.Contact."""
	if contact is None:
		return

	elasticity = jnp.zeros(contact.pos.shape[0])
	body1 = jnp.array(sys.geom_bodyid)[contact.geom1] - 1
	body2 = jnp.array(sys.geom_bodyid)[contact.geom2] - 1
	link_idx = (body1, body2)
	return Contact(link_idx=link_idx, elasticity=elasticity, **contact.__dict__)


def init(
	sys: System,
	q: jax.Array,
	qd: jax.Array,
	act,
	ctrl,
	unused_debug: bool = False,
) -> State:
	"""Initializes physics data.

	Args:
		sys: a brax System
		q: (q_size,) joint angle vector
		qd: (qd_size,) joint velocity vector
		act: actuator activations
		ctrl: actuator controls
		unused_debug: ignored

	Returns:
		data: initial physics data
	"""

	data = mjx.make_data(sys)
	data = data.replace(qpos=q, qvel=qd)
	if act is not None:
		data = data.replace(act=act)
	if ctrl is not None:
		data = data.replace(ctrl=ctrl)
	
	if act is None:
		data = data.replace(act = default_act())

	data = mjx.forward(sys, data)

	q, qd = data.qpos, data.qvel
	x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
	cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
	offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
	offset = Transform.create(pos=offset)
	xd = offset.vmap().do(cvel)

	brax_contact = _reformat_contact(sys, data.contact)
	data_args = data.__dict__
	data_args['contact'] = brax_contact

	return State(q=q, 
				 qd=qd, 
				 x=x, 
				 xd=xd, 
				 qvel_low_freq = jnp.zeros([29,]), 
				 f_stc = jnp.zeros([24,]),
				 **data_args)

#@partial(jax.jit, static_argnames = ('ids',))
def step(
		sys: System,
		state: State,
		act: jax.Array,
		unused_debug: bool = False,
		ids = bids
) -> State:
	ctrl, state = pd.step(sys, state, act, ids)
	data = state.replace(ctrl=ctrl)
	data = mjx.step(sys, data)
	q, qd = data.qpos, data.qvel
	x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
	cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
	offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
	offset = Transform.create(pos=offset)
	xd = offset.vmap().do(cvel)
	if data.ncon > 0:
		mjx_contact = data._impl.contact if hasattr(data, '_impl') else data.contact
		data = data.replace(contact=_reformat_contact(sys, mjx_contact))
	return data.replace(q=q, qd=qd, x=x, xd=xd)


def multistep(model, pipeline_state, n_frames, action, ids = bids):
	#pipeline_state = eefpbc.get_init_eef_des(model, pipeline_state, action)
	pipeline_state = pipeline_state.replace(
			qvel_low_freq = pipeline_state.qvel[ids["joint_vel_ids"]])
	def f(state, _):
		return (
			step(model, state, action, False),
			None,
		)

	return jax.lax.scan(f, pipeline_state, (), n_frames)[0]

class PDPipelineEnv(PipelineEnv):
	def __init__(self, 
					sys: base.System,
					n_frames: int = 1,):
		super().__init__(sys = sys,
							   backend = 'mjx',
							   n_frames = n_frames)
				
	def pipeline_init(
				self,
				q: jax.Array,
				qd: jax.Array,
				act: Optional[jax.Array] = None,
				ctrl: Optional[jax.Array] = None,
		) -> base.State:
		"""Initializes the pipeline state."""
		return init(self.sys, q, qd, act, ctrl, self._debug)
			  
	def pipeline_step(self, pipeline_state: Any, action: jax.Array) -> base.State:
		"""Takes a physics step using the physics pipeline."""
		return multistep(self.sys, pipeline_state, 5, action)
