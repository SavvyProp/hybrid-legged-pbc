import jax.numpy as jnp
from mujoco import mjx
import jax
import mujoco
from brax import math
from rewards.gait import create_stance_mask, height_target
from flax import linen as nn

# A few common variables for the reward functions:
# sys_dict: dict of ids
# data: mjx_data
# contact: jnp.array of whether left or right foot is in contact. 0 for not 1 for yes
# foot_frc: a 2 by 6 array of foot forces

def reward_foot_force(foot_frc, threshold: float = 500, max_reward: float = 400):
   """
   Reward for net force on foot, not dependent on phase
   """
   lin_frc = foot_frc[:, :3]
   lin_frc = jnp.linalg.norm(lin_frc, axis=1)
   frc_rew = jnp.where(lin_frc > threshold,
                       lin_frc - threshold,
                       0.0)
   frc_rew = jnp.clip(frc_rew, 0.0, max_reward)
   reward = jnp.sum(frc_rew)
   return reward

def reward_foot_height(ids, data, phase, std, halt):
   """
   Reward for tracking a foot height polynomial based on phase
   """
   standing_height = 0.0 # Means that terrain is all 0
   standing_position_toe_roll_z = 0.0
   offset = standing_height + standing_position_toe_roll_z

   stance_mask, mask2 = create_stance_mask(phase)
   swing_mask = 1 - stance_mask[0, :]

   left_foot_pos = data.site_xpos[ids["eef"]["left_foot"]["site_id"]]
   right_foot_pos = data.site_xpos[ids["eef"]["right_foot"]["site_id"]]

   current_foot_z = jnp.array([
      left_foot_pos[2],
      right_foot_pos[2]])

   #filt_foot = jnp.where(
   #   swing_mask == 1, current_foot_z, 0
   #)
   filt_foot = swing_mask * current_foot_z
   feet_z_value = jnp.sum(filt_foot)

   phase_mod = jnp.mod(phase, 0.5)

   feet_z_target = height_target(phase_mod[0]) + offset
   feet_z_target *= (1 - halt)

   error = jnp.square(feet_z_value - feet_z_target)
   reward = jnp.exp(-error / std**2)

   return reward

def reward_foot_contact(contacts, phase, pos_rw, neg_rw, halt):
   """
   Reward for having the expected number of feet contacting
   based on phase
   """
   stance_mask, mask2 = create_stance_mask(phase)
   reward = jnp.where(
      contacts == stance_mask[0, :], pos_rw, neg_rw
   )

   reward = jnp.mean(reward) * (1 - halt)

   return reward

def reward_foot_clearance(ids, data, 
                          target_height,
                          std,
                          tanh_mult,
                          halt):
   """
   Reward for swing a certain desired constant clearence above ground
   """
   standing_height = 0.0 # Means that terrain is all 0
   standing_position_toe_roll_z = 0.0155 #0.0626
   offset = standing_height + standing_position_toe_roll_z

   left_foot_pos = data.site_xpos[ids["eef"]["left_foot"]["site_id"]]
   right_foot_pos = data.site_xpos[ids["eef"]["right_foot"]["site_id"]]

   left_foot_vel = jnp.linalg.norm(
      data.xd.vel[ids["eef"]["right_foot"]["body_id"]])
   right_foot_vel = jnp.linalg.norm(
      data.xd.vel[ids["eef"]["right_foot"]["body_id"]])

   current_foot_z = jnp.array([
      left_foot_pos[2],
      right_foot_pos[2]])
   
   current_foot_spd = jnp.array([
      left_foot_vel,
      right_foot_vel])
   
   foot_z_err = jnp.square(
      jnp.clip(current_foot_z - (offset + target_height),
               min = None,
               max = 0.0)
   )

   xy_tanh = jnp.tanh(
      tanh_mult * current_foot_spd
   )

   reward = jnp.exp(
      -1 * jnp.sum(xy_tanh * foot_z_err) / std
   )
   reward *= (1 - halt)
   return reward

def reward_foot_slide(ids, data, contact):
   """
   Higher number for higher contact vel + angvel
   """
   # Vel of foot multiplied by contact state
   left_foot_site_id = ids["eef"]["left_foot"]["site_id"]
   right_foot_site_id = ids["eef"]["right_foot"]["site_id"]
   left_foot_vel = data.xd.vel[left_foot_site_id]
   left_foot_angvel = data.xd.ang[left_foot_site_id]
   right_foot_vel = data.xd.vel[right_foot_site_id]
   right_foot_angvel = data.xd.ang[right_foot_site_id]

   left_rew = jnp.linalg.norm(left_foot_vel) + jnp.linalg.norm(left_foot_angvel) * 0.1
   right_rew = jnp.linalg.norm(right_foot_vel) + jnp.linalg.norm(right_foot_angvel) * 0.1

   rew = jnp.array([left_rew, right_rew])
   return jnp.sum(rew * contact)  # Multiply by contact state to zero out when not in contact

def reward_foot_air_time(air_time, contact_time, threshold, halt):
   """
   Reward for having feet in the air for time (look at issaclab implementation)
   Notes:
   air_time, contact_time 2 dim array of times for l and r foot
   in_mode_time: 2 dim array contact time if in contact, air time if not
   single_stance: single num 1 for if only 1 contact
   reward: if not in ss 0 rew, else is the min of in mode time
   reward: reward is clamped and multiplied by idle constant
   """
   in_contact = contact_time > 0.0
   in_mode_time = jnp.where(
      in_contact, contact_time, air_time
   )
   single_stance = jnp.sum(in_contact) == 1
   reward = jnp.where(single_stance, in_mode_time, 0.0)
   reward = jnp.min(reward, axis = 0)
   reward = jnp.clip(reward, 0.0, threshold)
   reward = reward * (1 - halt)  # No reward if halting
   return reward

def reward_base_orien(data):
   inv_pelvis_rot = math.quat_inv(data.xquat[1])
   grav_vec = math.rotate(jnp.array([0,0,-1]), inv_pelvis_rot)
   # Should be penalty
   return jnp.sum(jnp.square(grav_vec[0:2]))

def height_termination(data, height_threshold):
   is_healthy = jnp.where(data.q[2] < height_threshold[0], 0.0, 1.0)
   is_healthy = jnp.where(data.q[2] > height_threshold[1], 0.0, is_healthy)
   termination = 1 - is_healthy
   return is_healthy, termination

def reward_xyvel(ids, data, target_vel, std):
   base_xyvel = data.xd.vel[ids["base_id"]][0:2]
   vel_err = jnp.square(base_xyvel - target_vel)
   return jnp.exp(-jnp.sum(vel_err) / (std**2))

def reward_xyvel_local(ids, data, target_vel, std, halt):
   base_xyvel = data.xd.vel[ids["base_id"]][0:2]
   inv_pelvis_rot = math.quat_inv(data.xquat[1])
   facing_vec = math.rotate(jnp.array([1,0,0]), inv_pelvis_rot)
   theta = jnp.arctan2(facing_vec[1], facing_vec[0])
   rot_mat = jnp.array([[jnp.cos(-theta), -jnp.sin(-theta)],
                        [jnp.sin(-theta), jnp.cos(-theta)]])
   local_vel = jnp.matmul(rot_mat , base_xyvel)
   vel_err = jnp.square(local_vel - target_vel * ( 1 - halt ))
   return jnp.exp(-jnp.sum(vel_err) / (std**2))

def reward_z_vel(ids, data):
   return jnp.square(data.xd.vel[ids["base_id"]][2])

def reward_track_z_angvel(ids, data, target_angvel, halt, std):
   z_angvel = data.xd.ang[ids["base_id"]][2]
   angvel_err = jnp.square(z_angvel - target_angvel * (1 - halt))
   return jnp.exp(-angvel_err / (std**2))

def reward_angvelxy_l2(ids, data):
   angvel = data.xd.ang[ids["base_id"]][0:2]
   angvel_err = jnp.sum(jnp.square(angvel))
   return angvel_err

def reward_jnt_frc_l2(joint_torques): #-1e-5
   return jnp.sum(jnp.square(joint_torques))

def reward_jnt_actuator_limits(jnt_pos, jnt_limits, prop):
   center = jnp.mean(jnt_limits[:, :], axis = 1)
   d_top = jnt_limits[:, 1] - center
   d_bottom = jnt_limits[:, 0] - center
   top = center + d_top * prop
   bottom = center + d_bottom * prop
   
   top_rew = jnp.clip(jnt_pos - top, min = 0, max = None)
   bottom_rew = jnp.clip(bottom - jnt_pos, min = 0, max = None)

   reward = jnp.sum(top_rew + bottom_rew)
   return reward

def reward_jnt_deviation(qpos, inds, weights, defaultpose):
   jnt_devs = jnp.abs(qpos - defaultpose)
   reward = jnp.sum(jnt_devs[inds] * weights)
   return reward

def reward_pbc_w_leg_only(w, contact):
   contact_prob = nn.sigmoid(w)
   contact_target = jnp.concatenate([
      contact, jnp.array([0., 0.])
   ], axis = 0)
   lse = jnp.sum(jnp.square(contact_prob - contact_target))
   return lse

def reward_action_rate(action, prev_action):
   rew = jnp.sum(jnp.square(jnp.tanh(action) - jnp.tanh(prev_action)))
   return rew

def reward_jnt_acc(qacc):
   return jnp.sum(jnp.square(qacc))

def reward_jnt_vel(jnt_vel):
   return jnp.sum(jnp.square(jnt_vel))