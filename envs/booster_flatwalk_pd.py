from pipelines.booster_pd import PDPipelineEnv
from brax.envs import State
import jax.numpy as jnp
import mujoco
from brax.io import mjcf
from brax import math
import jax
from rewards import mjx_col
from rewards import rewards
from lowctrl.eefpbc import ctrl2logits
from models.booster_t1.booster_ids import ids as bids

PHASE_DURATION = 0.64

weight_dict = {
    "lin_vel_xy": 2.,
    "angvel_z": 2.,
    "lin_vel_z_l2": -1.,
    "angvel_xy_l2": -0.05,
    "termination": -200,
    "joint_torques": -1.0e-5, #-1.0e-5,
    "joint_acc": -1.25e-7,
    "joint_vel": -5e-4,
    "joint_limits": -2.0,
    "base_flat_orien": -2.,
    "feet_slip": -1.0,
    "feet_airtime": 0.25,
    "feet_force": -3e-3,
    "feet_contact": 2.0,
    "feet_height_track": 0.5,
    "feet_clearance": 0.5,
    "action_rate": -0.01,
}

metrics_dict = {
    "reward": 0.0
}

for metric in weight_dict.keys():
    metrics_dict[metric] = 0.0

class FlatwalkEnv(PDPipelineEnv):
    def __init__(self):

        self.model = mujoco.MjModel.from_xml_path(
            "models/booster_t1/flat_scene.xml"
        )

        n_frames = 5

        system = mjcf.load_model(self.model)
        super().__init__(
            sys = system,
            n_frames = n_frames
        )

        self.initial_state = jnp.array(system.mj_model.keyframe('home').qpos)

        def get_sensor_data(sensor_name):
            sensor_id = system.mj_model.sensor(sensor_name).id
            sensor_adr = system.mj_model.sensor_adr[sensor_id]
            sensor_dim = system.mj_model.sensor_dim[sensor_id]
            return sensor_adr, sensor_dim
        
        self.gyro = get_sensor_data('angular-velocity')
        self.vel = get_sensor_data('linear-velocity')


        self.nv = system.nv
        self.nu = system.nu

        self.ids = bids

        self.jnt_num = self.ids["ctrl_num"]
        self.eef_num = self.ids["eef_num"]

    @property
    def action_size(self):
        return 2 * self.jnt_num

    def get_sensor_data(self, data, tuple):
        return data.sensordata[tuple[0]: tuple[0] + tuple[1]]

    def _get_obs(self, data, state = None):
        """
        Proprioception observations to feed into policy network

        """
        #inv_pelvis_rot = math.quat_inv(data.xquat[0])
        inv_pelvis_rot = math.quat_inv(data.qpos[3:7])
        vel = self.get_sensor_data(data, self.vel)
        angvel = self.get_sensor_data(data, self.gyro)

        qpos = data.qpos[self.ids["joint_pos_ids"]][7:]
        qvel = data.qvel[self.ids["joint_vel_ids"]][6:]

        grav_vec = math.rotate(jnp.array([0,0,-1]), inv_pelvis_rot)

        if state is None:
            phase = 0.0
            prev_action = jnp.zeros(self.action_size)
            vel_cmd = jnp.zeros([2,])
            angvel_cmd = jnp.zeros([1,])
            halt_cmd = jnp.array([0.0])
            phase_period = jnp.array([PHASE_DURATION,])
        else:
            phase = state.phase * 2 * jnp.pi
            prev_action = jnp.tanh(state.info["prev_action"])
            vel_cmd = state.info["cmd"]["vel"]
            angvel_cmd = state.info["cmd"]["angvel"]
            phase_period = state.info["cmd"]["phase_period"]
            halt_cmd = jnp.array([state.info["halt_cmd"]])

        phase_clock = jnp.array([jnp.sin(phase), jnp.cos(phase),
                                 jnp.sin(phase + jnp.pi), jnp.cos(phase + jnp.pi)])
        
        obs = jnp.concatenate([
            angvel, vel, grav_vec, qpos, qvel, 
            jnp.array([phase]), phase_clock, 
            vel_cmd, angvel_cmd, phase_period, halt_cmd,
            prev_action
        ], axis = 0)

        obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs
    

    def makeCmd(self, rng):
        rng, key1 = jax.random.split(rng)
        rng, key2 = jax.random.split(rng)
        rng, key3 = jax.random.split(rng)

        vel = jax.random.uniform(key1, shape=[2], minval = -1, maxval = 1)
        vel = vel * jnp.array([0.3, 0.3])
        #vel = vel + jnp.array([0.2, 0.0])
        angvel = jax.random.uniform(key2, shape=[1], minval=-0.3, maxval=0.3)
        #phase_period = jax.random.uniform(key3, shape=[1], minval=0.6, maxval=0.7)
        phase_period = PHASE_DURATION * jnp.ones([1,])
        cmd = {"vel": vel, "angvel": angvel, "phase_period": phase_period}
        return cmd, rng

    
    def updateCmd(self, state):
        rng = state.info["rng"]
        cmd, rng = self.makeCmd(rng)
        state.info["rng"] = rng
        tmod = jnp.mod(state.info["time"], 5.0)
        reroll_cmd = jnp.where(tmod > 5.0 - self.dt * 2, 1, 0)
        for key in cmd.keys():
            state.info["cmd"][key] = state.info["cmd"][key] * (1 - reroll_cmd) + cmd[key] * reroll_cmd
        return
    
    def periodicHalting(self, state):
        #period of ep[0] + ep[1]
        tmod = jnp.mod(state.info["time"], state.info["event_period"][0] +
                       state.info["event_period"][1])
        halt = jnp.where(tmod > state.info["event_period"][0], 1, 0)[0]
        state.info["halt_cmd"] = 0.0
        new_phase = jnp.zeros([1])
        state.info["phase"] = state.info["phase"] * (1 - halt) + new_phase * halt
        return
    
    def reset(self, rng: jax.Array):

        metrics = metrics_dict.copy()
        phase = jnp.zeros([1])
        prev_action = jnp.zeros(self.action_size)

        cmd, rng = self.makeCmd(rng)

        rng, key = jax.random.split(rng)
        event_period = jax.random.uniform(key, shape = [2], minval = 0, maxval = 1)
        event_period = event_period * jnp.array([3, 2.]) + jnp.array([4, 0.])
        
        state_info = {
            "rng": rng,
            "time": jnp.zeros([1]),
            "phase": phase,
            "air_time": jnp.zeros([2]),
            "contact_time": jnp.zeros([2]),
            "prev_action": prev_action,
            "cmd": cmd,
            "event_period": event_period,
            "halt_cmd": 0.0
        }

        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))

        obs = self._get_obs(pipeline_state)
        reward, done = jnp.zeros([2])
        state = State(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done,
            metrics = metrics,
            info = state_info
        )
        return state

    def step(self, state, action):
        """
        Step and reward function for Digit Environment
        """
        raw_action = action
        data0 = state.pipeline_state
        data1 = self.pipeline_step(data0, raw_action)

        obs = self._get_obs(data1)

        contact = mjx_col.feet_contact(data0,
            self.ids["col"]["floor"],
            self.ids["col"]["left_foot"],
            self.ids["col"]["right_foot"])
        
        left_frc, right_frc = mjx_col.get_feet_forces(
            data0,
            self.ids["col"]["floor"],
            self.ids["col"]["left_foot"],
            self.ids["col"]["right_foot"])

        reward, done = self.rewards(data1, state, contact, left_frc, right_frc, raw_action)

        state.info["time"] += self.dt
        state.info["phase"] = jnp.mod(state.info["phase"] + self.dt, 
                                      state.info["cmd"]["phase_period"]) / state.info["cmd"]["phase_period"]

        air_time = state.info["air_time"]
        contact_time = state.info["contact_time"]
        air_time, contact_time = mjx_col.update_feet_airtime(
            contact, air_time, contact_time, self.dt
        )

        state.info["air_time"] = air_time
        state.info["contact_time"] = contact_time

        self.updateCmd(state)
        self.periodicHalting(state)

        new_state = state.replace(
            pipeline_state = data1, obs = obs, reward = reward, done = done
        )

        return new_state

    def rewards(self, data0, state, contact, left_frc, right_frc, act):
        """
        Reward function for Digit Env.
        Derived From: https://github.com/evanzijianhe/leggedlab_direct/blob/main/legged_lab/envs/digit/env_config/
        1) track_lin_vel_xy_exp 2
        2) track_angvel_z_exp 2
        3) lin_vel_z_l2 -1
        4) angvel_xy_l2 -0.05
        5) termination -200
        6) joint_torque (qfrc) -1e-5
        7) pos limits -2
        10) base flat orien -2
        12) joint deviation (specific joints)
        13) feet slide/slip (-1.0)
        14) feet air time (0.25)
        15) feet force (Not periodic) 3e-3
        16) foot contact 2.0
        17) foot height track 0.5
        18) foot clearance (polynomial)
        # Pure function rewards func similar to legged lab
        """

        (des_pos_logit, 
         gnd_acc_logit, qp_weight_logit, tau_mix_logit, 
         w, oriens_logit, base_acc, select) = ctrl2logits(act, self.ids)


        info = state.info
        
        air_time = info["air_time"]
        contact_time = info["contact_time"]
        phase = info["phase"]
        
        foot_frc = jnp.concatenate([
            left_frc[None, :], right_frc[None, :]
        ], axis = 0)

        #target_vel = jnp.array([0.3, 0.0])
        #target_angvel = 0.0
        target_vel = info["cmd"]["vel"]
        target_angvel = info["cmd"]["angvel"][0]

        halt = info["halt_cmd"]

        reward_dict = {}
        reward_dict["lin_vel_xy"] = rewards.reward_xyvel_local(
            self.ids,
            data0,
            target_vel,
            0.5,
            halt
        ) * weight_dict["lin_vel_xy"]

        reward_dict["angvel_z"] = rewards.reward_track_z_angvel(
            self.ids,
            data0,
            target_angvel,
            0.5,
            halt
        ) * weight_dict["angvel_z"]

        reward_dict["lin_vel_z_l2"] = rewards.reward_z_vel(
            self.ids, data0
        ) * weight_dict["lin_vel_z_l2"]

        reward_dict["angvel_xy_l2"] = rewards.reward_angvelxy_l2(
            self.ids, data0
        ) * weight_dict["angvel_xy_l2"]

        _, done = rewards.height_termination(
            data0, (0.1, 1.2)
        )

        reward_dict["termination"] = done * weight_dict["termination"]

        reward_dict["joint_torques"] = rewards.reward_jnt_frc_l2(
            data0.actuator_force
        ) * weight_dict["joint_torques"]

        qc = data0.qpos[self.ids["joint_pos_ids"]][7:]
        qc_limits = self.ids["jnt_limits"]
        reward_dict["joint_limits"] = rewards.reward_jnt_actuator_limits(
            qc, qc_limits, 0.85
        ) * weight_dict["joint_limits"]

        reward_dict["base_flat_orien"] = rewards.reward_base_orien(
            data0
        ) * weight_dict["base_flat_orien"]

        reward_dict["feet_slip"] = rewards.reward_foot_slide(
            self.ids, data0, contact
        ) * weight_dict["feet_slip"]

        reward_dict["feet_airtime"] = rewards.reward_foot_air_time(
            air_time, contact_time, 0.34, halt
        ) * weight_dict["feet_airtime"]

        reward_dict["feet_force"] = rewards.reward_foot_force(
            foot_frc
        ) * weight_dict["feet_force"]

        pos_rw = 1.0
        neg_rw = -0.3
        reward_dict["feet_contact"] = rewards.reward_foot_contact(
            contact, phase, pos_rw, neg_rw, halt
        ) * weight_dict["feet_contact"]

        reward_dict["feet_height_track"] = rewards.reward_foot_height(
            self.ids, data0, phase, 0.5, halt
        ) * weight_dict["feet_height_track"]

        reward_dict["feet_clearance"] = rewards.reward_foot_clearance(
            self.ids, data0, 0.15, 0.5, 2.0, halt
        ) * weight_dict["feet_clearance"]

        reward_dict["action_rate"] = rewards.reward_action_rate(
            act, state.info["prev_action"]
        ) * weight_dict["action_rate"]

        reward_dict["joint_acc"] = rewards.reward_jnt_acc(
            data0.qacc[self.ids["joint_vel_ids"]][6:],
        ) * weight_dict["joint_acc"]

        reward_dict["joint_vel"] = rewards.reward_jnt_vel(
            data0.qvel[self.ids["joint_vel_ids"]][6:]
        ) * weight_dict["joint_vel"]


        state.info["prev_action"] = act
        
        reward = 0.0
        for key in reward_dict.keys():
            reward_dict[key] *= self.dt
            reward += jnp.nan_to_num(reward_dict[key],
                                     nan = 0.0, posinf = 0.0, neginf = 0.0)

        metric_dict = reward_dict.copy()
        metric_dict["reward"] = reward

        state.metrics.update(
            **metric_dict
        )

        return reward, done