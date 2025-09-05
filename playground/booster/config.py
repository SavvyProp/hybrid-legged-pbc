from mujoco_playground.config import locomotion_params
from ml_collections import config_dict

env_name = "T1JoystickFlatTerrain"

ppo_params = locomotion_params.brax_ppo_config(env_name)

ppo_params.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type = "normal",
        noise_std_type = "scalar"
    )

ppo_params.num_timesteps = 300_000_000
#ppo_params.normalize_observations = False