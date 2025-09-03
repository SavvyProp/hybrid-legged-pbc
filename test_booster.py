from mujoco_playground.config import locomotion_params

env_name = "T1JoystickFlatTerrain"

ppo_params = locomotion_params.brax_ppo_config(env_name)

print(ppo_params)