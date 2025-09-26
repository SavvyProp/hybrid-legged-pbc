from ml_collections import config_dict

ppo_params = config_dict.create(
      num_timesteps=200_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=500,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=0.005,
      num_envs=8192,
      batch_size=256,
      clipping_epsilon=0.2,
      max_grad_norm=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(512, 256, 256, 128),
          value_hidden_layer_sizes=(512, 256, 256, 128),
          policy_obs_key="state",
          value_obs_key="privileged_state",
          #distribution_type = "tanh_normal",
          #noise_std_type = "log"
      ),
      num_resets_per_eval=1,
  )


ppo_params.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(1024, 512, 512, 256, 256),
        value_hidden_layer_sizes=(512, 512, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type = "normal",
        noise_std_type = "scalar"
    )

ppo_params.num_timesteps = 200_000_000
#ppo_params.normalize_observations = False