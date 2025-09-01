from datetime import datetime
import functools
from brax import envs
from agents.ppo import train
#from brax.training.agents.ppo import networks as ppo_networks
import networks.mlp as mlp
from brax.io import model
from matplotlib import pyplot as plt
from envs.booster_flatwalk_pbc import FlatwalkPBCEnv, metrics_dict
import os

# ensure 'training' directory exists
training_dir = "training"
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

run_dir = "test_1"
if not os.path.exists(training_dir + "/" + run_dir):
    os.makedirs(training_dir + "/" + run_dir)

save_dir = os.path.join(training_dir, run_dir)

envs.register_environment('flatwalkpbc', FlatwalkPBCEnv)
env = envs.get_environment('flatwalkpbc')
eval_env = envs.get_environment('flatwalkpbc')

make_networks_factory = functools.partial(
    mlp.make_ppo_networks,
    policy_hidden_layer_sizes=(1024, 512, 512, 256, 128)
)



train_fn = functools.partial(
        train, num_timesteps=6000, num_evals=40, episode_length=1000,
        normalize_observations=True, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting = 0.97, learning_rate = 1e-3,
        entropy_cost=0.005, num_envs=2, batch_size=2, clipping_epsilon=0.2,
        num_resets_per_eval=1, action_repeat=1, max_grad_norm=1.0,
        reward_scaling=1.0,
        network_factory=make_networks_factory,
)

x_data = []
y_data = {}
for name in metrics_dict.keys():
    y_data[name] = []
prefix = "eval/episode_"
times = [datetime.now()]

count = 0

def progress(num_steps, metrics):
    print(metrics)
    times.append(datetime.now())
    x_data.append(num_steps)
    for key in y_data.keys():
        y_data[key].append(metrics[prefix + key])
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title('{}'.format(metrics['eval/episode_reward']))
    for key in y_data.keys():
        num = float(metrics[prefix + key])
        plt.plot(x_data, y_data[key], label = key + " {:.2f}".format(num))
    plt.legend()
    plt.savefig(save_dir + "/progress{}.png")
    print("saved plot")

make_inference_fn, params, _ = train_fn(
    environment = env,
    progress_fn = progress,
    eval_env = eval_env
)


model.save_params(save_dir + "/walk_policy", params)