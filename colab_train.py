
from datetime import datetime
import functools
from brax import envs
#from brax.training.agents.ppo import train as ppo
from agents.ppo import train
from brax.training.agents.ppo import networks as ppo_networks
import networks.mlp as mlp
from brax.io import model
from matplotlib import pyplot as plt
from envs.booster_flatwalk_pd import FlatwalkEnv, metrics_dict

def make_trainfn():
    envs.register_environment('FlatwalkEnv', FlatwalkEnv)
    env = envs.get_environment('FlatwalkEnv')
    eval_env = envs.get_environment('FlatwalkEnv')    
    make_networks_factory = functools.partial(
        #mlp.make_ppo_networks, 
        ppo_networks.make_ppo_networks,
        value_hidden_layer_sizes=(512, 256, 256, 128),
        policy_hidden_layer_sizes=(1024, 512, 512, 256, 256),
        distribution_type = "normal",
        noise_std_type = "scalar"
    )

    train_fn = functools.partial(
            train, num_timesteps=200000000, num_evals=15, episode_length=1000,
            normalize_observations=False, unroll_length=20, num_minibatches = 32,
            num_updates_per_batch = 4, discounting = 0.99, learning_rate = 1e-3,
            entropy_cost=0.005, num_envs=8192, batch_size=256, clipping_epsilon=0.2,
            num_resets_per_eval=1, action_repeat=1, max_grad_norm=1.0,
            reward_scaling=1.0,
            num_eval_envs = 256,
            normalize_advantage = True,
            network_factory=make_networks_factory,
    )

    x_data = []
    y_data = {}
    for name in metrics_dict.keys():
        y_data[name] = []
    prefix = "eval/episode_"
    times = [datetime.now()]

    y_entropy = [0.]
    y_value = [0.]
    y_policy_kl = [0.]

    def progress(num_steps, metrics):
        print(metrics)
        times.append(datetime.now())
        x_data.append(num_steps)
        for key in y_data.keys():
            y_data[key].append(metrics[prefix + key])
        try:
            y_entropy.append(metrics['training/entropy_loss'])
            y_value.append(metrics['training/v_loss'])
            y_policy_kl.append(metrics['training/policy_loss'])
        except:
            print("No training losses found in metrics")
        plt.xlim([0, train_fn.keywords['num_timesteps']])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title('{}'.format(metrics['eval/episode_reward']))
        for key in y_data.keys():
            num = float(metrics[prefix + key])
            plt.plot(x_data, y_data[key], label = key + " {:.2f}".format(num))
        plt.legend()
        plt.show()
        plt.clf()

        plt.xlabel('# environment steps')
        plt.ylabel('loss')
        plt.title("Training Losses")
        plt.plot(x_data, y_entropy, label = 'entropy loss {:.2f}'.format(y_entropy[-1]))
        plt.plot(x_data, y_value, label = 'value loss {:.2f}'.format(y_value[-1]))
        plt.plot(x_data, y_policy_kl, label = 'policy loss {:.2f}'.format(y_policy_kl[-1]))
        plt.legend()
        plt.show()

        #plt.savefig("trainplot.png")

    return train_fn, env, progress, eval_env