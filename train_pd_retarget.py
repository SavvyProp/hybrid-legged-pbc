from playground.booster import track
from playground.booster.track import ppo_params
from playground.booster.randomize_track import domain_randomize
from datetime import datetime
import functools
import matplotlib.pyplot as plt
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from brax.io import model

def make_trainfn():
    env = track.Track()
    eval_env = track.Track()

    ppo_params_ = env.override_config(ppo_params)


    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]


    def progress(num_steps, metrics):

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, ppo_params_["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
        plt.show()

    ppo_training_params = dict(ppo_params_)

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params_:
        del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params_.network_factory
    )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn = domain_randomize,
        progress_fn=progress,
    )

    return train_fn, env, eval_env, wrapper.wrap_for_brax_training

if __name__ == "__main__":
    train_fn, env, eval_env, wrapper = make_trainfn()
    make_inference_fn, params, metrics = train_fn(environment= env,
                                        eval_env= eval_env,
                                        wrap_env_fn=wrapper)

    model.save_params("walk_policy", params)