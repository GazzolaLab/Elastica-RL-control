__doc__ = """This script is to train or run a policy for the arm reaching to a fixed target with specific orientation. 
Case 2 in CoRL 2020 paper."""
import os
import numpy as np
import sys

import argparse
import matplotlib
import matplotlib.pyplot as plt

# Import stable baseline
from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicy_DDPG
from stable_baselines.td3.policies import MlpPolicy as MlpPolicy_TD3
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines import TRPO, DDPG, PPO1, TD3, SAC

# Import simulation environment
from set_environment import Environment


def get_valid_filename(s):
    import re

    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(title + ".png")
    plt.close()


parser = argparse.ArgumentParser()

########### training and data info ###########
parser.add_argument(
    "--total_timesteps", type=float, default=1e6,
)

parser.add_argument(
    "--SEED", type=int, default=0,
)

parser.add_argument(
    "--timesteps_per_batch", type=int, default=8000,
)

parser.add_argument(
    "--algo_name", type=str, default="TRPO",
)

args = parser.parse_args()

if args.algo_name == "TRPO":
    MLP = MlpPolicy
    algo = TRPO
    batchsize = "timesteps_per_batch"
    offpolicy = False
elif args.algo_name == "PPO":
    MLP = MlpPolicy
    algo = PPO1
    batchsize = "timesteps_per_actorbatch"
    offpolicy = False
elif args.algo_name == "DDPG":
    MLP = MlpPolicy_DDPG
    algo = DDPG
    batchsize = "nb_rollout_steps"
    offpolicy = True
elif args.algo_name == "TD3":
    MLP = MlpPolicy_TD3
    algo = TD3
    batchsize = "train_freq"
    offpolicy = True
elif args.algo_name == "SAC":
    MLP = MlpPolicy_SAC
    algo = SAC
    batchsize = "train_freq"
    offpolicy = True

# Mode 2 corresponds to fixed rotated fixed target
args.mode = 2

# Set simulation final time
final_time = 2
# Number of control points
number_of_control_points = 6
# target position
target_position = [-0.4, 0.6, 0.2]
# learning step skip
num_steps_per_update = 7

# alpha and beta spline scaling factors in normal/binormal and tangent directions respectively
args.alpha = 75
args.beta = 75

sim_dt = 2.0e-4

max_rate_of_change_of_activation = np.infty
print("rate of change", max_rate_of_change_of_activation)

# If True, train. Otherwise run trained policy
args.TRAIN = True

env = Environment(
    final_time=final_time,
    num_steps_per_update=num_steps_per_update,
    number_of_control_points=number_of_control_points,
    alpha=args.alpha,
    beta=args.beta,
    COLLECT_DATA_FOR_POSTPROCESSING=not args.TRAIN,
    mode=args.mode,
    target_position=target_position,
    target_v=0.5,
    boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
    E=1e7,
    sim_dt=sim_dt,
    n_elem=20,
    NU=30,
    dim=3.5,
    max_rate_of_change_of_activation=max_rate_of_change_of_activation,
)

name = str(args.algo_name) + "_3d-tracking_id"
identifer = name + "-" + str(args.timesteps_per_batch) + "_" + str(args.SEED)

if args.TRAIN:
    log_dir = "./log_" + identifer + "/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)


from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines import results_plotter

if args.TRAIN:
    if offpolicy:
        if args.algo_name == "TD3":
            items = {
                "policy": MLP,
                "buffer_size": int(args.timesteps_per_batch),
                "learning_starts": int(50e3),
            }
        else:
            items = {
                "policy": MLP,
                "buffer_size": int(args.timesteps_per_batch),
            }
    else:
        items = {
            "policy": MLP,
            batchsize: args.timesteps_per_batch,
        }

    model = algo(env=env, verbose=1, seed=args.SEED, **items)
    model.set_env(env)

    model.learn(total_timesteps=int(args.total_timesteps))
    # library helper
    plot_results(
        [log_dir],
        int(args.total_timesteps),
        results_plotter.X_TIMESTEPS,
        str(args.algo_name) + "_" + identifer,
    )
    plt.savefig("convergence_plot" + identifer + ".png")
    model.save("policy-" + identifer)

else:
    # Use trained policy for the simulation.
    model = TRPO.load("policy-" + identifer)

    for _ in range(9):
        obs = env.reset()

    done = False
    score = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        score += rewards
        if info["ctime"] > final_time:
            break
    print("Final Score:", score)
    env.post_processing(
        filename_video="video-" + identifer + ".mp4", SAVE_DATA=True,
    )
