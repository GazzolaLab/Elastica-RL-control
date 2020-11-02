__doc__ = """This script is to train or run a policy for the arm reaching a target in between obstacles. 
Case 3 supplementary information in CoRL 2020 paper."""
import os
import numpy as np
import sys

import argparse
import ast

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

    Parameters
    ----------
    values : numpy.ndarray
    window : int

    Returns
    -------
    numpy.ndarray
    """

    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """

    Parameters
    ----------
    log_folder : str
        the save location of the results to plot
    title : str
        the title of the task to plot
    Returns
    -------

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

########### timing info ###########
parser.add_argument(
    "--final_time", type=float, default=10.0,
)

parser.add_argument(
    "--sim_dt", type=float, default=5e-5,
)

parser.add_argument(
    "--per_update_time", type=float, default=0.01,
)
########### env arm info ###########
parser.add_argument(
    "--signal_scaling_factor", type=float, default=10.0,
)

parser.add_argument(
    "--number_of_muscle_segment", type=int, default=6,
)

parser.add_argument(
    "--muscle_segment_overlap", type=float, default=0.75,
)

parser.add_argument(
    "--alpha", type=float, default=2.0,
)

parser.add_argument(
    "--mode", type=int, default=4,
)
########### target info ###########
parser.add_argument(
    "--boundary", nargs="*", type=float, default=[-0.5, 0.5, 0.3, 0.8, -0.5, 0.5],
)

parser.add_argument(
    "--target_position", nargs="*", type=float, default=[0.75, 0.75, 0.0],
)

parser.add_argument(
    "--target_v", type=float, default=0.25,
)
########### physics info ###########
parser.add_argument(
    "--E", type=float, default=1e5,
)

parser.add_argument(
    "--NU", type=float, default=0.75,
)

parser.add_argument(
    "--max_rate_of_change_of_activation", type=float, default=float("inf"),
)
########### reward info ###########
parser.add_argument(
    "--acti_diff_coef", type=float, default=9e-1,
)

parser.add_argument(
    "--acti_coef", type=float, default=1e-1,
)
########### training and data info ###########
parser.add_argument(
    "--total_timesteps", type=float, default=1e6,
)

parser.add_argument(
    "--COLLECT_DATA_FOR_POSTPROCESSING", type=bool, default=True,
)

parser.add_argument(
    "--TRAIN", type=int, default=1,
)

parser.add_argument(
    "--SEED", type=int, default=0,
)

parser.add_argument(
    "--timesteps_per_batch", type=int, default=2048 * 2,
)

parser.add_argument(
    "--LAM", type=float, default=0.98,
)

parser.add_argument(
    "--act_fun_str", type=str, default="tanh",
)

parser.add_argument(
    "--net_arch", type=ast.literal_eval, default=[64, 64],
)

parser.add_argument(
    "--algo_name", type=str, default="TRPO",
)

parser.add_argument(
    "--number_of_control_points", type=int, default=4,
)

args = parser.parse_args()
args.final_time = 5.0

# learning step skip
args.num_steps_per_update = 14  # 14
args.sim_dt = 1e-4  # 1e-4 # use 2.5e-5 for rendering images
args.n_elem = 50  # 50

# args.algo_name = "SAC"
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

# target position
args.target_position = [-0.8, 0.5, 0.15]  # for contact maize 3 obstacles.
# alpha and beta spline scaling factors in normal/binormal and tangent directions respectively
args.alpha = 75
args.beta = 75
args.E = 1e7
args.NU = 30
args.target_v = 0.5
args.boundary = [-0.6, 0.6, 0.3, 0.9, -0.6, 0.6]

# mode 1 corresponds to fixed target
args.MODE = 1
args.TRAIN = True

env = Environment(
    final_time=args.final_time,
    num_steps_per_update=args.num_steps_per_update,
    number_of_control_points=args.number_of_control_points,
    alpha=args.alpha,
    beta=args.beta,
    COLLECT_DATA_FOR_POSTPROCESSING=not args.TRAIN,
    mode=args.MODE,
    target_position=args.target_position,
    target_v=args.target_v,
    boundary=args.boundary,
    E=args.E,
    sim_dt=args.sim_dt,
    n_elem=args.n_elem,
    NU=args.NU,
    num_obstacles=8,
    COLLECT_CONTROL_POINTS_DATA=not args.TRAIN,
)


name = str(args.algo_name) + "_Case3-ctrl_pts_id"
identifer = (
    name
    + str(args.mode)
    + "-"
    + str(args.number_of_control_points)
    + "_"
    + str(args.alpha)
    + "_"
    + str(args.NU)
    + "_"
    + str(args.total_timesteps)
    + "_"
    + str(args.timesteps_per_batch)
    + "_"
    + str(args.final_time)
    + "_"
    + str(args.SEED)
)


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
                "batch_size": int(1000),
            }
        else:
            items = {
                "policy": MLP,
                "buffer_size": int(args.timesteps_per_batch),
                "batch_size": int(1000),
            }
    else:
        items = {
            "policy": MLP,
            batchsize: args.timesteps_per_batch,
        }

    model = algo(env=env, verbose=1, seed=args.SEED, **items)

    model.set_env(env)
    print("Training for ", args.total_timesteps)

    model.learn(total_timesteps=int(args.total_timesteps))
    # library helper
    plot_results(
        [log_dir],
        int(args.total_timesteps),
        results_plotter.X_TIMESTEPS,
        "TRPO muscle" + identifer,
    )
    plt.savefig("convergence_plot" + identifer + ".png")
    model.save("policy-" + identifer)

else:

    model = TRPO.load("policy-" + identifer)
    video_name = identifer

    obs = env.reset()
    done = False
    score = 0
    while not done:
        action, _states = model.predict(obs)

        obs, rewards, done, info = env.step(action)
        score += rewards
        if info["ctime"] > args.final_time:
            break
    print("Final Score:", score)
    env.post_processing(filename_video="video-" + video_name + ".mp4",)
