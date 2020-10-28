##### biological simulation with meaningful parameters for octopus arm training file #####

import os
import gym
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("../../")

from set_environment_3d_nest_cylinder_obstacles_contact_with_state_2 import Environment

import argparse
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"

from stable_baselines.bench.monitor import Monitor, get_monitor_files, load_results
from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines.common.callbacks import BaseCallback

import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicy_DDPG
from stable_baselines.td3.policies import MlpPolicy as MlpPolicy_TD3
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines import TRPO, DDPG, PPO1, TD3, SAC

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

########### timing info ###########
parser.add_argument(
    "--final_time", type=float, default=10.0,
)

parser.add_argument(
    "--sim_dt", type=float, default=2.0e-4,
)

parser.add_argument(
    "--per_update_time", type=float, default=0.01,
)

parser.add_argument("--num_steps_per_update", type=float, default=7)

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
    "--alpha", type=float, default=75,
)

parser.add_argument(
    "--mode", type=int, default=1,
)
########### target info ###########
parser.add_argument(
    "--boundary", nargs="*", type=float, default=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
)

parser.add_argument(
    "--target_position", nargs="*", type=float, default=[0.3, 0.65, 0.0],
)

parser.add_argument(
    "--target_v", type=float, default=0.5,
)
########### physics info ###########
parser.add_argument(
    "--E", type=float, default=1e7,
)

parser.add_argument(
    "--NU", type=float, default=20,
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
    "--total_timesteps", type=float, default=2e6,
)

parser.add_argument(
    "--COLLECT_DATA_FOR_POSTPROCESSING", type=bool, default=False,
)

parser.add_argument(
    "--TRAIN", type=int, default=True,
)

parser.add_argument(
    "--SEED", type=int, default=0,
)

parser.add_argument(
    "--timesteps_per_batch", type=int, default=50000,
)

parser.add_argument(
    "--LAM", type=float, default=0.98,
)

parser.add_argument(
    "--act_fun_str", type=str, default="tanh",
)

parser.add_argument(
    '--net_arch', type=ast.literal_eval, default=[64, 64], 
)

parser.add_argument(
    '--algo_name', type=str, default="TRPO", 
)

parser.add_argument(
    '--number_of_control_points', type=int, default=4,
)

args = parser.parse_args()

# args.algo_name = "TRPO"
# args.timesteps_per_batch = 64000

# args.algo_name = "SAC"
# args.timesteps_per_batch = 1000000
# args.total_timesteps = 1e6
args.final_time = 5

args.algo_name = "TRPO"
args.number_of_control_points = 2
args.SEED = 5

# idx = "../random_cylinder_nest_to_send_Noel/policy-TRPO_nested_shifted-id-1-4_75_30_300000.0_16000_5_1"
idx = "../random_obstacles_hold/hold2/random_cylinder_nest_to_send_Noel/policy-TRPO_nested_id-1-2_75_30_2500000.0_64000_5.0_1"
# convergence_plotTRPO_nested_id-1-2_75_30_1000000.0_128000_2.0_1
# idx = "../ReacherSoft_Case2/policy-SAC_3d-orientation-v4_id-2000000-0.98_0_tanh_64_64"
video_name = "TRPO_success"
# MODE
args.MODE = 1
args.TRAIN = False


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

model = algo.load(idx)


# Signal scaling factor
args.signal_scaling_factor = 10
# Number of muscle segments
#args.number_of_control_points = 4
# target position
# target_position = [-0.5, 0.5, 0.0]
# args.target_position = [-0.5, 0.5, 0.25] # for reward engineering maize 3 obstacles
args.target_position = [-0.8, 0.5, 0.35] # for contact maize 3 obstacles.
# learning step skip
# num_steps_per_update = 25
args.num_steps_per_update = 14  # 7
# overlap of muscles is 0
args.muscle_segment_overlap = 0.75
# alpha: spline scaling factor in normal/binormal direction
args.alpha = 75
# alpha = 50
# beta: spline scaling factor in tangent direction
args.beta = 75
args.E = 1e7
args.sim_dt = 1e-4  # 2e-4
args.n_elem = 50
args.NU = 30
args.target_v = 0.5
args.boundary = [-0.6, 0.6, 0.3, 0.9, -0.6, 0.6]

env = Environment(
    final_time=args.final_time,
    signal_scaling_factor=args.signal_scaling_factor,
    num_steps_per_update=args.num_steps_per_update,
    number_of_control_points=args.number_of_control_points,
    alpha=args.alpha,
    beta=args.beta,
    COLLECT_DATA_FOR_POSTPROCESSING= not args.TRAIN,
    mode=args.MODE,
    target_position=args.target_position,
    target_v=args.target_v,
    boundary=args.boundary,
    E=args.E,
    sim_dt=args.sim_dt,
    n_elem=args.n_elem,
    NU=args.NU,
    num_obstacles=10,
    GENERATE_NEW_OBSTACLES=False,
)

name = str(args.algo_name) + "_nested_id-"
identifer = (
    name
    # "bio_case_mode_stiff_newtorque_strong-extra_min-vel_high-viscous_25_obstacle_grid_with_-2.0_"
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



from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines import results_plotter


if args.TRAIN:
    pass

else:

    obs = env.reset()
    done = False
    score = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        score += rewards
        if info["ctime"] > args.final_time:
            break
    # print(obs)
    print("Final score is:", score)
    print(video_name)
    env.post_processing(
        filename_video=video_name + ".mp4",
        filename_acti="video_" + video_name + ".mp4",
        x_limits = (-1.0, 1.0),
        y_limits = (-1.0, 1.0),
        z_limits = (-0.05, 1.0),
    )
