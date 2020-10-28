##### biological simulation with meaningful parameters for octopus arm training file #####

import os
import gym
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("../../../")

from set_environment_3d_nest_cylinder_obstacles_contact_with_state_POST import Environment

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

# MODE
args.MODE = 1
args.TRAIN = False


# Set simulation final time
args.final_time = 2
fps = 1000

args.algo_name = "TRPO"
args.SEED = 0
idx = "policy-"+args.algo_name +"_nested_refined_id-1-2_75_30_1000000.0_16000_5_" + str(args.SEED)
video_name = args.algo_name + "_seed-" + str(args.SEED)


##----------------------------------------------###
# args.sim_dt = 2e-4 
# args.n_elem = 20
# args.num_steps_per_update = 7

# args.sim_dt = 1e-4 
# args.n_elem = 50
# args.num_steps_per_update = 14 

timestep_refinement = 5
args.sim_dt = 1e-4/ timestep_refinement
args.n_elem = 50
args.num_steps_per_update = 14 * timestep_refinement
##----------------------------------------------###


# Unneeded training parameters
args.timesteps_per_batch = 16000
args.total_timesteps = 1e6

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

args.number_of_control_points = 2
# Signal scaling factor
args.signal_scaling_factor = 10
# target position
args.target_position = [-0.8, 0.5, 0.35] # for contact maize 3 obstacles.
# learning step skip
args.muscle_segment_overlap = 0.75
# alpha: spline scaling factor in normal/binormal direction
args.alpha = 75
# beta: spline scaling factor in tangent direction
args.beta = 75
args.E = 1e7
args.NU = 30
args.target_v = 0.5
args.boundary = [-0.6, 0.6, 0.3, 0.9, -0.6, 0.6]

env = Environment(
    final_time= args.final_time, #args.final_time,
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
    num_obstacles=12,
    GENERATE_NEW_OBSTACLES=False,
    fps = fps
)


if args.TRAIN:
    log_dir = "./log_" + identifer + "/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)


from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines import results_plotter


if args.TRAIN:
    pass
    
else:

    # model = TRPO.load(idx)
    # model = algo.load("policy-" + identifer)
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action, _states = model.predict(obs)
        # action = np.array([1,-1,1,-1,1,-1])
        obs, rewards, done, info = env.step(action)
        score+=rewards
        if info["ctime"] > args.final_time:
            break
    print("Final Score:", score)
    env.post_processing(
        filename_video='video-' + video_name + str(args.mode) + ".mp4",
        filename_acti="video_" + video_name + ".mp4",
        x_limits = (-1.0, 1.0),
        y_limits = (-1.0, 1.0),
        z_limits = (-0.05, 1.0),
    )
