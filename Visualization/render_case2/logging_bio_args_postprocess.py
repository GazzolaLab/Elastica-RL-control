##### biological simulation with meaningful parameters for octopus arm training file #####

import os
import gym
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("../")
from set_environment_3d_fixed_end_spline import Environment

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


# MLP = MlpPolicy_DDPG
# algo = DDPG
# algo_name = "DDPG"
# batchsize = "nb_rollout_steps"

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
if True:
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
        "--timesteps_per_batch", type=int, default=4000,
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


for test_num in [26, 33, 34, 39, 58, 73]:
    basenum = 0
    args = parser.parse_args()

    args.algo_name = "SAC"
    args.mode = 2
    target_position = [0.1, 0.9, 0.1]   #[x,y,z]
    args.plane_rotation = 0
    final_time = 1  # Signal scaling factor
    fps = 360

    # idx = "../ReacherSoft_3DTracking_OffPolicy_v2/policy-SAC_3d-tracking_buffer_id-2000000-0.98_0_tanh_64_64"
    # idx = "../ReacherSoft_Case2/policy-SAC_3d-orientation-v4_id-2000000-0.98_3_tanh_64_64"
    idx = "policy-SAC_3d-orientation-v4_id-2000000-0.98_3_tanh_64_64"

    video_name = "SAC-HQ-short-"+str(test_num + basenum) + "score_"

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

    model = algo.load(idx)



    signal_scaling_factor = 10
    # Number of muscle segments
    number_of_muscle_segments = 6
    # learning step skip
    num_steps_per_update = 7
    # overlap of muscles is 0
    muscle_segment_overlap = 0.75
    # alpha: basis function strength
    args.alpha = 75
    args.beta = 75 
    # MODE

    sim_dt = 2.0e-4

    # time_to_ramp = 0.1
    # max_rate_of_change_of_activation = sim_dt/time_to_ramp
    max_rate_of_change_of_activation = np.infty
    print('rate of change', max_rate_of_change_of_activation)

    args.TRAIN = False

    if args.mode == 4:
        dim = 3.0
    else:
        dim = 3.5 

    env = Environment(
        final_time=final_time,
        signal_scaling_factor=signal_scaling_factor,
        num_steps_per_update=num_steps_per_update,
        number_of_muscle_segment=number_of_muscle_segments,
        alpha=args.alpha, 
        beta=args.beta,
        muscle_segment_overlap=muscle_segment_overlap,
        COLLECT_DATA_FOR_POSTPROCESSING= not args.TRAIN,
        mode=args.mode,
        target_position=target_position,
        target_v = 0.5,
        boundary = [-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
        E = 1e7,
        sim_dt=sim_dt,
        n_elem = 20,
        NU=30,
        num_obstacles = 0,
        dim = dim,
        max_rate_of_change_of_activation=max_rate_of_change_of_activation,
        fps = fps,
        plane_rotation = args.plane_rotation,
    )

    from stable_baselines.results_plotter import ts2xy, plot_results
    from stable_baselines import results_plotter

    if args.TRAIN:
        pass
        
    else:

        for _ in range(test_num+1+basenum):
            obs = env.reset()
        done = False
        score = 0
        while not done:  
            action, _states = model.predict(obs)
            # action = np.array([0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

            obs, rewards, done, info = env.step(action) 
            score += rewards
            # print(score)
            if info["ctime"] > final_time:
                break
        print(' ')
        print("#################################################################")
        print("Final Score:", score)
        print("#################################################################")
        print(' ')

        env.post_processing(
            filename_video='video-' + video_name + str(int(score)) + "_MODE" + str(args.mode) + ".mp4",
            filename_acti="trpo_logging_acti_" + video_name + ".mp4",
            save_prefix=str(test_num)+"_1sec-360fps_"
        ) 
