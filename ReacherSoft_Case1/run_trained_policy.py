__doc__ = """This script is to train or run a policy for the arm following randomly moving target. 
Case 1 in CoRL 2020 paper."""
import os
import numpy as np
import sys

sys.path.append("../")

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"

# Import stable baseline
from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicy_DDPG
from stable_baselines.td3.policies import MlpPolicy as MlpPolicy_TD3
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines import TRPO, DDPG, PPO1, TD3, SAC
from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines import results_plotter

# Import simulation environment
from set_environment import Environment

########### training and data info ###########
parser = argparse.ArgumentParser()
parser.add_argument(
    "--SEED", type=int, default=0,
)
args = parser.parse_args()

args.algo_name = "PPO"
MLP = MlpPolicy
algo = PPO1
offpolicy = False


# filename = "policy-TRPO_3d-tracking_id_coarse_-16000_0.zip"
# filename = "policy-TRPO_3d-tracking_id_refined_-16000_0.zip"
filename = "policy-TRPO_3d-tracking_id_80_-16000_0.zip"
# filename = "policy-TRPO_3d-tracking_id_160_-1000_0.zip"
filename = "policy-PPO_2d-fixed_20_-500_0.zip"
filename = "policy-PPO_2d-1pt-fixed_20_-500_0.zip"

scale = 1
sim_dt = 2e-4 / scale
num_steps_per_update =  7 * scale
n_elem = 20 * scale
youngs_modulus = 1e7
identifer = 'test_small_trained_1pt_' + str(sim_dt) +'_'+str(n_elem) + "_" + str(youngs_modulus*1e-7)

# Mode 4 corresponds to randomly moving target
args.mode = 4
# Set simulation final time
final_time = 5
# Number of control points
number_of_control_points = 2
# target position
target_position = [-0.4, 0.6, 0.0]
# alpha and beta spline scaling factors in normal/binormal and tangent directions respectively
args.alpha = 75
args.beta = 75
args.TRAIN = False

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
    E=youngs_modulus,
    sim_dt=sim_dt,
    n_elem=n_elem,
    NU=30,
    num_obstacles=0,
    dim=2.0,
    max_rate_of_change_of_activation=np.infty,
)

# Use trained policy for the simulation.
model = algo.load(filename)
obs = env.reset()

done = False
score = 0
while not done:
    action, _states = model.predict(obs)
    # action = np.array([0,0])
    obs, rewards, done, info = env.step(action)
    score += rewards
    if info["ctime"] > final_time:
        break
print("Final Score:", score)
env.post_processing(
    filename_video="video-" + identifer + ".mp4", SAVE_DATA=True,
)
