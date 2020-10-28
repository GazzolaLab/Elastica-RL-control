__doc__ = "This script is to train multiple policies, and or hyper parameter study."

from multiprocessing import Pool
import subprocess
from datetime import datetime
import time

run_onpolicy = True
run_offpolicy = False
seed_list = [0, 1, 2, 3, 4]  # , 2, 3, 4]
batchsize_list_offpolicy = [2000000]
algo_list_offpolicy = ["DDPG", "SAC", "TD3"]

algo_list_onpolicy = ["TRPO", "PPO"]
batchsize_list_onpolicy = [16000]

points_list = [2]
timesteps = 1.0e6

time_list = [5]

run_comand_list = []

if run_onpolicy:
    for points in points_list:
        for seed in seed_list:
            for algo in algo_list_onpolicy:
                for batchsize in batchsize_list_onpolicy:
                    for final_time in time_list:
                        run_comand = (
                            "python3 logging_bio_args.py"
                            + " --total_timesteps="
                            + str(timesteps)
                            + " --SEED="
                            + str(seed)
                            + " --timesteps_per_batch="
                            + str(batchsize)
                            + " --algo="
                            + str(algo)
                        )
                        run_comand_list.append(run_comand)


if run_offpolicy:
    for points in points_list:
        for seed in seed_list:
            for algo in algo_list_offpolicy:
                for batchsize in batchsize_list_offpolicy:
                    for final_time in time_list:
                        run_comand = (
                            "python3 logging_bio_args.py"
                            + " --total_timesteps="
                            + str(timesteps)
                            + " --SEED="
                            + str(seed)
                            + " --timesteps_per_batch="
                            + str(batchsize)
                            + " --algo="
                            + str(algo)
                        )
                        run_comand_list.append(run_comand)


def run_command_fun(command):
    print(command)
    print("command started at:", datetime.now())
    start = time.time()
    subprocess.run(command, shell=True)
    done = time.time()
    print(
        "command finished at:",
        datetime.now(),
        "It took:",
        (done - start) / 60,
        "minutes",
    )
    print()


num_procs = (
    6  # make smaller than the number of cores to take advantage of multiple threads
)
pool = Pool(num_procs)
pool.map(run_command_fun, run_comand_list)
