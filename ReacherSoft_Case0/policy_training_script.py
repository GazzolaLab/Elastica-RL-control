__doc__ = "This script is to train multiple policies, and or hyper parameter study."

from multiprocessing import Pool
import subprocess
from datetime import datetime
import time

run_onpolicy = True
run_offpolicy = True
seed_list = [0, 1, 2, 3, 4]
batchsize_list_offpolicy = [100000, 200000, 500000, 1000000, 2000000]
algo_list_offpolicy = ["DDPG", "SAC", "TD3"]

batchsize_list_onpolicy = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
algo_list_onpolicy = ["PPO", "TRPO"]

timesteps = 10.0e6

run_comand_list = []

if run_offpolicy:
    for seed in seed_list:
        for algo in algo_list_offpolicy:
            for batchsize in batchsize_list_offpolicy:
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
                print(run_comand)

if run_onpolicy:
    for seed in seed_list:
        for algo in algo_list_onpolicy:
            for batchsize in batchsize_list_onpolicy:
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
                print(run_comand)


def run_command_fun(command):
    print(command)
    print("command started at:", datetime.now())
    start = time.time()
    subprocess.run(command, shell=True)
    # time.sleep(1)
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
    8  # make smaller than the number of cores to take advantage of multiple threads
)
pool = Pool(num_procs)
pool.map(run_command_fun, run_comand_list)
