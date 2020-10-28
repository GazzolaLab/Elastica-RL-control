__doc__ = """This script is to train multiple policies, and or hyper parameter study."""

from multiprocessing import Pool
import subprocess
from datetime import datetime
import time

run_onpolicy = True
seed_list = [0, 1, 2, 3, 4]  # 3, 4]
batchsize_list_onpolicy = [16000]
algo_list_onpolicy = ["PPO", "TRPO"]

timesteps = 0.5e6

run_comand_list = []

ctrl_pt_list = [2, 4, 6, 8]
if run_onpolicy:
    for seed in seed_list:
        for algo in algo_list_onpolicy:
            for batchsize in batchsize_list_onpolicy:
                for ctrl_pt in ctrl_pt_list:
                    run_comand = (
                        "python3 logging_bio_args.py"
                        + " --total_timesteps="
                        + str(timesteps)
                        + " --SEED="
                        + str(seed)
                        + " --timesteps_per_batch="
                        + str(batchsize)
                        + " --number_of_control_points="
                        + str(ctrl_pt)
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
    20  # make smaller than the number of cores to take advantage of multiple threads
)
pool = Pool(num_procs)
pool.map(run_command_fun, run_comand_list)
