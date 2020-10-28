__doc__ = "This script is to train multiple policies, and or hyper parameter study."

from multiprocessing import Pool
import subprocess
from datetime import datetime
import time

run_offpolicy = True
seed_list = [0, 1, 2, 3, 4]

timesteps = 0.5e6

batchsize_list_offpolicy = [50000, 5000000]
algo_list_offpolicy = ["SAC"]
tau_list = [0.005, 0.05]

dt_list = [1e-4]
n_elem_list = [50]

run_comand_list = []
if run_offpolicy:
    for seed in seed_list:
        for algo in algo_list_offpolicy:
            for batchsize in batchsize_list_offpolicy:
                for tau in tau_list:
                    run_comand = (
                        "python3 logging_bio_args_OffPolicy.py"
                        + " --total_timesteps="
                        + str(timesteps)
                        + " --SEED="
                        + str(seed)
                        + " --timesteps_per_batch="
                        + str(batchsize)
                        + " --algo="
                        + str(algo)
                        + " --tau="
                        + str(tau)
                        + " --sim_dt="
                        + str(dt_list[0])
                        + " --n_elem="
                        + str(n_elem_list[0])
                    )
                    run_comand_list.append(run_comand)
                    print(run_comand)

batchsize_list_offpolicy = [50000, 5000000]
algo_list_offpolicy = ["DDPG", "TD3"]
tau_list = [0.001, 0.005, 0.05]

if run_offpolicy:
    for seed in seed_list:
        for algo in algo_list_offpolicy:
            for batchsize in batchsize_list_offpolicy:
                for tau in tau_list:
                    run_comand = (
                        "python3 logging_bio_args_OffPolicy.py"
                        + " --total_timesteps="
                        + str(timesteps)
                        + " --SEED="
                        + str(seed)
                        + " --timesteps_per_batch="
                        + str(batchsize)
                        + " --algo="
                        + str(algo)
                        + " --tau="
                        + str(tau)
                        + " --sim_dt="
                        + str(dt_list[0])
                        + " --n_elem="
                        + str(n_elem_list[0])
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
    15  # make smaller than the number of cores to take advantage of multiple threads
)
pool = Pool(num_procs)
pool.map(run_command_fun, run_comand_list)
