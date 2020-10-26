## Supplementary code for "Elastica: A compliant mechanics environment for soft robotic control"

Elastica is a simulation environment to simulate assemblies of one-dimensional soft, slender structures using Cosserat rod theory. 
In this paper, Elastica has been interfaced with Stable Baselines using the typical OpenAI Gym format to investigate the ability 
of RL to dynamically control a compliant robotic arm.

Five different RL model-free algorithms from the Stable Baselines implementations are used. Two of them are on-policy 
algorithms: Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO) and 
three of them are off-policy algorithms: Soft Actor Critic (SAC), Deep Deterministic Policy Gradient (DDPG), and 
Twin Delayed DDPG (TD3). Four different cases are considered with detailed explanations given in the paper. 

### Case 1: 3D tracking of a randomly moving target
In this case, the arm is continuously tracking a randomly moving target in 3D space. Actuations are only allowed in 
normal and binormal directions with 6 control points in each direction. 

* To replicate training using different RL algorithms, run `logging_bio_args.py` located in the `ReacherSoft_Case1/ ` folder. 
You can train policies using the five RL algorithms considered by passing the algorithm name as a command-line argument i.e. `--algo_name TRPO`.
 Also, you can control the total number of training timesteps, the random seed, and the timestep 
per batch as command-line arguments, i.e. `--total_timesteps 1E6`, `--SEED 0`, `--timesteps_per_batch 2048`.
 In addition to that, you can choose a different number of control points or torque scaling factor by changing the 
`number_of_control_points` and `alpha` variables inside `logging_bio_args.py` respectively.  
* To replicate the hyperparameter tuning, run the code `ReacherSoft_Case1/policy_training_script.py`. 
Note that the number of CPUs should be edited appropriately in the script. Runtime is 12-24 hours per individual case.
* Code for initializing the Elastica simulation environment is located in `ReacherSoft_Case1/set_environment.py`. 
Specific details on how Case 1 was implemented are in this file. 
* Post-processing scripts are located in `ReacherSoft_Case1/post_processing.py`



### Case 2: Reaching to randomly located target with a defined orientation
In this case, the arm is reaching the randomly positioned stationary target, while re-orienting itself to match the orientation
of the target. Actuations are allowed in normal, binormal, and tangent directions with 6 control points in each direction.

* To replicate training using different RL algorithms, run `logging_bio_args.py` located in the `ReacherSoft_Case2/ ` folder. 
You can train policies using the five RL algorithms considered by passing the algorithm name as a command-line argument 
i.e. `--algo_name TRPO`. Also, you can control the total number of training timesteps, the random seed, and the timestep 
per batch as command-line arguments, i.e. `--total_timesteps 1E6`, `--SEED 0`, `--timesteps_per_batch 2048`. 
In addition to that, you can choose a different number of control points or torque scaling factor by changing the 
`number_of_control_points` and `alpha` variables inside `logging_bio_args.py` respectively.  
* To replicate the hyperparameter tuning, run the code `ReacherSoft_Case2/policy_training_script.py`. 
Note that the number of CPUs should be edited appropriately in the script. Runtime is 12-24 hours per individual case. 
* Code for initializing the Elastica simulation environment is located in `ReacherSoft_Case2/set_environment.py`. 
Specific details on how Case 2 was implemented are in this file. 
* Post-processing scripts are located in `ReacherSoft_Case2/post_processing.py`


### Case 3: Underactuated maneuvering between structured obstacles
In this case, the arm is reaching a stationary target placed behind an array of eight obstacles with an opening through
which the arm must maneuver to reach the target. Target is placed in the normal plane so that only in-plane actuation is
required. Thus actuation only in the normal direction is allowed. Case 3 has two subcases. First one is training using 
2 manually placed control points at 40% and 90% of the arm and the second one is training using 2, 4, 6, and 8 
equidistant control points. Code for manually selected control
points are located in `ReacherSoft_Case3/ReacherSoft_Case3_main-text/` folder and code for equidistant control points are
located in `ReacherSoft_Case3/ReacherSoft_Case3_SI-ctrl_pts/`.

* To replicate the manually placed two control points training:
   * Run `ReacherSoft_Case3/ReacherSoft_Case3_main-text/logging_bio_args_OnPolicy.py` and run
    `ReacherSoft_Case3/ReacherSoft_Case3_main-text/logging_bio_args_OffPolicy.py` for on-policy and off-policy algorithms
     respectively. You can train policies using the five RL algorithms considered by passing the algorithm name as a 
     command-line argument i.e. `--algo_name TRPO`. Also, you can control the total number of training timesteps, 
     the random seed, and the timestep per batch as command-line arguments, i.e. `--total_timesteps 1E6`, `--SEED 0`, `--timesteps_per_batch 2048`. 
     In addition to that, you can choose a different torque scaling factor by changing the `alpha` variable inside 
    `logging_bio_args_OnPolicy.py` or  `logging_bio_args_OffPolicy.py` depending on the policy. The number of control points is fixed for this case and it is two. 
    * To replicate the hyperparameter tuning, run the code `ReacherSoft_Case3/ReacherSoft_Case3_main-text/policy_training_script_OnPolicy.py`
    or `ReacherSoft_Case3/ReacherSoft_Case3_main-text/policy_training_script_OffPolicy.py`. 
    Note that the number of CPUs should be edited appropriately in the script. Runtime is 3-4 hours per individual case.  
    * Code for initializing the Elastica simulation environment is located in `ReacherSoft_Case3/ReacherSoft_Case3_main-text/set_environment.py`. 
    Specific details on how the first subcase of Case 3 was implemented are in this file. 
    * Post-processing scripts are located in `ReacherSoft_Case3/ReacherSoft_Case3_main-text/post_processing.py`.

* To replicate the equidistantly placed control points training:
   * Run `ReacherSoft_Case3/ReacherSoft_Case3_SI-ctrl_pts/logging_bio_args.py`. You can train policies using the five RL
    algorithms considered by passing the algorithm name as a command-line argument i.e. `--algo_name TRPO`. 
    Also, you can control the total number of training timesteps, the random seed, and the timestep per batch as 
    command-line arguments, i.e. `--total_timesteps 1E6`, `--SEED 0`, `--timesteps_per_batch 2048`. In addition to that,
     you can choose a different number of control points or torque scaling factor by changing the `number_of_control_points` 
     and `alpha` variables inside `logging_bio_args_OnPolicy.py` or  `logging_bio_args_OffPolicy.py`respectively. 
    * To replicate the hyperparameter tuning, run the code `ReacherSoft_Case3/ReacherSoft_Case3_SI-ctrl_pts/policy_training_script.py`.
    Note that the number of CPUs should be edited appropriately in the script. Runtime is 3-4 hours per individual case.  
    * Code for initializing the Elastica simulation environment is located in `ReacherSoft_Case3/ReacherSoft_Case3_SI-ctrl_pts/set_environment.py`. 
    Specific details on how the second subcase of Case 3 was implemented are in this file. 
    * Post-processing scripts are located in `ReacherSoft_Case3/ReacherSoft_Case3_SI-ctrl_pts/post_processing.py`. 


### Case 4: Underactuated maneuvering between unstructured obstacles
In this case, the arm is reaching a stationary target by maneuvering around an unstructured nest of twelve randomly located
obstacles. Actuation for this case is similar to Case 3, using two manually placed control points at 40% and 90% of the arm. 
Different than Case 3 actuations in normal and binormal directions are allowed. 

* To replicate training using different RL algorithms, run `logging_bio_args.py` located in the `ReacherSoft_Case4/ ` folder. 
You can train policies using the five RL algorithms considered by passing the algorithm name as a command-line argument 
i.e. `--algo_name TRPO`. Also, you can control the total number of training timesteps, the random seed, and the timestep 
per batch as command-line arguments, i.e. `--total_timesteps 1E6`, `--SEED 0`, `--timesteps_per_batch 2048`. 
In addition to that, you can choose a different torque scaling factor by changing the `alpha` variable inside 
`logging_bio_args.py`. The number of control points is fixed for this case and it is two. 
* To replicate the hyperparameter tuning, run the code `ReacherSoft_Case4/policy_training_script.py`. 
Note that the number of CPUs should be edited appropriately in the script. Runtime is 6-8 hours per individual case. 
* Code for initializing the Elastica simulation environment is located in `ReacherSoft_Case4/set_environment.py`. 
Specific details on how Case 2 was implemented are in this file. 
* Post-processing scripts are located in `ReacherSoft_Case4/post_processing.py`






