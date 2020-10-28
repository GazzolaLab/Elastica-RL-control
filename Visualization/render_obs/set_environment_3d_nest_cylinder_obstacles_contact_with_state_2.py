########### This file is for only normal torques: remove twisting and rotation in binormal direction ###########
########### This file is trying to match to OpenAI Gym interface ###########
########### Modify it to behave as moving target instead of fixed target ###########

"""
Available Tasks to be considered:
1. fixed target position to be reached
2. random fixed target position to be reached
3. fixed trajectory to be followed
4. random trajectory to followed
"""

import gym
from gym import spaces

import numpy as np
import numba
import copy
import sys

sys.path.append("../")


from single_arm_basis_postprocessing import (

    plot_video_with_sphere_cylinder,
    plot_video_with_sphere_obstacles
)

from MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *


# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


##################################### working on RL framework for basis function setup #####################################
class Environment(gym.Env):

    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    """
    FOUR modes: (specified by mode)
    1. fixed target position to be reached (default: need target_position parameter)
    2. random fixed target position to be reached
    3. fixed trajectory to be followed
    4. random trajectory to be followed (no need to worry in current phase)
    """

    def __init__(
        self,
        final_time,
        n_elem,
        signal_scaling_factor,
        num_steps_per_update,
        number_of_control_points,
        alpha,  # spline magnitude in normal/binormal direction
        beta,  # spline magnitude in tangent direction
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=2.0e-5,
        mode=1,
        num_obstacles=0,
        GENERATE_NEW_OBSTACLES=True,
        *args,
        **kwargs,
    ):
        super(Environment, self).__init__()
        self.dim = 3.0
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        self.n_elem = n_elem

        # Simulation parameters
        self.final_time = final_time
        self.h_time_step = sim_dt  # this is a stable timestep
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        print("Total steps", self.total_steps)

        # Video speed
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Number of muscle segments
        self.number_of_control_points = number_of_control_points

        # Spline scaling factor in normal/binormal direction
        self.alpha = alpha

        # Spline scaling factor in tangent direction
        self.beta = beta

        # signal scaling factor
        self.signal_scaling_factor = signal_scaling_factor

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        #### define action_space and observation_space ####
        ########## choose one or the other for 2D or 3D ############
        # normal, binormal and tangent activation (3D)
        if self.dim == 2.0:
            ## normal activation (2D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(self.number_of_control_points)
        if self.dim == 3.0 or self.dim == 2.5:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(2 * self.number_of_control_points)
        if self.dim == 3.5:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(3 * self.number_of_control_points)

        # ## normal activation (2D)
        # self.action_space = spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self.number_of_muscle_segment,),
        #     dtype=np.float64,
        # )

        ## includes the full 3D information
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(39,), dtype=np.float64
        # )
        ## includes only the 2D information
        # print('length', len(np.ones(40+1)[0 : : 2]))
        # num_rod_state = len(np.ones(50 + 1)[0::2])
        self.obs_state_points = 10
        num_points = int(self.n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(self.n_elem + 1)[0::num_points])
        # r_s_a[0 : 40 + 1 : 2]
        # print(num_rod_state*2 + 6)

        self.N_OBSTACLE = num_obstacles
        self.number_of_points_on_cylinder = 5

        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(num_rod_state * 3 + 3 + 3,),
        #     dtype=np.float64,
        # )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                num_rod_state * 3
                + 4  # 8
                + 3  # 11
                + self.N_OBSTACLE * self.number_of_points_on_cylinder * 3,
            ),
            dtype=np.float64,
        )

        # here we specify 4 tasks that can possibly used
        assert (
            "target_position" in kwargs
        ), "need to specify target_position in all modes"

        self.target_position = kwargs["target_position"]
        self.mode = mode

        if self.mode is 2:
            assert "boundary" in kwargs, "need to specify boundary in mode 2"
            self.boundary = kwargs["boundary"]

        if self.mode is 3:
            assert "target_v" in kwargs, "need to specify target_v in mode 3"
            self.target_v = kwargs["target_v"]

        if self.mode is 4:
            assert (
                "boundary" and "target_v" in kwargs
            ), "need to specify boundary and target_v in mode 4"
            self.boundary = kwargs["boundary"]
            self.target_v = kwargs["target_v"]

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

        self.time_tracker = np.float64(0.0)

        self.acti_diff_coef = kwargs.get("acti_diff_coef", 9e-1)

        self.acti_coef = kwargs.get("acti_coef", 1e-1)

        self.max_rate_of_change_of_activation = kwargs.get(
            "max_rate_of_change_of_activation", np.infty
        )
        self.max_rate_of_change_of_activation = 0.1

        self.E = kwargs.get("E", 1e7)

        self.NU = kwargs.get("NU", 10)

        # Create cylinder nest at the init step
        self.filename_obstacles = kwargs.get(
            "filename_obstacles", "obstacle_information_nest_11.npz"
        )
        if GENERATE_NEW_OBSTACLES == True:
            nest_start_pos_x = -0.40
            nest_end_pos_x = -0.55

            nest_start_pos_z = self.target_position[2] - 0.5
            nest_end_pos_z = self.target_position[2] + 0.5

            self.obstacle_direction = [None for _ in range(self.N_OBSTACLE)]
            self.obstacle_normal = [None for _ in range(self.N_OBSTACLE)]
            self.obstacle_length = [1.0 for _ in range(self.N_OBSTACLE)]
            self.obstacle_radii = [0.03 for _ in range(self.N_OBSTACLE)]
            self.obstacle_start = [None for _ in range(self.N_OBSTACLE)]

            for i in range(self.N_OBSTACLE):
                alpha = np.random.uniform(np.pi / 3, np.pi * 2 / 3)
                beta = np.random.uniform(np.pi / 3, np.pi * 2 / 3)

                direction = np.array(
                    [
                        np.cos(alpha) * np.cos(beta),
                        -np.sin(alpha),
                        np.cos(alpha) * np.sin(beta),
                    ]
                )
                direction /= np.linalg.norm(direction)

                normal = np.array(
                    [
                        np.sin(alpha) * np.cos(beta),
                        np.cos(alpha),
                        np.sin(alpha) * np.sin(beta),
                    ]
                )
                normal /= np.linalg.norm(normal)

                self.obstacle_direction[i] = direction.copy()
                self.obstacle_normal[i] = normal.copy()

                # if i <5:
                #     self.obstacle_direction[i] = np.array([np.cos(alpha), np.sin(alpha), 0.])
                #     self.obstacle_normal[i] = np.array([0.0, 0.0, 1.0])
                # else:
                #     self.obstacle_direction[i] = np.array([0., np.sin(alpha), np.cos(alpha)])
                #     self.obstacle_normal[i] = np.array([1.0, 0.0, 0.0])

                start = np.zeros((3))

                start[0] = np.random.uniform(nest_start_pos_x, nest_end_pos_x)
                start[1] = (
                    self.target_position[1]
                    - (0.5 * self.obstacle_length[i] * self.obstacle_direction[i])[1]
                )
                start[2] = np.random.uniform(nest_start_pos_z, nest_end_pos_z)

                self.obstacle_start[i] = start

            import os

            save_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(save_folder, exist_ok=True)

            np.savez(
                os.path.join(save_folder, self.filename_obstacles),
                N_OBSTACLE=self.N_OBSTACLE,
                obstacle_direction=self.obstacle_direction,
                obstacle_normal=self.obstacle_normal,
                obstacle_length=self.obstacle_length,
                obstacle_radii=self.obstacle_radii,
                obstacle_start=self.obstacle_start,
                allow_pickle=True,
            )

        else:
            # For post-processing load the trained obstacle nest.
            data = np.load(str("data/" + self.filename_obstacles), allow_pickle=True)
            self.N_OBSTACLE = data["N_OBSTACLE"]
            self.obstacle_direction = data["obstacle_direction"]
            self.obstacle_normal = data["obstacle_normal"]
            self.obstacle_length = data["obstacle_length"]
            self.obstacle_radii = data["obstacle_radii"]
            self.obstacle_start = data["obstacle_start"]

            assert self.N_OBSTACLE == num_obstacles, (
                "Number of obstacle loaded for post-processing "
                + str(self.N_OBSTACLE)
                + " is different than user input to initialize the environment  "
                + str(num_obstacles)
            )

    def reset(self):
        """
        This function, creates the simulation environment.
        First, rod intialized and then rod is modified to make it tapered.
        Second, muscle segments are intialized. Muscle segment position,
        number of basis functions and applied directions are set.
        Finally, friction plane is set and simulation is finalized.

        self_version: remove the cylinder component: just trying to reach a target position
        Returns
        -------

        """
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = self.n_elem
        start = np.zeros((3,))
        start[1] = self.target_position[1]
        direction = np.array([0.0, 0.0, 1.0])  # rod direction: pointing upwards
        normal = np.array([0.0, 1.0, 0.0])
        binormal = np.cross(direction, normal)

        density = 1000
        nu = self.NU  # dissipation coefficient
        E = self.E  # Young's Modulus
        poisson_ratio = 0.5

        # Set the arm properties after defining rods
        base_length = 1.0  # rod base length
        radius_tip = 0.05  # radius of the arm at the tip
        radius_base = 0.05  # radius of the arm at the base

        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)

        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius=radius_along_rod,
            density=density,
            nu=nu,
            youngs_modulus=E,
            poisson_ratio=poisson_ratio,
        )

        # Now rod is ready for simulation, append rod to simulation
        self.simulator.append(self.shearable_rod)
        # self.mode = 4
        if self.mode != 2:
            # fixed target position to reach
            target_position = self.target_position

        if self.mode == 2 or self.mode == 4:
            # random target position to reach with boundary
            t_x = np.random.uniform(self.boundary[0], self.boundary[1])
            t_y = np.random.uniform(self.boundary[2], self.boundary[3])
            if self.dim == 2.0 or self.dim == 2.5:
                t_z = np.random.uniform(self.boundary[4], self.boundary[5]) * 0
            elif self.dim == 3.0 or self.dim == 3.5:
                t_z = np.random.uniform(self.boundary[4], self.boundary[5])

            print("Target position:", t_x, t_y, t_z)
            target_position = np.array([t_x, t_y, t_z])

        # initialize sphere
        self.sphere = Sphere(
            center=target_position,  # initialize target position of the ball
            base_radius=0.05,
            density=1000,
        )

        if self.mode == 3:
            self.dir_indicator = 1
            self.sphere_initial_velocity = self.target_v
            self.sphere.velocity_collection[..., 0] = [
                self.sphere_initial_velocity,
                0.0,
                0.0,
            ]

        if self.mode == 4:
            # t_x = np.random.uniform(self.boundary[0], self.boundary[1])
            # t_y = np.random.uniform(self.boundary[2], self.boundary[3])
            # t_z = np.random.uniform(self.boundary[4], self.boundary[5])
            # target_position = np.array([t_x, t_y, t_z])
            # self.target_position = target_position

            self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
            if self.dim == 2.0 or self.dim == 2.5:
                self.rand_direction_2 = np.pi / 2.0
            elif self.dim == 3.0 or self.dim == 3.5:
                self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

            self.v_x = (
                self.target_v
                * np.cos(self.rand_direction_1)
                * np.sin(self.rand_direction_2)
            )
            self.v_y = (
                self.target_v
                * np.sin(self.rand_direction_1)
                * np.sin(self.rand_direction_2)
            )
            self.v_z = self.target_v * np.cos(self.rand_direction_2)

            self.sphere.velocity_collection[..., 0] = [
                self.v_x,
                self.v_y,
                self.v_z,
            ]
            self.boundaries = np.array(self.boundary)

        theta_x = np.random.uniform(0, 2 * np.pi) * 0
        theta_y = np.random.uniform(0, 2 * np.pi) * 0
        theta_z = np.random.uniform(0, 2 * np.pi) * 0
        theta = np.array([theta_x, theta_y, theta_z])
        R = np.array(
            [
                [
                    np.cos(theta[1]) * np.cos(theta[2]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2])
                    - np.sin(theta[2]) * np.cos(theta[0]),
                    np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2])
                    + np.sin(theta[0]) * np.sin(theta[2]),
                ],
                [
                    np.sin(theta[2]) * np.cos(theta[1]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2])
                    + np.cos(theta[0]) * np.cos(theta[2]),
                    np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0])
                    - np.sin(theta[0]) * np.cos(theta[2]),
                ],
                [
                    -np.sin(theta[1]),
                    np.sin(theta[0]) * np.cos(theta[1]),
                    np.cos(theta[0]) * np.cos(theta[1]),
                ],
            ]
        )
        # Set rod and sphere directors to each other.
        self.sphere.director_collection[
            ..., 0
        ] = self.shearable_rod.director_collection[..., 0]
        self.simulator.append(self.sphere)

        class WallBoundaryForSphere(FreeRod):
            def __init__(self, boundaries):
                self.x_boundary_low = boundaries[0]
                self.x_boundary_high = boundaries[1]
                self.y_boundary_low = boundaries[2]
                self.y_boundary_high = boundaries[3]
                self.z_boundary_low = boundaries[4]
                self.z_boundary_high = boundaries[5]

            def constrain_values(self, sphere, time):
                pos_x = sphere.position_collection[0]
                pos_y = sphere.position_collection[1]
                pos_z = sphere.position_collection[2]

                radius = sphere.radius

                vx = sphere.velocity_collection[0]
                vy = sphere.velocity_collection[1]
                vz = sphere.velocity_collection[2]

                if (pos_x - radius) < self.x_boundary_low:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_x + radius) > self.x_boundary_high:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_y - radius) < self.y_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_y + radius) > self.y_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_z - radius) < self.z_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

                if (pos_z + radius) > self.z_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

            def constrain_rates(self, sphere, time):
                pass

        if self.mode == 4:
            self.simulator.constrain(self.sphere).using(
                WallBoundaryForSphere, boundaries=self.boundaries
            )
        # Add external contact between rod and sphere
        # self.simulator.connect(self.shearable_rod, self.sphere).using(
        #     ExternalContact, 1e2, 0.1
        # )

        # Add boundary constraints as fixing one end
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_tangent_dir = defaultdict(list)
        self.spline_points_func_array_tangent_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_tangent_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_tangent_dir,
        )

        # Add call backs
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for tapered arm
            """

            def __init__(
                self,
                step_skip: int,
                callback_params: dict,
                signal_scaling_factor: int,
                target_position,
            ):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params
                self.signal_scaling_factor = signal_scaling_factor
                self.target_position = target_position

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    self.callback_params["signal"].append(
                        self.signal_scaling_factor
                        * (system.position_collection[..., -1] - self.target_position)
                    )

                    return

        # Add call backs
        class RigidSphereCallBack(CallBackBaseClass):
            """
            Call back function for two arm octopus
            """

            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        # Add call backs
        class RigidCylinderCallBack(CallBackBaseClass):
            """
            Call back function for two arm octopus
            """

            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    # self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    # self.callback_params["direction"].append(system.director_collection[2,:,0].copy())
                    #
                    # self.callback_params["height"].append(copy.deepcopy(system.length))

                    return

        """ Add Obstacles to the environment """

        # self.N_OBSTACLE = 5
        self.obstacle = [None for _ in range(self.N_OBSTACLE)]
        self.obstacle_histories = [defaultdict(list) for _ in range(self.N_OBSTACLE)]

        self.obstacle_states = np.zeros(
            (self.number_of_points_on_cylinder * self.N_OBSTACLE, 3)
        )

        for i in range(self.N_OBSTACLE):
            # if i == 6 or i == 7:
            #     print(self.obstacle_start[i])
            #     self.obstacle_start[i][2] = self.obstacle_start[i][2] - 0.3
            self.obstacle[i] = Cylinder(
                start=self.obstacle_start[i],
                direction=self.obstacle_direction[i],
                normal=self.obstacle_normal[i],
                base_length=self.obstacle_length[i],
                base_radius=self.obstacle_radii[i],
                density=1000,
            )

            # obstacle_states(idx_on_obstacle, coordinates_x_y_z)
            self.obstacle_states[
                i
                * self.number_of_points_on_cylinder : (i + 1)
                * self.number_of_points_on_cylinder,
                :,
            ] = (
                start.reshape(3, 1)
                + self.obstacle_direction[i].reshape(3, 1)
                * np.linspace(
                    0, self.obstacle_length[i], self.number_of_points_on_cylinder
                )
            ).T

            self.obstacle_histories[i]["radius"] = self.obstacle_radii[i]
            self.obstacle_histories[i]["height"] = self.obstacle_length[i]
            self.obstacle_histories[i]["direction"] = self.obstacle_direction[i].copy()

            # We use scatter plot to plot obstacles, thus we need to discretize the obstacle and
            # save the position of the discretized elements.

            n_elem_for_plotting = 10
            position_collection_for_plotting = np.zeros((3, n_elem_for_plotting))
            end = (
                self.obstacle_start[i]
                + self.obstacle_direction[i] * self.obstacle_length[i]
            )
            for k in range(0, 3):
                position_collection_for_plotting[k, ...] = np.linspace(
                    self.obstacle_start[i][k], end[k], n_elem_for_plotting
                )

            self.obstacle_histories[i][
                "position_plotting"
            ] = position_collection_for_plotting.copy()

            self.simulator.append(self.obstacle[i])

            # Constraint obstacle positions
            self.simulator.constrain(self.obstacle[i]).using(
                OneEndFixedRod,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

            # Add external contact
            self.simulator.connect(self.shearable_rod, self.obstacle[i]).using(
                ExternalContact, k=8e4, nu=4.0
            )

        """ Add Obstacles to the environment """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            # step_skip = 500  # collect data every # steps
            self.post_processing_dict_rod = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
                signal_scaling_factor=self.signal_scaling_factor,
                target_position=self.target_position,
            )

            self.post_processing_dict_sphere = defaultdict(list)

            # list which collected data will be append
            # set the diagnostics for cyclinder and collect data
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

            for i in range(self.N_OBSTACLE):
                self.simulator.collect_diagnostics(self.obstacle[i]).using(
                    RigidCylinderCallBack,
                    step_skip=self.step_skip,
                    callback_params=self.obstacle_histories[i],
                )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # set state
        state = self.get_state()

        # reset on_goal
        self.on_goal = 0
        # reset current_step
        self.current_step = 0
        # reset time_tracker
        self.time_tracker = np.float64(0.0)
        # reset previous_action
        self.previous_action = None

        # systems is like full observation state; we want to change it.
        systems = [self.shearable_rod, self.sphere]

        # return self.total_steps, self.total_learning_steps, state
        return state

    def sampleAction(self):
        """
        want to sample usable random actions of shape (3 * self.number_of_muscle_segment,), in range [-1, 1]
        return a np.array
        """
        random_action = (np.random.rand(2 * self.number_of_control_points) - 0.5) * 2
        return random_action

    def get_state(self):
        """
        return current state of the system:
        """

        rod_state = self.shearable_rod.position_collection
        r_s_a = rod_state[0]  # x_info
        r_s_b = rod_state[1]  # y_info
        r_s_c = rod_state[2]  # z_info

        num_points = int(self.n_elem / self.obs_state_points)
        ## get full 3D state information
        rod_compact_state = np.concatenate(
            (
                r_s_a[0 : len(r_s_a) + 1 : num_points],
                r_s_b[0 : len(r_s_b) + 1 : num_points],
                r_s_c[0 : len(r_s_b) + 1 : num_points],
            )
        )

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0.0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)]
        )
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0.0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )

        signal_state = self.signal_scaling_factor * (
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )
        signal_compact_state = signal_state[:].flatten()

        Q = self.shearable_rod.director_collection[..., -1]

        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        rod_tip_orientation = np.array([qw, qx, qy, qz])
        self.rod_tip_orientation = rod_tip_orientation

        Q = self.sphere.director_collection[..., 0]
        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        target_tip_orientation = np.array([qw, qx, qy, qz])
        self.target_tip_orientation = target_tip_orientation
        # print(Q)
        # obstacles_state = np.zeros([self.N_OBSTACLE, 3])
        # for i in range(self.N_OBSTACLE):
        #     obstacles_state[i, :] = self.obstacle[i].position_collection[:].flatten()
        # obstacle_data = obstacles_state.flatten()
        obstacle_data = self.obstacle_states.flatten()

        state = np.concatenate(
            (
                # rod information
                rod_compact_state,
                rod_compact_velocity_norm,
                rod_compact_velocity_dir,
                # rod_tip_orientation,
                sphere_compact_state,
                # sphere_compact_velocity_norm,
                # sphere_compact_velocity_dir,
                # target_tip_orientation,
                # obstacle information
                obstacle_data,
            )
        )
        # print(state)
        return state

    def step(self, action):

        # Activation array contains lists for activation in different directions
        # assign correct activation arrays to correct directions.
        # activation_array_list is to be learned

        # set binormal activations to 0 if solving 2D case
        if self.dim == 2.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_tangent_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 2.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_tangent_dir[:] = action[
                self.number_of_control_points :
            ]
        # apply binormal activations if solving 3D case
        elif self.dim == 3.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points :
            ]
            self.spline_points_func_array_tangent_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 3.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points : 2 * self.number_of_control_points
            ]
            self.spline_points_func_array_tangent_dir[:] = action[
                2 * self.number_of_control_points :
            ]

        prev_dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )

        # Do multiple time step of simulation for <one learning step>
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        if self.mode == 3:
            ##### (+1, 0, 0) -> (0, -1, 0) -> (-1, 0, 0) -> (0, +1, 0) -> (+1, 0, 0) #####
            if (
                self.current_step
                % (1.0 / (self.h_time_step * self.num_steps_per_update))
                == 0
            ):
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        -self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = [
                        -self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        +self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = [
                        +self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 1
                else:
                    print("ERROR")

        self.current_step += 1

        # observe current state: current as sensed signal
        state = self.get_state()

        # print(self.sphere.position_collection[..., 0])
        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )
        """ Reward Engineering """
        reward_dist = -np.square(dist).sum()
        reward_ctrl = -np.abs(action).sum() / (action.shape[0])
        reward_vel = (
            -np.linalg.norm(self.shearable_rod.velocity_collection, axis=0).sum()
            / self.n_elem
        )
        reward_c_dist = -np.abs(dist - prev_dist).sum()
        reward_c_ctrl = (
            -np.abs(action - self.previous_action).sum() / (action.shape[0])
            if self.previous_action is not None
            else -0.0
        )

        ## distance between orientations from https://math.stackexchange.com/questions/90081/quaternion-distance
        orientation_dist = -(
            ((1.0 - np.dot(self.rod_tip_orientation, self.target_tip_orientation) ** 2))
            ** 2
        )

        ## Obstacle reward engineering
        # reward_obstacle = 0.0
        # No need to do reward engineering this is with contact

        # reward = (
        #     reward_dist
        #     + self.acti_coef * reward_ctrl
        #     + self.acti_diff_coef * reward_c_ctrl
        # )
        reward = (
            1.0
            * reward_dist
            # + 0.2 * reward_ctrl  # * 0.0
            # + 0.2 * reward_c_ctrl  # * 0.0
            # + 0.05 * reward_vel
            # + 0.2 * reward_obstacle
            # + 0.1 * orientation_dist
        )

        # print(reward_dist, reward_ctrl, reward_c_ctrl)
        # print(reward_dist, 0.1 * reward_ctrl, 0.05 * reward_c_ctrl)
        """ Reward Engineering """

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation

        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        # if invalid_values_condition == True:
        #     print(" Nan detected, exiting simulation now")
        #     done = True

        if invalid_values_condition == True:
            print(" Nan detected, exiting simulation now")
            print(self.shearable_rod.position_collection)
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            print("zeroed", self.shearable_rod.position_collection)
            reward = -20000
            state = self.get_state()
            done = True

        if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():
            self.on_goal += self.time_step
            reward += 0.5
        # for this specific case, check on_goal parameter
        if np.isclose(dist, 0.0, atol=0.05).all():
            self.on_goal += self.time_step
            reward += 1.5
            # if self.on_goal > 0.1:
            #     reward += 1
            # if self.on_goal > 2.0:
            #     done = True
            #     print(" Goal Reached, exit simulation. Reward: %0.2f" % reward)
        else:
            self.on_goal = 0

        if self.current_step >= self.total_learning_steps:
            done = True
            if reward > 0:
                print(" Reward greater than 0! Reward: %0.2f" % reward)
            else:
                print(" Finished simulation. Reward: %0.2f" % reward)
        """ Done is a boolean to reset the environment before episode is completed """

        # set previous_action = action
        self.previous_action = action

        return state, reward, done, {"ctime": self.time_tracker}

    def render(self, mode="human"):
        return

    def post_processing(
        self, filename_video, filename_acti="activation_video.mp4", **kwargs
    ):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            # target_and_obstacle_postprocessing_list = [
            #     self.post_processing_dict_sphere
            # ] + self.obstacle_histories
            #
            # plot_video_with_sphere(
            #     [self.post_processing_dict_rod],
            #     target_and_obstacle_postprocessing_list,
            #     video_name="3d_" + filename_video,
            #     fps=self.rendering_fps,
            #     step=1,
            #     vis2D=False,
            #     **kwargs,
            # )
            #
            # plot_video_with_sphere_2D(
            #     [self.post_processing_dict_rod],
            #     target_and_obstacle_postprocessing_list,
            #     video_name="2d_" + filename_video,
            #     fps=self.rendering_fps,
            #     step=1,
            #     vis2D=False,
            #     **kwargs,
            # )

            plot_video_with_sphere_cylinder(
                [self.post_processing_dict_rod],
                self.obstacle_histories,
                [self.post_processing_dict_sphere],
                video_name=filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=True,
                vis3D=True,
                **kwargs,
            )
            import os

            save_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(save_folder, exist_ok=True)

            # Transform nodal to elemental positions
            position_rod = np.array(self.post_processing_dict_rod["position"])
            position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])


            np.savez(
                os.path.join(save_folder, "arm_data.npz"),
                position_rod=position_rod,
                radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                n_elems_rod=self.shearable_rod.n_elems,
                position_sphere=np.array(self.post_processing_dict_sphere["position"]),
                radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
            )

            # Obstacle history
            np.savez(os.path.join(save_folder, "obstacle_data.npz"),
                obstacle_history = self.obstacle_histories,
                allow_pickle=True,
            )

            # Target history
            np.savez(os.path.join(save_folder, "target_data.npz"),
                     target_position=np.array(self.sphere.position_collection),
                     target_radius = np.array(self.sphere.radius)
                     )

            np.savez(
                os.path.join(save_folder, "arm_activation.npz"),
                torque_mag=np.array(
                    self.torque_profile_list_for_muscle_in_normal_dir["torque_mag"]
                ),
                torque_muscle=np.array(
                    self.torque_profile_list_for_muscle_in_normal_dir["torque"]
                ),
            )


            

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )
