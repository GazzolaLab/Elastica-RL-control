__doc__ = """This file is for setting an environment for Elastica arm reaching to a fixed target and matching arms
orientation with the target. Actuation torques acting on arm can generate torques in normal, binormal and tangent 
direction. Environment set in this file is interfaced with stable-baselines and OpenAI Gym. It is shown that this
environment works with PPO, TD3, DDPG, TRPO and SAC."""


import gym
from gym import spaces

import copy
import sys

sys.path.append("../")

from post_processing import plot_video_with_sphere, plot_video_with_sphere_2D

from MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *

# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class Environment(gym.Env):
    """

    Custom environment that follows OpenAI Gym interface. This environment, generates an
    arm (Cosserat rod) and target (rigid sphere). Target position and orientation are fixed.
    Controller has to select control points (stored in action) and input to step class method.
    Control points have to be in between [-1,1] and are used to generate a beta spline. This beta spline is scaled
    by the torque scaling factor (alpha or beta) and muscle torques acting along arm computed. Muscle torques bend
    and twist the arm and to reach the target and match arm orientation with target.

    Attributes
    ----------
    dim : float
        Dimension of the problem.
        If dim=2.0 2D problem only muscle torques in normal direction is activated.
        If dim=2.5 or 3.0 3D problem muscle torques in normal and binormal direction are activated.
        If dim=3.5 3D problem muscle torques in normal, binormal and tangent direction are activated.
    n_elem : int
        Cosserat rod number of elements.
    final_time : float
        Final simulation time.
    time_step : float
        Simulation time-step.
    number_of_control_points : int
        Number of control points for beta-spline that generate muscle torques.
    alpha : float
        Muscle torque scaling factor for normal/binormal directions.
    beta : float
        Muscle torque scaling factor for tangent directions (generates twist).
    target_position :  numpy.ndarray
        1D (3,) array containing data with 'float' type.
        Initial target position, If mode is 2 or 4 target randomly placed.
    num_steps_per_update : int
        Number of Elastica simulation steps, before updating the actions by control algorithm.
    action : numpy.ndarray
        1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
        Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
        is number of torque directions, this is controlled by the dim.
    action_space : spaces.Box
        1D (n_torque_direction * number_of_control_poinst,) array containing data with 'float' type in range [-1., 1.].
    obs_state_points : int
        Number of arm (Cosserat rod) points used for state information.
    number_of_points_on_cylinder : int
        Number of cylinder points used for state information.
    observation_space : spaces.Box
        1D ( total_number_of_states,) array containing data with 'float' type.
        State information of the systems are stored in this variable.
    mode : int
        There are 4 modes available.
        mode=1 fixed target position to be reached (default)
        mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
        mode=3 moving target on fixed trajectory.
        mode=4 randomly moving target.
    COLLECT_DATA_FOR_POSTPROCESSING : boolean
        If true data from simulation is collected for post-processing. If false post-processing making videos
        and storing data is not done.
    E : float
        Young's modulus of the arm (Cosserat rod).
    NU : float
        Dissipation constant of the arm (Cosserat rod).
    COLLECT_CONTROL_POINTS_DATA : boolean
        If true actions or selected control points by the controller are stored throughout the simulation.
    total_learning_steps : int
        Total number of steps, controller is called. Also represents how many times actions changed throughout the
        simulation.
    control_point_history_array : numpy.ndarray
         2D (total_learning_steps, number_of_control_points) array containing data with 'float' type.
         Stores the actions or control points selected by the controller.
    shearable_rod : object
        shearable_rod or arm is Cosserat Rod object.
    sphere : object
        Target sphere is rigid Sphere object.
    spline_points_func_array_normal_dir : list
        Contains the control points for generating spline muscle torques in normal direction.
    torque_profile_list_for_muscle_in_normal_dir : defaultdict(list)
        Records, muscle torques and control points in normal direction throughout the simulation.
    spline_points_func_array_binormal_dir : list
        Contains the control points for generating spline muscle torques in binormal direction.
    torque_profile_list_for_muscle_in_binormal_dir : defaultdict(list)
        Records, muscle torques and control points in binormal direction throughout the simulation.
    spline_points_func_array_tangent_dir : list
        Contains the control points for generating spline muscle torques in tangent direction.
    torque_profile_list_for_muscle_in_tangent_dir : defaultdict(list)
        Records, muscle torques and control points in tangent direction throughout the simulation.
    post_processing_dict_rod : defaultdict(list)
        Contains the data collected by rod callback class. It stores the time-history data of rod and only initialized
        if COLLECT_DATA_FOR_POSTPROCESSING=True.
    post_processing_dict_sphere : defaultdict(list)
        Contains the data collected by target sphere callback class. It stores the time-history data of rod and only
        initialized if COLLECT_DATA_FOR_POSTPROCESSING=True.
    step_skip : int
        Determines the data collection step for callback functions. Callback functions collect data every step_skip.
    """

    # Required for OpenAI Gym interface
    metadata = {"render.modes": ["human"]}

    """
    FOUR modes: (specified by mode)
    1. fixed target position to be reached (default: need target_position parameter)
    2. random fixed target position to be reached
    3. fixed trajectory to be followed
    4. random trajectory to be followed 
    """

    def __init__(
        self,
        final_time,
        num_steps_per_update,
        number_of_control_points,
        alpha,
        beta,
        target_position,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=2.5e-4,
        n_elem=40,
        mode=1,
        dim=3.5,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        final_time : float
            Final simulation time.
        n_elem : int
            Arm (Cosserat rod) number of elements.
        num_steps_per_update : int
            Number of Elastica simulation steps, before updating the actions by control algorithm.
        number_of_control_points : int
            Number of control points for beta-spline that generate muscle torques.
        alpha : float
            Muscle torque scaling factor for normal/binormal directions.
        beta : float
            Muscle torque scaling factor for tangent directions (generates twist).
        target_position :  numpy.ndarray
            1D (3,) array containing data with 'float' type.
            Initial target position, If mode is 2 or 4 target randomly placed.
        COLLECT_DATA_FOR_POSTPROCESSING : boolean
            If true data from simulation is collected for post-processing. If false post-processing making videos
            and storing data is not done.
        sim_dt : float
            Simulation time-step
        mode : int
            There are 4 modes available.
            mode=1 fixed target position to be reached (default)
            mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
            mode=3 moving target on fixed trajectory.
            mode=4 randomly moving target.
        num_obstacles : int
            Number of rigid cylinder obstacles.
        COLLECT_CONTROL_POINTS_DATA : boolean
            If true actions or selected control points by the controller are stored throughout the simulation.
        *args
            Variables length arguments. Currently, *args are not used.
        **kwargs
            Arbitrary keyword arguments.
            * E : float
                Young's modulus of the arm (Cosserat rod). Default 1e7Pa
            * NU : float
                Dissipation constant of the arm (Cosserat rod). Default 10.
            * target_v : float
                Target velocity for moving taget, if mode=3,4 it is used.
            * boundary : numpy.ndarray
                1D (6,) array containing data with 'float' type.
                boundary used if mode=2,4. It determines the rectangular space, that target can move and  minimum
                and maximum of this space are given for x, y, and z coordinates. (xmin, xmax, ymin, ymax, zmin, zmax)

        """
        super(Environment, self).__init__()
        self.dim = dim
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        self.h_time_step = sim_dt  # this is a stable time step
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        print("Total steps", self.total_steps)

        # Video speed
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Number of control points
        self.number_of_control_points = number_of_control_points

        # Actuation torque scaling factor in normal/binormal direction
        self.alpha = alpha

        # Actuation torque scaling factor in tangent direction
        self.beta = beta

        # target position
        self.target_position = target_position

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        if self.dim == 2.0:
            # normal direction activation (2D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(self.number_of_control_points)
        if self.dim == 3.0 or self.dim == 2.5:
            # normal and/or binormal direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(2 * self.number_of_control_points)
        if self.dim == 3.5:
            # normal, binormal and/or tangent direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(3 * self.number_of_control_points)

        self.obs_state_points = 10
        num_points = int(n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(n_elem + 1)[0::num_points])

        # 8: 4 points for velocity and 4 points for orientation
        # 11: 3 points for target position plus 8 for velocity and orientation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 8 + 11,),
            dtype=np.float64,
        )

        # here we specify 4 tasks that can possibly used
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

        self.E = kwargs.get("E", 1e7)

        self.NU = kwargs.get("NU", 10)

        self.n_elem = n_elem

    def reset(self):
        """

        This class method, resets and creates the simulation environment. First,
        Elastica rod (or arm) is initialized and boundary conditions acting on the rod defined.
        Second, target and if there are obstacles are initialized and append to the
        simulation. Finally, call back functions are set for Elastica rods and rigid bodies.

        Returns
        -------

        """
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = self.n_elem
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction: pointing upwards
        normal = np.array([0.0, 0.0, 1.0])
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

        # Arm is shearable Cosserat rod
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

            self.trajectory_iteration = 0  # for changing directions
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

        if self.mode == 1:
            theta_x = 0
            theta_y = np.pi / 4
            theta_z = 0
        if self.mode == 2 or self.mode == 4:
            theta_x = 0
            theta_y = np.random.uniform(-np.pi / 2, np.pi / 2)
            theta_z = 0

        # set the orientation of target sphere
        theta = np.array([theta_x, theta_y, theta_z])
        print(theta)
        R = np.array(
            [
                [
                    -np.sin(theta[1]),
                    np.sin(theta[0]) * np.cos(theta[1]),
                    np.cos(theta[0]) * np.cos(theta[1]),
                ],
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
            ]
        )
        self.sphere.director_collection[..., 0] = R
        self.simulator.append(self.sphere)

        Q = self.sphere.director_collection[..., 0]
        # Compute target tip orientation using quaternions.
        # We add target and arm tip orientations difference to reward function.
        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        self.target_tip_orientation = np.array([qw, qx, qy, qz])

        class WallBoundaryForSphere(FreeRod):
            """

            This class generates a bounded space that sphere can move inside. If sphere
            hits one of the boundaries (walls) of this space, it is reflected in opposite direction
            with the same velocity magnitude.

            """

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

        # Add boundary constraints as fixing one end
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        # Add muscle torques acting on the arm for actuation.
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected by RL to
        # generate torques along the arm.
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
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
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
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_twist_dir,
        )

        # Call back function to collect arm data from simulation
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for Elastica rod or arm
            """

            def __init__(
                self, step_skip: int, callback_params: dict,
            ):
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
                    self.callback_params["directors"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        # Call back function to collect target sphere data from simulation
        class RigidSphereCallBack(CallBackBaseClass):
            """
            Call back function for target sphere
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
                    self.callback_params["directors"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            self.post_processing_dict_rod = defaultdict(list)
            # List which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_sphere = defaultdict(list)
            # List which collected data will be append
            # set the diagnostics for target sphere and collect data
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
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

        # After resetting the environment return state information
        return state

    def sampleAction(self):
        """
        Sample usable random actions are returned.

        Returns
        -------
        numpy.ndarray
            1D (3 * number_of_control_points,) array containing data with 'float' type, in range [-1, 1].
        """
        random_action = (np.random.rand(1 * self.number_of_control_points) - 0.5) * 2
        return random_action

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
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
            rod_compact_velocity_norm != 0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)]
        )
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )

        Q = self.shearable_rod.director_collection[..., -1]
        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        self.rod_tip_orientation = np.array([qw, qx, qy, qz])

        state = np.concatenate(
            (
                # rod information
                rod_compact_state,
                rod_compact_velocity_norm,
                rod_compact_velocity_dir,
                self.rod_tip_orientation,
                # target information
                sphere_compact_state,
                sphere_compact_velocity_norm,
                sphere_compact_velocity_dir,
                self.target_tip_orientation,
            )
        )

        return state

    def step(self, action):
        """
        This method integrates the simulation number of steps given in num_steps_per_update, using the actions
        selected by the controller and returns state information, reward, and done boolean.

        Parameters
        ----------
        action :  numpy.ndarray
            1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
            Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
            is number of torque directions, this is controlled by the dim.

        Returns
        -------
        state : numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        reward : float
            Reward after the integration.
        done: boolean
            Stops, simulation or training if done is true. This means, simulation reached final time or NaN is
            detected in the simulation.

        """

        # action contains the control points for actuation torques in different directions in range [-1, 1]
        self.action = action

        # set binormal activations to 0 if solving 2D case
        if self.dim == 2.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 2.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = action[
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
            self.spline_points_func_array_twist_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 3.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points : 2 * self.number_of_control_points
            ]
            self.spline_points_func_array_twist_dir[:] = action[
                2 * self.number_of_control_points :
            ]

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

        if self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                # print('changing direction')
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
                self.trajectory_iteration = 0

        self.current_step += 1

        # observe current state: current as sensed signal
        state = self.get_state()

        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )
        """ Reward Engineering """
        reward_dist = -np.square(dist).sum()

        ## distance between orientations from https://math.stackexchange.com/questions/90081/quaternion-distance
        orientation_dist = (
            1.0 - np.dot(self.rod_tip_orientation, self.target_tip_orientation) ** 2
        )
        orientation_penalty = -((orientation_dist) ** 2)

        reward = 1.0 * reward_dist + 0.5 * orientation_penalty
        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print(" Nan detected in the position, exiting simulation now")
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            reward = -10000
            state = self.get_state()
            done = True

        if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():
            reward += 0.5
            reward += 0.5 * (1 - orientation_dist)
            if np.isclose(orientation_dist, 0.0, atol=0.05 * 2.0).all():
                reward += 0.5

        # for this specific case, check on_goal parameter
        if np.isclose(dist, 0.0, atol=0.05).all():
            reward += 1.5
            reward += 1.5 * (1 - orientation_dist)
            if np.isclose(orientation_dist, 0.0, atol=0.05).all():
                reward += 1.5

        if self.current_step >= self.total_learning_steps:
            done = True
            if reward > 0:
                print(
                    " Reward greater than 0! Reward: %0.3f, Distance: %0.3f, Orientation: %0.3f -- %0.3f, %0.3f "
                    % (reward, dist, orientation_dist, reward_dist, orientation_penalty)
                )
            else:
                print(
                    " Finished simulation. Reward: %0.3f, Distance: %0.3f, Orientation: %0.3f -- %0.3f, %0.3f"
                    % (reward, dist, orientation_dist, reward_dist, orientation_penalty)
                )
        """ Done is a boolean to reset the environment before episode is completed """

        self.previous_action = action

        invalid_values_condition_state = _isnan_check(state)
        if invalid_values_condition_state == True:
            print(
                " Nan detected in the state other than position data, exiting simulation now"
            )
            reward = -10000
            state = np.zeros(state.shape)
            done = True

        return state, reward, done, {"ctime": self.time_tracker}

    def render(self, mode="human"):
        """
        This method does nothing, it is here for interfacing with OpenAI Gym.

        Parameters
        ----------
        mode

        Returns
        -------

        """
        return

    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):
        """
        Make video 3D of arm movement in time, and store the arm, target, obstacles, and actuation
        data.

        Parameters
        ----------
        filename_video : str
            Names of the videos to be made for post-processing.
        SAVE_DATA : boolean
            If true collected data in simulation saved.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            plot_video_with_sphere_2D(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="2d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            plot_video_with_sphere(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="3d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            if SAVE_DATA == True:
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
                    position_sphere=np.array(
                        self.post_processing_dict_sphere["position"]
                    ),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
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
