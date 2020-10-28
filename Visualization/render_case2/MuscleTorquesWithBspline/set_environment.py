from elastica._calculus import _isnan_check

from elastica.timestepper import extend_stepper_interface
from elastica import *


from examples.OctopusArmSensor.basis_gym.single_arm_basis_postprocessing import (
    plot_video_with_sphere,
)

from ReacherSoft3D.MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)

# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class Environment:
    def __init__(
        self, final_time, number_of_control_points, COLLECT_DATA_FOR_POSTPROCESSING=False,
    ):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        time_step = 1.0e-5  # this is a stable timestep
        self.total_steps = int(self.final_time / time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        # Video speed
        self.rendering_fps = 20
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        self.number_of_control_points = number_of_control_points

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

    def reset(self):
        """
        This function, creates the simulation environment.
        First, rod intialized and then rod is modified to make it tapered.
        Second, muscle segments are intialized. Muscle segment position,
        number of basis functions and applied directions are set.
        Finally, friction plane is set and simulation is finalized.
        Returns
        -------

        """
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = 120
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction
        normal = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(direction, normal)
        base_length = 1.0  # rod base length
        base_radius = 0.05  # rod base radius
        base_area = np.pi * base_radius ** 2
        density = 1000
        nu = 5.0  # dissipation coefficient
        E = 5e6  # Young's Modulus
        poisson_ratio = 0.5

        # Set the arm properties after defining rods
        radius_tip = 0.025  # radius of the arm at the tip
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

        self.sphere = Sphere(
            np.array([0.5, 0.5, 0.0]),  # cylinder  initial position
            base_radius=0.025,
            density=1000,
        )
        self.simulator.append(self.sphere)

        # Now rod is ready for simulation, append rod to simulation
        self.simulator.append(self.shearable_rod)

        # Constrain the rod
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        self.spline_points_func_array = []
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points = self.number_of_control_points,
            points_func_array=self.spline_points_func_array,
            muscle_torque_scale=-1.0,
            direction=str("normal"),
            step_skip=self.step_skip,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        # Add call backs
        class ArmMuscleBasisCallBack(CallBackBaseClass):
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
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
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
                    self.callback_params["radius"].append(system.radius)
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            # step_skip = 500  # collect data every # steps
            self.post_processing_dict_rod = defaultdict(
                list
            )  # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_sphere = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for cyclinder and collect data
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

        systems = [self.shearable_rod]

        return self.total_steps, systems

    def step(self, spline_points_list, time):

        self.spline_points_func_array[:] = spline_points_list

        # Do one time step of simulation
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        systems = [self.shearable_rod]

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print(" Nan detected, exiting simulation now")
            done = True
        """ Done is a boolean to reset the environment before episode is completed """

        return time, systems, done

    def post_processing(self, filename_video, **kwargs):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            empty_dict = defaultdict(list)

            plot_video_actiavation_muscle(
                self.torque_profile_list_for_muscle_in_normal_dir,
                video_name="arm_activation_muscle_torque_in_normal_dir.mp4",
                margin=0.2,
                fps=self.rendering_fps,
                step=1,
            )

            plot_video_with_sphere(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name=filename_video,
                fps=self.rendering_fps,
                step=1,
                **kwargs
            )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )


from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video_actiavation_muscle(
    torque_list: dict, video_name="video.mp4", margin=0.2, fps=15, step=100,
):
    import matplotlib.animation as manimation

    time = np.array(torque_list["time"])

    torque_mag = np.array(torque_list["torque_mag"])
    torque = np.array(torque_list["torque"])

    element_position = np.array(torque_list["element_position"])

    print("plot activation visualization video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.subplot(211)
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in tqdm(range(0, time.shape[0], int(step))):
            # ax1 = plt.subplot(2, 2, 1)
            # ax2 = plt.subplot(222, frameon=False)
            # x = activation[time][2]
            torq = torque[time][0]
            pos = element_position[time]
            fig.clf()
            plt.subplot(2, 1, 1)
            plt.plot(pos, torque_mag[time], "-")

            plt.subplot(2, 1, 2)
            plt.plot(pos, torq, "-")
            # plt.xlim([0 - margin, 2.5 + margin])

            writer.grab_frame()
