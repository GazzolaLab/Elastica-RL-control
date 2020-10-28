import numpy as np
import numba
from numba import njit
from elastica.external_forces import NoForces
from scipy.interpolate import make_interp_spline


class MuscleTorquesWithVaryingBetaSplines(NoForces):
    """

    This class compute the muscle torques using Beta spline.
    Points of beta spline can be changed through out the simulation, and
    every time it changes a new spline generated.

    """

    def __init__(
        self,
        base_length,
        number_of_control_points,
        points_func_array,
        muscle_torque_scale,
        direction,
        step_skip,
        max_rate_of_change_of_activation=0.01,
        **kwargs,
    ):
        super(MuscleTorquesWithVaryingBetaSplines, self).__init__()

        if direction == str("normal"):
            self.direction = int(0)
        elif direction == str("binormal"):
            self.direction = int(1)
        elif direction == str("tangent"):
            self.direction = int(2)
        else:
            raise NameError(
                "Please type normal, binormal or tangent as muscle torque direction. Input should be string."
            )

        self.points_array = (
            points_func_array
            if hasattr(points_func_array, "__call__")
            else lambda time_v: points_func_array
        )

        self.base_length = base_length
        self.muscle_torque_scale = muscle_torque_scale

        self.torque_profile_recorder = kwargs.get("torque_profile_recorder", None)
        self.step_skip = step_skip
        self.counter = 0  # for recording data from the muscles
        self.number_of_control_points = number_of_control_points
        self.points_cached = np.zeros((2, self.number_of_control_points+2))  # This caches the control points. Note that first and last control points are zero.
        self.points_cached[0,:] = np.linspace(0, self.base_length, self.number_of_control_points+2) # position of control points along the rod.
        self.points_cached[1, 1:-1] = np.zeros(
            self.number_of_control_points
        )  # initalize at a value that RL can not match
        # Max rate of change of activation determines, maximum change in activation
        # signal in one time-step.
        self.max_rate_of_change_of_activation = max_rate_of_change_of_activation

        # Purpose of this flag is to just generate spline even the control points are zero
        # so that code wont crash.
        self.initial_call_flag = 0

    def apply_torques(self, system, time: np.float = 0.0):

        # Check if RL algorithm changed the points we fit the spline at this time step
        # if points_array changed create a new spline. Using this approach we don't create a
        # spline every time step.
        # Make sure that first and last point y values are zero. Because we cannot generate a
        # torque at first and last nodes.
        # print('torque',self.max_rate_of_change_of_activation)

        if not np.array_equal(self.points_cached[1,1:-1], self.points_array(time)) or self.initial_call_flag==0:
            self.initial_call_flag=1

            # Apply filter to the activation signal, to prevent drastic changes in activation signal.
            self.filter_activation(
                                self.points_cached[1, 1:-1],
                                np.array((self.points_array(time))),
                                self.max_rate_of_change_of_activation,
                            )
            # self.points_cached[1,1:-1] = self.points_array(time)
            # print(self.points_cached[0], self.points_cached[1])
            self.my_spline = make_interp_spline(
                self.points_cached[0], self.points_cached[1]
            )
            # print(self.points_cached[0], self.points_cached[1])
            # print("updating torques:", time, self.direction, self.points_cached[1,1:-1], self.points_array(time))    
            cumulative_lengths = np.cumsum(system.lengths)

            # Compute the muscle torque magnitude from the beta spline.
            self.torque_magnitude_cache = self.muscle_torque_scale * self.my_spline(cumulative_lengths)
            # print(system.lengths)
            # print(system.lengths.shape, cumulative_lengths.shape)
            # print(cumulative_lengths, self.my_spline(cumulative_lengths))

        self.compute_muscle_torques(
            self.torque_magnitude_cache,
            self.direction,
            system.external_torques,
        )

        if self.counter % self.step_skip == 0:
            if self.torque_profile_recorder is not None:
                self.torque_profile_recorder["time"].append(time)

                # filter = np.ones(torque_magnitude.shape)
                self.torque_profile_recorder["torque_mag"].append(
                    self.torque_magnitude_cache.copy()
                )
                self.torque_profile_recorder["torque"].append(
                    system.external_torques.copy()
                )
                self.torque_profile_recorder["element_position"].append(
                    np.cumsum(system.lengths)
                )

        self.counter += 1

    @staticmethod
    @njit(cache=True)
    def compute_muscle_torques(
        torque_magnitude, direction, external_torques
    ):

        blocksize = torque_magnitude.shape[0]
        for k in range(blocksize):
            external_torques[direction, k] += torque_magnitude[k]
            # external_torques[direction, k] -= torque_magnitude[k + 1]

    @staticmethod
    @numba.njit()
    def filter_activation(signal, input_signal, max_signal_rate_of_change):
        signal_difference = input_signal - signal
        signal += np.sign(signal_difference) * np.minimum(
            max_signal_rate_of_change, np.abs(signal_difference)
        )
        # print(np.sign(signal_difference) * np.minimum(
        #     max_signal_rate_of_change, np.abs(signal_difference)
        # ))
        # print(max_signal_rate_of_change, np.abs(signal_difference))
