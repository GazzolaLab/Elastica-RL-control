import numpy as np
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

    def apply_torques(self, system, time: np.float = 0.0):

        # Check if RL algorithm changed the points we fit the spline at this time step
        # if points_array changed create a new spline. Using this approach we don't create a
        # spline every time step.
        # Make sure that first and last point y values are zero. Because we cannot generate a
        # torque at first and last nodes.
        if not np.array_equal(self.points_cached[1, 1:-1], self.points_array(time)[0]):
            self.points_cached[1, 1:-1] = self.points_array(time)[0]
            self.my_spline = make_interp_spline(
                self.points_cached[0], self.points_cached[1]
            )
            # Compute the muscle torque magnitude from the beta spline.
            self.torque_magnitude_cache = self.my_spline(np.cumsum(system.lengths))
            # print("updating torques:", time, self.torque_magnitude_cache)
        print(time, self.points_array)
        # cumulative_lengths = np.cumsum(system.lengths)
        # # Compute the muscle torque magnitude from the beta spline.
        # torque_magnitude = self.my_spline(cumulative_lengths)

        # torque = np.einsum("j,ij->ij", torque_magnitude[::-1], self.direction)
        # system.external_torques[..., 1:] += _batch_matvec(
        #     system.director_collection, torque
        # )[..., 1:]
        # system.external_torques[..., :-1] -= _batch_matvec(
        #     system.director_collection[..., :-1], torque[..., 1:]
        # )

        system.external_torques[self.direction, 1:] += self.torque_magnitude_cache[1:]
        system.external_torques[self.direction, :-1] -= self.torque_magnitude_cache[1:]

        if self.counter % self.step_skip == 0:
            if self.torque_profile_recorder is not None:
                self.torque_profile_recorder["time"].append(time)

                # filter = np.ones(torque_magnitude.shape)
                self.torque_profile_recorder["torque_mag"].append(self.torque_magnitude_cache)
                self.torque_profile_recorder["torque"].append(
                    system.external_torques.copy()
                )
                self.torque_profile_recorder["element_position"].append(
                    np.cumsum(system.lengths)
                )

        self.counter += 1
