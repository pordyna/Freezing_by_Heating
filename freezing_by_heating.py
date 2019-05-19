"""

"""


import numpy as np
from scipy.integrate import solve_ivp


def central_difference_derivative(function, x_0, h):
    return (function(x_0 + 0.5 * h) - function(x_0 - 0.5 * h)) / h


def calc_gradient(function, vec_val, h, *args):
    gradient = np.array([0, 0])
    gradient[0] = central_difference_derivative(
        lambda x_0: function(np.array[x_0, vec_val[1]], *args), vec_val[1], h)
    gradient[1] = central_difference_derivative(
        lambda x_0: function(np.array[vec_val[0], x_0], *args), vec_val[0], h)
    return gradient


class Simulation:
    # TODO adjust docstring length to PEP8
    """

    Some attributes:
        initial_state(np.ndarray[dim=3]): Describes particles positions and
          momenta. 1st axis 1st element position, 1st axis 2nd element
          momentum, 2nd axis - particles, 3rd axis spatial components (x,y).
          Example:
                   x[1][n][0] is p_x of the n-th particle.
                   x[0][n][1] is x_y of the n-th particle.
          Particles with a positive desired velocity come first.
          n_positive: Number of particles with the desired velocity > 0.
        walls(Callable[[np.ndarray], float]): For a given particle
          position should return the shortest distance to the nearest wall.
        boundary_condition(Callable[[np.ndarray], np.ndarrayvalue]): Takes a
          state (as defined for initial_state) and returns an adapted state
          that satisfies the boundary conditions.
        desired_direction(Callable[[np.ndarray], np.ndarray]): For a given
          position returns a unit vector pointing in the direction of the
          desired movement (for particles with a positive desired velocity).
    """
    def __init__(self, initial_state, n_positive,
                 desired_speed, desired_direction, relaxation_time,
                 noise_amplitude, param_factor, param_exponent, core_diameter,
                 walls, gradient_step, particle_mass):

        self.initial_state = initial_state
        self.desired_speed = desired_speed
        self.relaxation_time = relaxation_time
        self.noise_amplitude = noise_amplitude
        self.param_factor = param_factor
        self.param_exponent = param_exponent
        self.core_diameter = core_diameter
        self.particle_mass = particle_mass

        self.n_particles = self.initial_state.shape[1]
        self.n_positive = n_positive
        self.n_negative = self.n_particles - self.n_positive
        self.walls = walls
        self.desired_direction = desired_direction
        self.gradient_step = gradient_step

        self.ode_left_hand_side = np.zeros_like(self.initial_state)
        self.times = None
        self.states = None
        self.simulation_succeeded = False

        self.total_energies = None
        self.efficiencies = None
# use numpy ravel for solver

    def _particle_drive(self, momenta, orientation):
        desired_momentum = (self.desired_direction * self.desired_speed
                            * orientation * self.particle_mass)
        vectors = momenta - desired_momentum  # along 2nd axis of momenta
        vectors *= 1 / self.relaxation_time
        return vectors

    def _ppi_before_gradient(self, position, second_position):
        particles_distance = np.linalg.norm(second_position - position)
        value = particles_distance - self.core_diameter / 2
        value = value**self.param_exponent
        value *= self.param_factor
        return value

    def _pbi_before_gradient(self, position):
        value = self.walls(position) - self.core_diameter / 2
        value = value**self.param_exponent
        value *= self.param_factor
        return value

    def _particle_particle_interaction(self, positions):
        output = np.zeros_like(positions)
        for ii, position in enumerate(positions):
            summed = 0
            # one could exclude here the i=j case (see paper) but the
            # contribution is 0 anyway.
            for second_position in positions:
                summed += calc_gradient(self._ppi_before_gradient, position,
                                        self.gradient_step, second_position)
            output[ii] = summed
        return output

    def _particle_boundary_interaction(self, positions):
        output = np.zeros_like(positions)
        for ii, position in enumerate(positions):
            # Avoid going behind the walls while calculating the derivative.
            distance = self.walls(position)
            if distance < self.gradient_step / 2:
                step = distance
            else:
                step = self.gradient_step
            output[ii] = calc_gradient(self._pbi_before_gradient, position,
                                       step)
        return output

    def _calculate_momentum_derivative(self, old_momenta, old_positions):
        deriv_momenta = self.ode_left_hand_side[1, :, :]
        # only the particles with a positive v_0:
        particle_choice = slice(0, self.n_particles)
        deriv_momenta[particle_choice] = self._particle_drive(
            old_momenta[particle_choice], 1)
        # only the particles with a negative v_0:
        particle_choice = slice(self.n_positive, self.n_particles)
        deriv_momenta[particle_choice] = self._particle_drive(
            old_momenta[particle_choice], -1)
        # all particles:
        deriv_momenta += np.random.normal(0, self.noise_amplitude,
                                          deriv_momenta.shape)
        deriv_momenta += self._particle_particle_interaction(old_positions)
        deriv_momenta += self._particle_boundary_interaction(old_positions)

    def _ode_system(self, _time, state_1d):
        # Convert the scipy conform 1D state back in to the 3D array:
        # This shouldn't require any copying.
        state_3d = state_1d.reshape(self.initial_state.shape)
        self.ode_left_hand_side[0, :, :] = (state_3d[1, :, :]
                                            / self.particle_mass)
        self._calculate_momentum_derivative(state_3d[1, :, :],
                                            state_3d[0, :, :])
        # Return the calculated derivatives in 1D:
        # Again no copy is made since the array is still C-contiguous.
        return np.ravel(self.ode_left_hand_side)

    def run_simulation(self, integration_interval):
        initial_state_1d = np.ravel(self.initial_state)
        output = solve_ivp(self._ode_system, integration_interval,
                           initial_state_1d)
        self.simulation_succeeded = output.success
        if self.simulation_succeeded:
            self.times = output.t
            self.states = output.y
        # One can also try the continuous solution.

    def animate(self):
        ...

    def calculate_total_energies(self):
        ...

    def calculate_motion_efficiencies(self):
        ...

    def plot_efficiencies(self):
        ...

    def plot_total_energies(self):
        ...
