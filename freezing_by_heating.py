"""

"""
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from patches import circles


class SimulationStraightCorridor:
    """
    Some attributes:
        initial_state(np.ndarray[dim=3]): Describes particles positions
          and momenta. 1st axis 1st element position, 1st axis 2nd
          element momentum, 2nd axis - particles, 3rd axis spatial
          components (x,y). Example:
          * x[1][n][0] is p_x of the n-th particle.
          * x[0][n][1] is x_y of the n-th particle.
          Particles with a positive desired velocity come first.
        n_positive: Number of particles with the desired velocity > 0.
    """
    def __init__(self, n_positive, desired_speed,
                 relaxation_time, noise_amplitude, param_factor,
                 param_exponent, core_diameter, gradient_step, particle_mass,
                 in_core_force, n_drunk_positive, n_drunk_negative,
                 drunkness, initial_state=None, **kwargs):

        self.initial_state = initial_state
        self.desired_speed = desired_speed
        self.relaxation_time = relaxation_time
        self.noise_amplitude = noise_amplitude
        self.param_factor = param_factor
        self.param_exponent = param_exponent
        self.core_diameter = core_diameter
        self.particle_mass = particle_mass
        self.gradient_step = gradient_step
        self.in_core_force = in_core_force
        self.n_drunk_positive = n_drunk_positive
        self.n_drunk_negative = n_drunk_negative
        self.drunkness = drunkness
        self.setup_params = kwargs
        self.n_positive = n_positive

        if self.initial_state is None:
            self.n_particles = 2 * self.n_positive
            self.initial_state = np.zeros((2, self.n_particles, 2))
            self._set_default_initial_state()

        self.n_particles = self.initial_state.shape[1]

        self.n_negative = self.n_particles - self.n_positive

        self.ode_left_hand_side = np.zeros_like(self.initial_state)
        self.times = None
        self.states = None
        self.simulation_succeeded = False

        self.total_energies = None
        self.efficiencies = None

    def _set_default_initial_state(self):
        length = self.setup_params['corridor_length']
        width = self.setup_params['corridor_width']
        min_spacing = self.core_diameter / 2 + self.gradient_step

        n_single_line = width // self.core_diameter
        n_full_lines = self.n_positive // n_single_line
        n_last_line = self.n_positive % n_single_line
        full_line = np.linspace(-min_spacing, width - min_spacing,
                                n_single_line)
        last_line = np.linspace(-min_spacing, width - min_spacing,
                                n_last_line)
        for ii in range(n_full_lines):
            line = slice(ii, ii + n_single_line)
            self.initial_state[0, line, 1] = full_line
            self.initial_state[0, line, 0] = (min_spacing
                                              + ii * (self.core_diameter
                                                      + self.gradient_step))
        line = slice(n_full_lines * n_single_line,
                     n_full_lines * n_single_line + n_last_line)
        self.initial_state[0, line, 1] = last_line
        self.initial_state[0, line, 0] = (min_spacing + n_full_lines
                                          * (self.core_diameter
                                             + self.gradient_step))
        self.initial_state[0, self.n_positive:, 0] = \
            length - self.initial_state[0, 0:self.n_positive:, 0]
        self.initial_state[0, self.n_positive:, 1] = \
             self.initial_state[0, 0:self.n_positive:, 1]

        self.initial_state[1, 0: self.n_positive] = \
            self.desired_speed * self._desired_direction()
        self.initial_state[1, self.n_positive:] = \
            -1 * self.desired_speed * self._desired_direction()

    def _walls(self, position):
        """
        For a given particle position  returns the shortest
        distance to the nearest wall  and a unit vector along the
        shortest connecting line (pointing at the particle).
        """
        distance_1 = position[1]
        distance_2 = self.setup_params['corridor_width'] - position[1]
        if distance_1 < distance_2:
            return distance_1, np.array([0, 1])
        else:
            return distance_2, np.array([0, -1])

    def _boundary_condition(self, state):
        """

        Takes a state (as defined for initial_state) and returns an
        adapted state that satisfies the boundary conditions.
        """
        np.mod(state[0, :, :], self.setup_params['corridor_length'],
               where=[True, False], out=state[0, :, :])
        escaped_to_left = np.where(state[0, :, 0] < 0)
        state[0, :, 0][escaped_to_left] += self.setup_params['corridor_length']
        return state

    @staticmethod
    def _desired_direction():
        """

        For a given position returns a unit vector pointing in the
        direction of the desired movement (for particles with a positive
        desired velocity).
        """
        return np.array([1, 0])

    def _distance(self, position_1, position_2):
        """Calculates _distance b/w two particles.

        Takes positions as arguments.
        Returns distance and r_1 - r_2 / |r_1 - r_2|
        """
        corridor_length = self.setup_params['corridor_length']
        distance_1 = np.linalg.norm(position_2 - position_1)
        # transpose second particle
        position_2_tr = position_2
        if position_2[0] > position_1[0]:
            position_2_tr[0] -= corridor_length
        else:
            position_2_tr[0] += corridor_length
        # calculate distance over the periodic boundary
        distance_2 = np.linalg.norm(position_2_tr - position_1)
        if distance_2 < distance_1:
            return distance_2, (position_2_tr - position_1) / distance_2
        else:
            return distance_1, (position_2 - position_1) / distance_1

    def _particle_drive(self, momenta, orientation):
        desired_momentum = (self._desired_direction() * self.desired_speed
                            * orientation * self.particle_mass)
        vectors = desired_momentum - momenta  # along 2nd axis of momenta
        vectors *= 1 / self.relaxation_time
        return vectors

    def _particle_particle_interaction(self, positions):
        output = np.zeros_like(positions)
        for ii, position in enumerate(positions):
            summed = np.zeros(2)
            # one could exclude here the i=j case (see paper) but the
            # contribution is 0 anyway.
            for second_position in positions:
                distance, direction = self._distance(second_position, position)
                if distance <= self.core_diameter:
                    force = self.in_core_force
                else:
                    force = ((distance - self.core_diameter)
                             ** (-self.param_exponent - 1))
                    force *= self.param_factor * self.param_exponent
                summed += force * direction
            output[ii] = summed
        return output

    def _particle_boundary_interaction(self, positions):
        output = np.zeros_like(positions)
        for ii, position in enumerate(positions):
            # Avoid going behind the walls while calculating the derivative.
            distance, direction = self._walls(position)
            force = self.param_factor * self.param_exponent
            force *= ((distance - self.core_diameter/2)
                      ** (self.param_exponent - 1))
            output[ii] = force * direction
        return output

    def _calculate_momentum_derivative(self, old_momenta, old_positions):
        deriv_momenta = self.ode_left_hand_side[1, :, :]
        # only the particles with a positive v_0:
        particle_choice = slice(0, self.n_positive)
        deriv_momenta[particle_choice] = self._particle_drive(
            old_momenta[particle_choice], 1)

        # only the particles with a negative v_0:
        particle_choice = slice(self.n_positive, self.n_particles)
        deriv_momenta[particle_choice] = self._particle_drive(
            old_momenta[particle_choice], -1)
        # all particles:
        drunks = slice(-self.n_drunk_negative, self.n_drunk_positive)
        noise = (self.noise_amplitude
                 * np.random.normal(0, 1, deriv_momenta.shape))
        deriv_momenta += noise
        noise = ((self.drunkness - self.noise_amplitude)
                 * np.random.normal(0, 1, deriv_momenta[drunks].shape))
        deriv_momenta[drunks] += noise
        deriv_momenta += self._particle_particle_interaction(old_positions)
        deriv_momenta += self._particle_boundary_interaction(old_positions)
        self.ode_left_hand_side[1, :, :] = deriv_momenta

    # TODO add boundary conditions
    def _ode_system(self, _time, state_1d):
        # Convert the scipy conform 1D state back in to the 3D array:
        # This shouldn't require any copying.
        state_3d = state_1d.reshape(self.initial_state.shape)
        state_3d = self._boundary_condition(state_3d)
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
        else:
            print('no success')
        # One can also try the continuous solution.

    def animate(self, scale):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.setup_params['corridor_length'] + 0.5)
        plt.axis('equal')
        t_0 = 0.00001
        positions = self.initial_state[0, :, :]
        positive = slice(0, self.n_positive)
        negative = slice(self.n_positive, self.n_particles)
        circles(positions[positive, 0], positions[positive, 1],
                self.core_diameter, c='red', axis=ax)
        circles(positions[negative, 0], positions[negative, 1],
                self.core_diameter, c='blue', axis=ax)
        fig.canvas.draw()
        for ii, t in enumerate(self.times):
            ax.collections = []
            start_time = time.time()
            state_3d = self.states[:, ii].reshape(self.initial_state.shape)
            positions = state_3d[0, :, :]
            circles(positions[positive, 0], positions[positive, 1],
                    self.core_diameter, c='red', axis=ax)
            circles(positions[negative, 0], positions[negative, 1],
                    self.core_diameter, c='blue', axis=ax)
            fig.canvas.draw()
            try:
                plt.pause(abs((self.times[ii + 1] - t))
                          / scale - (start_time - time.time()))
            except IndexError:
                pass
        return fig, ax

    def calculate_total_energies(self):
        ...

    def calculate_motion_efficiencies(self):
        ...

    def plot_efficiencies(self):
        ...

    def plot_total_energies(self):
        ...
