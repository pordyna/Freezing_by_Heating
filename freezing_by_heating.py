"""

"""
import time
from math import sqrt, exp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from numba import jit, prange
import numba
from patches import circles
from functools import partial


def convert_truncated_normal(a,b, std, mean=0):
    return(a - mean) / std, (b - mean) / std


@jit(cache=True, nopython=True, parallel=False)
def _distance(position_1, position_2, corridor_length):
    """Calculates _distance b/w two particles.

    Takes positions as arguments.
    Returns distance and r_1 - r_2 / |r_1 - r_2|
    """
    distance_1 = np.linalg.norm(position_2 - position_1)
    # transpose second particle
    position_2_tr = np.copy(position_2)
    if position_2[0] > position_1[0]:
        position_2_tr[0] -= corridor_length
    else:
        position_2_tr[0] += corridor_length
    # calculate distance over the periodic boundary
    distance_2 = np.linalg.norm(position_2_tr - position_1)
    if distance_1 == 0 or distance_2 == 0:
        return float(0), np.array([0, 0], dtype=np.float64)
    elif distance_2 < distance_1:
        return distance_2, (position_2_tr - position_1) / distance_2

    return distance_1, (position_2 - position_1) / distance_1


@jit((numba.float64, numba.float64, numba.float64, numba.float64, numba.float64,
      numba.float64, numba.boolean, numba.float64, numba.float64[:, :]),
     cache=True, nopython=True, parallel=True)
def _particle_particle_interaction(core_diameter,
                                   in_core_force, exp_width, extra_potential, param_factor,
                                   param_exponent, state_preparation, corridor_length, positions):
    output = np.zeros_like(positions)
    for ii in prange(positions.shape[0]):
        position = positions[ii]
        summed = np.zeros(2)
        # one could exclude here the i=j case (see paper) but the
        # contribution is 0 anyway.
        for gg in prange(positions.shape[0]):
            second_position = positions[gg]
            distance, direction = _distance(position, second_position, corridor_length)
            # TODO decide on how this should look like. Remove exponential?
            if state_preparation:
                force = in_core_force
            else:
                force = ((distance - core_diameter)
                         ** (-param_exponent - 1))
                force *= param_factor * param_exponent
                if abs(distance - core_diameter) <= 200 * exp_width:
                    force += extra_potential / (1 + exp(abs(distance - core_diameter)/exp_width))
            summed -= force * direction
        output[ii] = summed
    return output


@jit(cache=True, nopython=True, parallel=False)
def _boundary_condition(corridor_length, state):
    """

    Takes a state (as defined for initial_state) and returns an
    adapted state that satisfies the boundary conditions.
    """
    np.fmod(state[0, :, 0], corridor_length, state[0, :, 0])
    escaped_to_left = np.where(state[0, :, 0] < 0)
    state[0, :, 0][escaped_to_left] += corridor_length
    return state


@jit(cache=True, nopython=True, parallel=False)
def _particle_drive(desired_speed,
                    particle_mass, relaxation_time,
                    momenta, orientation):
    desired_momentum = np.array([desired_speed * particle_mass * orientation, 0])
    vectors = desired_momentum - momenta  # along 2nd axis of momenta
    vectors = np.multiply(1 / relaxation_time, vectors)
    return vectors


@jit(cache=True, nopython=True, parallel=False)
def _walls(corridor_width, position):
    """
    For a given particle position  returns the shortest
    distance to the nearest wall  and a unit vector along the
    shortest connecting line (pointing at the particle).
    """
    distance_1 = position[1]
    distance_2 = corridor_width - position[1]
    if distance_1 < distance_2:
        return distance_1, np.array([0, 1])
    else:
        return distance_2, np.array([0, -1])


@jit((numba.float64, numba.float64, numba.float64, numba.float64,
      numba.float64, numba.float64, numba.uint, numba.float64, numba.float64,
      numba.float64[:], numba.float64[:]),
     cache=True, nopython=True, parallel=True)
def _jacobian(param_factor, param_exponent, core_diameter, particle_mass,
              corridor_length, corridor_width, n_positive, in_core_force,
              relaxation_time, _time, state1d):
    # shorter
    A = param_factor
    B = param_exponent
    D = core_diameter
    # building from blocks
    blocksize = int(state1d.shape[0] // 2)
    jac1 = np.zeros((blocksize, blocksize))
    jac2 = np.eye(blocksize) / particle_mass
    jac3 = np.zeros((blocksize, blocksize))
    # iterate over 2-particle combinations
    for i in prange(n_positive):
        position_i = state1d[2 * i:2 * i + 1]
        for j in prange(n_positive):
            position_j = state1d[2 * j:2 * j + 1]

            if i != j:
                # calculate useful terms that make up matrix elements
                dist, direction = _distance(position_i, position_j, corridor_length)
                # attention, unusual definition (other way round)
                direction *= -1
                # just for convenience
                f = A * B / dist * (dist - D)**(-B - 1)

                # contributions from particles i,j that enter into Jacobian:
                # for example: df_ijdivdx_i = "df_ij/dx_i":
                # force from particle j on particle i derivated after x_i (of
                # first particle)
                df_ijdivdx_i = (np.array([1, 0]) * f - direction[0] / dist
                                * position_i * f - (B + 1)
                                / (dist - D) * direction[0] * f)
                df_ijdivdy_i = (np.array([0, 1]) * f - direction[1] / dist
                                * position_i * f - (B + 1)
                                / (dist - D) * direction[1] * f)

                # similar, derivate after space coordinates of second particle
                # -> one term falls away
                df_ijdivdx_j = (0 - direction[0] / dist * position_i * f
                                - (B + 1) / (dist - D) * direction[0] * f)
                df_ijdivdy_j = (0 - direction[0] / dist * position_i * f
                                - (B + 1) / (dist - D) * direction[0] * f)

                # for convenience: write this in terms of 2x2 blocks
                block_first_arg = np.column_stack((df_ijdivdx_i, df_ijdivdy_i))
                block_second_arg = np.column_stack((df_ijdivdx_j, df_ijdivdy_j))

                # hard core: really would only have < and not <=, but should not
                # matter and probably more stable, but not sure
                # exp(600): just some high value: at border, full potential comes
                # in -> goes to infinity.
                # for spreading particles.
                if dist <= core_diameter:
                    block_first_arg = np.eye(2) * exp(600)
                    block_second_arg = -np.eye(2) * exp(600)

                # "correct", but probably annoying, if manages to tunnel into
                # other particle due to timestep issues
                #       if dist < core_diameter:
                #          block_first_arg = np.eye(2)*in_core_force*np.inf
                #         block_second_arg = -np.eye(2)*in_core_force*np.inf

                # add contributions in sum form particles i, j
                # TODO Correct indexing. jac3 is 2D not 3D
                # Should it be like this: ?
                # jac3[2 * i, 2 * i + 2] += block_first_arg
                # jac3[2 * j, 2 * j + 2] += block_second_arg
                # That's how it was:
                jac3[2 * i:2 * i + 2, 2 * i, 2 * i + 2] += block_first_arg
                jac3[2 * i:2 * i + 2, 2 * j, 2 * j + 2] += block_second_arg

    # quite similar
    for i in prange(n_positive):
        position = state1d[2 * i:2 * i + 1]
        dist_wall, direction = _walls(corridor_width, position)
        # use direction[1] in order to get right sign
        df_i_ydivdy_i = A * B * direction[1] * (dist_wall - D / 2)**(-B - 1)

        # same as above. walls are even harder than human bodys.
        if dist_wall <= 0:
            df_i_ydivdy_i = exp(600)

        # "correct"
        #      if dist_wall < 0:
        #          df_i_ydivdy_i = self.in_core_force*np.inf

        jac3[2 * i + 1, 2 * i + 1] += df_i_ydivdy_i

    jac4 = - np.eye(blocksize) * particle_mass / relaxation_time

    # build together
    jacobian = np.block([[jac1, jac2], [jac3, jac4]])

    return jacobian

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
                 relaxation_time, noise_variance, max_noise_val,
                 param_factor,
                 param_exponent, core_diameter, particle_mass,
                 in_core_force, exp_width, extra_potential, n_drunk_positive, n_drunk_negative,
                 drunkness, corridor_length, corridor_width, state_preparation=False, initial_state=None):

        self.initial_state = initial_state
        self.desired_speed = desired_speed
        self.relaxation_time = relaxation_time
        self.param_factor = param_factor
        self.param_exponent = param_exponent
        self.core_diameter = core_diameter
        self.particle_mass = particle_mass
        self.in_core_force = in_core_force
        self.n_drunk_positive = n_drunk_positive
        self.n_drunk_negative = n_drunk_negative
        self.drunk_std = sqrt(drunkness)
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.n_positive = n_positive
        self.exp_width =exp_width
        self.extra_potential = extra_potential
        self.state_preparation =state_preparation
        self._particle_particle_interaction = partial(
            _particle_particle_interaction, self.core_diameter,
                                   self.in_core_force, self.exp_width, self.extra_potential, self.param_factor,
                                   self.param_exponent, self.state_preparation, self.corridor_length)

        self._boundary_condition = partial(_boundary_condition,
                                           self.corridor_length)

        self._particle_drive = partial(_particle_drive,
                                       self.desired_speed, self.particle_mass,
                                       self.relaxation_time)

        self._walls = partial(_walls, corridor_width)

        self._jacobian = partial(_jacobian, self.param_factor, self.param_exponent, self.core_diameter, self.particle_mass,
             self.corridor_length, self.corridor_width, self.n_positive, self.in_core_force,
             self.relaxation_time)
        noise_std = sqrt(noise_variance)
        a,b = convert_truncated_normal(-max_noise_val, max_noise_val,
                                       noise_std)
        self.random_dist = truncnorm(a, b)

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
        length = self.corridor_length
        width = self.corridor_width
        min_spacing = self.core_diameter * 0.501
        sample = np.random.random(self.n_particles)
        sample = (width - 2 * min_spacing) * sample + min_spacing
        self.initial_state[0, :, 1] = sample
        sample = np.random.random(self.n_particles)
        sample = length * sample
        self.initial_state[0, :, 0] = sample
        # n_single_line = width // (self.core_diameter + min_spacing)
        # n_full_lines = self.n_positive // n_single_line
        # n_last_line = self.n_positive % n_single_line
        # full_line = np.linspace(min_spacing /, width - min_spacing,
        #                         n_single_line)
        # last_line = np.linspace(min_spacing, width - min_spacing,
        #                         n_last_line)
        # for ii in range(n_full_lines):
        #     line = slice(ii, ii + n_single_line)
        #     self.initial_state[0, line, 1] = full_line
        #     self.initial_state[0, line, 0] = (min_spacing
        #                                       + ii * (self.core_diameter
        #                                               * 1.01))
        # line = slice(n_full_lines * n_single_line,
        #              n_full_lines * n_single_line + n_last_line)
        # self.initial_state[0, line, 1] = last_line
        # self.initial_state[0, line, 0] = (min_spacing + n_full_lines
        #                                   * (self.core_diameter
        #                                      * 1.01))
        # self.initial_state[0, self.n_positive:, 0] = \
        #     length - self.initial_state[0, 0:self.n_positive:, 0]
        # self.initial_state[0, self.n_positive:, 1] = \
        #      self.initial_state[0, 0:self.n_positive:, 1]

        #self.initial_state[1, 0: self.n_positive] = \
        #    self.desired_speed * self._desired_direction()
        #self.initial_state[1, self.n_positive:] = \
        #    -1 * self.desired_speed * self._desired_direction()


    def _particle_boundary_interaction(self, positions):
        output = np.zeros_like(positions)
        for ii, position in enumerate(positions):
            # Avoid going behind the walls while calculating the derivative.
            distance, direction = self._walls(position)
            force = self.param_factor * self.param_exponent
            force *= ((distance - self.core_diameter/2)
                      ** (-self.param_exponent - 1))
            if abs(distance - self.core_diameter / 2) <= 200 * self.exp_width:
                force += self.extra_potential / (1 + exp((distance - self.core_diameter/2) / self.exp_width))
            output[ii] = force * direction
        return output

    def _calculate_momentum_derivative(self, old_momenta, old_positions):
        deriv_momenta = np.zeros_like(self.ode_left_hand_side[1, :, :])
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
        if max_noise_val!=0:
            noise = self.random_dist.rvs(size=deriv_momenta.size)
            noise = noise.reshape(deriv_momenta.shape)
        #print(noise.max(), noise.min())
            deriv_momenta += noise
        noise = self.random_dist.rvs(size=deriv_momenta[drunks].size)
        noise = noise.reshape(deriv_momenta[drunks].shape)
        deriv_momenta[drunks] += noise
        deriv_momenta += self._particle_particle_interaction(old_positions)
        deriv_momenta += self._particle_boundary_interaction(old_positions)
        #print(deriv_momenta.max(), deriv_momenta.min(), np.mean(deriv_momenta))
        return deriv_momenta

    # TODO add boundary conditions
    def _ode_system(self, _time, state_1d):
        # Convert the scipy conform 1D state back in to the 3D array:
        # This shouldn't require any copying.
        state_3d = state_1d.reshape(self.initial_state.shape)
        state_3d = self._boundary_condition(state_3d)
        self.ode_left_hand_side[0, :, :] = (state_3d[1, :, :]
                                            / self.particle_mass)
        self.ode_left_hand_side[1, :, :] = \
            self._calculate_momentum_derivative(state_3d[1, :, :],
                                                state_3d[0, :, :])
        # Return the calculated derivatives in 1D:
        # Again no copy is made since the array is still C-contiguous.

        return np.ravel(self.ode_left_hand_side)

    def run_simulation(self, integration_interval, **kwargs):
        initial_state_1d = np.copy(np.ravel(self.initial_state))
        output = solve_ivp(self._ode_system, integration_interval,
                           initial_state_1d, jac=self._jacobian, **kwargs)
        self.simulation_succeeded = output.success
        self.nfev = output.nfev
        self.njev = output.njev
        self.nlu = output.nlu
        if self.simulation_succeeded:
            self.times = output.t
            self.states = output.y
            print('finished. Applying boundary conditions now.')
            for ii in range(self.states.shape[1]):
                state_3d = self.states[:, ii].reshape(self.initial_state.shape)
                state_3d = self._boundary_condition(state_3d)
                self.states[:, ii] = np.ravel(state_3d)
        else:
            print('no success', output.status, output.message)
        # One can also try the continuous solution.

    def plot_initial_state(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.corridor_length + 0.5)
        ax.set_ylim(-.05, self.corridor_width + 0.5)
        plt.axis('equal')
        ax.hlines([0, self.corridor_width],
                  0, self.corridor_length)
        t_0 = 0.00001
        positions = self.initial_state[0, :, :]
        positive = slice(0, self.n_positive)
        negative = slice(self.n_positive, self.n_particles)
        circles(positions[positive, 0], positions[positive, 1],
                self.core_diameter/2, c='red', axis=ax)
        circles(positions[negative, 0], positions[negative, 1],
                self.core_diameter/2, c='blue', axis=ax)
        fig.canvas.draw()

    def animate(self, scale):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.corridor_length + 0.5)
        ax.set_ylim(-.05, self.corridor_width + 0.5)
        plt.axis('equal')
        ax.hlines([0, self.corridor_width],
                  0, self.corridor_length)
        t_0 = 0.00001
        positions = self.initial_state[0, :, :]
        positive = slice(0, self.n_positive)
        negative = slice(self.n_positive, self.n_particles)
        circles(positions[positive, 0], positions[positive, 1],
                self.core_diameter/2, c='red', axis=ax)
        circles(positions[negative, 0], positions[negative, 1],
                self.core_diameter/2, c='blue', axis=ax)
        fig.canvas.draw()
        for ii, t in enumerate(self.times[:-1]):
            start_time = time.time()
            ax.collections = []
            ax.hlines([0, self.corridor_width],
                      0, self.corridor_length)
            state_3d = self.states[:, ii].reshape(self.initial_state.shape)
            positions = state_3d[0, :, :]
            circles(positions[positive, 0], positions[positive, 1],
                    self.core_diameter/2, c='red', axis=ax)
            circles(positions[negative, 0], positions[negative, 1],
                    self.core_diameter/2, c='blue', axis=ax)
            fig.canvas.draw()
            #print(ii)
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
