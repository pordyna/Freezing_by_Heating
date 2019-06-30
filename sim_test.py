from freezing_by_heating import SimulationStraightCorridor
import numpy as np
sim = SimulationStraightCorridor(n_positive=10,
                                 desired_speed=1.4,
                                 relaxation_time=0.2,
                                 noise_variance=1,
                                 max_noise_val=0.1,
                                 param_factor=0.2,
                                 param_exponent=2,
                                 core_diameter=1,
                                 particle_mass=1,
                                 in_core_force=1e1,
                                 exp_width = 1e-4,
                                 extra_potential = 1e2,
                                 n_drunk_positive=0,
                                 n_drunk_negative=0,
                                 drunkness=0,
                                 corridor_width=5,
                                 corridor_length=25,
                                 initial_state = None,
                                 state_preparation = False
                                )

sim.run_simulation((0, 10), solver='RK45')#'LSODA')
sim.plot_initial_state()