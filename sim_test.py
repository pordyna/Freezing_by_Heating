from freezing_by_heating import SimulationStraightCorridor

sim = SimulationStraightCorridor(1, 150 ,0.000001, 1, 0, 2, 1, 0.001, 1, 10, 0, 0,0, corridor_width =6,corridor_length=3)
sim.run_simulation((0, 0.01))
