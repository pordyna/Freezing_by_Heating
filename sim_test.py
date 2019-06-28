from freezing_by_heating import SimulationStraightCorridor

sim = SimulationStraightCorridor(1, 5 ,1, 1, 0, 2, 1, 0.001, 1, 10, 0, 0,0, corridor_width =6,corridor_length=3)
sim.run_simulation((0, 4))
fig, ax = sim.animate(5)