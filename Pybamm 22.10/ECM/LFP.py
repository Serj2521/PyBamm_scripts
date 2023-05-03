#
# Compare and extract lithium-ion battery chemistries
#
import os
os.system('cls' if os.name == 'nt' else 'clear')
import pybamm as pb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pb.set_logging_level("INFO")

# load models
model = pb.lithium_ion.DFN()

# Load Chemistry
parameter_values = pb.ParameterValues("Prada2013")


#
# Test
#

experiment = pb.Experiment(
    [   "Discharge at 1C until 2.5 V",
        "Rest for 2 hours",
        "Charge at 1C until 3.65 V",
        "Charge at 3.65 V until 10 mA",
        "Rest for 2 hours",],
)

# Run simulations

sim = pb.Simulation(model, experiment=experiment, parameter_values=parameter_values, solver=pb.CasadiSolver())
sol=sim.solve()

#Data Generation

t = sol["Time [h]"].entries
V = sol["Terminal voltage [V]"].entries
Cap =sol["Discharge capacity [A.h]"].entries

print('\n')
print(V[0])
print('\n')
print(max(Cap))
print('\n')
print(parameter_values)



# Plot
pb.dynamic_plot(sim)

plt.plot(t, Cap, '--', color ='red', label ="Capacity")
plt.xlabel("Time [h]")
plt.ylabel("Discharge capacity [A.h]")
# plt.legend()
plt.show()



