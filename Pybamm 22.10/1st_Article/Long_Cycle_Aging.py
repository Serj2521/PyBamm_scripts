
import os
os.system('cls' if os.name == 'nt' else 'clear')
import pybamm
import matplotlib.pyplot as plt
import numpy as np

# Parameter identification
parameter_values = pybamm.ParameterValues("OKane2022")
parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-14})
parameter_values.update({"Ambient temperature [K]": 298.15})  # [268.15K = -5ºC, 283.15K= 10ºC, 298.15K = 25ºC, 308.15K = 25ºC]
parameter_values.update({"Upper voltage cut-off [V]": 4.21})

# MODEL
model = pybamm.lithium_ion.DFN({
    "SEI": "ec reaction limited",
    "SEI on cracks": "true",
    "SEI film resistance": "distributed",
    "particle mechanics": "swelling and cracking",
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",
    })

# Discretisation points
var_pts = {
    "x_n": 20,  # negative electrode
    "x_s": 20,  # separator 
    "x_p": 20,  # positive electrode
    "r_n": 30,  # negative particle
    "r_p": 30,  # positive particle
}

# Calculate stoichiometries at 100% SOC

param = model.param
esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

Vmin = 2.5
Vmax = 4.2
Cn = parameter_values.evaluate(param.n.cap_init)
Cp = parameter_values.evaluate(param.p.cap_init)
n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)

inputs={ "V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init}
esoh_sol = esoh_solver.solve(inputs)

print(f"Initial negative electrode SOC: {esoh_sol['x_100'].data[0]:.3f}")
print(f"Initial positive electrode SOC: {esoh_sol['y_100'].data[0]:.3f}")

# Update parameter values with initial conditions
c_n_max = parameter_values.evaluate(param.n.prim.c_max)
c_p_max = parameter_values.evaluate(param.p.prim.c_max)
parameter_values.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": esoh_sol["x_100"].data[0] * c_n_max,
        "Initial concentration in positive electrode [mol.m-3]": esoh_sol["y_100"].data[0] * c_p_max,
    }
)

pybamm.set_logging_level("NOTICE")

experiment = pybamm.Experiment([
    (f"Discharge at 1C until {Vmin}V",
     "Rest for 1 hour",
    f"Charge at 1C until {Vmax}V", 
    f"Hold at {Vmax}V until C/50")
] * 2,
termination="80% capacity"
)
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values,var_pts=var_pts,solver=pybamm.CasadiSolver(mode="safe"))
sol = sim.solve(save_at_cycles=2)


t = sol["Time [h]"].entries
# Cycle = sol['Cycle number'].entries
V = sol["Terminal voltage [V]"].entries
C =sol["Current [A]"].entries
Cap =sol["Discharge capacity [A.h]"].entries
SEI = sol["Loss of lithium to SEI [mol]"].entries + sol["Loss of lithium to SEI on cracks [mol]"].entries + sol["Loss of lithium to lithium plating [mol]"].entries
lithium_neg = sol["Total lithium in negative electrode [mol]"].entries
lithium_pos = sol["Total lithium in positive electrode [mol]"].entries
Tot_lithium_loss = sol["Loss of lithium inventory [%]"].entries


## Prints
print(param)
print('\n')
print(model.summary_variables)
print('\n')
print(len(t))
print('\n')
print(sol.summary_variables.keys())
# With integer
# sol_int = sim.solve(save_at_cycles=5)

fig, ax = plt.subplots(2, 2,sharex=False, sharey=False, figsize=(12,5))

ax[0, 0].plot(t,V)
ax[0, 0].set_xlabel("Time [h]")
ax[0, 0].set_ylabel("Terminal voltage [V]")
ax[1, 0].plot(t,C,'r')
ax[1, 0].set_xlabel("Time [h]")
ax[1, 0].set_ylabel("Current [A]")
ax[0, 1].plot(t,Tot_lithium_loss)
ax[0, 1].set_xlabel("Time [h]")
ax[0, 1].set_ylabel("Loss of lithium inventory [%]")
ax[1, 1].plot(t,lithium_neg+lithium_pos)
ax[1, 1].plot(t,lithium_neg[0]+lithium_pos[0]-SEI,linestyle="dashed")
ax[1, 1].set_xlabel("Time [h]")
ax[1, 1].set_ylabel("Total lithium in electrodes [mol]")

plt.show()