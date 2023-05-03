
# "SEI": "ec reaction limited",
#"SEI film resistance": "distributed",
# "lithium plating": "irreversible",
# plating_options = ["reversible", "irreversible", "partially reversible"]
# "lithium plating porosity change": "true",
# "particle": "Fickian diffusion", 
#"particle mechanics": "swelling and cracking",
        
import os
os.system('cls' if os.name == 'nt' else 'clear')
import pybamm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


## MODEL

model = pybamm.lithium_ion.DFN({
    "particle mechanics": "swelling and cracking",
    "SEI": "solvent-diffusion limited",
    "SEI on cracks": "true",
    "SEI film resistance": "distributed",
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",
})

## PARAMETER IDENTIFICATION

def Updated_cracking_rate_Ai2020(T_dim):
    """
    Particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    Eac_cr = 0  # to be implemented
    arrhenius = pybamm.exp(Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return 3*k_cr * arrhenius

def Updated_graphite_cracking_rate_Ai2020(T_dim):
    k_cr = 3.9e-20
    T_ref = 298.15
    Eac_cr = pybamm.Parameter(
        "Negative electrode activation energy for cracking rate [J.mol-1]"
    )
    arrhenius = pybamm.exp(Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / T_ref))
    return 3000*k_cr * arrhenius

param = pybamm.ParameterValues("OKane2022")
param.update({"Ambient temperature [K]": 298.15})                                           # [268.15K = -5ºC, 283.15K= 10ºC, 298.15K = 25ºC, 308.15K = 25ºC]
param.update({"Upper voltage cut-off [V]": 4.21})
param.update({"SEI kinetic rate constant [m.s-1]": 1e-12})                                  # Original SEI kinetic rate constant [m.s-1]: 1e-12
param.update({"Lithium plating transfer coefficient": 0.5})                                 # From Lithium plating Model: 0.5
param.update({"Dead lithium decay constant [s-1]": 1E-4})                                   # From Lithium plating Model: 1E-4 [s-1]
param.update({"Positive electrode cracking rate": Updated_cracking_rate_Ai2020})            # From Ai2020 parametrization
param.update({"Negative electrode cracking rate": Updated_graphite_cracking_rate_Ai2020})   # From Ai2020 parametrization



# Discretisation points
var_pts = {
    "x_n": 20,  # negative electrode
    "x_s": 20,  # separator 
    "x_p": 20,  # positive electrode
    "r_n": 30,  # negative particle
    "r_p": 30,  # positive particle
}


pybamm.set_logging_level("NOTICE")

exp = pybamm.Experiment(["Hold at 4.2 V until C/10", "Rest for 1 hour", "Discharge at 2C until 2.5 V", "Charge at 2C until 4.2 V"]*5)

fast_solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(model, parameter_values=param, experiment=exp, solver=fast_solver, var_pts=var_pts)
sol = sim.solve()


t = sol["Time [h]"].entries

V = sol["Terminal voltage [V]"].entries
C =sol["Current [A]"].entries
Cap =sol["Discharge capacity [A.h]"].entries
SEI = sol["Loss of lithium to SEI [mol]"].entries + sol["Loss of lithium to SEI on cracks [mol]"].entries + sol["Loss of lithium to lithium plating [mol]"].entries
lithium_neg = sol["Total lithium in negative electrode [mol]"].entries
lithium_pos = sol["Total lithium in positive electrode [mol]"].entries
Tot_lithium_loss = sol["Loss of lithium inventory [%]"].entries

#Dataframe
df = pd.DataFrame({0:t,1:V,2:C,3:Cap,4:(lithium_neg+lithium_pos),5:(SEI)})
df.columns = ['Time [h]', 'Voltage [V]','Current [A]', 'Discharge capacity [A.h]','Total Lithium in electrodes [mol]','Lithium Loss to SEI [mol]']
df_cycle = df[abs(df["Voltage [V]"]-2.5)<=0.0001]
df_cycle.reset_index(inplace = True, drop = True)

#Cycle function vectors
Cycle_Count=df_cycle.index.to_numpy()+1

Li_fCycl=np.concatenate((
   np.array([lithium_neg[0]+lithium_pos[0]]), df_cycle['Total Lithium in electrodes [mol]'].to_numpy()
))

SOH = 100*(Li_fCycl/(lithium_neg[0]+lithium_pos[0]))

## Prints
print(param)
#print('\n')
#print(model.summary_variables)
print('\n')
print(len(t))
print('\n')
print(df)
print('\n')
print(abs(df_cycle))
print('\n')
print(Cycle_Count)
print('\n')
print(np.array(np.linspace(0,Cycle_Count[-1],len(SOH))))
print('\n')
print(Li_fCycl)
# print(param["Upper voltage cut-off [V]"])
# pybamm.print_citations()


fig, ax = plt.subplots(2, 2,sharex=False, sharey=False, figsize=(12,5))

ax[0, 0].plot(t,V)
ax[0, 0].set_xlabel("Time [h]")
ax[0, 0].set_ylabel("Terminal voltage [V]")
ax[1, 0].plot(t,C,'r')
ax[1, 0].set_xlabel("Time [h]")
ax[1, 0].set_ylabel("Current [A]")
ax[0, 1].plot(SOH)
ax[0, 1].set_xlabel("Cycle No.")
ax[0, 1].set_ylabel("State of Health [%]")
ax[0, 1].set_xticks(np.array(np.arange(0,Cycle_Count[-1],2)))
ax[1, 1].plot(t,lithium_neg+lithium_pos)
ax[1, 1].plot(t,lithium_neg[0]+lithium_pos[0]-SEI,linestyle="dashed")
ax[1, 1].set_xlabel("Time [h]")
ax[1, 1].set_ylabel("Total lithium in electrodes [mol]")

plt.show()


