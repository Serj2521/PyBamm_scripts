# pip install scikit-learn
import os
os.system('cls' if os.name == 'nt' else 'clear')
import pybamm as pb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
#Change dataframe values

pb.set_logging_level("INFO")

# load models
model = pb.lithium_ion.DFN()

# Load Chemistry
param = pb.ParameterValues("Chen2020")

#
# Test
#

experiment_OCV = pb.Experiment(
[("Discharge at C/10 for 1 hour", "Rest for 2 hour")] ,
)

# Run simulations

sim = pb.Simulation(model, experiment=experiment_OCV, parameter_values=param, solver=pb.CasadiSolver())

sol=sim.solve()

#Data Generation

t = sol["Time [s]"].entries
V = sol["Terminal voltage [V]"].entries
C =sol["Current [A]"].entries
Cap =sol["Discharge capacity [A.h]"].entries
SOC_test=100*(1-Cap/param['Nominal cell capacity [A.h]'])
#SOC=np.linspace(0,100,step_test+1)

#Dataframe

df_test = pd.DataFrame({0:t,1:V,2:C,3:Cap,4:SOC_test})
df_test.columns = ['Time [h]', 'Voltage [V]','Current [A]', 'Discharge capacity [A.h]','State of Charge(%)']

df_polarization = df_test.loc[df_test['Voltage [V]'].idxmin()+1:]
df_polarization = df_polarization.reset_index()

# Define equivalent circuit model
def equivalent_circuit(t, R, C, V0):
    return V0 - R*np.exp(-t/(R*C))

print(df_test)
print('\n')
print(df_polarization)
print('\n')



# Plot


plt.plot(t,V,'g')
plt.plot(df_polarization["Time [h]"].to_numpy(),df_polarization["Voltage [V]"].to_numpy(),'r')

plt.xlabel("Time [s]")
plt.ylabel("Terminal voltage [V]")


plt.show()
