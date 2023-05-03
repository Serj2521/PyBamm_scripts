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
step_test=5
experiment_OCV = pb.Experiment(
[(f"Discharge at C/{step_test} for 1 hour", "Rest for 2 hour")* step_test] ,
)

# Run simulations

sim_OCV = pb.Simulation(model, experiment=experiment_OCV, parameter_values=param, solver=pb.CasadiSolver())
sol_OCV=sim_OCV.solve()

#Data Generation

t = sol_OCV["Time [h]"].entries
V = sol_OCV["Terminal voltage [V]"].entries
C =sol_OCV["Current [A]"].entries
Cap =sol_OCV["Discharge capacity [A.h]"].entries
SOC_test=100*(1-Cap/param['Nominal cell capacity [A.h]'])
SOC=np.linspace(0,100,step_test+1)

#Dataframe

df_test = pd.DataFrame({0:t,1:V,2:C,3:Cap,4:SOC_test})
df_test.columns = ['Time [h]', 'Voltage [V]','Current [A]', 'Discharge capacity [A.h]','State of Charge(%)']

# Iterating over SOC list
df_OCV  = pd.DataFrame({0:[],1:[],2:[],3:[],4:[]})
df_OCV .columns = ['Time [h]', 'Voltage [V]','Current [A]', 'Discharge capacity [A.h]','State of Charge(%)']
for i in SOC:
    df_OCV=pd.concat([df_OCV, df_test[abs(df_test["State of Charge(%)"]-i)<=0.0001]], ignore_index=True, sort=False)

df_OCV['State of Charge(%)'] = df_OCV['State of Charge(%)'].apply(lambda x: abs(round(x, 0)))
df_OCV['Voltage [V]'] = df_OCV['Voltage [V]'].apply(lambda x: abs(round(x, 3)))
df_OCV.iloc[0, 1] = 2.8
df_OCV.iloc[-1, 1] = 4.2

V_OCV=df_OCV["Voltage [V]"].to_numpy()
SOC_OCV=df_OCV["State of Charge(%)"].to_numpy()

data = np.column_stack((SOC_OCV, V_OCV))

# Get the unique x values in the dataset
unique_SOC = np.unique(data[:, 0])
# Combine the y values with the same x value
combined_data = np.zeros((unique_SOC.size, 2))
combined_data[:, 0] = unique_SOC
mean_OCV=[]
for i, x_val in enumerate(unique_SOC):
    y_vals = data[data[:, 0] == x_val, 1]
    mean_y=round(np.mean(y_vals),5)
    max_y=round(np.max(y_vals),5)
    mean_OCV.append(mean_y)

#Interpolation
X=np.linspace(0, 100, 201)
interp_func = interp1d(unique_SOC, mean_OCV)
interp_data = interp_func(X)

##Curve fitting

#Polyfit
z = np.polyfit(X, interp_data,7)
f = np.poly1d(z)
OCV_Poly = f(X)

# Define the Fourier series function to fit
def fourier_series(x, *a):
    N = 4  # degree of polynomial
    omega = 2 * np.pi / (2 * len(x))
    series = a[0]
    for i in range(1, N+1):
        series += a[i] * np.cos(i * omega * x) + a[N+i] * np.sin(i * omega * x)
    return series

# Perform Fourier series curve fitting
num_harmonics = 5 # choose an appropriate number of harmonics
a0 = np.mean(interp_data)
a_guess = [0] * (2*num_harmonics)
popt, pcov = curve_fit(fourier_series, X, interp_data, p0=a_guess, method='lm')
F_Fit = fourier_series(X, *popt)



# Calculate the mean squared error
mse_Poly = mean_squared_error(interp_data, OCV_Poly)
mse_Fourier = mean_squared_error(interp_data, F_Fit)
# Calculate the RMSE
rmse_Poly = np.sqrt(mse_Poly)
rmse_Fourier = np.sqrt(mse_Fourier)




print('\n')
print("RMSE Polyfit:", rmse_Poly)
print("RMSE Fourier:", rmse_Fourier)
print("Max OCV:", F_Fit[-1],"; Min OCV:", F_Fit[0])

# Parametrization
# Iterating over time list
t_param=np.linspace(0,t[-1],step_test+1)
df_param  = pd.DataFrame({0:[],1:[],2:[],3:[],4:[],5:[]})
df_param.columns = ['Time [h]', 'Voltage [V]','Current [A]', 'Discharge capacity [A.h]','State of Charge(%)','R0']
for i in t_param:
    df_param=pd.concat([df_param, df_test[abs(df_test["Time [h]"]-i)<=0.0001]], ignore_index=True, sort=False)
df_param['State of Charge(%)']=SOC[::-1]
# extract a range of rows from df_test based on a condition
df_cycle = df_test.loc[(df_test['Time [h]'] >= df_param.loc[0, 'Time [h]']) & 
                       (df_test['Time [h]'] <= df_param.loc[1, 'Time [h]'])]
df_polarization = df_cycle.loc[df_cycle['Voltage [V]'].idxmin()+1:]
df_polarization = df_polarization.reset_index()
# extract R0
df_param.loc[0,'R0']=((df_cycle.loc[0,'Voltage [V]']
                       -df_cycle.loc[1,'Voltage [V]'])
                        +(df_cycle['Voltage [V]'].iloc[df_cycle[df_cycle['Voltage [V]'] == df_cycle['Voltage [V]'].min()].index[0]]
                          -df_cycle['Voltage [V]'].min()))/2*df_cycle.loc[0,'Current [A]']

# Define the exponential function
def exponential(x, a, b, c, d):
    return a* np.exp(-b* x) + c* np.exp(-d* x)
popt_exp, pcov_exp = curve_fit(exponential, df_polarization.index.values,df_polarization["Voltage [V]"])
Exp_Fit = exponential(df_polarization.index.values, *popt_exp)
print('\n')
print(df_param)
print('\n')
print(df_cycle)
print('\n')
print(popt_exp)



# Plot
pb.dynamic_plot(sim_OCV)

fig, ax = plt.subplots(2, 2,sharex=False, sharey=False, figsize=(12,5))

ax[0, 0].plot(t,V)
ax[0, 0].set_xlabel("Time [h]")
ax[0, 0].set_ylabel("Terminal voltage [V]")
ax[0, 1].plot(df_cycle["Time [h]"].to_numpy(),df_cycle["Voltage [V]"].to_numpy(),'g')
ax[0, 1].plot(df_polarization["Time [h]"].to_numpy(),df_polarization["Voltage [V]"].to_numpy(),'r')
ax[0, 1].scatter(df_polarization["Time [h]"].to_numpy(), Exp_Fit, s=3.5, marker='x', color='k')
ax[0, 1].set_xlabel("Time [h]")
ax[0, 1].set_ylabel("Discharge capacity [A.h]")
ax[1, 0].plot(t,SOC_test,'r')
ax[1, 0].set_xlabel("Time [h]")
ax[1, 0].set_ylabel("SOC [%]")
ax[1, 1].scatter(SOC_OCV,V_OCV, s=0.5, color ='k', label ="Test data points")
ax[1, 1].scatter(unique_SOC, mean_OCV, s=3.5, marker='x', color='red', label="Mean OCV values")
ax[1, 1].scatter(X, interp_data, s=1, marker='x', color='blue', label="Interpolation")
ax[1, 1].plot(X, OCV_Poly, '--', color ='pink', label ="Polynomial fitting")
ax[1, 1].plot(X, F_Fit, '-.', color ='green', label ="Fourier fitting")
ax[1, 1].set_xlabel("SOC [%]")
ax[1, 1].set_ylabel("Voltage (V)")
ax[1, 1].legend(loc="lower right")

plt.show()
