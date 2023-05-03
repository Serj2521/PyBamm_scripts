
import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Generate synthetic battery data
time = np.linspace(0, 100, 101)
voltage = np.exp(-time/10) + np.random.normal(scale=0.01, size=101)

# Define equivalent circuit model
def equivalent_circuit(t, R, C, V0):
    return V0 - R*np.exp(-t/(R*C))

# Fit equivalent circuit model to synthetic data
params0 = [2.4, 1, 1]  # initial guess for R, C, V0
params_fit, cov = curve_fit(equivalent_circuit, time, voltage, p0=params0)

# Print the fitted parameters
print("Fitted parameters:", params_fit)

plt.plot(time, voltage, '-', color ='blue', label ="Data")
plt.scatter(time, equivalent_circuit(time,*params_fit), s=2, color ='red',  label ="Curve fit")
plt.show()
