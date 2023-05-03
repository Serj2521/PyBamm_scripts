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

# Define the exponential function
def exponential(x, a, b, c, d, e):
    return -a* np.exp(-b* x) - c* np.exp(-d* x)+e

X=np.linspace(0,10,100)
par_in=[1,0.5,1,0.5,3]

#original results
Y=exponential(X,*par_in)

#Plots
fig, ax = plt.subplots(2, 1, figsize=(12,12))

ax[0].scatter(X, Y, s=0.5, color='k', label="Test data points")
#ax[0].plot(x, F_Fit, '--', color='pink', label="Original Fourier fitting")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].legend()

#ax[1].bar(freqs, np.abs(signal_fft), width=0.5, color='blue', label="Original")

plt.show()