import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Create some sample data with multiple y values for each x value
x = np.linspace(0, 2 * np.pi, 50)
y = 3 * np.sin(x) + 2 * np.sin(2 * x) + np.random.normal(scale=0.5, size=len(x))

# Define the Fourier series function to fit
def fourier_series(x, *a):
    N = int(len(a)/2)  # degree of polynomial
    omega = 2 * np.pi / (2 * len(x))
    series = a[0]
    for i in range(1, N+1):
        series += a[i] * np.cos(i * omega * x) + a[N+i] * np.sin(i * omega * x)
    return series

# Perform Fourier series curve fitting
N = 8 # degree of polynomial for original equation
a0 = np.mean(x)
a_guess = [a0] + [0] * (2*N)
popt, pcov = curve_fit(fourier_series, x, y, p0=a_guess, method='lm')
F_Fit = fourier_series(x, *popt)

#FFT
signal_fft = np.fft.fft(F_Fit)
freqs = np.fft.fftfreq(len(F_Fit))

#Reduction
reduction = 0.09
mask = (freqs > -reduction) & (freqs < reduction)
signal_fft_trunc = signal_fft.copy()
signal_fft_trunc[~mask] = 0
reduced_signal = np.fft.ifft(signal_fft_trunc)

# Calculate the coefficients for the reduced-order Fourier series
N_reduced = int(reduction*len(freqs))  # degree of polynomial for reduced equation
a_guess_reduced = [a0] + [0] * (2*N_reduced)
popt_reduced, pcov_reduced = curve_fit(fourier_series, x, np.real(reduced_signal), p0=a_guess_reduced, method='lm')
Reduced_F_Fit = fourier_series(x, *popt_reduced)

# Print the coefficients for the reduced-order Fourier series
print(f"Coefficients for original Fourier series with {N_reduced} terms:")
for i in range(2*N+1):
    print(f"a{i} = {popt[i]}")

print('\n')

# Print the coefficients for the reduced-order Fourier series
print(f"Coefficients for reduced-order Fourier series with {N_reduced} terms:")
for i in range(2*N_reduced+1):
    print(f"a{i} = {popt_reduced[i]}")

#Plots
fig, ax = plt.subplots(2, 1, figsize=(12,12))

ax[0].scatter(x, y, s=0.5, color='k', label="Test data points")
ax[0].plot(x, F_Fit, '--', color='pink', label="Original Fourier fitting")
ax[0].plot(x, Reduced_F_Fit, '-.', color='blue', label="Reduced Fourier fitting")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].legend()

ax[1].bar(freqs, np.abs(signal_fft), width=0.5, color='blue', label="Original")
ax[1].bar(freqs, np.abs(signal_fft_trunc), width=0.5, color='red', label="Truncation")

plt.show()