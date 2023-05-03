import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the Fourier series function
def fourier_series(x, *a):
    N = len(a) // 2
    omega = 2 * np.pi / N
    series = a[0]
    for i in range(1, N):
        series += a[i] * np.cos(i * omega * x) + a[N+i] * np.sin(i * omega * x)
    return series

# Define the exponential function
def exp_func(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.tanh(d * x) + e

# Define the data
x = np.linspace(0, 2 * np.pi, 50)
y = 3 * np.sin(x) + 2 * np.sin(2 * x) + np.random.normal(scale=0.5, size=len(x))

# Find the Fourier coefficients
num_harmonics = 5
a_guess = np.zeros(2 * num_harmonics + 1)
a_guess[0] = np.mean(y)
for i in range(1, num_harmonics):
    a_guess[i] = np.cos(i * x) @ y * 2 / len(x)
    a_guess[num_harmonics+i] = np.sin(i * x) @ y * 2 / len(x)

popt, _ = curve_fit(fourier_series, x, y, p0=a_guess)

# Find the exponential decay coefficients
a_coeffs = popt[1:num_harmonics+1] + 1j * popt[num_harmonics+1:]
a_fit, _ = curve_fit(exp_func, np.arange(num_harmonics) + 1, np.abs(a_coeffs))

#Solve
F_Fit = fourier_series(x, *popt)
Exp_Fit = exp_func(x, *a_fit)

# Print the results
print(f"Fourier coefficients: {popt}")
print(f"Exponential decay coefficients: {a_fit}")

plt.scatter(x,y, s=0.5, color ='k', label ="Test data points")
plt.plot(x, F_Fit, '--', color ='pink', label ="Fourier fitting")
plt.plot(x, Exp_Fit, '-.', color ='green', label ="Exponential fitting")
plt.legend(loc="lower right")

plt.show()