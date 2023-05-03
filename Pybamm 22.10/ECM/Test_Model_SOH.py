import numpy as np
from scipy.integrate import odeint

# Battery Equivalent Circuit Model Parameters
R0 = 0.1 # Ohm
R1 = 0.01 # Ohm
C1 = 50 # Farad
R2 = 0.005 # Ohm
C2 = 200 # Farad

# Initial State of Charge (SOC) and State of Health (SOH)
SOC0 = 1.0
SOH0 = 1.0

# True Values of R0, R1, C1, R2, C2
R0_true = 0.1
R1_true = 0.01
C1_true = 50
R2_true = 0.005
C2_true = 200

# Load Current (Ampere)
def load_current(t):
    return 10.0*np.sin(2*np.pi*t/3600.0)

# Battery Model Differential Equations
def battery_model(x, t, R0, R1, C1, R2, C2, I, SOH):
    SOC = x[0]
    V1 = x[1]
    V2 = x[2]

    # State of Charge (SOC) Equation
    I_norm = I/(SOH*SOC0*3600)
    dSOC = -I_norm

    # Capacitor Voltage (V1) Equation
    dV1 = (V2-V1)/(R1*C1)
    V1_init = I_norm*R1

    # Capacitor Voltage (V2) Equation
    dV2 = (-V2-R2*I_norm)/C2

    # Open-Circuit Voltage (OCV) Equation
    OCV = 4.1 - 0.01*(SOC*100)

    # Total Voltage Equation
    V_tot = OCV - V1 - V2 - I_norm*R0

    dxdt = [dSOC, dV1, dV2]
    return dxdt

# Generate Synthetic Voltage and Current Data
dt = 1.0 # second
t = np.arange(0, 3600, dt) # 1 hour
I = load_current(t)
x0 = [SOC0, 0, 0] # initial conditions
V_true = []
for i in range(len(t)):
    x = odeint(battery_model, x0, [t[i], t[i]+dt], args=(R0_true, R1_true, C1_true, R2_true, C2_true, I[i], SOH0))
    V_true.append(x[-1][-1])
    x0 = x[-1]

# Add Noise to Voltage Data
mean = 0.0
stddev = 0.05 # 5% of the true voltage
V = V_true + np.random.normal(mean, stddev, len(t))

# Extended Kalman Filter (EKF) for SOH Estimation
def ekf_battery_model(x, P, V_meas, I, dt):
    SOC = x[0]
    SOH = x[1]

    # Prediction Step
    I_norm = I/(SOH*SOC0*3600)
    dSOC = -I_norm
    F = np.array([[1, 0], [0, 1]])
    Q = np.zeros((2,2))
    Q[0,0] = (0.01*dt)**2
    Q[1,1] = (

# State Transition Matrix
F = np.array([[1, 0], [0, 1]])

# Process Noise Covariance Matrix
Q = np.zeros((2,2))
Q[0,0] = (0.01*dt)**2
Q[1,1] = (0.0001*dt)**2

# Measurement Matrix
H = np.array([[OCV_model(SOC)], [0]])

# Measurement Noise Covariance Matrix
R = np.array([[stddev**2]])

# Innovation Covariance
S = H @ P @ H.T + R

# Kalman Gain
K = P @ H.T @ np.linalg.inv(S)

# Residual
y = V_meas - OCV_model(SOC)

# State Update
x = x + K @ y

# Covariance Update
P = (np.eye(2) - K @ H) @ P

return x, P

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10,8))

ax[0].plot(t, SOC_est, 'r-', label='Estimated')
ax[0].plot(t, [SOC0]*len(t), 'k--', label='Initial')
ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('SOC')
ax[0].set_ylim([0, 1])
ax[0].legend()

ax[1].plot(t, SOH_est, 'r-', label='Estimated')
ax[1].plot(t, [SOH0]*len(t), 'k--', label='Initial')
ax[1].set_xlabel('Time (sec)')
ax[1].set_ylabel('SOH')
ax[1].set_ylim([0, 1])
ax[1].legend()

plt.show()