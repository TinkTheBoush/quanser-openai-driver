from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from numba import jit,jitclass,float32
import numpy as np


# Motor
Rm = 8.4  # Resistance
kt = 0.042  # Current-torque (N-m/A)
km = 0.042  # Back-emf constant (V-s/rad)

# Rotary Arm
mr = 0.095  # Mass (kg)
Lr = 0.085  # Total length (m)
Jr = mr * Lr**2 / 8 + 4.6e-6 # Moment of inertia about pivot (kg-m^2)
Dr = 0.0015  # Equivalent viscous damping coefficient (N-m-s/rad)

# Pendulum Link
mp = 0.024  # Mass (kg)
Lp = 0.129  # Total length (m)
Jp = mp * Lp**2 / 12  # Moment of inertia about pivot (kg-m^2)
Dp = 0.0005  # Equivalent viscous damping coefficient (N-m-s/rad)
#Dp = 0.000125  # 튜닝됨.

g = 9.81  # Gravity constant

@jit
#def forward_model(theta, alpha, theta_dot, alpha_dot, Vm, dt, euler_steps):
def forward_model(state, Vm, dt, euler_steps):
    dt /= euler_steps
    for step in range(euler_steps):
        tau = (km * (Vm - km * state[2])) / Rm  # torque

        theta_dot_dot = -(
            Lp * Lr * mp *
            (8.0 * Dp * state[3] - Lp**2 * mp * state[2]**2 * np.sin(
                2.0 * state[1]) + 4.0 * Lp * g * mp * np.sin(state[1])) *
            np.cos(state[1]) + (4.0 * Jp + Lp**2 * mp) *
            (4.0 * Dr * state[2] +
             Lp**2 * state[3] * mp * state[2] * np.sin(2.0 * state[1]) +
             2.0 * Lp * Lr * state[3]**2 * mp * np.sin(state[1]) - 4.0 * tau)) / (
                 4.0 * Lp**2 * Lr**2 * mp**2 * np.cos(state[1])**2 +
                 (4.0 * Jp + Lp**2 * mp) *
                 (4.0 * Jr + Lp**2 * mp * np.sin(state[1])**2 + 4.0 * Lr**2 * mp))
        alpha_dot_dot = (
            4.0 * Lp * Lr * mp *
            (2.0 * Dr * state[2] + 0.5 * Lp**2 * state[3] * mp * state[2] *
             np.sin(2.0 * state[1]) + Lp * Lr * state[3]**2 * mp * np.sin(state[1])
             - 2.0 * tau) * np.cos(state[1]) -
            (4.0 * Jr + Lp**2 * mp * np.sin(state[1])**2 + 4.0 * Lr**2 * mp) *
            (4.0 * Dp * state[3] - 0.5 * Lp**2 * mp * state[2]**2 *
             np.sin(2.0 * state[1]) + 2.0 * Lp * g * mp * np.sin(state[1]))) / (
                 4.0 * Lp**2 * Lr**2 * mp**2 * np.cos(state[1])**2 +
                 (4.0 * Jp + Lp**2 * mp) *
                 (4.0 * Jr + Lp**2 * mp * np.sin(state[1])**2 + 4.0 * Lr**2 * mp))

        state[2] += theta_dot_dot * dt
        state[3] += alpha_dot_dot * dt

        state[0] += state[2] * dt
        state[1] += state[3] * dt


    return state

@jitclass([('_euler_steps',float32),
           ('_time_step',float32),
           ('state',float32[:]),
           ])
class QubeServo2Simulator(object):
    '''Simulator that has the same interface as the hardware wrapper.'''
    def __init__(self,
                 euler_steps=10,
                 frequency=1000):
        self._time_step = 1.0 / frequency
        self._euler_steps = euler_steps
        self.state = np.zeros(4,dtype=float32)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return None

    def reset_encoders(self):
        pass

    def action(self, action):
        a = forward_model(
            self.state,
            action[0],
            self._time_step,
            self._euler_steps)
        encoders = self.state[:2]   # [theta, alpha]
        currents = [action / 8.4]  # 8.4 is resistance
        if encoders[0] >= np.pi - 0.5 or encoders[0] <= -np.pi + 0.5:
            others = [4000.0 ]
        else:
            others = [0.]

        return currents, encoders, others

    def reset_sim_down(self):
        self.state = np.zeros(4,dtype=float32)

    def reset_sim_up(self):
        self.state = np.zeros(4, dtype=float32)
        self.state[1] = +np.pi


