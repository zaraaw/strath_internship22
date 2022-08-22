# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:10:58 2022

@author: zaraw

"""
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np
from sklearn.metrics import mean_squared_error

z0 = [0,0]                              # initial conditions
ts = np.arange(0,50,0.01)              # period of integration
F = 10
c = 4
k = 2
m = 3
n = 1
def spring_damp_sin(z, t, F=F, c=c, k=k, m=m, n=n):
    return (z[1], ((F-c*z[1]-k*z[0]-np.sin(n*t))/m))

ts = np.arange(0,20,0.01)                           # change (if needed) when changing n
zs = odeint(spring_damp_sin, z0, ts)                # solving ode
xs = zs[:,0]                        # x (disp) 1st column
x_dots = zs[:,1]                    # x' (vel) 2nd column


plt.plot(ts, xs)                            # plot displacemet
plt.xlabel('time')
plt.ylabel('x values')
plt.title("displacement")
plt.show()

plt.plot(ts, x_dots)                        # plot velocity 
plt.xlabel('time')
plt.ylabel('x dot values')
plt.title("velocity")
plt.show()

accel = (F - c*x_dots - k*xs - np.sin(n*ts))/m

plt.plot(ts, accel)                        # plot acceleration
plt.xlabel('time')
plt.ylabel('x dot dot values')
plt.title("acceleration")
plt.show()

#%% add noise (random noise)
noise = np.random.normal(0, .1, xs.shape)

# rmse noise
rmse = mean_squared_error(xs, np.zeros(xs.shape), squared=False)
noise = np.random.normal(0, rmse / 100.0, xs.shape)

noisy_xs = xs + noise
plt.plot(ts, noisy_xs)                            # plot noisy displacemet
plt.xlabel('time')
plt.ylabel('x values')
plt.title("noisy displacement")
plt.show()

noisy_x_dots = x_dots + noise
plt.plot(ts, noisy_x_dots)                        # plot noisy velocity 
plt.xlabel('time')
plt.ylabel('x dot values')
plt.title("noisy velocity")
plt.show()

noisy_accel = accel + noise
plt.plot(ts, noisy_accel)                        # plot noisy acceleration
plt.xlabel('time')
plt.ylabel('x dot dot values')
plt.title("noisy acceleration")
plt.show()

