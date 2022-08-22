# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:25:52 2022

@author: zaraw
function for the lotka volterra equations
"""

# def lotkavolterra(x, t, a = 0.05, b = 0.005, c = 0.005, d = 0.2):
#     return[
#         a * x[0] - b * x[0] * x[1],
#         c * x[0] * x[1] - d * x[1]]

def lotkavolterra(x, t, a = 0.05, b = 0.005, c = 0.005, d = 0.2):
    return[
        a * x[0] - b * x[0] * x[1],
        c * x[0] * x[1] - d * x[1]]
"""
# from sindySA example - change eqn slightly

def lotka_volterra(X, t, alpha, beta, delta, gamma):
	x, y = X
	dXdt = [alpha*x - beta*x*y,
             delta*x*y - gamma*y
	]
	return dXdt
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

#solve 
x0 = [15,1]                      # initial conditions
ts = np.arange(0,200, 0.1)           # period of integration
xs = odeint(lotkavolterra, x0, ts)              # solving ode
x = xs[:,0]                    # defining x as first column of xs
y = xs[:,1]

plt.plot(ts, x)
plt.plot(ts, y)
plt.xlabel('time')
plt.legend(['rabbits', 'foxes'])
plt.show()
#%%
plt.plot(x, y)
plt.title('phase plot')
plt.xlabel('rabbits')
plt.ylabel('foxes')
#"""