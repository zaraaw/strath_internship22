# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:44:37 2022

@author: zaraw

from first principles (F = ma), trying different forcing terms 
"""
from matplotlib import pyplot as plt
from scipy.integrate import odeint
#import sympy as sp
import numpy as np

#same for each system
z0 = [0,0]                              # initial conditions
ts = np.arange(0,50,0.01)              # period of integration

#%% constant forcing function

F = 10
m = 3
def constant(z, t, F=F, m=m):
    return (z[1], F/m)

zs = odeint(constant, z0, ts)       # solving ode
xs = zs[:,0]                        # x (disp) 1st column
x_dots = zs[:,1]                    # x' (vel) 2nd column


plt.plot(ts, xs)                            # plot displacemet
plt.xlabel('time')
plt.ylabel('x values')
plt.title('displacement in x - constant F')
plt.show()

plt.plot(ts, x_dots)                        # plot velocity 
plt.xlabel('time')
plt.ylabel('x dot values')
plt.title('velocity in x - constant F')
plt.show()

# cant plot acceleration
# accel = F/m
# plt.plot(ts, accel)                         
# plt.xlabel('time')
# plt.ylabel('x values')
# plt.title('acceleration in x - constant F')
# plt.show()
#%% spring force (kx')
F = 10
k = 2
m = 3
def spring(z, t, F=F, k=k, m=m):
    return (z[1], (F-k*z[0])/m)

zs = odeint(spring, z0, ts)       # solving ode
xs = zs[:,0]                        # x (disp) 1st column
x_dots = zs[:,1]                    # x' (vel) 2nd column


plt.plot(ts, xs)                            # plot displacemet
plt.xlabel('time')
plt.ylabel('x values')
plt.title("displacement in x - F = kx (spring)")
plt.show()

plt.plot(ts, x_dots)                        # plot velocity 
plt.xlabel('time')
plt.ylabel('x dot values')
plt.title("velocity in x - F = kx (spring)")
plt.show()

"""
plt.plot(ts, zs)                        # plot disp & vel
plt.xlabel('time')
plt.title("F = kx(spring)")
plt.legend(['displacement','velocity'])
plt.show()"""

accel = (F - k*xs)/m
plt.plot(ts, accel)                        # plot acceleration
plt.xlabel('time')
plt.ylabel('x dot dot values')
plt.title("acceleration in x - F = kx(spring)")
plt.show()
#%% damping force (cx)
F = 10
c = 4
m = 3
def damping(z, t, F=F, c=c, m=3):
    return (z[1], (F-c*z[1])/m)

zs = odeint(damping, z0, ts)       # solving ode
xs = zs[:,0]                        # x (disp) 1st column
x_dots = zs[:,1]                    # x' (vel) 2nd column

plt.plot(ts, xs)                            # plot displacemet
plt.xlabel('time')
plt.ylabel('x values')
plt.title("displacement in x - F = cx' (damping)")
plt.show()

plt.plot(ts, x_dots)                        # plot velocity 
plt.xlabel('time')
plt.ylabel('x dot values')
plt.title("velocity in x - F = cx' (damping)")
plt.show()

accel = (F - c*x_dots)/m
plt.plot(ts, accel)                        # plot acceleration
plt.xlabel('time')
plt.ylabel('x dot dot values')
plt.title("acceleration in x - F = cx' (damping)")
plt.show()
#%% periodic
F = 0.1
A = 8
m = 3
n = 1
def periodic(z, t, F=F, A=A, m=m, n=n):
    return (z[1],
            (F - A*np.cos(n*t)/m))
            #(F - A*np.sin(n*t)/m))

zs = odeint(periodic, z0, ts)       # solving ode
xs = zs[:,0]                        # x (disp) 1st column
x_dots = zs[:,1]                    # x' (vel) 2nd column

plt.plot(ts, xs)                            # plot displacemet
plt.xlabel('time')
plt.ylabel('x values')
plt.title('displacement in x - F = F - cos(nt)')
#plt.title('displacement in x - F = sin(nt)')
plt.show()

plt.plot(ts, x_dots)                        # plot velocity 
plt.xlabel('time')
plt.ylabel('x dot values')
plt.title('velocity in x - F = F - cos(nt)')
#plt.title('velocity in x - F = sin(nt)')
plt.show()

#accel = (F - np.sin(n*ts))/m
accel = (F - np.cos(n*ts))/m
plt.plot(ts, accel)                        # plot acceleration
plt.xlabel('time')
plt.ylabel('x dot dot values')
#plt.title("acceleration in x - F = sin(nt)")
plt.title("acceleration in x - F = F - cos(nt)")
plt.show()
#%% spring mass damper example 

# Initialization
tstart = 0
tstop = 60
increment = 0.1
# Initial condition
x_init = [0,0]
t = np.arange(tstart,tstop+1,increment)
# Function that returns dx/dt
def mydiff(x, t):
    c = 4 # Damping constant
    k = 2 # Stiffness of the spring
    m = 20 # Mass
    F = 5
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m
    dxdt = [dx1dt, dx2dt]
    return dxdt
# Solve ODE
x = odeint(mydiff, x_init, t)
x1 = x[:,0]
x2 = x[:,1]
# Plot the Results
plt.plot(t,x1)
plt.plot(t,x2)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(["x1 (position)", "x2 (velocity)"])
plt.grid()
plt.show()

#%% spring damper & periodic term
F = 10
c = 4
k = 2
m = 3
n = 1
def spring_damp_sin(z, t, F=F, c=c, k=k, m=m, n=n):
    return (z[1], ((F-c*z[1]-k*z[0]-np.sin(n*t))/m))

ts = np.arange(0,15,0.01)                           # change (if needed) when changing n
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
"""
plt.plot(ts, zs)                        # plot disp & vel
plt.xlabel('time')
#plt.title("")
plt.legend(['displacement','velocity'])
plt.show()


plt.plot(ts, zs)                        # plot disp & vel
plt.plot(ts, accel)                     # plot accel
plt.xlabel('time')
#plt.title("increased c (damping term)")
#plt.title('increased k (spring term)')
plt.title('increased frequency (of sin term)')
plt.legend(['displacement','velocity', 'acceleration'])
plt.show()
"""
#%%
F = 10
c = 4
k = 2
m = 3
n = 0.5
accel = (F - c*x_dots - k * xs -np.sin(n))/m
plt.plot(ts, accel)
plt.xlabel('time')
plt.ylabel('x dot dot values')
plt.title("acceleration")
plt.show()