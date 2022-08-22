# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:31:34 2022

@author: zaraw

===============================================
applying sindy to the mass spring damper system 
==============================================
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error

import pysindy as ps

# ignore user warnings
import warnings
from scipy.integrate.odepack import ODEintWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ODEintWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

#%%
F = 10
c = 4
k = 2
m = 3
n = 1
def spring_damp_sin(t, z, F=F, c=c, k=k, m=m, n=n):
    return (z[1], ((F-c*z[1]-k*z[0]-np.sin(n*t))/m))

#%%
# Generate training data

dt = 0.001
t_train = np.arange(0, 100, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [0, 0]
x_train = solve_ivp(spring_damp_sin, t_train_span, 
                    x0_train, t_eval=t_train, **integrator_keywords).y.T
# add noise at this stage
rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train = x_train + np.random.normal(0, rmse / 100.0, x_train.shape)
print("Noisy")

x_dot_train_measured = np.array(
    [spring_damp_sin(0, x_train[i]) for i in range(t_train.size)]
    )

#%% Noisy models (copied from example)
# Fit the models and simulate

# library = ps.PolynomialLibrary(degree=5)
# threshold = 0.05

# noise_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# models = []
# t_sim = np.arange(0, 20, dt)
# x_sim = []
# for eps in noise_levels:
#     model = ps.SINDy(
#         optimizer=ps.STLSQ(threshold=threshold),
#         feature_library=library,
#     )
#     model.fit(
#         x_train,
#         t=dt,
#         x_dot=x_dot_train_measured
#         + np.random.normal(scale=eps, size=x_train.shape),             # noise gets added here
#         quiet=True,
#     )
#     print("model with noise level = ", eps)
#     model.print()
#     models.append(model)
#     x_sim.append(model.simulate(x_train[0], t_sim))


# # Plot results

# fig = plt.figure(figsize=(12, 5))
# model_idx = 2
# ax = fig.add_subplot(221)
# ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
# ax.plot(t_sim, x_sim[model_idx][:, 0], "k--")
# plt.title(f"noise level, $\eta$={noise_levels[model_idx]:.2f}")
# plt.ylabel("x0")
# plt.legend(['true', 'model'])

# ax = fig.add_subplot(223)
# ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
# ax.plot(t_sim, x_sim[model_idx][:, 1], "k--")
# plt.xlabel("time")
# plt.ylabel("x1")
# plt.legend(['true', 'model'])

# model_idx = 4
# ax = fig.add_subplot(222)
# ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
# ax.plot(t_sim, x_sim[model_idx][:, 0], "k--")
# plt.title(f"noise level, $\eta$={noise_levels[model_idx]:.2f}")
# plt.ylabel("x0")
# plt.legend(['true', 'model'])

# ax = fig.add_subplot(224)
# ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
# ax.plot(t_sim, x_sim[model_idx][:, 1], "k--")
# plt.xlabel("time")
# plt.ylabel("x1")
# plt.legend(['true', 'model'])

# fig.show()


#%% moiseless model - use to change library

# # change library
# lib = ps.FourierLibrary()
# lib = ps.PolynomialLibrary(degree=6)
# lib = ps.PolynomialLibrary(degree=2) + ps.FourierLibrary()
library= ps.PolynomialLibrary()

# library_functions = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x*x,
#     lambda x: np.sin(t_train),
# ]                
# library_function_names = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x + "^2",
#     lambda x: "sin(t)", 
# ]

# library = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names)
#     # + ps.PolynomialLibrary()
    
# # print("Custom Library: ")
# print(custom_library.get_feature_names(), "\n")

# # PDE library
# print("PDE Library")
# functions = [
#     lambda x : np.exp(x),
#     lambda x, y : np.sin(x*y),
#     # lambda x : np.sin(t_train)
#     ]
# function_names = [
#     lambda x: "exp(" + x + ")",
#     lambda x, y: "sin(" + x + y + ")",
#     # lambda x : "sin(t)"
#     ]
# lib = ps.PDELibrary(
#     library_functions=functions,
#     function_names=function_names,
#     temporal_grid=t_train,
#     derivative_order=1,
#     implicit_terms=True
#     )

threshold = 0.05

library = library.fit(x_train)
library.transform(x_train)
print("Library:")
print(library.get_feature_names(), "\n")
#%%
# # change threshold
# threshold = 0.0005

# fitting model
t_sim = np.arange(0, 20, dt)
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=library,
)
model.fit(
    x_train,
    t=dt,
    # x_dot=x_dot_train_measured,
    ensemble=True,
    quiet=True,
)
# x_sim = model.simulate(x0_train, t_sim)
x_sim = model.simulate(x_train[0], t_sim)

print("Model (w/ noise)")
model.print()
#%%
# plotting
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
ax.plot(t_sim, x_sim[:, 0], "k--")
plt.xlabel("time")
plt.ylabel("x0")
plt.title("Velocity")

ax = fig.add_subplot(122)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
ax.plot(t_sim, x_sim[:, 1], "k--")
plt.xlabel("time")
plt.ylabel("x1")
plt.legend(['true', 'model'])
plt.title("Acceleration")