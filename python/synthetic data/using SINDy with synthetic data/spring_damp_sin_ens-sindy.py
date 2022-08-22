# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:17:11 2022

@author: zaraw
=======================================================================
synthetic noisy data (using spring damp sin system) in ensemble sindy
- following pysindy example 13 ensembling 
=======================================================================
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error

# Ignore integration and solver convergence warnings
import warnings
from scipy.integrate.odepack import ODEintWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ODEintWarning)

import pysindy as ps

# Seed the random number generators for reproducibility
np.random.seed(100)

# integration keywords for solve_ivp, typically needed for chaotic systems
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

F = 10
c = 4
k = 2
m = 3
n = 1
def spring_damp_sin(t, z, F=F, c=c, k=k, m=m, n=n):
    return (z[1], ((F-c*z[1]-k*z[0]-np.sin(n*t))/m))
#%% generate the training data
dt = .001
t_train = np.arange(0, 100, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [0, 0]
x_train = solve_ivp(spring_damp_sin, t_train_span, x0_train, 
                    t_eval=t_train, **integrator_keywords).y.T

# add 1% noise
rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train = x_train + np.random.normal(0, rmse / 100.0, x_train.shape)
print("Noisy")

# x_dot_train_measured = np.array(
#     [spring_damp_sin(0, x_train[i]) for i in range(t_train.size)]
#     )

# # Evolve the equations in time using a different initial condition (test data) - not used?
# t_test = t_train
# t_test_span = (t_test[0], t_test[-1])
# x0_test = np.array([1, 1])
# x_test = solve_ivp(spring_damp_sin, t_test_span, x0_test, 
#                    t_eval=t_test, **integrator_keywords).y.T

# Instantiate and fit the SINDy model 
# feature_names = ['x0', 'x1'] #not needed bc these r the default
optimizer = ps.STLSQ(threshold=0.01)
# t_sim = np.arange(0, 20, dt)
t_sim = np.arange(0, len(x_train), 1)

# # library = ps.PolynomialLibrary()
# library_functions = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x*x,
#     # lambda x: x*x*x,
#     # lambda x: x*x*x*x,
#     # lambda x: x*x*x*x*x,
# ]                
# library_function_names = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x + "^2",
#     # lambda x: x + "^3",
#     # lambda x: x + "^4",
#     # lambda x: x + "^5",
# ]
# sindy_library = ps.PDELibrary(
#     library_functions=library_functions,
#     temporal_grid=t_train,
#     function_names=library_function_names,
#     include_bias=True,
#     implicit_terms=True,
#     derivative_order=1,
#     include_interaction=False
# )
# lib = sindy_library
lib = ps.PolynomialLibrary()

# lib = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names,
#     )
# lib = ps.IdentityLibrary() + ps.FourierLibrary()
#%%
lib = lib.fit(x_train)
lib.transform(x_train)
print("Library: ")
print(lib.get_feature_names(), "\n")

# lib_t = lib.transform(x_train)


#%% 
# No ensembling
# optimizer = ps.STLSQ(threshold=0.005)
model = ps.SINDy(#feature_names=feature_names,
                 optimizer=optimizer,
                 feature_library=lib)
model.fit(x_train, t=dt, ensemble=False, quiet=True)
x_sim = model.simulate(x_train[0], t_sim)
print("No ensembling")
model.print()
print("\n")

# print no ensembling model
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
plt.legend(['true', 'SINDy (original) model'])
plt.title("Acceleration")
#%%
# Ensemble with replacement (V1)
modelV1 = ps.SINDy(#feature_names=feature_names,
                 optimizer=optimizer,
                 feature_library=lib)
modelV1.fit(x_train, t=dt, ensemble=True, quiet=True)
x_simV1 = modelV1.simulate(x_train[0], t_sim)
print("Ensemble with replacement (V1)")
modelV1.print()
print("\n")
ensemble_coefsV1 = modelV1.coef_list

# print V1 model
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
ax.plot(t_sim, x_simV1[:, 0], "k--")
plt.xlabel("time")
plt.ylabel("x0")
plt.title("Velocity")

ax = fig.add_subplot(122)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
ax.plot(t_sim, x_simV1[:, 1], "k--")
plt.xlabel("time")
plt.ylabel("x1")
plt.legend(['true', 'V1 model'])
plt.title("Acceleration")

#%%
# Ensemble without replacement (V2)
modelV2 = ps.SINDy(#feature_names=feature_names,
                 optimizer=optimizer,
                 feature_library=lib)
modelV2.fit(x_train, t=dt, ensemble=True, replace=False, quiet=True)
x_simV2 = modelV2.simulate(x_train[0], t_sim)
print("Ensemble without replacement (V2)")
modelV2.print()
print("\n")
ensemble_coefsV2 = modelV2.coef_list

# print V2 model 
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
ax.plot(t_sim, x_simV2[:, 0], "k--")
plt.xlabel("time")
plt.ylabel("x0")
plt.title("Velocity")

ax = fig.add_subplot(122)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
ax.plot(t_sim, x_simV2[:, 1], "k--")
plt.xlabel("time")
plt.ylabel("x1")
plt.legend(['true', 'V2 model'])
plt.title("Acceleration")

#%%
# library ensemble V3
modelV3 = ps.SINDy(#feature_names=feature_names, 
                  optimizer=optimizer,
                  feature_library=lib)
modelV3.fit(x_train, t=dt, library_ensemble=True, quiet=True)
x_simV3 = modelV3.simulate(x_train[0], t_sim)
print("Library Ensemble (V3)")
modelV3.print()
print("\n")
library_ensemble_coefs = modelV3.coef_list

# print V3 model 
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
ax.plot(t_sim, x_simV3[:, 0], "k--")
plt.xlabel("time")
plt.ylabel("x0")
plt.title("Velocity")

ax = fig.add_subplot(122)
# ax.plot(t_test, x_train[: t_test.size, 1], "r")
# ax.plot(t_test, x_test[:, 1], "k--")
ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
ax.plot(t_sim, x_sim[:, 1], "k--")
plt.xlabel("time")
plt.ylabel("x1")
plt.legend(['true', 'V3 model'])
plt.title("Acceleration")

#%%
# combination ensemble V4
# optimizer = ps.STLSQ(threshold=0.5)       # failed when threshold is .05
modelV4 = ps.SINDy(#feature_names=feature_names, 
                  optimizer=optimizer,
                  feature_library=lib)
modelV4.fit(x_train, t=dt, library_ensemble=True, ensemble=True, 
          n_candidates_to_drop=2, quiet=True)
x_simV4 = modelV4.simulate(x_train[0], t_sim)
print("Combination Ensemble (V4)")
modelV4.print()
print("\n")
double_ensemble_coefs = modelV4.coef_list

# print V4 model 
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(t_sim, x_train[: t_sim.size, 0], "r")
ax.plot(t_sim, x_simV4[:, 0], "k--")
plt.xlabel("time")
plt.ylabel("x0")
plt.title("Velocity")

ax = fig.add_subplot(122)
ax.plot(t_sim, x_train[: t_sim.size, 1], "r")
ax.plot(t_sim, x_simV4[:, 1], "k--")
plt.xlabel("time")
plt.ylabel("x1")
plt.legend(['true', 'V4 model'])
plt.title("Acceleration")
