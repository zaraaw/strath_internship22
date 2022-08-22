# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:52:25 2022

@author: zaraw

applying ensemble SINDy to normal operation case western data
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pysindy as ps

np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


#%%

import scipy.io as sio
file97 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\97.mat"
mat_0hp = sio.loadmat(file97) 

# full_data_set = mat_0hp["X097_DE_time"]
# x_train, x_test = train_test_split(full_data_set, test_size=0.2)

# add fault data 
full_data_norm = mat_0hp["X097_DE_time"]
x_train_norm, x_test_norm = np.split(full_data_norm, 2)

file105 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\105.mat"
mat_7mil_0hp_inner = sio.loadmat(file105) 

full_data_fault = mat_7mil_0hp_inner["X105_DE_time"]
x_train_fault, x_test_fault = np.split(full_data_norm, 2)

x_train = np.concatenate((x_train_norm, x_train_fault))
x_test = np.concatenate((x_test_norm, x_test_fault))


dt = 1.0                # change to match 12k sampling rate ? 
t_train = np.arange(0, len(x_train), dt)
t_train_span = (t_train[0], t_train[-1])
# x0_train = 0                              # setting inital condition as 0
x0_train = x_train[0]                       # # setting inital condition as first datapoint

opt = ps.STLSQ(threshold=0.01)
#opt = ps.SR3(threshold=0.1, thresholder='l1')
#opt = ps.ConstrainedSR3()
# library_functions = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x*x,
#     lambda x: x*x*x,
#     #lambda x: np.sin(t_train),
# ]                
# library_function_names = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x + "^2",
#     lambda x: x + "^3",
#     #lambda x: "sin(t)", 
# ]

# # Define PDELibrary which generates up to first order temporal derivatives
# library = ps.PDELibrary(
#     library_functions=library_functions,
#     temporal_grid=t_train,
#     function_names=library_function_names,
#     include_bias=True,
#     implicit_terms=True,
#     derivative_order=2,
#     include_interaction=False
# )

# library = ps.CustomLibrary(library_functions=library_functions,
#                            function_names=library_function_names)
library = ps.PolynomialLibrary(degree=5)
lib = library.fit(x_train)
lib.transform(x_train)
print("Library:")
print(lib.get_feature_names(), "\n")

# normal SINDy - no ensembling 
# model = ps.SINDy(optimizer=opt, feature_library=library)
# model.fit(x_train, t=dt)
# print('Training Data')
# model.print()

plt.plot(t_train, x_train, 'k')
plt.title('Training Data')
plt.show


# define for the test data --- where is this used? 
# t_test = np.arange(0, len(x_test), dt)
# t_test_span = (t_test[0], t_test[-1])
# x0_test = [0, 0]

# Instantiate and fit the SINDy model 
feature_names = ['x']
t_sim = np.arange(0, len(x_train), dt)
# t_sim = np.arange(0, 1000, dt)
# model = ps.SINDy(feature_names=feature_names, optimizer=opt)

#%%
print('Ensemble with replacement (V1)')
model = ps.SINDy(optimizer=opt, feature_library=library)
# model.fit(x_train, t=dt, ensemble=True, quiet=True)
model.fit(x_train, t=t_train, ensemble=True, quiet=True)
x_sim = model.simulate(x_train[0], t=t_train) 
# x_sim = model.simulate(x_test[0], t_sim)
model.print()
print("\n")
ensemble_coefs = model.coef_list

# print V1 model
fig = plt.figure(figsize=(12, 5))

plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
plt.plot(t_sim, x_sim[:, 0], "k--")
# plt.plot(t_test, x_train[: t_test.size, 0], "r")
# plt.plot(t_test, x_test[:, 0], "k--")
plt.xlabel("Time")
plt.ylabel("Amplitude")
# plt.title("Normal Baseline 0hp")
plt.title("Normal Baseline 0hp + 7mil 0hp inner race DE fault")
plt.legend(['true', 'V1 model'])



#%%
# without replacement (V2)
print('Ensemble model without replacement (V2)')
# ensemble_optimizer = ps.STLSQ()
# model = ps.SINDy(feature_names=feature_names, optimizer=ensemble_optimizer)
model.fit(x_train, t=dt, ensemble=True, replace=False, quiet=True)
x_sim = model.simulate(x_train[0], t=t_train)
model.print()
print("\n")
ensemble_coefs = model.coef_list

# print V2 model 
fig = plt.figure(figsize=(12, 5))

# ax = fig.add_subplot(121)
plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
plt.plot(t_sim, x_sim[:, 0], "k--")
plt.xlabel("Time")
plt.ylabel("Amplitude")
# plt.title("Normal Baseline 0hp")
plt.title("Normal Baseline 0hp + 7mil 0hp inner race DE fault")
plt.legend(['true', 'V2 model'])

#%%
# V3 ensembling
# library_ensemble_optimizer = ps.STLSQ()
# model = ps.SINDy(feature_names=feature_names, 
#                  optimizer=library_ensemble_optimizer)
model.fit(x_train, t=dt, library_ensemble=True, quiet=True)
library_ensemble_coefs = model.coef_list

x_sim = model.simulate(x_train[0], t_sim)
# x_sim = model.simulate(x_train[0], t_sim)
print('Ensemble library (V3)')
model.print()
print("\n")


# plot V3 model 
fig = plt.figure(figsize=(12, 5))

# ax = fig.add_subplot(121)
plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
# plt.scatter(t_sim, x_train[: t_sim.size, 0], marker="." )       #scatter
plt.plot(t_sim, x_sim[:, 0], "k--")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Normal Baseline 0hp + 7mil 0hp inner race DE fault")
plt.legend(['true', 'V3 ensemble model'])

# V4 ensembling
# double_ensemble_optimizer = ps.STLSQ()
# model = ps.SINDy(feature_names=feature_names, 
#                  optimizer=double_ensemble_optimizer)
model.fit(x_train, t=dt, library_ensemble=True, ensemble=True, 
          n_candidates_to_drop=2, quiet=True)
print('Both types of ensembling w/ candidate drops (V4)')
model.print()
print("\n")
double_ensemble_coefs = model.coef_list

x_sim = model.simulate(x_train[0], t_sim)
# plot V4 model 
fig = plt.figure(figsize=(12, 5))

# ax = fig.add_subplot(121)
plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
# plt.scatter(t_sim, x_train[: t_sim.size, 0], marker="." )       #scatter
plt.plot(t_sim, x_sim[:, 0], "k--")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Normal Baseline 0hp + 7mil 0hp inner race DE fault")
plt.legend(['true', 'V4 ensemble model'])


# #%%
# # Plot the fit of the derivative
# #x_train = np.asarray(x_train_multi)[0, :, :]
# x_dot_train = model.differentiate(x_train, t_train)
# x_dot_train_pred = ensemble_optimizer.Theta_[:len(t_train) - 2, :] @ ensemble_optimizer.coef_.T
# plt.figure()
# #inds = [ind1, ind2]
# inds = [0, 1]               # manually writing indices rather that getting them 
# r = 1
# for i in range(r):
#     plt.subplot(1, r, i + 1)
#     plt.plot(t_train, x_dot_train[:, i], 'k')
#     plt.plot(t_train[1:-1], x_dot_train_pred[:, inds[i]], 'r--')
# plt.title('Normal operation 0 hp')
# plt.legend(['true', 'prediction'])
# plt.show()    
    
# #%% taken from ex 1 feature overview - ensembling section
# # Default is to sample the entire time series with replacement, generating 10 models on roughly 
# # 60% of the total data, with duplicates. 

# # Custom feature names
# np.random.seed(100)
# #feature_names = ['x', 'y', 'z']

# ensemble_optimizer = ps.EnsembleOptimizer(
#     ps.STLSQ(threshold=1e-3,normalize_columns=False),
#     bagging=True,
#     n_subset=int(0.6*x_train.shape[0]))

# model = ps.SINDy(optimizer=ensemble_optimizer,
#                  feature_names=feature_names)
# model.fit(x_train, t=dt)
# ensemble_coefs = ensemble_optimizer.coef_list
# mean_ensemble = np.mean(ensemble_coefs, axis=0)
# std_ensemble = np.std(ensemble_coefs, axis=0)

# # Now we sub-sample the library. The default is to omit a single library term.
# library_ensemble_optimizer=ps.EnsembleOptimizer(ps.STLSQ(threshold=1e-7,            # changed this threshold
#                                                          normalize_columns=False),
#                                                 library_ensemble=True)
# model = ps.SINDy(optimizer=library_ensemble_optimizer,
#                  feature_names=feature_names)

# model.fit(x_train, t=dt)
# library_ensemble_coefs = library_ensemble_optimizer.coef_list
# mean_library_ensemble = np.mean(library_ensemble_coefs, axis=0)
# std_library_ensemble = np.std(library_ensemble_coefs, axis=0)

#%% Plot results
# plt.plot()

#%%
# xticknames = model.get_feature_names()
# for i in range(10):
#     xticknames[i] = '$' + xticknames[i] + '$'
# plt.figure(figsize=(10, 4))
# colors = ['b', 'r']
# plt.subplot(1, 2, 1)
# plt.title('ensembling')
# for i in range(3):
#     plt.errorbar(range(10), mean_ensemble[i, :], yerr=std_ensemble[i, :], 
#                  fmt='o', color=colors[i],
#                  label=r'Equation for $\dot{' + feature_names[i] + r'}$')
# ax = plt.gca()
# plt.grid(True)
# ax.set_xticks(range(10))
# ax.set_xticklabels(xticknames, verticalalignment='top')
# plt.subplot(1, 2, 2)
# plt.title('library ensembling')
# for i in range(3):
#     plt.errorbar(range(10), mean_library_ensemble[i, :], yerr=std_library_ensemble[i, :], 
#                  fmt='o', color=colors[i], 
#                  label=r'Equation for $\dot{' + feature_names[i] + r'}$')
# ax = plt.gca()
# plt.grid(True)
# plt.legend()
# ax.set_xticks(range(10))
# ax.set_xticklabels(xticknames, verticalalignment='top');