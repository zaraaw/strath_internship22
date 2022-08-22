# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:40:00 2022

@author: zaraw

case western fault data - 105.mat is:
    fault diameter: 7 mil
    load: 0hp, motor speed: 1797 rpm
    accelerometer location: inner race
    
    
"""
# Import libraries.
import numpy as np
from matplotlib import pyplot as plt
import pysindy as ps 
from sklearn.model_selection import train_test_split

# For reproducibility
np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# define training data
import scipy.io as sio
file105 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\105.mat"
mat_7mil_0hp_inner = sio.loadmat(file105) 

full_data = mat_7mil_0hp_inner["X105_DE_time"]    
x_train, x_test = train_test_split(full_data, test_size=0.2)

dt = 1
t_train = np.arange(0, len(x_train), dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [0,0]

#%%SINDy (original)
print("================ \n Original SINDy \n================")
poly_order = 5
library = ps.PolynomialLibrary(degree=poly_order)
threshold = 5e-4

lib = library.fit(x_train)
lib.transform(x_train)
print("Library:")
print(lib.get_feature_names(), "\n")

# fitting model
t_sim = np.arange(0, len(x_train), dt)
# t_sim = np.arange(0, 50000, dt)
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=library,
)
model.fit(
    x_train,
    t=dt,
    #x_dot=x_dot_train_measured,
    quiet=True,
)
x_sim = model.simulate(x_train[0], t_sim)

print("SINDy Model")
model.print()

# plotting
plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
plt.plot(t_sim, x_sim[:, 0], "k--")
plt.legend(['True', 'SINDy model'])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("7mil, 0hp, inner race, DE fault")


#%% SINDyPI
print("\n================ \n SINDy-PI \n================")

# build library
library_functions = [
    # lambda x: x,
    # lambda x: x * x,
    lambda x: np.sin(t_train),
    ]
library_function_names = [
    # lambda x: x,
    # lambda x: x + x,
    lambda x: "sin(t)", 
    ]

# PDELibrary has temporal derivatives
sindy_library = ps.PDELibrary(
    library_functions=library_functions,
    temporal_grid=t_train,
    function_names=library_function_names,
    include_bias=True,
    implicit_terms=True,
    derivative_order=2,
    include_interaction=False
)
library = sindy_library + ps.PolynomialLibrary(degree=6)
lib = library.fit(x_train)
lib.transform(x_train)
print("Library: ")
print(lib.get_feature_names(), "\n")

# define optimiser
sindy_opt = ps.SINDyPI(
    threshold=1e-3,
    tol=1e-5,
    thresholder="l1",
    max_iter=6000,
    #normalize_columns=True
)

# generate model
model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=library,
)
model.fit(x_train, t=t_train)
model.print(precision=4)
#%%
t_sim = np.arange(0,len(x_train),dt)            # use to plot shorter amounts
x_dot_true = model.differentiate(x_train, t=t_train)
x_dot_pred = model.predict(x_train)  # returns d/dt of all library features

# plot
plt.figure()
plt.plot(x_dot_true[: t_sim.size, 0], 'r')
plt.plot(x_dot_pred[: t_sim.size, 3], 'k--')       # have to look at library
plt.ylabel("Amplitude")
plt.legend(['true', 'SINDy-PI model'])
plt.title('7mil, 0hp, inner race, DE fault')
plt.show()
         
sindy_library.get_feature_names()

#%% ENSEMBLE SINDy
print("\n================ \n Ensemble SINDy \n================")

# define optimiser
opt = ps.STLSQ(threshold=0.00)
#opt = ps.SR3(threshold=0.1, thresholder='l1')
#opt = ps.ConstrainedSR3()

# # define library 
# library_functions = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x*x,
#     lambda x: x*x*x,
#     lambda x: np.sin(t_train),
# ]                
# library_function_names = [
#     lambda x: 1,
#     lambda x: x,
#     lambda x: x + "^2",
#     lambda x: x + "^3",
#     lambda x: "sin(t)", 
# ]
# library = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names,
#     )

# # print library
# lib = library.fit(x_train)
# lib.transform(x_train)
# print("Library:")
# print(lib.get_feature_names(), "\n")

library = ps.PolynomialLibrary(degree=5)
lib = library.fit(x_train)
lib.transform(x_train)
print("Library:")
print(lib.get_feature_names(), "\n")


plt.plot(t_train, x_train, 'r')
plt.title('Training Data')
plt.show

# define for the test data --- where is this used? 
# t_test = np.arange(0, len(x_test), dt)
# t_test_span = (t_test[0], t_test[-1])
# x0_test = [0, 0]

# Instantiate and fit the SINDy model 
feature_names = ['x']
t_sim = np.arange(0, len(x_train), 1)
# t_sim = np.arange(0, 1000, dt)



# fit the SINDy model with V1 ensembling
model = ps.SINDy(optimizer=opt, feature_library=library)
model.fit(x_train, t=t_train, ensemble=True, quiet=True)
# model.fit(x_train[: t_sim.size, 0], t=t_sim, ensemble=True, quiet=True)
x_sim = model.simulate(x_train[0], t_train)
# x_sim = model.simulate(x_test[0], t_sim)
print('Ensemble with replacement (V1)')
model.print()
print("\n")
ensemble_coefs = model.coef_list

# plot V1 model
fig = plt.figure(figsize=(12, 5))

plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
plt.plot(t_sim, x_sim[:, 0], "k--")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("7mil, 0hp, inner race, DE fault")
plt.legend(['true', 'V1 ensemble model'])


# fit the SINDy modeling with V2 ensembling
model.fit(x_train, t=t_train, ensemble=True, replace=False, quiet=True)
x_sim = model.simulate(x_train[0], t_train)
print('Ensemble model without replacement (V2)')
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
plt.title("7mil, 0hp, inner race, DE fault")
plt.legend(['true', 'V2 ensemble model'])




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
plt.title("7mil, 0hp, inner race, DE fault")
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
plt.title("7mil, 0hp, inner race, DE fault")
plt.legend(['true', 'V4 ensemble model'])

#%% ??
# opt = ps.STLSQ(threshold=0)
# #opt = ps.SR3(threshold=0.1, thresholder='l1')
# #opt = ps.ConstrainedSR3()
# library_functions = [
#     # lambda x: 1,
#     # lambda x: x,
#     # lambda x: x*x,
#     # lambda x: x*x*x,
#     lambda x: np.sin(t_train),
# ]                
# library_function_names = [
#     # lambda x: 1,
#     # lambda x: x,
#     # lambda x: x + "^2",
#     # lambda x: x + "^3",
#     lambda x: "sin(t)", 
# ]
# library = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names,
#     )

# lib = library.fit(x_train)
# lib.transform(x_train)
# print("Library:")
# print(lib.get_feature_names(), "\n")

# model = ps.SINDy(optimizer=opt, feature_library=library)
# model.fit(x_train, t=t_train)
# print('Training Data')
# model.print()

# plt.plot(t_train, x_train, 'k')
# plt.title('Training Data')
# plt.show

# #%%
# # define for the test data --- where is this used? 
# t_test = np.arange(0, len(x_test), dt)
# t_test_span = (t_test[0], t_test[-1])
# x0_test = [0, 0]

# # Instantiate and fit the SINDy model 
# feature_names = ['x']
# # t_sim = np.arange(0, len(x_train), dt)
# t_sim = np.arange(0, 1000, dt)
# # model = ps.SINDy(feature_names=feature_names, optimizer=opt)


# print('Ensemble with replacement (V1)')
# model.fit(x_train, t=t_train, ensemble=True, quiet=True)
# x_sim = model.simulate(x_train[0], t_train)
# # x_sim = model.simulate(x_test[0], t_sim)
# model.print()
# print("\n")
# ensemble_coefs = model.coef_list

# # print V1 model
# fig = plt.figure(figsize=(12, 5))

# plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
# plt.plot(t_sim, x_sim[:, 0], "k--")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Normal Baseline 0hp")
# plt.legend(['true', 'V1 model'])


# #%%
# # without replacement (V2)
# print('Ensemble model without replacement (V2)')
# model.fit(x_train, t=t_train, ensemble=True, replace=False, quiet=True)
# model.print()
# print("\n")
# ensemble_coefs = model.coef_list

# # print V2 model 
# fig = plt.figure(figsize=(12, 5))

# plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
# plt.plot(t_sim, x_sim[:, 0], "k--")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Normal Baseline 0hp")
# plt.legend(['true', 'V2 model'])