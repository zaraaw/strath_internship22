# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:40:28 2022

@author: zaraw
"""

# Import libraries.
import numpy as np
from matplotlib import pyplot as plt
import pysindy as ps 

# For reproducibility
np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

#%%
import scipy.io as sio

file97 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\97.mat"
mat_0hp = sio.loadmat(file97) 

# add fault data 
full_data_norm = mat_0hp["X097_DE_time"]
x_train_norm, x_test_norm = np.split(full_data_norm, 2)

file105 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\105.mat"
mat_7mil_0hp_inner = sio.loadmat(file105) 

full_data_fault = mat_7mil_0hp_inner["X105_DE_time"]
x_train_fault, x_test_fault = np.split(full_data_norm, 2)

x_train = np.concatenate((x_train_norm, x_train_fault))
x_test = np.concatenate((x_test_norm, x_test_fault))

dt = 1
# x_train = mat_0hp["X097_DE_time"]               # normal 0 hp DE data
t_train = np.arange(0, len(x_train), dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [0,0]


# Only train on the data for x, chop the other variable!
# x_train = x_train[:, 0].reshape(len(t_train), 1)

# Define custom functions up to quadratic terms
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

# Define PDELibrary which generates up to first order temporal derivatives
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
#%%
sindy_opt = ps.SINDyPI(
    threshold=1e-3,
    tol=1e-5,
    thresholder="l1",
    max_iter=6000,
    #normalize_columns=True
)

model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=library,
)
model.fit(x_train, t=t_train)
model.print(precision=4)

t_sim = np.arange(0,5000,dt)
x_dot_true = model.differentiate(x_train, t=t_train)
x_dot_pred = model.predict(x_train)  # default returns d/dt of all 15 features!

#%% plotting section 
plt.figure()
plt.plot(x_dot_true, 'r')
# plt.title('true')
# plt.show()
plt.plot(x_dot_pred[:, 3], '--k')
# plt.title('model')
plt.legend(['true', 'model'])
# plt.title('DE 0hp')
plt.title("Normal Baseline 0hp + 7mil 0hp inner race DE fault")
plt.show()
         
sindy_library.get_feature_names()


