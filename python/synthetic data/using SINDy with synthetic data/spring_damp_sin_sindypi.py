# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:10:30 2022

@author: zaraw
"""
# Import libraries.
import numpy as np
from matplotlib import pyplot as plt
import pysindy as ps 
#import sympy as sp
from scipy.integrate import solve_ivp

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# For reproducibility
np.random.seed(100)

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

# Generate measurement data
dt = 0.05
t_train = np.arange(0, 50, dt)
t_train_span = (t_train[0], t_train[-1])

x0_train = [1, 0]
x_train = solve_ivp(spring_damp_sin, t_train_span, x0_train,
                    t_eval=t_train, **integrator_keywords).y.T
# x_dot_train_measured = np.array(
#     [spring_damp_sin(0, x_train[i]) for i in range(t_train.size)]
#     )
#%%
# adding noise
noise = np.random.randn(x_train.shape[0], x_train.shape[1])
x_train = x_train + 0.5*noise

# need to work out how to limit the noise (set bounds like you can for randint)
#%% library options
# Define custom functions up to quadratic terms
library_functions = [
    lambda x: x, 
    lambda x: np.sin(t_train),
    #lambda y: y,
    #lambda x: x * x
]                
library_function_names = [
    lambda x: x,
    lambda x: "sin(t)", 
    #lambda y: y,
    #lambda x: x + x
]

# Define PDELibrary which generates up to first order time derivatives
sindy_library = ps.PDELibrary(
    library_functions=library_functions,
    temporal_grid=t_train,
    function_names=library_function_names,
    include_bias=True,
    implicit_terms=True,
    derivative_order=1,
    include_interaction=False
)

# library_functions = [
#     lambda x : 1,
#     lambda x : x,
#     lambda x : np.sin(x),
#     # lambda x,y : np.sin(x + y),
#     # lambda x : x*x
# ]
# library_function_names = [
#     lambda x : 1,
#     lambda x : x,
#     lambda x : 'sin(' + x + ')',
#     # lambda x,y : 'sin(' + x + ',' + y + ')',
#     # lambda x : x + '^2'
# ]

# custom_library = ps.CustomLibrary(library_functions=library_functions,
#                                   function_names=library_function_names).fit(x_train)
# custom_library.transform(x_train)

# print("Custom Library: ")
# print(custom_library.get_feature_names(), "\n")

# # use fourier library
# sindy_library2 = ps.FourierLibrary(n_frequencies=3)
# # use combined library
# sindy_library = sindy_library1 + sindy_library2

# print library
lib = sindy_library.fit(x_train)
lib.transform(x_train)
print("Library: ")
print(lib.get_feature_names(), "\n")


#%% fit model

sindy_opt = ps.SINDyPI(
    threshold=0.001,
    tol=1e-5,
    thresholder="l1",
    max_iter=6000,
    #normalize_columns=True
)

model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
)
model.fit(x_train,
          t=t_train,
)
          #x_dot=x_dot_train_measured + np.random.normal(0.01, size=x_train.shape))

model.print(precision=4)

x_dot_true = model.differentiate(x_train, t=t_train)
x_dot_pred = model.predict(x_train)  # default returns d/dt of all features

#%% plotting section 

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(t_train, x_dot_true[:, 0], 'k') 
ax.plot(t_train, x_dot_pred[:, 5], '--r')
plt.xlabel("time")
plt.ylabel("x0")
plt.title("Velocity")

ax = fig.add_subplot(122)
ax.plot(t_train, x_dot_true[:, 1], 'k')
ax.plot(t_train, x_dot_pred[:, 6], '--r')
plt.xlabel("time")
plt.ylabel("x1")
plt.legend(['true', 'model'])
plt.title("Acceleration")

# the positions 5, 6 in the library are 'x0_t', 'x1_t'
# so you would need to know what lhs you want (in theory there would only be a
# couple sparse equations so it would be obvious)