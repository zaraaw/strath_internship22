# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:08:37 2022

@author: zaraw

============
to use these you would do 'model = ps.SINDy(feature_library= ... )'
============
"""

import numpy as np
import pysindy as ps

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

t = np.linspace(0, 1, 100)
x = np.ones((100, 2))

#%% polynomial library

poly_library = ps.PolynomialLibrary(include_interaction=False).fit(x)
poly_library.transform(x)

print("Polynomial Library")
print(poly_library.get_feature_names(), "\n")


#%%fourier library - trig terms
#fourier_library = ps.FourierLibrary(n_frequencies=3).fit(x)
fourier_library = ps.FourierLibrary().fit(x)
fourier_library.transform(x)

print("Fourier Library")
print(fourier_library.get_feature_names(), "\n")


#%% Identity Library
identity_library = ps.IdentityLibrary().fit(x)
identity_library.transform(x)

print("Identity Library")
print(identity_library.get_feature_names(), "\n")

#%% concatenate 2 libraries

identity_library = ps.IdentityLibrary()
fourier_library = ps.FourierLibrary()
combined_library = identity_library + fourier_library

combined_library.fit(x)
fourier_library.transform(x)

print("Contatenated Library (identity and fourier)")
print(combined_library.get_feature_names(), "\n")

#%% tensor 2 libraries
identity_library = ps.PolynomialLibrary(include_bias=False)
fourier_library = ps.FourierLibrary()
combined_library = identity_library * fourier_library

combined_library.fit(x)
fourier_library.transform(x)

print("Tensored Library (identity and fourier)")
print(combined_library.get_feature_names(), "\n")

#%% SINDy-PI
print("SINDy-PI library")
# Functions to be applied to the data x
functions = [
    lambda x: np.exp(x),
    lambda x, y: np.sin(x + y)
    ]

# Functions to be applied to the data x_dot
x_dot_functions = [lambda x: x]

lib = ps.SINDyPILibrary(
    library_functions=functions,
    x_dot_library_functions=x_dot_functions,
    t=t,
).fit(x)
lib.transform(x)
print("Without function names: ")
print(lib.get_feature_names(), "\n")

# Function names includes both the x and x_dot functions
function_names = [
    lambda x: "exp(" + x + ")",
    lambda x, y: "sin(" + x + y + ")",
    lambda x: x,
]
lib = ps.SINDyPILibrary(
    library_functions=functions,
    x_dot_library_functions=x_dot_functions,
    function_names=function_names,
    t=t,
).fit(x)
lib.transform(x)
print("With function names: ")
print(lib.get_feature_names(), "\n")


#%% PDE Library not working 
# SINDyPILibrary is now deprecated, 
# use the PDE or WeakPDE library instead.
print("PDE Library")
functions = [
    lambda x : np.exp(x),
    lambda x, y : np.sin(x*y),
    lambda x : np.sin(t)
    ]
function_names = [
    lambda x: "exp(" + x + ")",
    lambda x, y: "sin(" + x + y + ")",
    lambda x : "sin(t)"
    ]
lib = ps.PDELibrary(
    library_functions=functions,
    function_names=function_names,
    temporal_grid=t,
    derivative_order=1,
    implicit_terms=True
    )
lib = lib.fit(x)
lib.transform(x)
print("With function names: ")
print(lib.get_feature_names(), "\n")

print("WeakPDE Library")
# Repeat with the weak form library
lib = ps.WeakPDELibrary(
    library_functions=functions,
    function_names=function_names,
    spatiotemporal_grid=t,
    K=2, 
    derivative_order=1,
    implicit_terms=True
)
lib = lib.fit(x)
lib.transform(x)
print("With function names: ")
print(lib.get_feature_names(), "\n")


#%% Custom Library
print("Custom Library")

library_functions = [
    lambda x : np.exp(x),
    lambda x : 1. / x,
    lambda x : x,
    lambda x,y : np.sin(x + y)
]
library_function_names = [
    lambda x : 'exp(' + x + ')',
    lambda x : '1/' + x,
    lambda x : x,
    lambda x,y : 'sin(' + x + ',' + y + ')'
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
).fit(x)
custom_library.transform(x)

print("With function names: ")
print(custom_library.get_feature_names(), "\n")

# #%% fully custom library -- doesn't work
# print("Generalised Library")
# # Initialize two libraries
# poly_library = ps.PolynomialLibrary(include_bias=False)
# fourier_library = ps.FourierLibrary()

# # Initialize the default inputs, i.e. each library
# # uses all the input variables
# inputs_temp = np.tile([0, 1, 2], 2)
# inputs_per_library = np.reshape(inputs_temp, (2, 3))

# # Don't use the x0 input for generating the Fourier library
# inputs_per_library[1, 0] = 1

# # Tensor all the polynomial and Fourier library terms together
# tensor_array = [[1, 1]]

# # Initialize this generalized library, all the work hidden from the user!
# generalized_library = ps.GeneralizedLibrary([poly_library, fourier_library], 
#                                             tensor_array=tensor_array,
#                                             inputs_per_library=inputs_per_library)

# generalized_library.fit(x)
# generalized_library.transform(x)

# print("Generalised Library")
# print(generalized_library.get_feature_names(), "\n")