# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:19:35 2022

@author: zaraw

taking the normal baseline 0hp Case Western Data and applying original SINDy
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

#%%
import scipy.io as sio
file97 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\97.mat"
mat_0hp = sio.loadmat(file97) 

# # for using normal operation data only 
# full_data_set = mat_0hp["X097_DE_time"]
# x_train, x_test = train_test_split(full_data_set, test_size=0.2)

# for appending fault data 
full_data_norm = mat_0hp["X097_DE_time"]
x_train_norm, x_test_norm = np.split(full_data_norm, 2)

file105 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\105.mat"
mat_7mil_0hp_inner = sio.loadmat(file105) 

full_data_fault = mat_7mil_0hp_inner["X105_DE_time"]
x_train_fault, x_test_fault = np.split(full_data_norm, 2)

x_train = np.concatenate((x_train_norm, x_train_fault))
x_test = np.concatenate((x_test_norm, x_test_fault))


dt = 1
t_train = np.arange(0, 100, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [0, 0]

# x_dot_train_measured = np.array(
#     [spring_damp_sin(0, x_train[i]) for i in range(t_train.size)]
#     )

#%% define library
poly_order = 5
library = ps.PolynomialLibrary(degree=poly_order)
threshold = 0.0005

lib = library.fit(x_train)
lib.transform(x_train)
print("Library:")
print(lib.get_feature_names(), "\n")
#%% fit model

t_sim = np.arange(0, len(x_train), dt)
# t_sim = np.arange(0, 5000, dt)
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
#%% plotting

plt.plot(t_sim, x_train[: t_sim.size, 0], "r")
plt.plot(t_sim, x_sim[:, 0], "k--")
plt.legend(['true', 'model'])
plt.xlabel("Time")
plt.ylabel("Amplitude")
# plt.title("Normal Baseline 0hp")
plt.title("Normal Baseline 0hp + 7mil 0hp inner race DE fault")


