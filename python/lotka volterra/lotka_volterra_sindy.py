# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:14:02 2022

@author: zaraw

===============================================
applying sindy to the lotka-volterra system  
==============================================
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
# from lotkavolterra import lotkavolterra

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

def lotka_volterra(t, x, a = 0.05, b = 0.005, c = 0.005, d = 0.2):
    return[
        (a * x[0] - b * x[0] * x[1]),
        (c * x[0] * x[1] - d * x[1])]
#%%
# generate training data 

dt = 0.01                                   # define time
t_train = np.arange(0, 500, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [1, 15]                           # initial conditions
x_train = solve_ivp(lotka_volterra, t_train_span,        # solve ODEs
                    x0_train, t_eval=t_train,
                    **integrator_keywords).y.T

# # add noise at this stage
# rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
# x_train = x_train + np.random.normal(0, rmse / 100.0, x_train.shape)
# print("Noisy")

x_dot_train_measured = np.array(
    [lotka_volterra(0, x_train[i]) for i in range(t_train.size)]
    )

# Fit the model
poly_order = 5
threshold = 0.00015
library = ps.PolynomialLibrary(degree=poly_order)

library = library.fit(x_train)
library.transform(x_train)
print("Library:")
print(library.get_feature_names(), "\n")

t_sim = np.arange(0, 20, dt)
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=library,
    
)
model.fit(x_train, 
          t=dt,
          # x_dot=x_dot_train_measured,
          quiet=True,)
model.print()

# Simulate and plot the results

x_sim = model.simulate(x_train[0], t_sim)

plot_kws = dict(linewidth=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_sim, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_sim, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
fig.show()
