# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:58:25 2022

@author: zaraw

importing case western datasets from matlab
"""
import numpy as np
from matplotlib import pyplot as plt
# using scipy.io to import .mat file
## note use of 2 \\ in file location
import scipy.io as sio
file97 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\97.mat"
mat = sio.loadmat(file97) 

X097_DE_time = mat["X097_DE_time"]
X097_FE_time = mat["X097_FE_time"]

#print(mat["X097RPM"])

#print drive end normal baseline 0 hp data
t = np.arange(0, len(X097_DE_time), 1)
plt.plot(t, X097_DE_time)
plt.title('Normal Drive End (0 hp)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plt.hist(X097_DE_time, bins=50)
plt.title('Normal Drive End (0 hp)')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.show()

#%% print fan end normal baseline 0 hp data
t = np.arange(0, len(X097_FE_time), 1)
plt.plot(t, X097_FE_time)
plt.title('Normal Fan End (0 hp)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plt.hist(X097_DE_time, bins=50)
plt.title('Normal Fan End (0 hp)')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.show()
