# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:18:17 2022

@author: zaraw

from '12k Drive End Bearing Fault Data'
"""
#%% 7 mils, 0 hp, inner race
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio


file105 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\MATLAB\\case western\\105.mat"
inner_7_0_12 = sio.loadmat(file105)

X105_BA_time = inner_7_0_12["X105_BA_time"]         #base plate
X105_DE_time = inner_7_0_12["X105_DE_time"]         #drive end
X105_FE_time = inner_7_0_12["X105_FE_time"]         #fan end 

plt.hist(X105_DE_time, bins=100)
plt.title('DE (Inner race fault: 7 mils, 0 hp, 12k samp/sec) Drive End Operation')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.show()

t = np.arange(0, len(X105_DE_time), 1)
plt.plot(t, X105_DE_time)
plt.title('DE (Inner race fault: 7 mils, 0 hp, 12k samp/sec) Drive End Operation')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
