# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:42:32 2022

@author: zaraw
fan end faults through fan end sensors
--from '12k Fan End Bearing Fault Data'
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

#%% 7 mils, 0 hp, ball 
file278 = "C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\MATLAB\\case western\\278.mat"
ball_7_0_12 = sio.loadmat(file278)

#%%
X278_BA_time = ball_7_0_12["X278_BA_time"]         #base plate
X278_DE_time = ball_7_0_12["X278_DE_time"]         #drive end
X278_FE_time = ball_7_0_12["X278_FE_time"]         #fan end 

plt.hist(X278_FE_time, bins=100)
plt.title('FE (Ball fault: 7 mils, 0 hp, 12k samp/sec) Fan End Operation')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.show()

t = np.arange(0, len(X278_FE_time), 1)
plt.plot(t, X278_FE_time)
plt.title('FE (Ball fault: 7 mils, 0 hp, 12k samp/sec) Fan End Operation')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()