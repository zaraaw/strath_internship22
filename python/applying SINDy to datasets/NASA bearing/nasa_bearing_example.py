# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:09:14 2022

@author: zaraw
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import entropy

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

def calculate_rms(df):
    result = []
    for col in df:
        r = np.sqrt((df[col]**2).sum() / len(df[col]))
        result.append(r)
    return result

# extract shannon entropy (cut signals to 500 bins)
def calculate_entropy(df):
    ent = []
    for col in df:
        ent.append(entropy(pd.cut(df[col], 500).value_counts()))
    return np.array(ent)

def load_data(path):
    series = []
    chnum = 0
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullname = os.path.join(dirname, filename)
            timestamp = datetime.datetime.strptime(filename, '%Y.%m.%d.%H.%M.%S')
            content = pd.read_csv(fullname, sep='\t', header=None)
            chnum=len(content.columns)
            
            #features extraction
            mean_abs = content.abs().mean()
            max_abs = content.abs().max()
            rms = calculate_rms(content)
            entropy = calculate_entropy(content)
            kurtosis = np.array(content.kurtosis())
            shape = rms / mean_abs
            impulse = max_abs / mean_abs
            crest = max_abs/rms
            #print(kurtosis.tolist())
            series.append([timestamp] + mean_abs.tolist() + rms + kurtosis.tolist() + shape.tolist() + impulse.tolist() + crest.tolist() + entropy.tolist())
            #break
            
    ch_prefix=['ch{}','RMS_ch{}','KUR_ch{}','SHP_ch{}','IMP_ch{}','CRS_ch{}','ENT_ch{}']
    cols =['timestamp']
    for c in ch_prefix:
        cols+=[c.format(x+1) for x in range(chnum)]

    
    df = pd.DataFrame(series, columns=cols)
    df.set_index('timestamp',inplace = True)      
    df.sort_index(inplace = True)
    return df

def get_datanofail(df, thresholdmin, thresholdmax):
    res = df[df.index<=thresholdmax]
    return res[res.index>=thresholdmin]
    
def get_datawithfail(df, threshold):
    return df[df.index>threshold]

def calctimeleft(df):
    failure_datetime = df.index.max()
    return (failure_datetime - df.index)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
df = load_data('C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\NASA bearing datasets\\1st_test\\1st_test')
# df = load_data('C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\NASA bearing datasets\\2nd_test\\2nd_test')
#df3 = load_data('C:\\Users\\zaraw\\OneDrive - University of Strathclyde\\INTERNSHIP-Zaras_Laptop\\NASA bearing datasets\\3rd_test\\4th_test\\txt')
print(df.head)

#%%
#plot test 1 data
c = ["ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

c = ["RMS_ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

c = ["KUR_ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

c = ["SHP_ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

c = ["IMP_ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

c = ["CRS_ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

c = ["ENT_ch{}".format(x+1) for x in range(8)]
df[c].plot(figsize=(14,8))

#%% plot test 2 (raw data)
c = ["ch{}".format(x+1) for x in range(4)]
df[c].plot(figsize=(14,4))

#%% EXTRACT TRAIN DATA AND TEST DATA SETS

WIN_LEN = 18
df_nofail = get_datanofail(df,'2003-10-25','2003-11-25')
df_nofail = df_nofail.rolling(WIN_LEN, min_periods=1).mean()
#df_ok = df_ok[['ch5','ch6','RMS_ch5','RMS_ch6','ENT_ch5','ENT_ch6']]
#df_ok = df_ok[['ch5','ch6','RMS_ch5','RMS_ch6']]

df_test_B1 = df_nofail[['RMS_ch1','RMS_ch2','SHP_ch1','SHP_ch2','IMP_ch1','IMP_ch2']]
df_test_B2 = df_nofail[['RMS_ch3','RMS_ch4','SHP_ch3','SHP_ch4','IMP_ch3','IMP_ch4']]
df_test_B1.rename(columns = {'RMS_ch1':'F1','RMS_ch2':'F2','SHP_ch1':'F3','SHP_ch2':'F4','IMP_ch1':'F5','IMP_ch2':'F6'}, inplace = True)
df_test_B2.rename(columns = {'RMS_ch3':'F1','RMS_ch4':'F2','SHP_ch3':'F3','SHP_ch4':'F4','IMP_ch3':'F5','IMP_ch4':'F6'}, inplace = True)


#df_test_B1 = df_nofail[['RMS_ch1','RMS_ch2','SHP_ch1','SHP_ch2','IMP_ch1','IMP_ch2','ENT_ch1','ENT_ch2','KUR_ch1','KUR_ch2','CRS_ch1','CRS_ch2']]
#df_test_B2 = df_nofail[['RMS_ch3','RMS_ch4','SHP_ch3','SHP_ch4','IMP_ch3','IMP_ch4','ENT_ch3','ENT_ch4','KUR_ch3','KUR_ch4','CRS_ch3','CRS_ch4']]
#df_test_B1.rename(columns = {'RMS_ch1':'F1','RMS_ch2':'F2','SHP_ch1':'F3','SHP_ch2':'F4','IMP_ch1':'F5','IMP_ch2':'F6','ENT_ch1':'F7','ENT_ch2':'F8','KUR_ch1':'F9','KUR_ch2':'F10','CRS_ch1':'F11','CRS_ch2':'F12'}, inplace = True)
#df_test_B2.rename(columns = {'RMS_ch3':'F1','RMS_ch4':'F2','SHP_ch3':'F3','SHP_ch4':'F4','IMP_ch3':'F5','IMP_ch4':'F6','ENT_ch3':'F7','ENT_ch4':'F8','KUR_ch3':'F9','KUR_ch4':'F10','CRS_ch3':'F11','CRS_ch4':'F12'}, inplace = True)
dfs = [df_test_B1,df_test_B2]
df_train=pd.concat(dfs)
df_fail = get_datawithfail(df,'2003-10-25')
df_fail = df_fail.rolling(WIN_LEN, min_periods=1).mean()
df_fail['timeleft'] = calctimeleft(df_fail)
df_fail['timetofail_s'] = df_fail['timeleft'].dt.total_seconds()
print(df_fail.head)


df_test_B3 = df_fail[['RMS_ch5','RMS_ch6','SHP_ch5','SHP_ch6','IMP_ch5','IMP_ch6']]
df_test_B3.rename(columns = {'RMS_ch5':'F1','RMS_ch6':'F2','SHP_ch5':'F3','SHP_ch6':'F4','IMP_ch5':'F5','IMP_ch6':'F6'}, inplace = True)
df_test_B4 = df_fail[['RMS_ch7','RMS_ch8','SHP_ch7','SHP_ch8','IMP_ch7','IMP_ch8']]
df_test_B4.rename(columns = {'RMS_ch7':'F1','RMS_ch8':'F2','SHP_ch7':'F3','SHP_ch8':'F4','IMP_ch7':'F5','IMP_ch8':'F6'}, inplace = True)

#df_test_B3 = df_fail[['RMS_ch5','RMS_ch6','SHP_ch5','SHP_ch6','IMP_ch5','IMP_ch6','ENT_ch5','ENT_ch6','KUR_ch5','KUR_ch6','CRS_ch5','CRS_ch6']]
#df_test_B3.rename(columns = {'RMS_ch5':'F1','RMS_ch6':'F2','SHP_ch5':'F3','SHP_ch6':'F4','IMP_ch5':'F5','IMP_ch6':'F6','ENT_ch5':'F7','ENT_ch6':'F8','KUR_ch5':'F9','KUR_ch6':'F10','CRS_ch5':'F11','CRS_ch6':'F12'}, inplace = True)
#df_test_B4 = df_fail[['RMS_ch7','RMS_ch8','SHP_ch7','SHP_ch8','IMP_ch7','IMP_ch8','ENT_ch7','ENT_ch8','KUR_ch7','KUR_ch8','CRS_ch7','CRS_ch8']]
#df_test_B4.rename(columns = {'RMS_ch7':'F1','RMS_ch8':'F2','SHP_ch7':'F3','SHP_ch8':'F4','IMP_ch7':'F5','IMP_ch8':'F6','ENT_ch7':'F7','ENT_ch8':'F8','KUR_ch7':'F9','KUR_ch8':'F10','CRS_ch7':'F11','CRS_ch8':'F12'}, inplace = True)
print(df_test_B3.head)
print(df_train.head)
df_train.plot(figsize=(14,8))
# df_train.plot(figsize=(14,8), kind='scatter') #error
df_test_B3.plot(figsize=(14,8))
df_test_B4.plot(figsize=(14,8))

#%% make channels individual csv files 

# ch1 is test 1, bearing 1, x axis data (doesnt fail)
ch1_df = df[['ch1']].copy()
ch1_df.to_csv('NASA_test1_ch1.csv', index=False)

ch2_df = df[['ch2']].copy()
ch2_df.to_csv('NASA_test1_ch2.csv', index=False)

ch3_df = df[['ch3']].copy()
ch3_df.to_csv('NASA_test1_ch3.csv', index=False)

ch4_df = df[['ch4']].copy()
ch4_df.to_csv('NASA_test1_ch4.csv', index=False)

# ch5 is test, 1 bearing 3, x axis data (inner race defect)
ch5_df = df[['ch5']].copy()
ch5_df.to_csv('NASA_test1_ch5.csv', index=False)

ch6_df = df[['ch6']].copy()
ch6_df.to_csv('NASA_test1_ch6.csv', index=False)

ch7_df = df[['ch7']].copy()
ch7_df.to_csv('NASA_test1_ch7.csv', index=False)

ch8_df = df[['ch8']].copy()
ch8_df.to_csv('NASA_test1_ch8.csv', index=False)

#%%
# for test 2 data 
ch1_df = df[['ch1']].copy()
ch1_df.to_csv('NASA_test2_ch1.csv', index=False)

ch2_df = df[['ch2']].copy()
ch2_df.to_csv('NASA_test2_ch2.csv', index=False)

ch3_df = df[['ch3']].copy()
ch3_df.to_csv('NASA_test2_ch3.csv', index=False)

ch4_df = df[['ch4']].copy()
ch4_df.to_csv('NASA_test2_ch4.csv', index=False)