#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.2 16:57
# @Author  : Tony
# @Site    : 
# @File    : DEAP_visualize.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import pywt

participant_data = pd.read_csv(r"E:\\Programming\\python program\\EEG_emotion\\DEAP\\participant_ratings.csv")
print(participant_data)

a = participant_data[participant_data['Participant_id']==1]
# b = participant_data[participant_data['Trial'] == 1]
# c = pd.merge(a, b)
# print(type(np.int(c.Experiment_id)-1))
print(a)

raw_data = pk.load(open(r"E:\\Programming\\python program\\EEG_emotion\\DEAP\\data_preprocessed_python\\s01.dat", 'rb'), encoding='latin1')
print(type(raw_data))
print(raw_data['data'][0])
print(raw_data['data'][0].shape)
print(raw_data['labels'])
print(raw_data['labels'].shape)

channel_data = raw_data['data'][0]
print(channel_data.shape)
len = channel_data.shape[0]
print(len)
for i in range(0, len):
    signal = channel_data[i]
    print(signal.shape)
    print("channel {}".format(i+1))
    fig = plt.figure(figsize=(12, 6))
    plt.plot(signal)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# test

test_signal = channel_data[0][384:]
print(test_signal.shape)
[cA1_l1, cD1_11] = pywt.wavedec(test_signal, 'db4', 'smooth', level = 1)
print(cD1_11.shape)
fig = plt.figure(figsize=(12, 6))
plt.plot(cD1_11)
plt.xlabel('Hz')
plt.ylabel('amplitude')
plt.show()

level1_only = [cD1_11, None]
level1 = pywt.waverec(level1_only, 'db4', 'smooth')
print(level1.shape)
fig = plt.figure(figsize=(12, 6))
plt.plot(level1)
plt.xlabel('Time')
plt.ylabel('amplitude')
plt.show()