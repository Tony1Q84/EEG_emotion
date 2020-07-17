#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.26 23:46
# @Author  : Tony
# @Site    : 
# @File    : Convlstm.py
# @Software: PyCharm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import ConvLSTM2D, Convolution2D
from keras.layers import MaxPooling3D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Activation


# keras EEG models