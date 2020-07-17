#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.29 22:04
# @Author  : Tony
# @Site    : 
# @File    : data_generator.py
# @Software: PyCharm

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, train_type, train_mode, path_IDs, batch_size=32,
                n_classes=2, shuffle=True, ):

        self.train_type = train_type
        self.train_mode = train_mode
        self.path_IDs = path_IDs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        if self.train_mode == 'data_2':
            self.redim = (64, 4, 9, 9)
            self.dim = (64, 9, 9, 4)

        elif self.train_mode == 'data_4':
            self.redim = (128, 4, 9, 9)
            self.dim= (128, 9, 9, 4)

    def __len__(self):

        return int(np.floor(len(self.path_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        path_IDs_temp = [self.path_IDs[k] for k in indexes]

        X, y = self.__data_generation(path_IDs_temp)

        return X, y


    def on_epoch_end(self):

        self.indexes = np.arange(len(self.path_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, path_IDs_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, path_ID in enumerate(path_IDs_temp):
            # Store sample
            signal_data = np.load(path_ID)
            signal_image = signal_data['signal_image']
            valence_label = int(signal_data['valence_label'])
            arousal_label = int(signal_data['arousal_label'])

            if self.train_mode == 'data_2':
                signal_image = signal_image.reshape(self.redim)

            elif self.train_mode == 'data_4':
                signal_image = signal_image.reshape(self.redim)

            else:
                raise Exception('Incorrect train mode, please input right train mode!')

            signal_image = signal_image.swapaxes(1, 3)

            if self.train_type == 'valence':
                X[i,] = signal_image
                y[i] = valence_label

            elif self.train_type == 'arousal':
                X[i,] = signal_image
                y[i] = arousal_label

            else:
                raise Exception('Incorrect train type, please input right train type!')

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
