#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.29 22:55
# @Author  : Tony
# @Site    : 
# @File    : path_IDs.py
# @Software: PyCharm

import os
import numpy as np

class PathIds(object):
    def __init__(self, usage, dataset_name, train_mode, base_path = None, dataset_path = None,
                validation_split = 0.3, shuffle = True):

        self.usage = usage
        self.dataset_name = dataset_name
        self.train_mode = train_mode
        self.base_path = base_path
        self.dataset_path = dataset_path
        self.validation_split = validation_split
        self.shuffle = shuffle

        if self.train_mode == 'data_2':
            self.base_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data2\\'
        elif self.train_mode == 'data_4':
            self.base_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data4\\'
        else:
            raise Exception('Incorrect train mode, please input right mode')

        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'raw_data':
            self.dataset_path = self.base_path + 'raw_data\\'
        elif self.dataset_name == 'Alpha':
            self.dataset_path = self.base_path + 'Alpha\\'
        elif self.dataset_name == 'Beta':
            self.dataset_path = self.base_path + 'Beta\\'
        elif self.dataset_name == 'Gamma':
            self.dataset_path = self.base_path + 'Gamma\\'
        elif self.dataset_name == 'Theta':
            self.dataset_path = self.base_path + 'Theta\\'
        else:
            raise Exception('Incorrect dataset name, please input right directory')

    def generator_path_IDs(self):
        all_path_IDs = []

        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.npz')):
                    all_path_IDs.append(os.path.join(folder, filename))

        shuffled_IDs = self.shuffle_data(all_path_IDs, True)
        train_path_IDs, test_path_IDs = self.split_data(shuffled_IDs, self.validation_split)

        if self.usage == 'Train':
            return train_path_IDs

        elif self.usage == 'Validation':
            return test_path_IDs

        else:
            raise Exception('Incorrect usage, pleaswe input right usage!')

    def split_data(self, all_path_IDs, validation_split=0.3):

        num_samples = len(all_path_IDs)

        num_train_samples = int((1 - validation_split) * num_samples)
        train_path_IDs = all_path_IDs[:num_train_samples]
        test_path_IDs = all_path_IDs[num_train_samples:]

        return train_path_IDs, test_path_IDs

    def shuffle_data(self, path_IDs, do_shuffle=True):

        num_paths = len(path_IDs)

        if do_shuffle == True:
            shuffle_indices = np.random.permutation(np.arange(num_paths))
            shuffled_data = np.asarray(path_IDs)[shuffle_indices]
            shuffled_data = shuffled_data.tolist()

        else:
            shuffled_data = path_IDs

        return shuffled_data

