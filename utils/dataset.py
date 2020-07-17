#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.27 10:34
# @Author  : Tony
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

import numpy as np
import os

ROWS = 9
COLS = 9
CHANNELS = 1
TIME_STEPS_DATA2 = 64
TIME_STEPS_DATA4 = 128
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.3

class DataManager(object):
    def __init__(self, train_type, dataset_name, train_mode, base_path = None, dataset_path = None):

        self.train_type = train_type
        self.dataset_name = dataset_name
        self.train_mode = train_mode
        self.base_path = base_path
        self.dataset_path = dataset_path

        if self.train_mode == 'data_2':
            self.base_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\npz_data_2\\'
        elif self.train_mode == 'data_4':
            self.base_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\npz_data_4\\'
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


    def get_data(self):
        train_data, test_data = self._load_rawData()

        return train_data, test_data

    def _load_rawData(self):

        file_paths = []
        for folder,subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.npz')):
                    file_paths.append(os.path.join(folder, filename))

        file_paths = shuffle_data(file_paths, True)
        num_signals = len(file_paths)

        signal_images = []
        valence_labels = []
        arousal_labels = []

        for file_arg, file_path in enumerate(file_paths):
            print('Processing {} / {} signals'.format(file_arg+1, num_signals))
            signal_data = np.load(file_path)
            signal_image = signal_data['signal_image']
            valence_label = int(signal_data['valence_label'])
            arousal_label = int(signal_data['arousal_label'])

            if self.train_mode == 'data_2':
                signal_image = signal_image.reshape((TIME_STEPS_DATA2, CHANNELS, ROWS, COLS))
                # signal_image.swapaxes(3,1)
            elif self.train_mode == 'data_4':
                signal_image = signal_image.reshape((TIME_STEPS_DATA4, CHANNELS, ROWS, COLS))

            else:
                raise Exception('Incorrect train mode, please input right mode!')

            signal_image = signal_image.swapaxes(1, 3)
            signal_images.append(signal_image)
            valence_labels.append(valence_label)
            arousal_labels.append(arousal_label)


        if self.train_type == 'valence':
            signal_images = np.asarray(signal_images)
            valence_labels = np.asarray(valence_labels)
            valence_labels = one_hot(valence_labels, NUM_CLASSES)

            train_data, test_data = split_data(signal_images, valence_labels, VALIDATION_SPLIT)

        elif self.train_type == 'arousal':
            signal_images = np.asarray(signal_images)
            valence_labels = np.asarray(valence_labels)
            valence_labels = one_hot(valence_labels, NUM_CLASSES)

            train_data, test_data = split_data(signal_images, valence_labels, VALIDATION_SPLIT)

        else:
            raise Exception('Incorrect train type, please input right type!')

        print('Data loading is finished!')

        return train_data, test_data


def shuffle_data(file_paths, do_shuffle = True):

    num_paths = len(file_paths)

    if do_shuffle == True:
        shuffle_indices = np.random.permutation(np.arange(num_paths))
        shuffled_data = np.asarray(file_paths)[shuffle_indices]
        shuffled_data = shuffled_data.tolist()

    else:
        shuffled_data = file_paths

    return shuffled_data


def split_data(signals, labels, validation_split = 0.2):
    num_samples = len(signals)

    num_train_samples = int((1 - validation_split)*num_samples)
    train_signals = signals[:num_train_samples]
    train_labels = labels[:num_train_samples]

    test_signals = signals[num_train_samples:]
    test_labels = labels[num_train_samples:]

    train_data = (train_signals, train_labels)
    test_data = (test_signals, test_labels)

    return train_data, test_data

def one_hot(arr, num_classes=None):
    arr = np.array(arr, dtype='int')
    input_shape = arr.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    arr = arr.ravel()
    if not num_classes:
        num_classes = np.max(arr) + 1
    n = arr.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), arr] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical