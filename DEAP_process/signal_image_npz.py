#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.27 22:33
# @Author  : Tony
# @Site    : 
# @File    : signal_image_npz.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.27 10:34
# @Author  : Tony
# @Site    :
# @File    : dataset.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from random import shuffle
import os
import csv

csv.field_size_limit(500 * 1024 * 1024)

WIDTH = 9
HEIGHT = 9
END_POINT = 245760
ROWS_NUM = 32
COLS_NUM = 7680
STEP_SIZE = 256
WINDOW_SIZE = 512
CHANNEL_ORDER = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
               'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6',
               'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

CHANNEL_POSITION = {'Fp1':(0,3), 'Fp2':(0,5), 'AF3':(1,3), 'AF4':(1,5), 'F7':(2,0), 'F3':(2,2),
                  'Fz':(2,4), 'F4':(2,6), 'F8':(2,8), 'FC5':(3, 1), 'FC1':(3,3), 'FC2':(3,5),
                  'FC6':(3,7), 'T7':(4,0), 'C3':(4,2), 'Cz':(4,4), 'C4':(4,6), 'T8':(4,8),
                  'CP5':(5,1), 'CP1':(5,3), 'CP2':(5,5), 'CP6':(5,7), 'P7':(6,0), 'P3':(6,2),
                  'Pz':(6,4), 'P4':(6,6), 'P8':(6,8), 'PO3':(7,3), 'PO4':(7,5), 'O1':(8,3),
                  'Oz':(8,4), 'O2':(8,5)}


class npzData(object):
    def __init__(self, dataset_name, dataset_path = None, dst_path = None):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dst_path = dst_path

        if self.dataset_path != None:
            self.dataset_path = dataset_path
            self.dst_path = dst_path
        elif self.dataset_name == 'raw_data':
            self.dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\raw_data\\'
            self.dst_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data4\\raw_data\\'
        elif self.dataset_name == 'Alpha':
            self.dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\Alpha\\'
            self.dst_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data4\\Alpha\\'
        elif self.dataset_name == 'Beta':
            self.dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\Beta\\'
            self.dst_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data4\\Beta\\'
        elif self.dataset_name == 'Gamma':
            self.dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\Gamma\\'
            self.dst_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data4\\Gamma\\'
        elif self.dataset_name == 'Theta':
            self.dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\Theta\\'
            self.dst_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\all_npz_data4\\Theta\\'
        else:
            raise Exception('Incorrect dataset name, please input right directory')


    def write_data(self):
        if self.dataset_name == 'raw_data':
            self._save_rawData()
        elif self.dataset_name == 'Alpha':
            self._save_Alpha()
        elif self.dataset_name == 'Beta':
            self._save_Beta()
        elif self.dataset_name == 'Theta':
            self._save_Theta()
        elif self.dataset_name == 'Gamma':
           self._save_Gamma()


    def _save_rawData(self):

        file_paths = []
        for folder,subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.csv')):
                    file_paths.append(os.path.join(folder, filename))

        num_len = len(file_paths)


        for file_arg, file_path in enumerate(file_paths):
            all_singal_images = []
            all_valence_labels = []
            all_arousal_labels = []
            print("Precessing {} / {}".format(file_arg+1, num_len))
            with open(file_path, 'r') as csvin:
                data = csv.reader(csvin)
                for line_arg, row in enumerate(data):
                    if line_arg != 0:
                        valence_label = int(row[2])
                        arousal_label = int(row[3])
                        signal_list = [np.float32(signal) for signal in row[1].strip().split(' ')]

                        signal_images = signal_to_image(signal_list)
                        all_singal_images.append(signal_images)
                        all_valence_labels.append(valence_label)
                        all_arousal_labels.append(arousal_label)

                    else:
                        continue
                print('Finish Loading {} / {}!'.format(file_arg+1, num_len))

            base_path = self.dst_path + str(file_arg + 1) + '-'

            save_single_image(all_singal_images, all_valence_labels, all_arousal_labels, base_path)

            print('Finish Writing {} / {}!'.format(file_arg+1, num_len))

    def _save_Alpha(self):

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.csv')):
                    file_paths.append(os.path.join(folder, filename))

        num_len = len(file_paths)

        for file_arg, file_path in enumerate(file_paths):
            all_singal_images = []
            all_valence_labels = []
            all_arousal_labels = []
            print("Precessing {} / {}".format(file_arg + 1, num_len))
            with open(file_path, 'r') as csvin:
                data = csv.reader(csvin)
                for line_arg, row in enumerate(data):
                    if line_arg != 0:
                        valence_label = int(row[2])
                        arousal_label = int(row[3])
                        signal_list = [np.float32(signal) for signal in row[1].strip().split(' ')]

                        signal_images = signal_to_image(signal_list)
                        all_singal_images.append(signal_images)
                        all_valence_labels.append(valence_label)
                        all_arousal_labels.append(arousal_label)

                    else:
                        continue
                print('Finish Loading {} / {}!'.format(file_arg + 1, num_len))

            base_path = self.dst_path + str(file_arg + 1) + '-'

            save_single_image(all_singal_images, all_valence_labels, all_arousal_labels, base_path)

            print('Finish Writing {} / {}!'.format(file_arg + 1, num_len))


    def _save_Beta(self):

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.csv')):
                    file_paths.append(os.path.join(folder, filename))

        num_len = len(file_paths)

        for file_arg, file_path in enumerate(file_paths):
            all_singal_images = []
            all_valence_labels = []
            all_arousal_labels = []
            print("Precessing {} / {}".format(file_arg + 1, num_len))
            with open(file_path, 'r') as csvin:
                data = csv.reader(csvin)
                for line_arg, row in enumerate(data):
                    if line_arg != 0:
                        valence_label = int(row[2])
                        arousal_label = int(row[3])
                        signal_list = [np.float32(signal) for signal in row[1].strip().split(' ')]

                        signal_images = signal_to_image(signal_list)
                        all_singal_images.append(signal_images)
                        all_valence_labels.append(valence_label)
                        all_arousal_labels.append(arousal_label)

                    else:
                        continue
                print('Finish Loading {} / {}!'.format(file_arg + 1, num_len))

            base_path = self.dst_path + str(file_arg + 1) + '-'

            save_single_image(all_singal_images, all_valence_labels, all_arousal_labels, base_path)

            print('Finish Writing {} / {}!'.format(file_arg + 1, num_len))


    def _save_Gamma(self):

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.csv')):
                    file_paths.append(os.path.join(folder, filename))

        num_len = len(file_paths)

        for file_arg, file_path in enumerate(file_paths):
            all_singal_images = []
            all_valence_labels = []
            all_arousal_labels = []
            print("Precessing {} / {}".format(file_arg + 1, num_len))
            with open(file_path, 'r') as csvin:
                data = csv.reader(csvin)
                for line_arg, row in enumerate(data):
                    if line_arg != 0:
                        valence_label = int(row[2])
                        arousal_label = int(row[3])
                        signal_list = [np.float32(signal) for signal in row[1].strip().split(' ')]

                        signal_images = signal_to_image(signal_list)
                        all_singal_images.append(signal_images)
                        all_valence_labels.append(valence_label)
                        all_arousal_labels.append(arousal_label)

                    else:
                        continue
                print('Finish Loading {} / {}!'.format(file_arg + 1, num_len))

            base_path = self.dst_path + str(file_arg + 1) + '-'

            save_single_image(all_singal_images, all_valence_labels, all_arousal_labels, base_path)

            print('Finish Writing {} / {}!'.format(file_arg + 1, num_len))


    def _save_Theta(self):

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.csv')):
                    file_paths.append(os.path.join(folder, filename))

        num_len = len(file_paths)

        for file_arg, file_path in enumerate(file_paths):
            all_singal_images = []
            all_valence_labels = []
            all_arousal_labels = []
            print("Precessing {} / {}".format(file_arg + 1, num_len))
            with open(file_path, 'r') as csvin:
                data = csv.reader(csvin)
                for line_arg, row in enumerate(data):
                    if line_arg != 0:
                        valence_label = int(row[2])
                        arousal_label = int(row[3])
                        signal_list = [np.float32(signal) for signal in row[1].strip().split(' ')]

                        signal_images = signal_to_image(signal_list)
                        all_singal_images.append(signal_images)
                        all_valence_labels.append(valence_label)
                        all_arousal_labels.append(arousal_label)

                    else:
                        continue
                print('Finish Loading {} / {}!'.format(file_arg + 1, num_len))

            base_path = self.dst_path + str(file_arg + 1) + '-'

            save_single_image(all_singal_images, all_valence_labels, all_arousal_labels, base_path)

            print('Finish Writing {} / {}!'.format(file_arg + 1, num_len))



def save_single_image(all_signal_images, all_valence_labels, all_arousal_labels, base_path):

    signal_len = len(all_signal_images)
    start = 0
    end = COLS_NUM - WINDOW_SIZE + 1
    for i in range(signal_len):
        print("Processing {} / {} signals_images".format(i+1, signal_len))
        to_split_image = np.asarray(all_signal_images)[i]
        valence_label = all_valence_labels[i]
        arousal_label = all_arousal_labels[i]

        for j in range(start, end, STEP_SIZE):
            one_image = to_split_image[start:start+WINDOW_SIZE]
            npz_name = base_path + str(i+1) + '-' + str(j+1) + '.npz'
            np.savez(npz_name, signal_image = one_image, valence_label = valence_label, arousal_label = arousal_label)



def signal_to_image(signal_list):
    signal_needed = signal_list[0:END_POINT]
    signal_needed = np.asarray(signal_needed).reshape(ROWS_NUM, COLS_NUM)

    start = 0
    end = COLS_NUM
    signal_images = []

    for i in range(start, end):
        signal_image = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
        one_sample = signal_needed[:, i]
        one_sample = preprocess_signal(one_sample).tolist()

        for j in range(ROWS_NUM):
            channel_name = CHANNEL_ORDER[j]
            channel_position = CHANNEL_POSITION[channel_name]
            row = channel_position[0]
            col = channel_position[1]
            signal_image[row][col] = one_sample[j]

        signal_images.append(signal_image)

    return signal_images


def preprocess_signal(one_signal):
    min_value = np.min(one_signal)
    max_value = np.max(one_signal)
    one_signal = ((one_signal - min_value) / (max_value - min_value)) - 0.5

    return one_signal


dataset_name = ['raw_data', 'Alpha', 'Beta', 'Gamma', 'Theta']
# dataset_name = ['raw_data']
for i in dataset_name:
    data_process = npzData(i)
    data_process.write_data()