#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.17 23:59
# @Author  : Tony
# @Site    : 
# @File    : srcdata_to_csv.py
# @Software: PyCharm

import os
import pickle as pk
import numpy as np
import csv

VIDEO_NUM = 40
CHANNEL = 32
START_POINT = 384
BATH_PATH = "E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\raw_data\\"
DEP_VALUE = 4.50

dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\data_preprocessed_python\\'

def data_to_csv(path):

    file_paths = []
    for folder, subfolders, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(('.dat')):
                file_paths.append(os.path.join(folder, filename))

    total_num = len(file_paths)
    for file_arg, file_path in enumerate(file_paths):
        print("Processing: {}/{}".format(file_arg, total_num))
        Header = ['orders', 'signals', 'valence_labels', 'arousal_labels', 'dominance_labels']
        rows = []
        (base_path, tempfilename) = os.path.split(file_path)
        (filename, extension) = os.path.splitext(tempfilename)
        csv_path = BATH_PATH + filename + '.csv'
        raw_data = pk.load(open(file_path, 'rb'), encoding='latin1')
        signals = raw_data['data']
        labels = raw_data['labels']
        for i in range(0, VIDEO_NUM):
            details = {}
            details['orders'] = i + 1

            (valence_label, arousal_label, dominance_label) = (labels[i][0], labels[i][1], labels[i][2])
            if valence_label <= DEP_VALUE:
                details['valence_labels'] = 0
            else:
                details['valence_labels'] = 1

            if arousal_label <= DEP_VALUE:
                details['arousal_labels'] = 0
            else:
                details['arousal_labels'] = 1

            if dominance_label <= DEP_VALUE:
                details['dominance_labels'] = 0
            else:
                details['dominance_labels'] = 1

            raw_signal = signals[i]
            final_signal = []
            for j in range(0, CHANNEL):
                temp_signal = raw_signal[j][START_POINT:]
                # min_value = np.min(temp_signal)
                # max_value = np.max(temp_signal)
                # temp_signal = ((temp_signal - min_value) / (max_value - min_value)).tolist()
                temp_signal = temp_signal.tolist()
                final_signal.append(temp_signal)


            final_signal = list_to_str(final_signal)
            details['signals'] = final_signal

            rows.append(details)

        with open(csv_path, 'w', newline='') as f:
            f_csv = csv.DictWriter(f, Header)
            f_csv.writeheader()
            f_csv.writerows(rows)

            f.close()

        print('Finish')

def list_to_str(src_list):
    src_array = np.array(src_list).flatten()
    dst_list = src_array.tolist()
    dst_str = " ".join(str(round(i,2)) for i in dst_list)

    return dst_str

data_to_csv(dataset_path)