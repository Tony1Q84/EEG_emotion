#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.18 21:53
# @Author  : Tony
# @Site    : 
# @File    : diffBand_EEG_csv.py
# @Software: PyCharm

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
import pywt

VIDEO_NUM = 40
CHANNEL = 32
START_POINT = 384
BATH_PATH = "E:\\Programming\\python program\\EEG_emotion\\DEAP\\raw_deap\\"
DEP_VALUE = 4.50

dataset_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\data_preprocessed_python\\'

def data_to_csv(path, band):

    file_paths = []
    for folder, subfolders, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(('.dat')):
                file_paths.append(os.path.join(folder, filename))

    total_num = len(file_paths)
    for file_arg, file_path in enumerate(file_paths):
        print("Processing: {} signals".format(band))
        print("Processing: {}/{}".format(file_arg, total_num))
        Header = ['orders', 'signals', 'valence_labels', 'arousal_labels', 'dominance_labels']
        rows = []
        (base_path, tempfilename) = os.path.split(file_path)
        (filename, extension) = os.path.splitext(tempfilename)
        csv_path = BATH_PATH + band + "\\" +  filename + '.csv'
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

                if band == 'Gamma':
                    Gamma_signal = Gamma_decompos_signal(temp_signal)
                    # print("Gamma_signal: {}".format(Gamma_signal))
                    # min_value = np.min(Gamma_signal)
                    # max_value = np.max(Gamma_signal)
                    # Gamma_signal = ((Gamma_signal - min_value) / (max_value - min_value)).tolist()
                    # print(np.min(Gamma_signal))
                    # print(np.max(Gamma_signal))
                    Gamma_signal = Gamma_signal.tolist()
                    final_signal.append(Gamma_signal)

                elif band == 'Beta':
                    Beta_signal = Beta_decompos_signal(temp_signal)
                    # min_value = np.min(Beta_signal)
                    # max_value = np.max(Beta_signal)
                    # Beta_signal = ((Beta_signal - min_value) / (max_value - min_value)).tolist()
                    # print(np.min(Beta_signal))
                    # print(np.max(Beta_signal))
                    Beta_signal = Beta_signal.tolist()
                    final_signal.append(Beta_signal)

                elif band == 'Alpha':
                    Alpha_signal = Alpha_decompos_signal(temp_signal)
                    # min_value = np.min(Alpha_signal)
                    # max_value = np.max(Alpha_signal)
                    # Alpha_signal = ((Alpha_signal - min_value) / (max_value - min_value)).tolist()
                    # print(np.min(Alpha_signal))
                    # print(np.max(Alpha_signal))
                    Alpha_signal = Alpha_signal.tolist()
                    final_signal.append(Alpha_signal)

                elif band == 'Theta':
                    Theta_signal = Theta_decompos_signal(temp_signal)
                    # min_value = np.min(Theta_signal)
                    # max_value = np.max(Theta_signal)
                    # Theta_signal = ((Theta_signal - min_value) / (max_value - min_value)).tolist()
                    # print(np.min(Theta_signal))
                    # print(np.max(Theta_signal))
                    Theta_signal = Theta_signal.tolist()
                    final_signal.append(Theta_signal)


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

def Gamma_decompos_signal(signal):
    coefficients_level2 = pywt.wavedec(signal, 'db4', 'smooth', level=2)
    [cA2_l2, cD2_l2, cD1_l2] = coefficients_level2
    Gamma_signal_only = [None, cD2_l2, None]
    Gamma_signal = pywt.waverec(Gamma_signal_only, 'db4', 'smooth')

    return Gamma_signal

def Beta_decompos_signal(signal):
    coefficients_level3 = pywt.wavedec(signal, 'db4', 'smooth', level=3)
    [cA3_l3, cD3_l3, cD2_l3, cD1_l3] = coefficients_level3
    Beta_signal_only = [None, cD3_l3, None, None]
    Beta_signal = pywt.waverec(Beta_signal_only, 'db4', 'smooth')

    return Beta_signal

def Alpha_decompos_signal(signal):
    coefficients_level4 = pywt.wavedec(signal, 'db4', 'smooth', level=4)
    [cA4_l4, cD4_l4, cD3_l4, cD2_l4, cD1_l4] = coefficients_level4
    Alpha_signal_only = [None, cD4_l4, None, None, None]
    Alpha_signal = pywt.waverec(Alpha_signal_only, 'db4', 'smooth')

    return Alpha_signal

def Theta_decompos_signal(signal):
    coefficients_level5 = pywt.wavedec(signal, 'db4', 'smooth', level=5)
    [cA5_l5, cD5_l5, cD4_l5, cD3_l5, cD2_l5, cD1_l5] = coefficients_level5
    Theta_signal_only = [None, cD5_l5, None, None, None, None]
    Theta_signal = pywt.waverec(Theta_signal_only, 'db4', 'smooth')

    return Theta_signal


Bands = ['Gamma', 'Beta', 'Alpha', 'Theta']
for band in Bands:
    data_to_csv(dataset_path, band)