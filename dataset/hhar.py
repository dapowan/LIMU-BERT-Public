 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 15:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : hhar.py
# @Description : http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition

import os
import numpy as np
import pandas as pd


DATASET_PATH = r'D:\Dataset_Mobility\HHAR'


def extract_sensor(data, time_index, time_tag, window_time):
    index = time_index
    while index < len(data) and abs(data.iloc[index]['Creation_Time'] - time_tag) < window_time:
        index += 1
    if index == time_index:
        return None, index
    else:
        data_slice = data.iloc[time_index:index]
        if data_slice['User'].unique().size > 1 or data_slice['gt'].unique().size > 1:
            return None, index
        else:
            data_sensor = data_slice[['x', 'y', 'z']].to_numpy()
            sensor = np.mean(data_sensor, axis=0)
            label = data_slice[['User', 'Model', 'gt']].iloc[0].values
            return np.concatenate([sensor, label]), index


def transform_to_index(label, print_label=False):
    labels_unique = np.unique(label)
    if print_label:
        print(labels_unique)
    for i in range(labels_unique.size):
        label[label == labels_unique[i]] = i


def separate_data_label(data_raw):
    labels = data_raw[:, :, -3:].astype(np.str)
    transform_to_index(labels[:, :, 0])
    transform_to_index(labels[:, :, 1], print_label=True)
    transform_to_index(labels[:, :, 2], print_label=True)
    data = data_raw[:, :, :6].astype(np.float)
    labels = labels.astype(np.float)
    return data, labels


# 'Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt'
def preprocess_hhar(path, path_save, version, window_time=50, seq_len=40, jump=0):
    accs = pd.read_csv(path + '\Phones_accelerometer.csv')
    gyros = pd.read_csv(path + '\Phones_gyroscope.csv') #, nrows=200000
    time_tag = min(accs.iloc[0, 2], gyros.iloc[0, 2])
    time_index = [0, 0] # acc, gyro
    window_num = 0
    data = []
    data_temp = []
    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc, time_index_new_acc = extract_sensor(accs, time_index[0], time_tag, window_time=window_time * pow(10, 6))
        gyro, time_index_new_gyro = extract_sensor(gyros, time_index[1], time_tag, window_time=window_time * pow(10, 6))
        time_index = [time_index_new_acc, time_index_new_gyro]
        if acc is not None and gyro is not None and np.all(acc[-3:] == gyro[-3:]):
            time_tag += window_time * pow(10, 6)
            window_num += 1
            data_temp.append(np.concatenate([acc[:-3], gyro[:-3], acc[-3:]]))
            if window_num == seq_len:
                data.append(np.array(data_temp))
                if jump == 0:
                    data_temp.clear()
                    window_num = 0
                else:
                    data_temp = data_temp[-jump:]
                    window_num -= jump
        else:
            if window_num > 0:
                data_temp.clear()
                window_num = 0
            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(accs.iloc[time_index[0], 2], gyros.iloc[time_index[1], 2])
            else:
                break
    data_raw = np.array(data)
    data_new, label_new = separate_data_label(data_raw)
    np.save(os.path.join(path_save, 'data_' + version + '.npy'), np.array(data_new))
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), np.array(label_new))
    return data_new, label_new

# label: ('User', 'Model', 'gt')
# ['nexus4' 's3' 's3mini']
# ['bike' 'sit' 'stairsdown' 'stairsup' 'stand' 'walk']
# acc + gyro
path_save = 'hhar'
version = '20_120'
data, label = preprocess_hhar(DATASET_PATH, path_save, version, window_time=50, seq_len=120)

