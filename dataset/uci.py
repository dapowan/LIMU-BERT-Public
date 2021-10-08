 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 15:01
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : uci.py
# @Description : http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions

import os
import numpy as np
import pandas as pd


DATASET_PATH = r'F:\Dataset_Mobility\UCI HAR Dataset Raw\RawData'


def down_sample(data, window_sample, start, end):
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(start, end - window, window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = int(start)
        while int(start) <= i + window + 1 < int(end):
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)


def preprocess(path, path_save, version, raw_sr=50, target_sr=20, seq_len=20):
    labels = np.loadtxt(os.path.join(DATASET_PATH, 'labels.txt'), delimiter=' ')
    data = []
    label = []
    window_sample = raw_sr / target_sr
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith('acc'):
                tags = name.split('.')[0].split('_')
                exp_num = int(tags[1][-2:])
                exp_user = int(tags[2][-2:])
                label_index = (labels[:, 0] == exp_num) & (labels[:, 1] == exp_user)
                label_stat = labels[label_index, :]
                for i in range(label_stat.shape[0]):
                    index_start = label_stat[i, 3]
                    index_end = label_stat[i, 4]

                    exp_data_acc = np.loadtxt(os.path.join(root, name), delimiter=' ') * 9.80665
                    exp_data_gyro = np.loadtxt(os.path.join(root, 'gyro' + name[3:]), delimiter=' ')
                    exp_data = down_sample(np.concatenate([exp_data_acc, exp_data_gyro], 1), window_sample, index_start, index_end)
                    if exp_data.shape[0] > seq_len and label_stat[i, 2] <= 6:
                        exp_data = exp_data[:exp_data.shape[0] // seq_len * seq_len, :]
                        exp_data = exp_data.reshape(exp_data.shape[0] // seq_len, seq_len, exp_data.shape[1])
                        exp_label = np.ones((exp_data.shape[0], exp_data.shape[1], 1))
                        exp_label = np.concatenate([exp_label * label_stat[i, 2], exp_label * label_stat[i, 1]], 2)
                        data.append(exp_data)
                        label.append(exp_label)
    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)
    label[:, :, 0] = label[:, :, 0] - np.min(label[:, :, 0])
    label[:, :, 1] = label[:, :, 1] - np.min(label[:, :, 1])
    print('All data processed. Size: %d' % (data.shape[0]))
    np.save(os.path.join(path_save, 'data_' + version + '.npy'), np.array(data))
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), np.array(label))
    return data, label


# activity, user
path_save = r'uci'
version = r'20_120'
data, label = preprocess(DATASET_PATH, path_save, version, target_sr=20, seq_len=120)


