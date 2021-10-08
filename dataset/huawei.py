#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 16:31
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : huawei.py
# @Description : http://www.shl-dataset.org/download/#shldataset-preview
import numpy as np
import pandas as pd
import os

SAMPLE_WINDOW = 20


def read_data(file_path):
    column_names = ['Time','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'
        ,'mag_x','mag_y','mag_z','ori_w','ori_x','ori_y','ori_z'
        ,'gra_x','gra_y','gra_z','lin_x','lin_y','lin_z'
        ,'pre','alt','temp']
    use_cols = ['Time','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','mag_x','mag_y','mag_z','gra_x','gra_y','gra_z']
    data = pd.read_csv(file_path, header=None, names=column_names, sep=' ', usecols=use_cols)

    return data


def read_label(file_path):
    column_names = ['Time', 'Coarse_label', 'Fine_label', 'Road_label', 'Traffic_label', 'Tunnels_label', 'Social_label', 'Food_label']
    use_cols = ['Time', 'Coarse_label']
    data = pd.read_csv(file_path, header=None, names=column_names, sep=' ', usecols=use_cols)
    # for key in column_names:
    #     data[key] = [ float(k) for k in data[key]]
    return data


def down_sample(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data.iloc[i: i + window, :]
            result.append(slice.mean(0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 < len(data):
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data.iloc[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(slice.mean(0))
                i += window + 1
            else:
                slice = data.iloc[i: i + window, :]
                result.append(slice.mean(0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return pd.concat(result, 1).T


def read_and_save_data(path):
    # 数据路径
    date_list = ['User1/220617/', 'User1/260617/', 'User1/270617/',
                 'User2/140617/', 'User2/140717/', 'User2/180717/',
                 'User3/030717/', 'User3/070717/', 'User3/140617/']
    user_num = 3 # 每个user3份数据
    phone_list = ['Bag_', 'Hand_', 'Torso_', 'Hips_'] #
    total_data = None
    for (d, date) in enumerate(date_list):
        for p, phone in enumerate(phone_list):
            data_path = 'D:/Dataset_Mobility/Huawei/SHLDataset_preview_v1/' + date + phone + 'Motion.txt'
            label_path = 'D:/Dataset_Mobility/Huawei/SHLDataset_preview_v1/' + date + 'Label.txt'
            print('【【【【data source:', date, phone, '】】】')

            labels = read_label(label_path)
            data = read_data(data_path)
            data = data.dropna(axis=0, how='any')
            total_data_temp = data.join(labels.set_index('Time'), on='Time')
            total_data_temp['Position_label'] = pd.Series(np.ones(len(total_data_temp['Coarse_label'])) * p
                                                          , index=total_data_temp.index)
            total_data_temp['User_label'] = pd.Series(np.ones(len(total_data_temp['Coarse_label'])) * (d // user_num)
                                                          , index=total_data_temp.index)

            if date == date_list[0] and phone == phone_list[0]:
                total_data = pd.DataFrame(columns=total_data_temp.columns)

            # 取 步行coarse_label=2， 骑行（4-bike， 5-car）
            total_data_temp = total_data_temp[(total_data_temp['Coarse_label']==1.0)  | (total_data_temp['Coarse_label'] ==2.0)
                                              | (total_data_temp['Coarse_label'] ==3.0) | (total_data_temp['Coarse_label'] ==4.0)]
                                              # | (total_data_temp['Coarse_label'] ==5.0) | (total_data_temp['Coarse_label'] ==6.0)]
            # 降采样 20ms -> 100ms
            # total_data_temp = total_data_temp.iloc[::5]
            # return total_data_temp
            # downsample to 50ms
            # total_data_down = down_sample(total_data_temp, 50)
            total_data_down = total_data_temp

            print('-----------before concat------------')
            print('still：', len(total_data[(total_data['Coarse_label'] == 1.0)]))
            print('walk：', len(total_data[(total_data['Coarse_label'] == 2.0)]))
            print('run：', len(total_data[(total_data['Coarse_label'] == 3.0)]))
            print('bike：', len(total_data[(total_data['Coarse_label'] == 4.0)]))
            # print('car/bus：', len(total_data[(total_data['Coarse_label'] == 5.0)])
            #       + len(total_data[(total_data['Coarse_label'] == 6.0)]))
            print('all：', len(total_data))

            total_data = pd.concat([total_data, total_data_down])
            print('-----------after concat------------')
            print('still：', len(total_data[(total_data['Coarse_label'] == 1.0)]))
            print('walk：', len(total_data[(total_data['Coarse_label'] == 2.0)]))
            print('run：', len(total_data[(total_data['Coarse_label'] == 3.0)]))
            print('bike：', len(total_data[(total_data['Coarse_label'] == 4.0)]))
            # print('car/bus：', len(total_data[(total_data['Coarse_label'] == 5.0)])
            #       + len(total_data[(total_data['Coarse_label'] == 6.0)]))
            print('all：', len(total_data))

#         pause = input()
    total_data.to_csv(path, index=False)
    return total_data


def preprocess_huawei(path_read, path_save, version, seq_len=100, jump=10, process_func=None,
                      columns=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','mag_x','mag_y','mag_z']):
    data = pd.read_csv(path_read)
    # data = data.iloc[:20000,:]
    data_new = []
    label_new = []
    positions = np.unique(data['Position_label'].values)
    users = np.unique(data['User_label'].values)
    for p in range(positions.size):
        for u in range(users.size):
            print('Start Position-%d, User-%d' % (p, u))
            slice = data[(data['Position_label'] == p) & (data['User_label'] == u)]
            slice = slice.sort_values(by=['Time'])
            tags = np.zeros(slice.shape[0])
            tag = 0
            for i in range(1, slice.shape[0]):
                if np.abs(slice.iloc[i]['Time'] - slice.iloc[i - 1]['Time']) < 150:
                    tag += 1
                    tags[i] = tag
                else:
                    tag = 0
                    tags[i] = tag
            for i in range(0, slice.shape[0] - 1, jump):
                print('Progress: %2.2f %%' % (i / slice.shape[0] * 100))
                if i + seq_len - 1 < slice.shape[0] and tags[i + seq_len - 1] - tags[i] == seq_len - 1:
                    labels = (slice.iloc[i:i + seq_len])[['Coarse_label', 'Position_label', 'User_label']].values
                    if np.unique(labels[:, 0]).size == 1:
                        # labels[labels[:, 0] == 6, 0] = 5
                        labels[:, 0] = labels[:, 0] - 1
                        label_new.append(labels)
                        if process_func:
                            data_new.append(process_func(slice.iloc[i:i + seq_len][columns].values))
                        else:
                            data_new.append(slice.iloc[i:i + seq_len][columns].values)

    print('All data processed. Size: %d' % (len(data_new)))
    np.save(os.path.join(path_save, 'data_' + version + '.npy'), np.array(data_new))
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), np.array(label_new))
    return np.array(data_new), np.array(label_new)


# First Step, merge data and save as csv file
total_data = read_and_save_data(r'D:\Dataset_Mobility\Huawei\50.csv')


# Second Step, csv to npy
path = r'D:\Dataset_Mobility\Huawei\50.csv'
path_save = r'huawei'
version = r'50_300'
# data_raw = pd.read_csv(path)
data, label = preprocess_huawei(path, path_save, version, seq_len=300, jump=120)



