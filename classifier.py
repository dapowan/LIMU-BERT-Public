#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : classifier.py
# @Description :
import argparse
from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import train
from config import  load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix

from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv \
    , IMUDataset, load_classifier_config, prepare_classifier_dataset


def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate
                                     , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
                                     , balance=balance)
    data_set_train = IMUDataset(data_train, label_train)
    data_set_vali = IMUDataset(data_vali, label_vali)
    data_set_test = IMUDataset(data_test, label_test)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test

def augment_with_speed_variation(embedding_data, min_speed=0.8, max_speed=1.2):
    """
    Adds speed variations to embedded data by stretching/compressing temporal dimension.
    
    Parameters:
        embedding_data (numpy.ndarray): Embeddings of shape (samples, timesteps, embedding_dim)
        min_speed (float): Minimum speed factor
        max_speed (float): Maximum speed factor
    """
    num_samples, timesteps, embedding_dim = embedding_data.shape
    augmented_data = []

    for sample in range(num_samples):
        speed_factor = np.random.uniform(min_speed, max_speed)
        new_timesteps = int(timesteps * speed_factor)

        # Interpolate each embedding dimension 
        interpolated_sample = []
        for dim in range(embedding_dim):
            time_original = np.linspace(0, 1, timesteps)
            time_new = np.linspace(0, 1, new_timesteps)
            interpolator = interp1d(time_original, embedding_data[sample, :, dim], kind='linear')
            interpolated_channel = interpolator(time_new)
            interpolated_sample.append(interpolated_channel)

        interpolated_sample = np.stack(interpolated_sample, axis=1)

        # Resample back to original timesteps
        resample_time = np.linspace(0, 1, timesteps)
        interpolated_time = np.linspace(0, 1, new_timesteps)
        resampled_sample = []
        for dim in range(embedding_dim):
            resampler = interp1d(interpolated_time, interpolated_sample[:, dim], kind='linear')
            resampled_channel = resampler(resample_time)
            resampled_sample.append(resampled_channel)

        augmented_data.append(np.stack(resampled_sample, axis=1))

    return np.array(augmented_data)

def augment_with_jitter_and_jerk(embedding_data, jitter_std=0.01, jerk_magnitude=0.02):
    """
    Adds jitter and jerk to embedded data.
    Note: Using smaller magnitudes since we're working with embeddings
    
    Parameters:
        embedding_data (numpy.ndarray): Embeddings of shape (samples, timesteps, embedding_dim)
        jitter_std (float): Standard deviation for noise
        jerk_magnitude (float): Magnitude of sudden changes
    """
    # Add subtle jitter to embeddings
    jitter = np.random.normal(loc=0.0, scale=jitter_std, size=embedding_data.shape)
    augmented_data = embedding_data + jitter

    # Add sparse jerks
    jerk_mask = np.random.choice([0, 1], size=embedding_data.shape, p=[0.995, 0.005])
    jerk = jerk_mask * np.random.uniform(-jerk_magnitude, jerk_magnitude, size=embedding_data.shape)
    augmented_data += jerk

    return augmented_data

def augment_embedding_data(embedding_data, n_augmentations=1):
    """
    Creates multiple augmented versions of the embedding data.
    
    Parameters:
        embedding_data (numpy.ndarray): Original embeddings of shape (samples, timesteps, embedding_dim)
        n_augmentations (int): Number of augmented copies to create
        
    Returns:
        numpy.ndarray: Augmented embeddings with shape (samples * n_augmentations, timesteps, embedding_dim)
    """
    augmented_samples = []
    
    for _ in range(n_augmentations):
        # Apply speed variation with random parameters
        aug_data = augment_with_speed_variation(
            embedding_data,
            min_speed=np.random.uniform(0.8, 0.9),
            max_speed=np.random.uniform(1.1, 1.2)
        )
        
        # Apply subtle jitter and jerks
        aug_data = augment_with_jitter_and_jerk(
            aug_data,
            jitter_std=np.random.uniform(0.005, 0.015),
            jerk_magnitude=np.random.uniform(0.01, 0.03)
        )
        
        augmented_samples.append(aug_data)
    
    # Stack all augmented versions
    augmented_data = np.concatenate(augmented_samples, axis=0)
    
    return augmented_data


if __name__ == "__main__":

    training_rate = 0.8 # unlabeled sample / total sample
    ## changed from 0.01 to 0.1
    label_rate = 0.001 # labeled sample / unlabeled sample
    balance = True

    mode = "base"
    method = "gru"
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
    print("size of embedding: ", embedding.shape, "size of labels: ", labels.shape)


    n_augmentations = 5

    # Create augmented versions
    augmented_embedding = augment_embedding_data(embedding, n_augmentations=n_augmentations)

    # Combine original and augmented data
    combined_embedding = np.concatenate([embedding, augmented_embedding], axis=0)
    combined_labels = np.tile(labels, (n_augmentations + 1, 1, 1))  

    # Now pass to the classifier
    label_test, label_estimate_test = classify_embeddings(args, combined_embedding, combined_labels, 
                                                        args.label_index, training_rate, label_rate, 
                                                        balance=balance, method=method)
    
    #label_test, label_estimate_test = classify_embeddings(args, embedding, labels, args.label_index,
    #                                                      training_rate, label_rate, balance=balance, method=method)

    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)


