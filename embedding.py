# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : embedding.py
# @Description : generate embeddings using pretrained LIMU-BERT models
import os

import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from features import detect_nucleus, compute_energy

import train
from config import load_dataset_label_names
from models import LIMUBertModel4Pretrain
from plot import plot_reconstruct_sensor, plot_embedding
from utils import LIBERTDataset4Pretrain, load_pretrain_data_config, get_device, handle_argv, \
    Preprocess4Normalization, IMUDataset


def fetch_setup(args, output_embed):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg, output_embed=output_embed)
    criterion = nn.MSELoss(reduction='none')
    return data, labels, data_loader, model, criterion, train_cfg

def generate_nucleus_mask(seq_len, batch_nucleus_points):
    """
    Generate a binary mask for the nucleus for each sequence in the batch.

    Args:
        seq_len: Length of each sequence
        batch_nucleus_points: List of lists, where each inner list contains start and end points for the nucleus of a sequence

    Returns:
        A binary mask tensor with shape (batch_size, seq_len), where 1 indicates the nucleus region.
    """
    batch_size = len(batch_nucleus_points)
    nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, nucleus_points in enumerate(batch_nucleus_points):
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            nucleus_mask[i, start:end] = 1  # Mark nucleus region with 1

    return nucleus_mask


from features import detect_nucleus, compute_energy  # Import both nucleus detection and energy computation

def generate_embedding_or_output(args, save=False, output_embed=True):
    data, labels, data_loader, model, criterion, train_cfg = fetch_setup(args, output_embed)

    optimizer = None
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        seqs, label = batch

        # Compute the energy for each sequence in the batch
        energy = compute_energy(seqs)

        # Detect nucleus points for each sequence in the batch
        batch_nucleus_points = detect_nucleus(energy)

        # Generate the batched nucleus mask
        nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)

        # Pass sequences and batched nucleus mask to the model
        embed = model(seqs, nucleus_mask=nucleus_mask)
        return embed, label

    output = trainer.run(func_forward, None, data_loader, args.pretrain_model)

    if save:
        save_name = 'embed_' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version
        np.save(os.path.join('embed', save_name + '.npy'), output)

    return data, output, labels


def load_embedding_label(model_file, dataset, dataset_version):
    embed_name = 'embed_' + model_file + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    embed = np.load(os.path.join('embed', embed_name + '.npy')).astype(np.float32)
    labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
    return embed, labels


if __name__ == "__main__":
    save = True
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    data, output, labels = generate_embedding_or_output(args=args, output_embed=True, save=save)

    label_index = 0  #put activity_label_index from data_config.json here
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, label_index)
    data_tsne, labels_tsne = plot_embedding(output, labels, label_index=label_index, reduce=1000, label_names=label_names)
