#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import (
    set_seeds, get_device, LIBERTDataset4Pretrain, 
    handle_argv, load_pretrain_data_config, 
    prepare_classifier_dataset, prepare_pretrain_dataset, 
    Preprocess4Normalization, Preprocess4Mask
)
from semantic_utils import prepare_bert_embeddings
from losses import SemanticAwareLoss

def main(args, training_rate):
    # Load configurations and data
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    # Define activity mapping
    label_to_activity = {
        0: "biking",
        1: "sitting",
        2: "walking downstairs",
        3:"walking upstairs",
        4: "standing",
        5: "sitting"
    }


    # Get semantic embeddings and relationships
    label = labels[:, 0, 2]

    semantic_embeds, similarity_matrix, activity_relationships = prepare_bert_embeddings(
        label,  # Now shape is (n,)
        label_to_activity,
        output_dim=model_cfg.hidden
    )

    # Prepare data
    pipeline = [
        Preprocess4Normalization(model_cfg.feature_num),
        Preprocess4Mask(mask_cfg)
    ]
    
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(
        data, labels, training_rate, seed=train_cfg.seed
    )

    # Create datasets and dataloaders
    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    
    data_loader_train = DataLoader(
        data_set_train, 
        shuffle=True, 
        batch_size=train_cfg.batch_size
    )
    data_loader_test = DataLoader(
        data_set_test, 
        shuffle=False, 
        batch_size=train_cfg.batch_size
    )

    # Initialize model, loss, and optimizer
    model = LIMUBertModel4Pretrain(model_cfg)
    loss_computer = SemanticAwareLoss(device=get_device(args.gpu))
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=train_cfg.lr
    )

    # Setup trainer
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)
    trainer.set_semantic_info(
        semantic_embeds, 
        similarity_matrix, 
        activity_relationships
    )

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch
        
        # Print shapes of batch variables
        print("Batch shape - mask_seqs:", mask_seqs.shape)
        print("Batch shape - masked_pos:", masked_pos.shape)
        print("Batch shape - seqs:", seqs.shape)
        
        # Print shapes of semantic information
        print("Semantic embeds shape:", trainer.semantic_embeds.shape)
        print("Similarity matrix shape:", trainer.similarity_matrix.shape)
        
        seq_recon = model(mask_seqs, masked_pos, trainer.semantic_embeds)
        
        semantic_info = (
            trainer.semantic_embeds,
            trainer.similarity_matrix,
            trainer.activity_relationships
        )
        
        loss, loss_components = loss_computer.compute_loss(
            seq_recon,
            seqs,
            semantic_info
        )
        return loss, loss_components

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos, trainer.semantic_embeds)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss = F.mse_loss(predict_seqs, seqs)
        return loss.mean().cpu().numpy()

    # Start training
    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(
            func_loss, 
            func_forward, 
            func_evaluate,
            data_loader_train, 
            data_loader_test,
            model_file=args.pretrain_model
        )
    else:
        trainer.pretrain(
            func_loss, 
            func_forward, 
            func_evaluate,
            data_loader_train, 
            data_loader_test,
            model_file=None
        )

if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)