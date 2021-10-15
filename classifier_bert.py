#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : classifier_bert.py
# @Description :
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

import train
from config import load_dataset_label_names
from models import BERTClassifier, fetch_classifier

from statistic import stat_acc_f1
from utils import get_device,  handle_argv \
    , IMUDataset, load_bert_classifier_data_config, Preprocess4Normalization, \
    prepare_classifier_dataset


def bert_classify(args, label_index, training_rate, label_rate, frozen_bert=False, balance=True):
    data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg = load_bert_classifier_data_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)

    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                     label_rate=label_rate, merge=model_classifier_cfg.seq_len, seed=train_cfg.seed
                                     , balance=balance)
    pipeline = [Preprocess4Normalization(model_bert_cfg.feature_num)]
    data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    classifier = fetch_classifier(method, model_classifier_cfg, input=model_bert_cfg.hidden, output=label_num)
    model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=frozen_bert)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
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

    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali
                        , model_file=args.pretrain_model, load_self=True)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test


if __name__ == "__main__":
    train_rate = 0.8
    label_rate = 0.01
    balance = True
    frozen_bert = False
    method = "base_gru"
    args = handle_argv('bert_classifier_' + method, 'bert_classifier_train.json', method)
    if args.label_index != -1:
        label_index = args.label_index
    label_test, label_estimate_test = bert_classify(args, args.label_index, train_rate, label_rate
                                                    , frozen_bert=frozen_bert, balance=balance)


