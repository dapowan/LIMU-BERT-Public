# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : config.py
# @Description :

import json
from typing import NamedTuple
import os
# from bunch import bunchify


class PretrainModelConfig(NamedTuple):
    "Configuration for BERT model"
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    feature_num: int = 0  # Factorized embedding parameterization

    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = True

    @classmethod
    def from_json(cls, js):
        return cls(**js)


class ClassifierModelConfig(NamedTuple):
    "Configuration for classifier model"
    seq_len: int = 0
    input: int = 0

    num_rnn: int = 0
    num_layers: int = 0
    rnn_io: list = []

    num_cnn: int = 0
    conv_io: list = []
    pool: list = []
    flat_num: int = 0

    num_attn: int = 0
    num_head: int = 0
    atten_hidden: int = 0

    num_linear: int = 0
    linear_io: list = []

    activ: bool = False
    dropout: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)


class TrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 0  # random seed
    batch_size: int = 0
    lr: int = 0  # learning rate
    n_epochs: int = 0  # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0
    save_steps: int = 0  # interval for saving model
    total_steps: int = 0  # total number of steps to train
    lambda1: float = 0
    lambda2: float = 0


    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class MaskConfig(NamedTuple):
    """ Hyperparameters for training """
    mask_ratio: float = 0  # masking probability
    mask_alpha: int = 0  # How many tokens to form a group.
    max_gram: int = 0  # number of max n-gram to masking
    mask_prob: float = 1.0
    replace_prob: float = 0.0

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class DatasetConfig(NamedTuple):
    """ Hyperparameters for training """
    sr: int = 0  # sampling rate
    # dataset = Narray with shape (size, seq_len, dimension)
    size: int = 0  # data sample number
    seq_len: int = 0  # seq length
    dimension: int = 0  # feature dimension

    activity_label_index: int = -1  # index of activity label
    activity_label_size: int = 0  # number of activity label
    activity_label: list = []  # names of activity label.

    user_label_index: int = -1  # index of user label
    user_label_size: int = 0  # number of user label

    position_label_index: int = -1  # index of phone position label
    position_label_size: int = 0  # number of position label
    position_label: list = []  # names of position label.

    model_label_index: int = -1  # index of phone model label
    model_label_size: int = 0  # number of model label

    @classmethod
    def from_json(cls, js):
        return cls(**js)


def create_io_config(args, dataset_name, version, pretrain_model=None, target='pretrain'):
    data_path = os.path.join('dataset', dataset_name, 'data_' + version + '.npy')
    label_path = os.path.join('dataset', dataset_name, 'label_' + version + '.npy')
    args.data_path = data_path
    args.label_path = label_path

    save_path = os.path.join('saved', target + "_" + dataset_name + "_" + version)  # + "_temp"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args.save_path = os.path.join(save_path, args.save_model)

    # log_path = os.path.join('log', target + "_" + dataset_name + "_" + version)  # + "_temp"
    # if not os.path.exists(log_path):
    #     os.mkdir(log_path)
    # args.log_dir = log_path

    if pretrain_model is not None:
        if target.count('_') > 2: # bert_classifier
            model_path = os.path.join('saved', 'pretrain_' + target.split('_')[2] + "_" + dataset_name + "_" + version, pretrain_model)
        else:
            model_path = os.path.join(save_path, pretrain_model)
        args.pretrain_model = model_path
    else:
        args.pretrain_model = None
    return args


def load_model_config(target, prefix, version
                      , path_bert='config/limu_bert.json', path_classifier='config/classifier.json'):
    if "bert" not in target: # pretrain or pure classifier
        if "pretrain" in target:
            model_config_all = json.load(open(path_bert, "r"))
        else:
            model_config_all = json.load(open(path_classifier, "r"))
        name = prefix + "_" + version
        if name in model_config_all:
            if "pretrain" in target:
                return PretrainModelConfig.from_json(model_config_all[name])
            else:
                return ClassifierModelConfig.from_json(model_config_all[name])
        else:
            return None
    else: # pretrain + classifier for fine-tune
        model_config_bert = json.load(open(path_bert, "r"))
        model_config_classifier = json.load(open(path_classifier, "r"))
        prefixes = prefix.split('_')
        versions = version.split('_')
        bert_name = prefixes[0] + "_" + versions[0]
        classifier_name = prefixes[1] + "_" + versions[1]
        if bert_name in model_config_bert and classifier_name in model_config_classifier:
            return [PretrainModelConfig.from_json(model_config_bert[bert_name])
                , ClassifierModelConfig.from_json(model_config_classifier[classifier_name])]
        else:
            return None


def load_dataset_stats(dataset, version):
    path = 'dataset/data_config.json'
    dataset_config_all = json.load(open(path, "r"))
    name = dataset + "_" + version
    if name in dataset_config_all:
        return DatasetConfig.from_json(dataset_config_all[name])
    else:
        return None


def load_dataset_label_names(dataset_config, label_index):
    for p in dir(dataset_config):
        if getattr(dataset_config, p) == label_index and "label_index" in p:
            temp = p.split("_")
            label_num = getattr(dataset_config, temp[0] + "_" + temp[1] + "_size")
            if hasattr(dataset_config, temp[0] + "_" + temp[1]):
                return getattr(dataset_config, temp[0] + "_" + temp[1]), label_num
            else:
                return None, label_num
    return None, -1

