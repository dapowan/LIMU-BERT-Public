#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 9:16
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : benchmark.py
# @Description : In this script, the implementation of BERT refers from https://github.com/dhlee347/pytorchic-bert

from utils import split_last, merge_last

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LIMUBertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer(cfg) # encoder
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm


class ClassifierLSTM(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('lstm' + str(i), nn.LSTM(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('lstm' + str(i),
                                 nn.LSTM(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.seq_len))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            lstm = self.__getattr__('lstm' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h, _ = lstm(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None, feats=False):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierAttn(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.embd = nn.Embedding(cfg.seq_len, input)
        self.proj_q = nn.Linear(input, cfg.atten_hidden)
        self.proj_k = nn.Linear(input, cfg.atten_hidden)
        self.proj_v = nn.Linear(input, cfg.atten_hidden)
        self.attn = nn.MultiheadAttention(cfg.atten_hidden, cfg.num_head)
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.flatten = nn.Flatten()
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        seq_len = input_seqs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        pos = pos.unsqueeze(0).expand(input_seqs.size(0), seq_len)  # (S,) -> (B, S)
        h = input_seqs + self.embd(pos)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        h, weights = self.attn(q, k, v)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            if i == self.num_linear - 1:
                h = self.flatten(h)
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN2D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i), nn.Conv2d(1, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i), nn.Conv2d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm2d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool2d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = bn(self.pool(h))
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN1D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.seq_len, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool1d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = self.pool(h)
            # h = bn(h)
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class BERTClassifier(nn.Module):

    def __init__(self, bert_cfg, classifier=None, frozen_bert=False):
        super().__init__()
        self.transformer = Transformer(bert_cfg)
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier

    def forward(self, input_seqs, training=False): #, training
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


class BenchmarkDCNN(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 40, (5, 1))
        self.bn2 = nn.BatchNorm2d(40)
        if cfg.seq_len <= 20:
            self.conv3 = nn.Conv2d(40, 20, (2, 1))
        else:
            self.conv3 = nn.Conv2d(40, 20, (3, 1))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d((2, 1))
        self.lin1 = nn.Linear(input * cfg.flat_num, 400)
        self.lin2 = nn.Linear(400, output)

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        h = F.relu(F.tanh(self.conv1(h)))
        h = self.bn1(self.pool(h))
        h = F.relu(F.tanh(self.conv2(h)))
        h = self.bn2(self.pool(h))
        h = F.relu(F.tanh(self.conv3(h)))
        h = h.view(h.size(0), h.size(1), h.size(2) * h.size(3))
        h = self.lin1(h)
        h = F.relu(F.tanh(torch.sum(h, dim=1)))
        h = self.normalize(h[:, :, None, None])
        h = self.lin2(h[:, :, 0, 0])
        return h

    def normalize(self, x, k=1, alpha=2e-4, beta=0.75):
        # x = x.view(x.size(0), x.size(1) // 5, 5, x.size(2), x.size(3))#
        # y = x.clone()
        # for s in range(x.size(0)):
        #     for j in range(x.size(1)):
        #         for i in range(5):
        #             norm = alpha * torch.sum(torch.square(y[s, j, i, :, :])) + k
        #             norm = torch.pow(norm, -beta)
        #             x[s, j, i, :, :] = y[s, j, i, :, :] * norm
        # x = x.view(x.size(0), x.size(1) * 5, x.size(3), x.size(4))
        return x


class BenchmarkDeepSense(nn.Module):

    def __init__(self, cfg, input=None, output=None, num_filter=8):
        super().__init__()
        self.sensor_num = input // 3
        for i in range(self.sensor_num):
            self.__setattr__('conv' + str(i) + "_1", nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__('conv' + str(i) + "_2", nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__('conv' + str(i) + "_3", nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__('bn' + str(i) + "_1", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_2", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_3", nn.BatchNorm2d(num_filter))
        self.conv1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, (2, 1))
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(cfg.flat_num, 12)
        self.lin2 = nn.Linear(12, output)


    def forward(self, input_seqs, training=False):
        h = input_seqs.view(input_seqs.size(0), input_seqs.size(1), self.sensor_num, 3)
        hs = []
        for i in range(self.sensor_num):
            t = h[:, :, i, :]
            t = torch.unsqueeze(t, 1)
            for j in range(3):
                cv = self.__getattr__('conv' + str(i) + "_" + str(j + 1))
                bn = self.__getattr__('bn' + str(i) + "_" + str(j + 1))
                t = bn(F.relu(cv(t)))
            hs.append(self.flatten(t)[:, :, None])
        h = torch.cat(hs, dim=2)
        h = h.unsqueeze(1)
        h = self.bn1(F.relu(self.conv1(h)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.flatten(h)
        h = self.lin2(F.relu(self.lin1(h)))
        return h


class BenchmarkTPNPretrain(nn.Module):
    def __init__(self, cfg, task_num, input=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=2)
        self.flatten = nn.Flatten()
        for i in range(task_num):
            self.__setattr__('slin' + str(i) + "_1", nn.Linear(96, 256))
            self.__setattr__('slin' + str(i) + "_2", nn.Linear(256, 1))
        self.task_num = task_num

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv3(h))
        h = F.dropout(h, p=0.1, training=training)
        h = self.flatten(torch.max(h, 2)[0])
        hs = []
        for i in range(self.task_num):
            lin1 = self.__getattr__('slin' + str(i) + "_1")
            lin2 = self.__getattr__('slin' + str(i) + "_2")
            hl = F.relu(lin1(h))
            hl = F.sigmoid(lin2(hl))
            hs.append(hl)
        hf = torch.stack(hs)[:, :, 0]
        hf = torch.transpose(hf, 0, 1)
        return hf


class BenchmarkTPNClassifier(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96, 1024)
        self.fc2 = nn.Linear(1024, output)
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.conv2.parameters():
            p.requires_grad = False
        for p in self.conv3.parameters():
            p.requires_grad = False

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv3(h))
        h = F.dropout(h, p=0.1, training=training)
        h = self.flatten(torch.max(h, 2)[0])
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


def fetch_classifier(method, model_cfg, input=None, output=None, feats=False):
    if 'lstm' in method:
        model = ClassifierLSTM(model_cfg, input=input, output=output)
    elif 'gru' in method:
        model = ClassifierGRU(model_cfg, input=input, output=output)
    elif 'dcnn' in method:
        model = BenchmarkDCNN(model_cfg, input=input, output=output)
    elif 'cnn2' in method:
        model = ClassifierCNN2D(model_cfg, output=output)
    elif 'cnn1' in method:
        model = ClassifierCNN1D(model_cfg, output=output)
    elif 'deepsense' in method:
        model = BenchmarkDeepSense(model_cfg, input=input, output=output)
    elif 'attn':
        model = ClassifierAttn(model_cfg, input=input, output=output)
    else:
        model = None
    return model
