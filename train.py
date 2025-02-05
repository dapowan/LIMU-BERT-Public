#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : train.py
# @Description :
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import count_model_parameters


class Trainer(object):
    """Enhanced Training Helper Class with semantic awareness capabilities"""
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        
        # Initialize semantic information containers
        self.semantic_embeds = None
        self.similarity_matrix = None
        self.alpha = 0.7  # Weight for balancing reconstruction and semantic losses

    def set_semantic_info(self, semantic_embeds, similarity_matrix):
        """Set semantic information for training"""
        self.semantic_embeds = semantic_embeds.to(self.device)
        self.similarity_matrix = similarity_matrix.to(self.device)

    def compute_combined_loss(self, model_output, target_seqs):
        """Compute combined reconstruction and semantic losses"""
        # Reconstruction loss
        recon_loss = F.mse_loss(model_output, target_seqs, reduction='none')
        
        if self.semantic_embeds is None:
            return recon_loss

        # Get current embeddings similarities
        model_embeddings = model_output.mean(dim=1)
        current_similarities = F.cosine_similarity(
            model_embeddings.unsqueeze(1),
            model_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Semantic consistency loss
        semantic_loss = F.mse_loss(
            current_similarities,
            self.similarity_matrix[None].expand(model_output.size(0), -1, -1),
            reduction='none'
        )
        
        # Combine losses
        total_loss = self.alpha * recon_loss + (1-self.alpha) * semantic_loss
        return total_loss

    def pretrain(self, func_loss, func_forward, func_evaluate,
                data_loader_train, data_loader_test, model_file=None, 
                data_parallel=False):
        """Enhanced Train Loop with semantic awareness"""
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        global_step = 0
        best_loss = 1e6
        model_best = model.state_dict()

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.
            time_sum = 0.0
            self.model.train()
            
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                
                # Get model output with semantic information
                mask_seqs, masked_pos, seqs = batch
                seq_recon = model(mask_seqs, masked_pos, self.semantic_embeds)
                
                # Compute combined loss
                loss = self.compute_combined_loss(seq_recon, seqs)
                
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                
                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return

            loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader_train), loss_eva))

            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
                
        model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')

    def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, load_self=load_self)
        # print(count_model_parameters(self.model))
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        labels = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                result, label = func_forward(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                labels.append(label)
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        if func_evaluate:
            return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
        else:
            return torch.cat(results, 0).cpu().numpy()

    def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali
              , model_file=None, data_parallel=False, load_self=False):
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        vali_acc_best = 0.0
        best_stat = None
        model_best = model.state_dict()
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1 = self.run(func_forward, func_evaluate, data_loader_vali)
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, self.cfg.n_epochs, loss_sum / len(data_loader_train), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, i=0):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '.pt')

