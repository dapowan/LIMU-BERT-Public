import argparse
import json
import copy
from sklearn.model_selection import ParameterGrid

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import train
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset

def create_config_with_params(original_cfg, params):
    # Create a dictionary with the original config's attributes
    config_dict = {
        'seed': original_cfg.seed,
        'batch_size': params.get('batch_size', original_cfg.batch_size),
        'lr': params.get('lr', original_cfg.lr),
        'n_epochs': params.get('n_epochs', original_cfg.n_epochs),
        'warmup': params.get('warmup', original_cfg.warmup),
        'save_steps': original_cfg.save_steps,
        'total_steps': original_cfg.total_steps,
        'lambda1': original_cfg.lambda1,
        'lambda2': params.get('lambda2', original_cfg.lambda2)
    }
    return argparse.Namespace(**config_dict)

def grid_search_classifier(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    # Define parameter grid
    param_grid = {
        'batch_size': [32, 64, 128, 256],
        'lr': [1e-4, 1e-3, 5e-3],
        'n_epochs': [200, 500, 1000],
        'warmup': [0.1, 0.2],
        'lambda2': [0.001, 0.005, 0.01]
    }
    
    # Create all combinations of parameters
    grid = ParameterGrid(param_grid)
    
    # Variables to track best performance
    best_f1 = 0
    best_params = None
    best_model = None
    
    # Load original configs
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    
    # Prepare validation data once
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_classifier_dataset(
        data, labels, 
        label_index=label_index, 
        training_rate=training_rate,
        label_rate=label_rate, 
        merge=model_cfg.seq_len, 
        seed=train_cfg.seed,
        balance=balance
    )

    # Create datasets
    data_set_train = IMUDataset(data_train, label_train)
    data_set_vali = IMUDataset(data_vali, label_vali)
    data_set_test = IMUDataset(data_test, label_test)

    criterion = nn.CrossEntropyLoss()
    
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

    print("Starting grid search...")
    
    # Try each parameter combination
    for params in grid:
        print(f"\nTrying parameters: {params}")
        
        # Create new config with current parameters
        current_cfg = create_config_with_params(train_cfg, params)
            
        # Create data loaders with current batch size
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=params['batch_size'])
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=params['batch_size'])
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=params['batch_size'])
        
        # Initialize model and optimizer
        model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
        
        # Create trainer
        trainer = train.Trainer(current_cfg, model, optimizer, args.save_path, get_device(args.gpu))
        
        try:
            # Train and evaluate
            trainer.train(func_loss, func_forward, func_evaluate, 
                         data_loader_train, data_loader_test, data_loader_vali)
            
            # Get final test predictions
            label_estimate_test = trainer.run(func_forward, None, data_loader_test)
            
            # Calculate metrics
            acc, matrix, f1 = stat_results(label_test, label_estimate_test)
            
            print(f"F1 Score: {f1:.4f}")
            
            # Update best if current is better
            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = copy.deepcopy(model)
                
                # Save best parameters
                with open('best_params.json', 'w') as f:
                    json.dump(best_params, f, indent=4)
                    
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
                
    print("\nGrid search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    # Use best model for final predictions
    if best_model is not None:
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=best_params['batch_size'])
        trainer = train.Trainer(create_config_with_params(train_cfg, best_params), 
                              best_model, optimizer, args.save_path, get_device(args.gpu))
        label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    else:
        print("Warning: No successful parameter combination found")
        label_estimate_test = None
    
    return label_test, label_estimate_test, best_params, best_f1


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


if __name__ == "__main__":
    training_rate = 0.8  # unlabeled sample / total sample
    label_rate = 0.1  # labeled sample / unlabeled sample
    balance = True
    
    mode = "base"
    method = "gru"
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
    print("size of embedding: ", embedding.shape)
    
    # Run grid search instead of single training
    label_test, label_estimate_test, best_params, best_f1 = grid_search_classifier(
        args, embedding, labels, args.label_index,
        training_rate, label_rate, balance=balance, method=method
    )

    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)