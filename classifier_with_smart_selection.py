import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Tuple
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter

import train
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, set_seeds


def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)

def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)
    
def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(labels)
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                               , label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


@dataclass
class SelectorConfig:
    features_to_use: List[str]
    data_percentage: float = 0.1
    diversity_weight: float = 0.5
    time_step: float = 0.01
    use_normalization: bool = False
    jitter_threshold: float = 100
    jitter_replacement: float = 0.001
    smoothing_window: int = 100

def calculate_significant_axis(seqs):
    # Calculate the axis with maximum rotational activity
    abs_rotations = torch.abs(seqs[:, :, 3:6])
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)
    return sig_axis

def calculate_jerk(data):
    time_step = 0.01
    diff_acceleration = torch.from_numpy(data).diff() if isinstance(data, np.ndarray) else data.diff()
    diff_acceleration = diff_acceleration / time_step
    jerk_data = diff_acceleration.diff() / time_step
    return jerk_data.numpy() if torch.is_tensor(jerk_data) else jerk_data

def calculate_velocity(data):
    time_step = 0.01
    data_tensor = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    velocity_data = torch.cumsum(data_tensor, dim=1) * time_step
    return velocity_data.numpy() if torch.is_tensor(velocity_data) else velocity_data

def smoothing(arr):
    f = 100
    B = np.ones(f) / f
    x = lfilter(B, 1, arr)
    return x

def calculate_jitter(smooth_accel, data, threshold=100, replacement=0.001):
    if isinstance(smooth_accel, torch.Tensor):
        smooth_accel = smooth_accel.numpy()
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    mse = np.mean((smooth_accel - data)**2)
    jitter_s = mse / data
    jitter_s[np.abs(jitter_s) >= threshold] = replacement
    jitter_s = np.mean(np.abs(jitter_s))
    return jitter_s

class GestureSelector:
    def __init__(self, config: SelectorConfig):
        self.config = config
        self.scaler = StandardScaler() if config.use_normalization else None
    
    def _extract_sig_axis_data(self, data: torch.Tensor, sig_axes: torch.Tensor) -> torch.Tensor:
        batch_size = data.shape[0]
        sig_axis_data = torch.zeros((batch_size, data.shape[1]))
        for i in range(batch_size):
            sig_axis_data[i] = data[i, :, sig_axes[i] + 3]
        return sig_axis_data
    
    def _calculate_features(self, data: torch.Tensor, sig_axes: torch.Tensor) -> Dict[str, torch.Tensor]:
        sig_axis_data = self._extract_sig_axis_data(data, sig_axes)
        features = {}
        
        for feat in self.config.features_to_use:
            if feat == 'jerk':
                jerk_values = calculate_jerk(sig_axis_data)
                features['jerk'] = torch.tensor([np.mean(np.abs(jerk_values[i])) 
                                              for i in range(len(sig_axis_data))])
            elif feat == 'jitter':
                features['jitter'] = torch.tensor([calculate_jitter(
                    smoothing(sig_axis_data[i].numpy()),
                    sig_axis_data[i].numpy(),
                    self.config.jitter_threshold,
                    self.config.jitter_replacement)
                    for i in range(len(sig_axis_data))])
            elif feat == 'velocity':
                velocity_values = calculate_velocity(sig_axis_data)
                features['velocity'] = torch.tensor([np.mean(np.abs(velocity_values[i]))
                                                  for i in range(len(sig_axis_data))])
            elif feat == 'range':
                features['range'] = torch.max(sig_axis_data, dim=1)[0] - torch.min(sig_axis_data, dim=1)[0]
        
        return features
    
    def _normalize_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_matrix = torch.stack([feat for feat in features.values()], dim=1)
        if self.config.use_normalization:
            feature_matrix = torch.tensor(self.scaler.fit_transform(feature_matrix))
        return feature_matrix
    
    def _calculate_diversity_scores(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        class_features = features[indices]
        distances = torch.tensor(cdist(class_features.numpy(), class_features.numpy(), metric='euclidean'))
        diversity_scores = torch.mean(distances, dim=1)
        return diversity_scores
    
    def _calculate_quality_scores(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        class_features = features[indices]
        mean_features = torch.mean(class_features, dim=0)
        distances = torch.norm(class_features - mean_features, dim=1)
        quality_scores = 1 - (distances - torch.min(distances)) / (torch.max(distances) - torch.min(distances) + 1e-8)
        return quality_scores

    def select_samples(self, data: Union[torch.Tensor, np.ndarray], 
                      labels: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            if isinstance(labels, np.ndarray):
                if len(labels.shape) == 3:
                    labels = torch.from_numpy(labels[:, 0, 0]).long()
                else:
                    labels = torch.from_numpy(labels).long()
            
            sig_axes = calculate_significant_axis(data)
            features_dict = self._calculate_features(data, sig_axes)
            feature_matrix = self._normalize_features(features_dict)
            selected_indices = self._select_per_class(feature_matrix, labels)
            
            return selected_indices, features_dict
        except Exception as e:
            print(f"Error in selection process: {e}")
            return self._random_selection(data, labels), {}

    def _select_per_class(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        selected_indices = []
        unique_labels = torch.unique(labels)
        total_samples = len(labels)
        total_to_select = int(round(total_samples * self.config.data_percentage))
        
        class_counts = {label.item(): torch.sum(labels == label).item() 
                       for label in unique_labels}
        
        class_selections = {}
        remaining_selections = total_to_select
        
        for label, count in class_counts.items():
            proportion = count / total_samples
            n_select = max(1, int(round(total_to_select * proportion)))
            remaining_selections -= n_select
            class_selections[label] = n_select
        
        for label in unique_labels:
            class_mask = labels == label
            class_indices = torch.where(class_mask)[0]
            n_select = min(class_selections[label.item()], len(class_indices))
            
            if len(class_indices) > 1:
                diversity_scores = self._calculate_diversity_scores(features, class_indices)
                quality_scores = self._calculate_quality_scores(features, class_indices)
                
                diversity_scores = (diversity_scores - torch.min(diversity_scores)) / (
                    torch.max(diversity_scores) - torch.min(diversity_scores) + 1e-8)
                quality_scores = (quality_scores - torch.min(quality_scores)) / (
                    torch.max(quality_scores) - torch.min(quality_scores) + 1e-8)
                
                combined_scores = (self.config.diversity_weight * diversity_scores + 
                                 (1 - self.config.diversity_weight) * quality_scores)
                
                top_k_indices = torch.topk(combined_scores, n_select).indices
                selected_indices.extend(class_indices[top_k_indices].tolist())
            elif len(class_indices) == 1 and n_select > 0:
                selected_indices.extend(class_indices.tolist())
    
        return torch.tensor(sorted(selected_indices))


def select_samples_with_timeseries(data_timeseries, data_embedding, labels, label_rate, selector_config):
    """
    Select samples using time series features but return embeddings
    """
    try:
        if selector_config is not None:
            selector = GestureSelector(selector_config)
            selector.config.data_percentage = label_rate
            selected_indices, _ = selector.select_samples(data_timeseries, labels)
            
            # Use the same indices for embeddings
            return data_embedding[selected_indices], labels[selected_indices]
        else:
            data_train_label, label_train_label, _, _ = \
                prepare_simple_dataset(data_embedding, labels, training_rate=label_rate)
            return data_train_label, label_train_label
            
    except Exception as e:
        print(f"Feature-based selection failed: {e}. Falling back to random selection.")
        data_train_label, label_train_label, _, _ = \
            prepare_simple_dataset(data_embedding, labels, training_rate=label_rate)
        return data_train_label, label_train_label

def select_balanced_samples_with_timeseries(data_timeseries, data_embedding, labels, label_rate, selector_config):
    """
    Select balanced samples using time series features but return embeddings
    """
    try:
        if selector_config is not None:
            selector = GestureSelector(selector_config)
            selector.config.data_percentage = label_rate
            selected_indices, _ = selector.select_samples(data_timeseries, labels)
            
            # Use the same indices for embeddings
            return data_embedding[selected_indices], labels[selected_indices]
        else:
            data_train_label, label_train_label, _, _ = \
                prepare_simple_dataset_balance(data_embedding, labels, training_rate=label_rate)
            return data_train_label, label_train_label
            
    except Exception as e:
        print(f"Feature-based balanced selection failed: {e}. Falling back to random balanced selection.")
        data_train_label, label_train_label, _, _ = \
            prepare_simple_dataset_balance(data_embedding, labels, training_rate=label_rate)
        return data_train_label, label_train_label
    

def prepare_classifier_dataset_with_selector(data_timeseries, data_embedding, labels, selector_config, label_index=0, 
                                        training_rate=0.8, label_rate=1.0, change_shape=True,
                                        merge=0, merge_mode='all', seed=None, balance=False):
    """
    Enhanced version of prepare_classifier_dataset that uses time series data for selection
    but returns embeddings for training
    """
    set_seeds(seed)
    
    # First split into train/vali/test
    data_timeseries_train, label_train, data_timeseries_vali, label_vali, data_timeseries_test, label_test \
        = partition_and_reshape(data_timeseries, labels, label_index=label_index, 
                              training_rate=training_rate, vali_rate=0.1,
                              change_shape=change_shape, merge=merge, 
                              merge_mode=merge_mode)
    
    # Get the same splits for embeddings using the same random seed
    data_embedding_train, _, data_embedding_vali, _, data_embedding_test, _ \
        = partition_and_reshape(data_embedding, labels, label_index=label_index, 
                              training_rate=training_rate, vali_rate=0.1,
                              change_shape=change_shape, merge=merge, 
                              merge_mode=merge_mode)
    
    set_seeds(seed)
    
    if balance:
        data_train_label, label_train_label = select_balanced_samples_with_timeseries(
            data_timeseries_train, data_embedding_train, label_train, label_rate, selector_config)
    else:
        data_train_label, label_train_label = select_samples_with_timeseries(
            data_timeseries_train, data_embedding_train, label_train, label_rate, selector_config)
            
    return data_train_label, label_train_label, data_embedding_vali, label_vali, data_embedding_test, label_test



def select_samples(data, labels, label_rate, selector_config):
    try:
        if selector_config is not None:
            selector = GestureSelector(selector_config)
            selector.config.data_percentage = label_rate
            selected_indices, _ = selector.select_samples(data, labels)
            return data[selected_indices], labels[selected_indices]
        else:
            data_train_label, label_train_label, _, _ = \
                prepare_simple_dataset(data, labels, training_rate=label_rate)
            return data_train_label, label_train_label
    except Exception as e:
        print(f"Feature-based selection failed: {e}. Falling back to random selection.")
        data_train_label, label_train_label, _, _ = \
            prepare_simple_dataset(data, labels, training_rate=label_rate)
        return data_train_label, label_train_label

def select_balanced_samples(data, labels, label_rate, selector_config):
    try:
        if selector_config is not None:
            selector = GestureSelector(selector_config)
            selector.config.data_percentage = label_rate
            selected_indices, _ = selector.select_samples(data, labels)
            return data[selected_indices], labels[selected_indices]
        else:
            data_train_label, label_train_label, _, _ = \
                prepare_simple_dataset_balance(data, labels, training_rate=label_rate)
            return data_train_label, label_train_label
    except Exception as e:
        print(f"Feature-based balanced selection failed: {e}. Falling back to random balanced selection.")
        data_train_label, label_train_label, _, _ = \
            prepare_simple_dataset_balance(data, labels, training_rate=label_rate)
        return data_train_label, label_train_label


def classify_embeddings(args, data_timeseries, data_embedding, labels, label_index, training_rate, label_rate, balance=False, method=None):
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    
    # Create selector config if feature-based selection is enabled
    selector_config = None
    if hasattr(train_cfg, 'use_feature_selection') and train_cfg.use_feature_selection:
        selector_config = SelectorConfig(
            features_to_use=train_cfg.features_to_use,
            data_percentage=label_rate,
            diversity_weight=train_cfg.diversity_weight if hasattr(train_cfg, 'diversity_weight') else 0.5,
            use_normalization=train_cfg.use_normalization if hasattr(train_cfg, 'use_normalization') else False
        )
    
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset_with_selector(
            data_timeseries, data_embedding, labels, 
            selector_config=selector_config,
            label_index=label_index, 
            training_rate=training_rate,
            label_rate=label_rate, 
            merge=model_cfg.seq_len, 
            seed=train_cfg.seed,
            balance=balance
        )

    # Rest of the training process remains the same
    data_set_train = IMUDataset(data_train, label_train)
    data_set_vali = IMUDataset(data_vali, label_vali)
    data_set_test = IMUDataset(data_test, label_test)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
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

    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test

if __name__ == "__main__":
    training_rate = 0.8
    label_rate = 0.01
    balance = True
    mode = "base"
    method = "gru"
    
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    
    # Load both time series data and embeddings
    data_timeseries = np.load('dataset/blind_user/data_20_120.npy')
    embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
    print("Time series data shape:", data_timeseries.shape)
    print("Embedding shape:", embedding.shape)
    
    label_test, label_estimate_test = classify_embeddings(
        args, data_timeseries, embedding, labels, args.label_index,
        training_rate, label_rate, balance=balance, method=method
    )

    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)

