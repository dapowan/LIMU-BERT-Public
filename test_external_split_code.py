from utils import get_device, handle_argv \
    , IMUDataset, load_classifier_config, prepare_classifier_dataset, prepare_classifier_dataset_ext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def test_splits():
   data = np.random.rand(1439, 120, 72)
   # Generate more controlled labels (0,1,2)
   base_labels = np.random.randint(0, 3, size=(1439, 1))  # One label per sequence
   labels = np.repeat(base_labels, 120, axis=1).reshape(1439, 120, 1)
   labels = np.concatenate([labels, labels], axis=2)  # Duplicate for 2 label dimensions
   print(f"Data: {data.shape}, Labels: {labels.shape}")
   
   training_rate = 0.8
   label_rate = 0.05

   # Internal split
   internal_results = prepare_classifier_dataset(
       data=data, 
       labels=labels,
       label_index=0,
       training_rate=training_rate,
       label_rate=label_rate,
       merge=20,
       seed=3431,
       balance=True
   )
   
   # Create external splits
   train_size = int(len(data) * training_rate)
   train_data = data[:train_size]
   train_labels = labels[:train_size]
   test_data = data[train_size:]
   test_labels = labels[train_size:]
   
   # Save splits
   np.save('train_data.npy', train_data)
   np.save('train_labels.npy', train_labels)
   np.save('test_data.npy', test_data)
   np.save('test_labels.npy', test_labels)
   
   # External split
   external_results = prepare_classifier_dataset_ext(
       split_mode='external',
       train_data=train_data,
       train_labels=train_labels,
       test_data=test_data,
       test_labels=test_labels,
       label_index=0,
       label_rate=label_rate,
       merge=20,
       seed=3431,
       balance=True
   )

   # Print results
   for mode, results in [("Internal", internal_results), ("External", external_results)]:
       data_train, label_train, data_vali, label_vali, data_test, label_test = results
       print(f"\n{mode} Split Results:")
       print(f"Train: {data_train.shape}, Labels: {np.unique(label_train)}")
       print(f"Validation: {data_vali.shape}, Labels: {np.unique(label_vali)}")
       print(f"Test: {data_test.shape}, Labels: {np.unique(label_test)}")


if __name__ == "__main__":
    test_splits()