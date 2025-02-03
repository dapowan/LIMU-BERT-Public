import numpy as np
from raw_data_augmentation import augment_and_split_data
from embedding import generate_embedding_or_output
from classifier import classify_embeddings
from utils import handle_argv
from config import  load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
import copy

def run_complete_pipeline(args, augment_factor=2):
    """Run complete pipeline from raw data to classification"""
    
    print("Step 1: Augmenting raw data and creating splits...")
    split_paths = augment_and_split_data(
        args.data_path,
        args.label_path,
        save_dir='augmented_data',
        augment_factor=augment_factor
    )
    
    print("\nStep 2: Generating embeddings for each split...")
    # Modify args for each split
    train_args = copy.deepcopy(args)
    val_args = copy.deepcopy(args)
    test_args = copy.deepcopy(args)
    
    train_args.data_path = split_paths['train_data']
    train_args.label_path = split_paths['train_labels']
    val_args.data_path = split_paths['val_data']
    val_args.label_path = split_paths['val_labels']
    test_args.data_path = split_paths['test_data']
    test_args.label_path = split_paths['test_labels']
    
    # Generate embeddings
    print("Generating embeddings for training data...")
    train_data, train_output, train_labels = generate_embedding_or_output(train_args, save=True)
    print("Generating embeddings for validation data...")
    val_data, val_output, val_labels = generate_embedding_or_output(val_args, save=True)
    print("Generating embeddings for test data...")
    test_data, test_output, test_labels = generate_embedding_or_output(test_args, save=True)
    
    print("\nStep 3: Running classification...")
    # Combine all data maintaining the original split order
    all_data = np.concatenate([train_output, val_output, test_output])
    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    
    # Load the original split indices
    split_indices = np.load(split_paths['split_indices'], allow_pickle=True).item()
    
    # Train classifier
    label_test, label_estimate_test = classify_embeddings(
        args, 
        all_data,
        all_labels,
        args.label_index,
        training_rate=0.8,
        label_rate=0.1,
        balance=True,
        method="gru"
    )
    
    return label_test, label_estimate_test

if __name__ == "__main__":
    mode = "base"
    method = "gru"
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    
    # Run complete pipeline
    label_test, label_estimate_test = run_complete_pipeline(args, augment_factor=2)
    
    # Evaluate results
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)
    print(f"\nFinal Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")