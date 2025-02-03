import numpy as np
from scipy.interpolate import interp1d
import os

def augment_and_split_data(data_path, label_path, save_dir='augmented_data', 
                          training_rate=0.8, val_rate=0.1, augment_factor=2,
                          label_rate=0.1, balance=True, seed=42):
    """
    Augment raw IMU data and create train/val/test splits.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    imu_data = np.load(data_path)
    labels = np.load(label_path)
    
    print("Original data shapes:")
    print(f"IMU data: {imu_data.shape}")
    print(f"Labels: {labels.shape}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Calculate split sizes
    total_samples = imu_data.shape[0]
    train_size = int(training_rate * total_samples)
    val_size = int(val_rate * total_samples)
    
    # Create random indices for splitting
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split data
    train_data = imu_data[train_indices]
    val_data = imu_data[val_indices]
    test_data = imu_data[test_indices]
    
    # Handle labels - need to transpose for correct indexing
    # Labels shape is (L, N, T) where L is label types, N is samples, T is timesteps
    labels_reordered = np.transpose(labels, (1, 0, 2))  # Now (N, L, T)
    
    train_labels = labels_reordered[train_indices]
    val_labels = labels_reordered[val_indices]
    test_labels = labels_reordered[test_indices]
    
    # Augment only training data
    augmented_data = [train_data]
    augmented_labels = [train_labels]
    
    for i in range(augment_factor - 1):
        print(f"Performing augmentation {i+1}/{augment_factor-1}")
        aug_data = augment_with_speed_variation(train_data)
        aug_data = augment_with_jitter_and_jerk(aug_data)
        aug_data = augment_with_planar_rotation(aug_data)
        
        augmented_data.append(aug_data)
        augmented_labels.append(train_labels)
    
    # Combine augmented training data
    train_data_aug = np.concatenate(augmented_data, axis=0)
    train_labels_aug = np.concatenate(augmented_labels, axis=0)
    
    print("\nFinal shapes after augmentation:")
    print(f"Train data: {train_data_aug.shape}")
    print(f"Train labels: {train_labels_aug.shape}")
    print(f"Val data: {val_data.shape}")
    print(f"Val labels: {val_labels.shape}")
    print(f"Test data: {test_data.shape}")
    print(f"Test labels: {test_labels.shape}")
    
    # Save splits
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    label_base = os.path.splitext(os.path.basename(label_path))[0]
    
    # Save data splits
    np.save(os.path.join(save_dir, f'{base_name}_train.npy'), train_data_aug)
    np.save(os.path.join(save_dir, f'{base_name}_val.npy'), val_data)
    np.save(os.path.join(save_dir, f'{base_name}_test.npy'), test_data)
    
    # Save label splits - transpose back to original format before saving
    np.save(os.path.join(save_dir, f'{label_base}_train.npy'), 
            np.transpose(train_labels_aug, (1, 0, 2)))
    np.save(os.path.join(save_dir, f'{label_base}_val.npy'), 
            np.transpose(val_labels, (1, 0, 2)))
    np.save(os.path.join(save_dir, f'{label_base}_test.npy'), 
            np.transpose(test_labels, (1, 0, 2)))
    
    # Save indices for consistent splits
    np.save(os.path.join(save_dir, 'split_indices.npy'), 
            {'train': train_indices, 'val': val_indices, 'test': test_indices})
    
    print(f"\nAugmented and split data saved in {save_dir}")
    
    return {
        'train_data': os.path.join(save_dir, f'{base_name}_train.npy'),
        'val_data': os.path.join(save_dir, f'{base_name}_val.npy'),
        'test_data': os.path.join(save_dir, f'{base_name}_test.npy'),
        'train_labels': os.path.join(save_dir, f'{label_base}_train.npy'),
        'val_labels': os.path.join(save_dir, f'{label_base}_val.npy'),
        'test_labels': os.path.join(save_dir, f'{label_base}_test.npy'),
        'split_indices': os.path.join(save_dir, 'split_indices.npy')
    }

def augment_with_speed_variation(imu_data, min_speed=0.8, max_speed=1.2):
    """Adds speed variations to IMU data"""
    num_samples, timesteps, channels = imu_data.shape
    augmented_data = []

    for sample in range(num_samples):
        speed_factor = np.random.uniform(min_speed, max_speed)
        new_timesteps = int(timesteps * speed_factor)
        
        interpolated_sample = []
        for channel in range(channels):
            time_original = np.linspace(0, 1, timesteps)
            time_new = np.linspace(0, 1, new_timesteps)
            interpolator = interp1d(time_original, imu_data[sample, :, channel], kind='linear')
            interpolated_channel = interpolator(time_new)
            interpolated_sample.append(interpolated_channel)
        
        interpolated_sample = np.stack(interpolated_sample, axis=1)
        
        resample_time = np.linspace(0, 1, timesteps)
        interpolated_time = np.linspace(0, 1, new_timesteps)
        resampled_sample = []
        for channel in range(channels):
            resampler = interp1d(interpolated_time, interpolated_sample[:, channel], kind='linear')
            resampled_channel = resampler(resample_time)
            resampled_sample.append(resampled_channel)
        
        augmented_data.append(np.stack(resampled_sample, axis=1))
    
    return np.array(augmented_data)

def augment_with_jitter_and_jerk(imu_data, jitter_std=0.05, jerk_magnitude=0.1):
    """Adds jitter and jerk to IMU data"""
    jitter = np.random.normal(loc=0.0, scale=jitter_std, size=imu_data.shape)
    augmented_data = imu_data + jitter
    
    jerk_mask = np.random.choice([0, 1], size=imu_data.shape, p=[0.99, 0.01])
    jerk = jerk_mask * np.random.uniform(-jerk_magnitude, jerk_magnitude, size=imu_data.shape)
    augmented_data += jerk
    
    return augmented_data

def augment_with_planar_rotation(imu_data, angle_range=(-10, 10)):
    """Adds planar rotation to IMU data"""
    angle = np.radians(np.random.uniform(angle_range[0], angle_range[1]))
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    augmented_data = []
    for sample in imu_data:
        accel = sample[:, :3] @ rotation_matrix.T
        gyro = sample[:, 3:] @ rotation_matrix.T
        augmented_data.append(np.hstack((accel, gyro)))
    
    return np.array(augmented_data)