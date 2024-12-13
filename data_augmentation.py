import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

def augment_with_speed_variation(imu_data, min_speed=0.8, max_speed=1.2):
    """
    Adds speed variations to IMU data by stretching or compressing along the time axis.

    Parameters:
        imu_data (numpy.ndarray): Original IMU data of shape (samples, timesteps, channels).
        min_speed (float): Minimum speed factor (e.g., 0.8 for 80% of the original speed).
        max_speed (float): Maximum speed factor (e.g., 1.2 for 120% of the original speed).

    Returns:
        numpy.ndarray: Augmented IMU data with the same shape as input.
    """
    num_samples, timesteps, channels = imu_data.shape
    augmented_data = []

    for sample in range(num_samples):
        speed_factor = np.random.uniform(min_speed, max_speed)
        new_timesteps = int(timesteps * speed_factor)

        # Interpolate the data to simulate speed changes
        interpolated_sample = []
        for channel in range(channels):
            time_original = np.linspace(0, 1, timesteps)
            time_new = np.linspace(0, 1, new_timesteps)
            interpolator = interp1d(time_original, imu_data[sample, :, channel], kind='linear')
            interpolated_channel = interpolator(time_new)
            interpolated_sample.append(interpolated_channel)

        interpolated_sample = np.stack(interpolated_sample, axis=1)

        # Resample back to the original number of timesteps
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
    """
    Adds jitter and jerk to IMU data.

    Parameters:
        imu_data (numpy.ndarray): Original IMU data of shape (samples, timesteps, channels).
        jitter_std (float): Standard deviation of the Gaussian noise added as jitter.
        jerk_magnitude (float): Magnitude of the jerk added as a sudden change to the data.

    Returns:
        numpy.ndarray: Augmented IMU data with the same shape as input.
    """
    # Add jitter (Gaussian noise)
    jitter = np.random.normal(loc=0.0, scale=jitter_std, size=imu_data.shape)
    augmented_data = imu_data + jitter

    # Add jerk (random spikes in the data)
    jerk_mask = np.random.choice([0, 1], size=imu_data.shape, p=[0.99, 0.01])  # Sparse jerks
    jerk = jerk_mask * np.random.uniform(-jerk_magnitude, jerk_magnitude, size=imu_data.shape)
    augmented_data += jerk

    return augmented_data

def augment_imu_data(imu_data, min_speed=0.8, max_speed=1.2, jitter_std=0.05, jerk_magnitude=0.1):
    """
    Combines speed variations, jitter, and jerk to augment IMU data.

    Parameters:
        imu_data (numpy.ndarray): Original IMU data of shape (samples, timesteps, channels).
        min_speed (float): Minimum speed factor for speed variation.
        max_speed (float): Maximum speed factor for speed variation.
        jitter_std (float): Standard deviation of the Gaussian noise added as jitter.
        jerk_magnitude (float): Magnitude of the jerk added as a sudden change to the data.

    Returns:
        numpy.ndarray: Augmented IMU data with the same shape as input.
    """
    # Apply speed variation
    imu_data = augment_with_speed_variation(imu_data, min_speed, max_speed)

    # Apply jitter and jerk
    imu_data = augment_with_jitter_and_jerk(imu_data, jitter_std, jerk_magnitude)

    return imu_data

def compare_statistics(original_data, augmented_data):
    for i in range(original_data.shape[2]):
        print(f"Channel {i+1}:")
        print(f"  Original Mean: {np.mean(original_data[:, :, i]):.4f}")
        print(f"  Augmented Mean: {np.mean(augmented_data[:, :, i]):.4f}")
        print(f"  Original Std: {np.std(original_data[:, :, i]):.4f}")
        print(f"  Augmented Std: {np.std(augmented_data[:, :, i]):.4f}\n")



def plot_sample(original_data, augmented_data, sample_idx=0, channel_idx=0):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data[sample_idx, :, channel_idx], label="Original")
    plt.plot(augmented_data[sample_idx, :, channel_idx], label="Augmented")
    plt.title(f"Sample {sample_idx} - Channel {channel_idx}")
    plt.legend()
    plt.show()



# Apply combined augmentations to the data
imu_data = np.load('dataset/blind_user/data_20_120.npy')
augmented_imu_data = augment_imu_data(imu_data)

# Ensure output directory exists
output_dir = 'dataset/augmented_data'
os.makedirs(output_dir, exist_ok=True)
augmented_data_path = os.path.join(output_dir, 'augmented_combined_data_20_120.npy')

# Save the augmented data in the same format
np.save(augmented_data_path, augmented_imu_data)

print(f"Augmented IMU data with speed variations, jitter, and jerk saved to: {augmented_data_path}")



compare_statistics(imu_data, augmented_imu_data)
plot_sample(imu_data, augmented_imu_data, sample_idx=0, channel_idx=0)