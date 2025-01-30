import numpy as np
from scipy.signal import lfilter
import torch



def calculate_significant_axis(seqs):
    # Calculate the axis with maximum rotational activity (x=0, y=1, z=2)
    abs_rotations = torch.abs(seqs[:, :, 3:6])  # Assumes last three features are rotations
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)  # Shape: (batch_size,)
    return sig_axis


def calculate_range_of_motion(data):
    max_values = np.max(data)
    min_values = np.min(data)
    range_of_motion = max_values - min_values
    return range_of_motion

def calculate_jerk(data):
    # Time step between data points
    time_step = 0.01  # Adjust this based on your data's time resolution

    # Calculate acceleration differences
    diff_acceleration = data.diff()
    diff_acceleration /= time_step

    # Calculate jerk
    jerk_data = diff_acceleration.diff()
    jerk_data /= time_step

    return jerk_data

def calculate_velocity(data):
    # Time step between data points
    time_step = 0.01  # Adjust this based on your data's time resolution

    velocity_data = data.cumsum() * time_step

    return velocity_data

def smoothing(arr):
    f = 100
    B = np.ones(f) / f
    x = lfilter(B, 1, arr)
    return x

def calculate_jitter(smooth_accel, data, threshold=100, replacement=0.001):
    mse = np.mean((smooth_accel - data)**2)
    jitter_s = mse / data
    jitter_s[np.abs(jitter_s) >= threshold] = replacement
    jitter_s = np.mean(np.abs(jitter_s))
    return jitter_s



def calculate_rms(data, window_size):
    squared_data = data**2
    rolling_sum = np.cumsum(squared_data)
    window_sum = np.concatenate(([0], rolling_sum[window_size:] - rolling_sum[:-window_size]))
    return np.sqrt(window_sum / window_size)

def detect_rms_change_points(data, window_size, threshold):
    rms_values = calculate_rms(data, window_size)
    diff_rms = np.diff(rms_values)
    change_points = np.where(diff_rms > threshold)[0] + window_size // 2
    return change_points
    
def calculate_pauses(accel_data, gyro_data, threshold):
    pauses = []

    max_gx = np.max(gyro_data['gx'])
    max_gy = np.max(gyro_data['gy'])
    max_gz = np.max(gyro_data['gz'])

    max_g = np.argmax([max_gx, max_gy, max_gz])
    significant_axis_g = gyro_data.iloc[:, max_g]

    try:
        change_points = detect_rms_change_points(significant_axis_g, window_size=20, threshold=10)
        gyro_data = gyro_data.iloc[change_points[0]:change_points[-1]]
        accel_data = accel_data.iloc[change_points[0]:change_points[-1]]
    except:
        gyro_data = gyro_data
        accel_data = accel_data

    total_samples = len(accel_data)  # Assuming accel_data and gyro_data have the same length
    
    for i in range(total_samples):
        # Calculate the magnitude of acceleration and gyroscope readings

        accel_magnitude = (accel_data.iloc[i][0]**2 + accel_data.iloc[i][1]**2 + accel_data.iloc[i][2]**2)**0.5
        gyro_magnitude = (gyro_data.iloc[i][0]**2 + gyro_data.iloc[i][1]**2 + gyro_data.iloc[i][2]**2)**0.5
        
        # Check if both acceleration and gyroscope readings are below the threshold
        if accel_magnitude < 9.9 and gyro_magnitude < threshold:
            pauses.append(i)
    
    return len(pauses)