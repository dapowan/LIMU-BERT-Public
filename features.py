import numpy as np
import torch

def compute_energy(seqs):
    """
    Compute energy of the input sequence.
    Args:
        seqs: Tensor (batch_size, seq_len, feature_size) representing IMU sequences.
    
    Returns:
        energy: A list or tensor representing the energy of the sequence.
    """
    energy = torch.sqrt((seqs ** 2).sum(dim=-1))  # Simple norm-based energy calculation
    return energy


def detect_nucleus(energy, window=20, nucleus_thres=0.4):
    """
    Detects the nucleus of gestures based on changes in signal energy.

    Parameters:
    - energy: Tensor (batch_size, sequence_length) containing energy values
    - window: int, window size for detecting changes
    - nucleus_thres: float, threshold for significant energy change
    
    Returns:
    - filtered_change_pts: list of lists, each containing start and end points of the nucleus for each sequence
    """
    batch_nucleus_points = []

    # Loop over each sequence in the batch
    for sequence_energy in energy:
        change_pts = []

        # Convert each sequence to a list of scalars (optional if already 1D)
        sequence_energy = sequence_energy.cpu().numpy() if sequence_energy.is_cuda else sequence_energy.numpy()

        # Sliding window to detect energy changes
        for i in range(len(sequence_energy) - 15):
            if abs(sequence_energy[i + 15] - sequence_energy[i]) > nucleus_thres:
                change_pts.append(i)

        # If no change points are detected, use default nucleus points
        if not change_pts:
            filtered_change_pts = [0, min(len(sequence_energy), window)]
        else:
            # Adjust detected change points
            change_pts = list(map(lambda x: x + window, change_pts))
            
            # Filter close change points
            filtered_change_pts = [change_pts[0]]
            for i in range(1, len(change_pts)):
                if change_pts[i] - filtered_change_pts[-1] >= window:
                    filtered_change_pts.append(change_pts[i])

            filtered_change_pts = filtered_change_pts[:2]

            # Adjust if only one change point detected
            if len(filtered_change_pts) == 1:
                filtered_change_pts.append(change_pts[-1] + 10)

        batch_nucleus_points.append(filtered_change_pts)

    return batch_nucleus_points  # Returns nucleus points for each sequence in the batch

# Example usage
#filtered_change_pts = detect_nucleus(energy)

def calculate_significant_axis(seqs):
    # Calculate the axis with maximum rotational activity (x=0, y=1, z=2)
    abs_rotations = torch.abs(seqs[:, :, 3:6])  # Assumes last three features are rotations
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)  # Shape: (batch_size,)
    return sig_axis
