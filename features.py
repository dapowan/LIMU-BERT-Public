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
    Detects the nucleus of a gesture based on changes in signal energy.

    Parameters:
    - energy: list of signal energies
    - window: int, window size for detecting changes
    - nucleus_thres: float, threshold for detecting significant changes in energy
    
    Returns:
    - filtered_change_pts: list of start and end points of the nucleus
    """
    change_pts = []

    # Sliding window to detect energy changes
    for i in range(len(energy) - 15):
        if abs(energy[i + 15] - energy[i]) > nucleus_thres:
            change_pts.append(i)

    change_pts = list(map(lambda x: x + window, change_pts))
    print("Actual change points: ", change_pts)

    # Filter out change points that are too close to each other
    filtered_change_pts = [change_pts[0]]
    for i in range(1, len(change_pts)):
        if change_pts[i] - filtered_change_pts[-1] >= window:
            filtered_change_pts.append(change_pts[i])

    filtered_change_pts = filtered_change_pts[:2]
    print("Filtered change points: ", filtered_change_pts)

    # If only one change point is detected, adjust the end of the nucleus
    if len(filtered_change_pts) == 1:
        filtered_change_pts.append(change_pts[-1] + 10)

    return filtered_change_pts

# Example usage
#filtered_change_pts = detect_nucleus(energy)

def significant_axis(df):
    """
    Detects the axis with the maximum rotation in the gyroscope data.

    Parameters:
    - df: pandas DataFrame with columns 'gx', 'gy', 'gz' representing gyroscope data

    Returns:
    - sig_axis: Series, the axis with the most significant rotation
    - axis: int, the axis number (1 for x, 2 for y, 3 for z)
    """
    max_x = np.max(df['gx'])
    max_y = np.max(df['gy'])
    max_z = np.max(df['gz'])

    max_axis = np.max([max_x, max_y, max_z])
    
    if max_axis == max_x:
        sig_axis = df['gx']
        axis = 1
    elif max_axis == max_y:
        sig_axis = df['gy']
        axis = 2
    else:
        sig_axis = df['gz']
        axis = 3

    return sig_axis, axis

# Example usage
#sig_axis, axis = significant_axis(df)



def mag(vectors, axis):
    return np.sqrt(np.sum(vectors ** 2, axis=axis))

def frenet(x, y, z=None):
    """
    Calculates the Frenet-Serret invariants: Tangent (T), Normal (N), Binormal (B), 
    Curvature (k), and Torsion (t) of a 3D space curve.

    Parameters:
    - x, y, z: numpy arrays representing the coordinates of the curve in 3D space

    Returns:
    - T: Tangent vector
    - N: Normal vector
    - B: Binormal vector
    - k: Curvature
    - t: Torsion
    """
    if z is None:
        z = np.zeros_like(x)

    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    dr = np.column_stack((dx, dy, dz))

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)

    ddr = np.column_stack((ddx, ddy, ddz))

    # Tangent
    T = dr / mag(dr, axis=1)[:, None]

    # Derivative of Tangent
    dT = np.gradient(T, axis=0)

    # Normal
    N = dT / mag(dT, axis=1)[:, None]

    # Binormal
    B = np.cross(T, N)

    # Curvature
    k = mag(np.cross(dr, ddr), axis=1) / (mag(dr, axis=1) ** 3)

    # Torsion
    t = np.einsum('ij,ij->i', -np.gradient(B, axis=0), N)

    return T, N, B, k, t

# Example usage
#T, N, B, k, t = frenet(x, y, z)


