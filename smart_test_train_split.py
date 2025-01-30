import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.signal import lfilter

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
    # Calculate the axis with maximum rotational activity (x=0, y=1, z=2)
    abs_rotations = torch.abs(seqs[:, :, 3:6])  # Assumes last three features are rotations
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)  # Shape: (batch_size,)
    return sig_axis

def calculate_jerk(data):
    # Time step between data points
    time_step = 0.01  # Adjust this based on your data's time resolution

    # Calculate acceleration differences
    diff_acceleration = torch.from_numpy(data).diff() if isinstance(data, np.ndarray) else data.diff()
    diff_acceleration = diff_acceleration / time_step

    # Calculate jerk
    jerk_data = diff_acceleration.diff()
    jerk_data = jerk_data / time_step

    return jerk_data.numpy() if torch.is_tensor(jerk_data) else jerk_data

def calculate_velocity(data):
    # Time step between data points
    time_step = 0.01  # Adjust this based on your data's time resolution
    
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
        """Extract data from significant axes for each sample."""
        batch_size = data.shape[0]
        sig_axis_data = torch.zeros((batch_size, data.shape[1]))
        
        for i in range(batch_size):
            sig_axis_data[i] = data[i, :, sig_axes[i] + 3]  # Add 3 to get rotational components
            
        return sig_axis_data
    
    def _calculate_features(self, data: torch.Tensor, sig_axes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate selected features for each sample."""
        sig_axis_data = self._extract_sig_axis_data(data, sig_axes)
        features = {}
        
        for feat in self.config.features_to_use:
            if feat == 'jerk':
                jerk_values = calculate_jerk(sig_axis_data)
                features['jerk'] = torch.tensor([
                    np.mean(np.abs(jerk_values[i])) 
                    for i in range(len(sig_axis_data))
                ])
            elif feat == 'jitter':
                features['jitter'] = torch.tensor([
                    calculate_jitter(
                        smoothing(sig_axis_data[i].numpy()),
                        sig_axis_data[i].numpy(),
                        self.config.jitter_threshold,
                        self.config.jitter_replacement
                    )
                    for i in range(len(sig_axis_data))
                ])
            elif feat == 'velocity':
                velocity_values = calculate_velocity(sig_axis_data)
                features['velocity'] = torch.tensor([
                    np.mean(np.abs(velocity_values[i]))
                    for i in range(len(sig_axis_data))
                ])
            elif feat == 'range':
                features['range'] = torch.max(sig_axis_data, dim=1)[0] - torch.min(sig_axis_data, dim=1)[0]
                
        return features
    
    def _normalize_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine and normalize features."""
        feature_matrix = torch.stack([feat for feat in features.values()], dim=1)
        
        if self.config.use_normalization:
            feature_matrix = torch.tensor(
                self.scaler.fit_transform(feature_matrix))
        
        return feature_matrix
    
    def _calculate_diversity_scores(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Calculate diversity scores for samples."""
        class_features = features[indices]
        
        distances = torch.tensor(cdist(
            class_features.numpy(), 
            class_features.numpy(), 
            metric='euclidean'
        ))
        
        diversity_scores = torch.mean(distances, dim=1)
        return diversity_scores
    
    def _calculate_quality_scores(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Calculate quality scores based on feature statistics."""
        class_features = features[indices]
        mean_features = torch.mean(class_features, dim=0)
        distances = torch.norm(class_features - mean_features, dim=1)
        
        quality_scores = 1 - (distances - torch.min(distances)) / (
            torch.max(distances) - torch.min(distances) + 1e-8)
        
        return quality_scores
    
    def _select_per_class(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Select samples maintaining total percentage while preserving class proportions."""
        selected_indices = []
        unique_labels = torch.unique(labels)
        total_samples = len(labels)
        total_to_select = int(round(total_samples * self.config.data_percentage))
        
        # Calculate class distributions and selections
        class_counts = {label.item(): torch.sum(labels == label).item() 
                       for label in unique_labels}
        
        class_selections = {}
        remaining_selections = total_to_select
        
        for label, count in class_counts.items():
            proportion = count / total_samples
            n_select = max(1, int(round(total_to_select * proportion)))
            remaining_selections -= n_select
            class_selections[label] = n_select
            
        # Select samples using diversity and quality scores
        for label in unique_labels:
            class_mask = labels == label
            class_indices = torch.where(class_mask)[0]
            n_select = min(class_selections[label.item()], len(class_indices))
            
            if len(class_indices) > 1:
                diversity_scores = self._calculate_diversity_scores(features, class_indices)
                quality_scores = self._calculate_quality_scores(features, class_indices)
                
                # Normalize scores
                diversity_scores = (diversity_scores - torch.min(diversity_scores)) / (
                    torch.max(diversity_scores) - torch.min(diversity_scores) + 1e-8)
                quality_scores = (quality_scores - torch.min(quality_scores)) / (
                    torch.max(quality_scores) - torch.min(quality_scores) + 1e-8)
                
                combined_scores = (
                    self.config.diversity_weight * diversity_scores + 
                    (1 - self.config.diversity_weight) * quality_scores
                )
                
                top_k_indices = torch.topk(combined_scores, n_select).indices
                selected_indices.extend(class_indices[top_k_indices].tolist())
            elif len(class_indices) == 1 and n_select > 0:
                selected_indices.extend(class_indices.tolist())
    
        return torch.tensor(sorted(selected_indices))
    
    def select_samples(self, data: Union[torch.Tensor, np.ndarray], 
                      labels: Union[torch.Tensor, np.ndarray]
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Main selection method."""
        try:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            if isinstance(labels, np.ndarray):
                # Take only the first label for each sample if labels are 3D
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
            print("Falling back to random selection")
            return self._random_selection(data, labels), {}
    
    def _random_selection(self, data: Union[torch.Tensor, np.ndarray],
                         labels: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Fallback random selection maintaining total percentage."""
        if isinstance(labels, np.ndarray):
            if len(labels.shape) == 3:
                labels = torch.from_numpy(labels[:, 0, 0]).long()
            else:
                labels = torch.from_numpy(labels).long()
            
        total_samples = len(labels)
        total_to_select = int(round(total_samples * self.config.data_percentage))
        
        unique_labels = torch.unique(labels)
        class_counts = {label.item(): torch.sum(labels == label).item() 
                       for label in unique_labels}
            
        class_selections = {}
        remaining_selections = total_to_select
        
        for label, count in class_counts.items():
            proportion = count / total_samples
            n_select = max(1, int(round(total_to_select * proportion)))
            remaining_selections -= n_select
            class_selections[label] = n_select
                        
        selected_indices = []
        for label in unique_labels:
            class_mask = labels == label
            class_indices = torch.where(class_mask)[0]
            n_select = min(class_selections[label.item()], len(class_indices))
            
            if n_select > 0:
                selected = torch.randperm(len(class_indices))[:n_select]
                selected_indices.extend(class_indices[selected].tolist())
        
        return torch.tensor(sorted(selected_indices))
    
    def visualize_selection(self, data: torch.Tensor, labels: torch.Tensor, 
                          selected_indices: torch.Tensor, 
                          features_dict: Dict[str, torch.Tensor]):
        """Visualize the selection results."""
        if not features_dict:
            print("No features available for visualization")
            return
            
        self._plot_feature_distributions(features_dict, selected_indices)
        self._plot_example_gestures(data, labels, selected_indices)
        if len(features_dict) >= 2:
            self._plot_feature_space_coverage(features_dict, selected_indices)
    
    def _plot_feature_distributions(self, features_dict: Dict[str, torch.Tensor], 
                              selected_indices: torch.Tensor):
        # Get random indices with same size as selected indices
        random_indices = torch.randperm(len(next(iter(features_dict.values()))))[
            :len(selected_indices)]
        
        n_features = len(features_dict)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
        if n_features == 1:
            axes = [axes]
            
        for i, (feat_name, feat_values) in enumerate(features_dict.items()):
            sns.kdeplot(feat_values.numpy(), ax=axes[i], 
                    label='All samples', color='blue')
            sns.kdeplot(feat_values[selected_indices].numpy(), 
                    ax=axes[i], label='Selected samples (Our method)', 
                    color='green')
            sns.kdeplot(feat_values[random_indices].numpy(), 
                    ax=axes[i], label='Selected samples (Random)', 
                    color='red')
            axes[i].set_title(f'{feat_name} Distribution')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_example_gestures(self, data: torch.Tensor, labels: torch.Tensor, 
                         selected_indices: torch.Tensor):
    # Get random indices with same size as selected indices
        random_indices = torch.randperm(len(labels))[:len(selected_indices)]
        
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        
        fig, axes = plt.subplots(n_classes, 3, figsize=(15, 5*n_classes))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, label in enumerate(unique_labels):
            class_mask = labels[selected_indices] == label
            class_selected = selected_indices[class_mask]
            
            random_class_mask = labels[random_indices] == label
            random_class_selected = random_indices[random_class_mask]
            
            if len(class_selected) > 0:
                # Plot rotational components for our method
                for j in range(3):
                    axes[i, 0].plot(data[class_selected[0], :, j+3].numpy(), 
                                label=f'Rotation {j}')
                axes[i, 0].set_title(f'Class {label.item()} - Our Method Example')
                axes[i, 0].legend()
                
                # Plot rotational components for random selection
                if len(random_class_selected) > 0:
                    for j in range(3):
                        axes[i, 1].plot(data[random_class_selected[0], :, j+3].numpy(), 
                                    label=f'Rotation {j}')
                    axes[i, 1].set_title(f'Class {label.item()} - Random Example')
                    axes[i, 1].legend()
                
                # Plot mean rotational components
                mean_gesture = torch.mean(data[class_selected], dim=0)
                mean_random = torch.mean(data[random_class_selected], dim=0) if len(random_class_selected) > 0 else None
                
                for j in range(3):
                    axes[i, 2].plot(mean_gesture[:, j+3].numpy(), 
                                label=f'Our Method - Rotation {j}', 
                                linestyle='-')
                    if mean_random is not None:
                        axes[i, 2].plot(mean_random[:, j+3].numpy(), 
                                    label=f'Random - Rotation {j}', 
                                    linestyle='--')
                axes[i, 2].set_title(f'Class {label.item()} - Mean Comparison')
                axes[i, 2].legend()
        
        plt.tight_layout()
        plt.show()

    def _plot_feature_space_coverage(self, features_dict: Dict[str, torch.Tensor], 
                               selected_indices: torch.Tensor):
        # Get random indices with same size as selected indices
        random_indices = torch.randperm(len(next(iter(features_dict.values()))))[
            :len(selected_indices)]
        
        feature_names = list(features_dict.keys())
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot all samples
        ax.scatter(features_dict[feature_names[0]], 
                features_dict[feature_names[1]], 
                alpha=0.3, label='All samples', color='blue')
        
        # Plot our method's selected samples
        ax.scatter(features_dict[feature_names[0]][selected_indices], 
                features_dict[feature_names[1]][selected_indices], 
                alpha=0.8, label='Selected samples (Our method)', 
                color='green')
        
        # Plot random selected samples
        ax.scatter(features_dict[feature_names[0]][random_indices], 
                features_dict[feature_names[1]][random_indices], 
                alpha=0.8, label='Selected samples (Random)', 
                color='red')
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend()
        plt.show()



def main():
    config = SelectorConfig(
        features_to_use=['jerk', 'jitter', 'velocity', 'range'],
        data_percentage=0.2,
        diversity_weight=0.5,
        use_normalization=True
    )
    
    selector = GestureSelector(config)
    
    try:
        data = np.load('dataset/blind_user/data_20_120.npy')
        labels = np.load('dataset/blind_user/label_20_120.npy')

        print("Data shape:", data.shape)
        print("Labels shape:", labels.shape)

        # Print original class distribution
        unique_labels, counts = np.unique(labels[:, 0, 0], return_counts=True)
        print("\nOriginal class distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Class {label}: {count} samples")

        # Select samples
        selected_indices, features_dict = selector.select_samples(data, labels)
        
        print(f"\nSelected {len(selected_indices)} samples")
        if len(selected_indices) > 0:
            labels_tensor = torch.from_numpy(labels[:, 0, 0]).long()
            print("Class distribution of selected samples:")
            unique_labels, counts = torch.unique(labels_tensor[selected_indices], return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"Class {label.item()}: {count.item()} samples")
        
            # Visualize results
            """selector.visualize_selection(
                torch.from_numpy(data).float(),
                labels_tensor,
                selected_indices,
                features_dict
            )"""
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        
if __name__ == "__main__":
    main()