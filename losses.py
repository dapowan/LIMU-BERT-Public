import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_enhanced_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes enhanced similarity between embeddings using both cosine similarity
    and relative distances. This provides a more robust measure of semantic relationships.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        
    Returns:
        Tensor of shape (batch_size, batch_size) containing pairwise similarities
    """
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        dim=2
    )
    
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # Normalize distances to [0, 1]
    distances = distances / distances.max()
    
    # Combine both metrics (higher value means more similar)
    similarity = (cos_sim + (1 - distances)) / 2
    
    return similarity


class SemanticAwareLoss:
    """
    Enhanced loss computation with normalized semantic loss and relationship monitoring.
    """
    def __init__(self, alpha=0.7, device='cuda'):
        self.alpha = alpha
        self.device = device
        self.running_stats = {
            'semantic_min': float('inf'),
            'semantic_max': float('-inf'),
            'reconstruction_values': [],
            'semantic_values': []
        }
    
    def normalize_semantic_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Normalizes semantic loss to [0, 1] range using running statistics.
        """
        current_min = min(self.running_stats['semantic_min'], loss.min().item())
        current_max = max(self.running_stats['semantic_max'], loss.max().item())
        
        self.running_stats['semantic_min'] = current_min
        self.running_stats['semantic_max'] = current_max
        
        # Avoid division by zero
        if current_max == current_min:
            return torch.zeros_like(loss)
            
        return (loss - current_min) / (current_max - current_min)
    
    def compute_loss(
        self,
        model_output: torch.Tensor,
        target_seqs: torch.Tensor,
        semantic_info: Tuple[torch.Tensor, torch.Tensor, Dict]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes combined loss with semantic awareness.
        """
        # Reconstruction loss
        print("Model output shape:", model_output.shape)
        print("Target seqs shape:", target_seqs.shape)
        print("Semantic embeds shape:", semantic_info[0].shape)
        print("Similarity matrix shape:", semantic_info[1].shape)
        recon_loss = F.mse_loss(model_output, target_seqs)
        
        if semantic_info is None:
            return recon_loss, self._create_loss_stats(recon_loss)
        
        semantic_embeds, similarity_matrix, relationships = semantic_info
        
        # Ensure tensors are on correct device
        semantic_embeds = semantic_embeds.to(self.device).squeeze(1)  # [num_activities, embedding_dim]
        similarity_matrix = similarity_matrix.to(self.device)
        
        # Compute embeddings for current batch sequences
        # Average the model output across masked positions to get sequence-level representation
        batch_embeds = model_output.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Compute semantic regularization loss
        # Option 1: Compare batch-level embedding distribution to global semantic relationships
        try:
            # Compute global embedding statistics from batch
            global_embedding_mean = batch_embeds.mean(dim=0)
            global_embedding_std = batch_embeds.std(dim=0)
            
            # Convert similarity matrix to a loss-compatible format
            target_stats = torch.tensor([
                similarity_matrix.mean(),
                similarity_matrix.std()
            ], device=self.device)
            
            current_stats = torch.tensor([
                global_embedding_mean.mean(),
                global_embedding_std.mean()
            ], device=self.device)
            
            # Compute loss between global statistics
            semantic_loss = F.mse_loss(current_stats, target_stats)
        except Exception as e:
            print(f"Semantic loss computation error: {e}")
            semantic_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses with adaptive weighting
        total_loss = self.alpha * recon_loss + (1 - self.alpha) * semantic_loss
        
        # Create loss statistics
        loss_stats = self._create_loss_stats(
            torch.tensor(recon_loss), 
            torch.tensor(semantic_loss)
        )
        
        return total_loss, loss_stats

    
    def _create_loss_stats(
        self,
        recon_loss: torch.Tensor,
        semantic_loss: torch.Tensor = None
    ) -> Dict:
        """
        Creates detailed loss statistics for monitoring.
        """
        stats = {
            'reconstruction': recon_loss.mean().item(),
            'reconstruction_std': recon_loss.std().item(),
        }
        
        if semantic_loss is not None:
            stats.update({
                'semantic': semantic_loss.mean().item(),
                'semantic_std': semantic_loss.std().item(),
                'semantic_min': self.running_stats['semantic_min'],
                'semantic_max': self.running_stats['semantic_max']
            })
        
        return stats