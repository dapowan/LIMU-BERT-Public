import torch
import torch.nn.functional as F

class SemanticAwareLoss:
    """
    A class that computes and manages losses combining IMU reconstruction
    and semantic consistency.
    
    This class handles:
    1. Traditional reconstruction loss for IMU signals
    2. Semantic consistency loss using BERT embeddings
    3. Weighted combination of both losses
    
    The class structure allows easy integration with existing training loops
    while keeping loss computation modular.
    """
    def __init__(self, alpha=0.7, beta=0.3, device='cuda'):
        self.alpha = alpha  # weight for reconstruction loss
        self.beta = beta    # weight for semantic loss
        self.device = device
        
    def compute_loss(self, model_output, target_seqs, semantic_info=None):
        """
        Computes the combined loss or just reconstruction loss if no semantic info.
        
        Args:
            model_output: Model's predicted sequences
            target_seqs: Ground truth sequences
            semantic_info: Tuple of (embeddings, similarity_matrix) or None
        
        Returns:
            torch.Tensor: The computed loss
            dict: Loss components for monitoring
        """
        # Basic reconstruction loss
        recon_loss = F.mse_loss(model_output, target_seqs, reduction='none')
        
        # If no semantic information, return only reconstruction loss
        if semantic_info is None:
            return recon_loss, {'total': recon_loss.mean().item(),
                              'reconstruction': recon_loss.mean().item(),
                              'semantic': 0.0}
        
        semantic_embeds, similarity_matrix = semantic_info
        semantic_embeds = semantic_embeds.to(self.device)
        similarity_matrix = similarity_matrix.to(self.device)
        
        # Compute current embeddings similarities
        model_embeddings = model_output.mean(dim=1)
        current_similarities = F.cosine_similarity(
            model_embeddings.unsqueeze(1),
            model_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Semantic consistency loss
        semantic_loss = F.mse_loss(
            current_similarities,
            similarity_matrix[None].expand(model_output.size(0), -1, -1),
            reduction='none'
        )
        
        # Combine losses
        total_loss = self.alpha * recon_loss + self.beta * semantic_loss
        
        loss_components = {
            'total': total_loss.mean().item(),
            'reconstruction': recon_loss.mean().item(),
            'semantic': semantic_loss.mean().item()
        }
        
        return total_loss, loss_components