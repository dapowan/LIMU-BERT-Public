import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from typing import Dict, List, Tuple, Union


# In semantic_utils.py
def prepare_bert_embeddings(
    activity_labels: np.ndarray,  # shape: (n,)
    label_to_activity: Dict[int, str],
    output_dim: int = 72,
    bert_model_name: str = 'bert-base-uncased'
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Prepares BERT embeddings for activity recognition.
    """
    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertModel.from_pretrained(bert_model_name)
    model.eval()

    # Get unique activity labels
    unique_labels = np.unique(activity_labels)
    activities = [label_to_activity[label] for label in unique_labels]
    
    # Create activity relationships dict
    activity_relationships = {}
    
    bert_embeddings = []
    with torch.no_grad():
        for activity in activities:
            inputs = tokenizer(activity, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            bert_embeddings.append(embedding)
    
    # Stack and normalize
    bert_embeddings = torch.stack(bert_embeddings)  # Shape: [num_activities, 768]
    bert_embeddings = normalize(bert_embeddings.numpy())
    bert_embeddings = torch.FloatTensor(bert_embeddings)
    
    # Project to lower dimension
    projection = torch.nn.Linear(768, output_dim)
    torch.nn.init.xavier_uniform_(projection.weight)
    projected_embeddings = projection(bert_embeddings)  # Shape: [num_activities, output_dim]
    
    # Add sequence length dimension for attention
    projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: [num_activities, 1, output_dim]
    
    # Calculate similarity matrix
    similarity_matrix = torch.nn.functional.cosine_similarity(
        projected_embeddings,
        projected_embeddings.transpose(0, 1),
        dim=2
    )
    
    # Create activity relationships dictionary
    for i, activity in enumerate(activities):
        similarities = similarity_matrix[i]
        most_similar = torch.argsort(similarities, descending=True)[1:4]  # Top 3 similar
        activity_relationships[activity] = {
            'most_similar': [activities[j] for j in most_similar],
            'similarity_scores': similarities[most_similar].tolist()
        }
    print("Projected embeddings shape:", projected_embeddings.shape)
    print("Similarity matrix shape:", similarity_matrix.shape)
    print("Number of activities:", len(activities))
    return projected_embeddings.squeeze(1), similarity_matrix, activity_relationships