import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Union

def prepare_bert_embeddings(
    activity_labels: np.ndarray,
    label_to_activity: Dict[int, str],
    output_dim: int = 72,
    bert_model_name: str = 'bert-base-uncased'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares BERT embeddings for activity recognition by processing labels 
    and projecting them to a lower dimensional space compatible with IMU embeddings.
    
    This function performs several key operations:
    1. Converts numeric activity labels to text descriptions using provided mapping
    2. Generates BERT embeddings for each activity description
    3. Projects the 768-dimensional BERT embeddings down to the specified output dimension
    4. Creates a similarity matrix between activities based on their semantic relationships
    
    Args:
        activity_labels (np.ndarray): Array of activity label indices from the dataset
        label_to_activity (Dict[int, str]): Mapping from label indices to activity descriptions
                                          e.g., {0: "walking", 1: "running", ...}
        output_dim (int): Desired dimensionality of the output embeddings (default: 72)
        bert_model_name (str): Name of the BERT model to use (default: 'bert-base-uncased')
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - projected_embeddings: Tensor of shape (num_activities, output_dim) 
              containing the reduced-dimension embeddings for each activity
            - similarity_matrix: Tensor of shape (num_activities, num_activities)
              containing pairwise semantic similarities between activities
    
    Example:
        >>> labels = np.array([0, 1, 2])
        >>> label_map = {0: "walking", 1: "running", 2: "jumping"}
        >>> embeddings, similarities = prepare_bert_embeddings(labels, label_map)
    
    Notes:
        - The function uses the [CLS] token embedding as the sentence representation
        - Embeddings are L2 normalized before projection to ensure consistent scales
        - The projection matrix is initialized using Xavier initialization
        - The similarity matrix uses cosine similarity between the original BERT embeddings
    """
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertModel.from_pretrained(bert_model_name)
    model.eval()  # Set to evaluation mode

    # Get unique activity labels
    unique_labels = np.unique(activity_labels)
    
    # Convert labels to activity descriptions
    activities = [label_to_activity[label] for label in unique_labels]
    
    # Generate BERT embeddings
    bert_embeddings = []
    with torch.no_grad():
        for activity in activities:
            # Tokenize and get BERT embedding
            inputs = tokenizer(activity, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            bert_embeddings.append(embedding)
    
    # Stack embeddings and normalize
    bert_embeddings = torch.stack(bert_embeddings)
    bert_embeddings = normalize(bert_embeddings.numpy())
    bert_embeddings = torch.FloatTensor(bert_embeddings)
    
    # Create projection matrix from 768 to output_dim
    projection = torch.nn.Linear(768, output_dim)
    torch.nn.init.xavier_uniform_(projection.weight)
    
    # Project embeddings to lower dimension
    projected_embeddings = projection(bert_embeddings)
    
    # Calculate similarity matrix using cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(
        bert_embeddings.unsqueeze(1),
        bert_embeddings.unsqueeze(0),
        dim=2
    )
    
    return projected_embeddings, similarity_matrix


