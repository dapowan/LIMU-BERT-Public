import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define movement classes
movement_classes = [
    "Forearm up", "Forearm down", "Forearm left", "Forearm right",
    "Rotate wrist & move arm right", "Rotate wrist & move arm left",
    "Flick and forearm up", "Flick and forearm down", 
    "Flick and forearm left", "Flick and forearm right",
    "Square", "Circle", "Triangle", "Question mark", "Infinity"
]

# Get BERT embeddings
embeddings = []

for movement in movement_classes:
    inputs = tokenizer(movement, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# PCA
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)
embeddings_2d = pca_2d.fit_transform(embeddings)
embeddings_3d = pca_3d.fit_transform(embeddings)

# 2D Plot
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, txt in enumerate(movement_classes):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.title("2D PCA of BERT Embeddings for Movement Classes")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
for i, txt in enumerate(movement_classes):
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], txt)
ax.set_title("3D PCA of BERT Embeddings for Movement Classes")
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
plt.tight_layout()
plt.show()

# Print explained variance
print("2D PCA explained variance ratio:", pca_2d.explained_variance_ratio_)
print("3D PCA explained variance ratio:", pca_3d.explained_variance_ratio_)
