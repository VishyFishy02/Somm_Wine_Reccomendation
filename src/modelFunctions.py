import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import math
from tqdm import tqdm
import faiss
import numpy as np

wine = pd.read_csv('winemag-data-130k-v2.csv', encoding='utf-8', on_bad_lines='skip')
wine.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

# # Generating and Exporting Description Embeddings to a Tensor File

# This code was used to create the description embeddings tensor file.
# You don't have to run it again.
# It takes about an hour to run.

# # KNN Search with Faiss

knn_vectors = torch.load("description_embeddings.pt")
knn_labels = wine.id.astype(str).tolist()

# Convert the data to NumPy arrays for use with faiss
vectors_np = knn_vectors.numpy()

# Determine the dimension of the vectors
dimension = vectors_np.shape[1]  # This represents the dimension of the vectors

# Build the Faiss index
index = faiss.IndexFlatL2(dimension)  # Create a Faiss index with L2 (Euclidean) distance metric
index.add(vectors_np)  # Add the data vectors to the index

# Initialize a sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode the text query to obtain a vector
query_text = 'melon and bitter'
query_vector = model.encode([query_text])[0]

# Query the index with the vector
k = 10  # Number of neighbors to return

# Perform a nearest neighbor search to find the closest neighbors to the query vector(s).
# D will contain squared L2 distances between the query vector and its neighbors.
# I will contain the indices of the nearest neighbors in the dataset.
D, I = index.search(np.array([query_vector]), k)

# Get labels of the neighbors
neighbor_labels = [knn_labels[i] for i in I[0]]

# Extract rows from the wine DataFrame
result = wine.iloc[I[0]]

result['description'].values