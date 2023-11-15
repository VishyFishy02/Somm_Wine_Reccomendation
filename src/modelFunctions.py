from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import math
from tqdm import tqdm
import faiss
import numpy as np
import pandas as pd

def load_embeddings_and_wine():
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    # Construct the path to the CSV file
    data_file = current_dir.parent / 'Data/Raw Data/winemag-data-130k-v2.csv'
    # Read the CSV file
    wine = pd.read_csv(data_file, encoding='utf-8', on_bad_lines='skip')
    wine.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    # # KNN Search with Faiss
    vector_file = current_dir.parent / 'Data/description_embeddings.pt'
    knn_vectors = torch.load(vector_file)
    knn_labels = wine.id.astype(str).tolist()

    # Convert the data to NumPy arrays for use with faiss
    vectors_np = knn_vectors.numpy()
    # Determine the dimension of the vectors
    dimension = vectors_np.shape[1]  # This represents the dimension of the vectors
    faiss.normalize_L2(vectors_np)

    # Build the Faiss index
    #index = faiss.IndexFlatL2(dimension)  # Create a Faiss index with L2 (Euclidean) distance metric
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)  # Add the data vectors to the index
    
    return wine, index 

def get_predictions(query_text):
    # Encode the text query to obtain a vector
    query_vector = model.encode([query_text])[0]

    # Query the index with the vector
    k = 10  # Number of neighbors to return

    # Perform a nearest neighbor search to find the closest neighbors to the query vector(s).
    # D will contain squared L2 distances between the query vector and its neighbors.
    # I will contain the indices of the nearest neighbors in the dataset.
    D, I = index.search(np.array([query_vector]), k)
    similarity_scores = D[0]
    
    # Get labels of the neighbors
    # neighbor_labels = [knn_labels[i] for i in I[0]]
    
    # Extract rows from the wine DataFrame
    result = wine.iloc[I[0]]
    result_df = result[['title', 'description', 'variety', 'province']]
    result_df['Similarity Score'] = similarity_scores
    result_df = result_df.rename(columns={
        'title': 'Wine Name',
        'description': 'Description',
        'province': 'Province',
        'variety': 'Grape Variety'
    })

    return result_df

# Initialize a sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
wine, index = load_embeddings_and_wine()
