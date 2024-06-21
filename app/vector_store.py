import json
import numpy as np

VECTOR_STORE_FILEPATH = 'data/vector_store.json'

def cosine_similarity(query_vector, vectors):
    """Calculates the cosine similarity between a query vector and a list of vectors."""
    query_vector = np.array(query_vector)
    vectors = np.array(vectors)
    return np.dot(vectors, query_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector))

class VectorStore:
    def __init__(self):
        self.store = []

    def reset(self):
        self.store = []
    
    def add(self, vectors):
        self.store.extend(vectors)
    
    def save(self, file_path=VECTOR_STORE_FILEPATH):
        with open(file_path, 'w') as f:
            json.dump(self.store, f)

    def load(self, file_path=VECTOR_STORE_FILEPATH):
        with open(file_path, 'r') as f:
            self.store = json.load(f)
    
    def query(self, vector, top_k=5):
        vectors = [v['vector'] for v in self.store]
        similarities = cosine_similarity(vector, vectors)
        print('Shape', similarities.shape)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{**self.store[i], 'score': similarities[i]} for i in top_k_indices]
