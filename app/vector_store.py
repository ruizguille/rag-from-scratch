import json
import numpy as np

VECTOR_STORE_FILEPATH = 'data/vector_store.json'

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