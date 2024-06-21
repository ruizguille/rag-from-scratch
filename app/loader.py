import os
from tqdm import tqdm
from pdfminer.high_level import extract_text

DATA_DIR = 'data'

def load_docs():
    docs = []
    for filename in tqdm(os.listdir(DATA_DIR)):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DATA_DIR, filename)
            text = extract_text(file_path)
            docs.append(text)

    print(f'Loaded {len(docs)} PDF documents')
    return docs
