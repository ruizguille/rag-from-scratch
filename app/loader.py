import os
from tqdm import tqdm
from pdfminer.high_level import extract_text
from nomic import embed
from app.config import settings
from app.splitter import TextSplitter
from app.vector_store import VectorStore

DOCS_DIR = 'data/docs'

def main():
    docs = []
    for filename in tqdm(os.listdir(DOCS_DIR)):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DOCS_DIR, filename)
            text = extract_text(file_path)
            docs.append(text)
    print(f'Loaded {len(docs)} PDF documents')

    chunks = []
    text_splitter = TextSplitter(chunk_size=512)
    print('\nSplitting documents into chunks')
    for i, doc in enumerate(docs):
        doc_chunks = text_splitter(doc)
        chunks += doc_chunks
        print(f'Doc {i+1}: {len(doc_chunks)} chunks')
    print('Total chunks:', len(chunks))

    embed_res = embed.text(
        texts=chunks,
        model='nomic-embed-text-v1.5',
        task_type='search_document',
        inference_mode=settings.NOMIC_INFERENCE_MODE
    )
    print(f'\nCreated {len(embed_res['embeddings'])} vector embeddings, '
          f'{embed_res['usage']['total_tokens']} total tokens')
    
    vector_store = VectorStore()
    vectors = [
        {'vector': vector, 'text': text} for vector, text in zip(embed_res['embeddings'], chunks)
    ]
    vector_store.add(vectors)
    print(f'{len(vectors)} vector embeddings added to vector store')
    vector_store.save()
