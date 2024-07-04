# Retrieval Augmented Generation from Scratch: Inception RAG

This repository contains the code for a simple Retrieval Augmented Generation (RAG) system built from scratch using Python and open-source tools. It uses Llama 3 as a language model, powered by Groq, and Nomic embeddings. It includes Christopher Nolan's Inception movie script as a source document in the `data/docs` folder to build the RAG app, but you can add any documents of your choice.

For a detailed explanation of the code and the concepts behind RAG, check out [this blog post](https://codeawake.com/blog/rag-from-scratch).

This project was developed by [CodeAwake](https://codeawake.com).

## Structure

The code is contained in the `app/` folder and organized into the following files:

- `config.py`: Configuration settings for the application.
- `loader.py`: Loads and processes the pdf source documents for RAG (chunking, embedding and storing in vector store).
- `splitter.py`: Text chunking functionality.
- `vector_store.py`: Simple vector store implementation.
- `rag.py`: Core RAG functionality and interactive Q&A loop.

## Installation

### Prerequisites ✅

- Python 3.11 or higher
- Poetry (Python package manager)

### Instructions

1. Install the dependencies using Poetry:

    ```bash
    poetry install
    ```

2. Create a `.env` file in the backend folder copying the `.env.example` file provided and set the required environment variable:
    - `GROQ_API_KEY`: Your Groq API key.
  
## Running the Application

To load the source pdf documents and process them for RAG:

```bash
poetry run load-docs
```

To run the interactive Q&A RAG app:

```bash
poetry run rag
```