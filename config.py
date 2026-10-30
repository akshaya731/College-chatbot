
import os

# Project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

print("College RAG FAQ Bot Configuration Loaded!")