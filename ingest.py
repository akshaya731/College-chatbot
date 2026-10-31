
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import DOCUMENTS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL

import os
import shutil


def create_vector_database():

    documents = []

    # Step 1: Load all PDF documents
    for file in os.listdir(DOCUMENTS_DIR):

        if file.lower().endswith(".pdf"):

            path = os.path.join(DOCUMENTS_DIR, file)

            loader = PyPDFLoader(path)
            loaded_docs = loader.load()

            documents.extend(loaded_docs)

            print(f"Loaded: {file}")

    if not documents:
        print("No PDF documents found!")
        return

    # Step 2: Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Step 4: Remove old sample database
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print("Old vector database removed!")

    # Step 5: Create a fresh vector database
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("MIST vector database created successfully!")


if __name__ == "__main__":
    create_vector_database()
