from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DOCUMENTS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL
import os

def create_vector_database():

    documents = []

    for file in os.listdir(DOCUMENTS_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DOCUMENTS_DIR, file)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
            print(f"Loaded: {file}")

    if not documents:
        print("No PDF documents found!")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("Vector database created successfully!")

if __name__ == "__main__":
    create_vector_database()
