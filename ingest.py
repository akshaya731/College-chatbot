import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from functools import lru_cache

# Ensure Windows stdout/stderr uses UTF-8 to prevent charmap encoding crashes
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import chromadb

from config import (
    DOCUMENTS_DIR,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


@lru_cache(maxsize=1)
def get_embedding_function():
    """Returns one reusable Hugging Face embedding instance per app process."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_database(
    docs_dir: Optional[Path] = None,
    db_dir: Optional[Path] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Ingests PDF documents from docs_dir, chunks them, computes embeddings,
    and stores them into ChromaDB in a Windows-safe, process-safe manner.
    """
    start_time = time.time()
    docs_path = Path(docs_dir) if docs_dir else Path(DOCUMENTS_DIR)
    db_path = str(db_dir) if db_dir else str(CHROMA_DB_DIR)

    if progress_callback:
        progress_callback("Scanning documents directory...", 0.1)

    if not docs_path.exists():
        return {
            "success": False,
            "error": f"Documents folder '{docs_path}' does not exist.",
            "documents_count": 0,
            "chunks_count": 0,
            "elapsed_seconds": 0,
        }

    pdf_files = sorted(list(docs_path.glob("*.pdf")))
    if not pdf_files:
        return {
            "success": False,
            "error": f"No PDF documents found in '{docs_path}'.",
            "documents_count": 0,
            "chunks_count": 0,
            "elapsed_seconds": 0,
        }

    # Step 1: Load documents
    loaded_documents = []
    file_details = []
    total_files = len(pdf_files)

    for idx, pdf_path in enumerate(pdf_files):
        try:
            if progress_callback:
                progress_callback(
                    f"Loading ({idx + 1}/{total_files}): {pdf_path.name}",
                    0.1 + 0.3 * ((idx + 1) / total_files),
                )
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Clean null bytes or rogue control characters from text
            for doc in docs:
                doc.page_content = doc.page_content.replace("\x00", "")
                doc.metadata["source_filename"] = pdf_path.name
            loaded_documents.extend(docs)
            file_details.append({"filename": pdf_path.name, "pages": len(docs), "status": "loaded"})
        except Exception as e:
            file_details.append({"filename": pdf_path.name, "pages": 0, "status": f"error: {str(e)}"})

    if not loaded_documents:
        return {
            "success": False,
            "error": "Failed to extract text from any PDF documents.",
            "documents_count": 0,
            "chunks_count": 0,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "files": file_details,
        }

    failed_files = [item for item in file_details if item["status"] != "loaded"]
    if failed_files:
        failure_summary = "; ".join(
            f"{item['filename']}: {item['status']}" for item in failed_files
        )
        return {
            "success": False,
            "error": (
                "Indexing was cancelled because one or more PDFs could not be read. "
                "The existing vector index was left unchanged. " + failure_summary
            ),
            "documents_count": sum(1 for item in file_details if item["status"] == "loaded"),
            "chunks_count": 0,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "files": file_details,
        }

    # Step 2: Split documents into chunks
    if progress_callback:
        progress_callback("Splitting documents into optimized chunks...", 0.45)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(loaded_documents)

    # Step 3: Embeddings
    if progress_callback:
        progress_callback("Initializing embedding model...", 0.6)

    embeddings = get_embedding_function()

    # Step 4: Safely reset or recreate collection in ChromaDB without Windows file locking crashes
    if progress_callback:
        progress_callback("Updating Chroma vector store...", 0.75)

    try:
        client = chromadb.PersistentClient(path=db_path)
        existing_collections = [c.name for c in client.list_collections()]
        if CHROMA_COLLECTION_NAME in existing_collections:
            client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception as reset_err:
        # If collection deletion fails or folder is fresh, continue
        pass

    # Step 5: Index documents
    if progress_callback:
        progress_callback(f"Generating embeddings for {len(chunks)} chunks...", 0.85)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    elapsed = round(time.time() - start_time, 2)
    if progress_callback:
        progress_callback(f"Knowledge base ready! ({len(chunks)} chunks in {elapsed}s)", 1.0)

    return {
        "success": True,
        "documents_count": len(pdf_files),
        "chunks_count": len(chunks),
        "elapsed_seconds": elapsed,
        "files": file_details,
    }


def create_vector_database():
    """CLI entrypoint for building the vector database."""
    print("=" * 60)
    print("CampusConnect AI - Building MIST Vector Knowledge Base")
    print("=" * 60)

    result = build_vector_database(
        progress_callback=lambda msg, pct: print(f"[{int(pct * 100):3d}%] {msg}")
    )

    if result["success"]:
        print(f"\nSUCCESS: Indexed {result['documents_count']} documents into {result['chunks_count']} chunks.")
        print(f"Elapsed time: {result['elapsed_seconds']} seconds.")
        print(f"Storage path: {CHROMA_DB_DIR}")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error during ingestion.')}")


if __name__ == "__main__":
    create_vector_database()
