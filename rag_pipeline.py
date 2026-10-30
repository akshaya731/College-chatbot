
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from config import CHROMA_DB_DIR, EMBEDDING_MODEL


# Load API key from the project's .env file
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Number of relevant document chunks
TOP_K = 3


def initialize_rag():

    # Check API key
    if not GROQ_API_KEY:
        raise ValueError(
            "Groq API key not found. Please check your .env file."
        )

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Connect to existing ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # Retrieve relevant document chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    # Connect to Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=GROQ_API_KEY,
        temperature=0.2
    )

    return retriever, llm


def get_answer(question):

    if not question or not question.strip():
        return "Please enter a question."

    try:
        # Initialize RAG components
        retriever, llm = initialize_rag()

        # Search college documents
        documents = retriever.invoke(question)

        if not documents:
            return (
                "Sorry, I could not find relevant information "
                "in the college documents."
            )

        # Combine retrieved content
        context = "\n\n".join(
            document.page_content for document in documents
        )

        # Prepare prompt
        prompt = f"""
You are a helpful College FAQ Assistant.

Answer the student's question using ONLY the information
provided in the college document context below.

Rules:
1. Do not invent college information.
2. Give clear and simple answers.
3. If the answer is not available in the context,
   say that the information is not available in the college documents.

COLLEGE DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

        # Generate answer
        response = llm.invoke(prompt)

        return response.content

    except Exception as error:
        print("RAG Pipeline Error:", error)
        return (
            "Sorry, something went wrong while generating your answer. "
            "Please check the terminal for the error details."
        )
