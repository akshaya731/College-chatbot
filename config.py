import os
import sys
from pathlib import Path

# Ensure Windows stdout/stderr uses UTF-8 to prevent charmap encoding errors
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# College Identity & Metadata
COLLEGE_NAME = "Mother Teresa Institute of Science and Technology (MIST)"
COLLEGE_SHORT_NAME = "MIST"
COLLEGE_ESTD = "2001 (by MCM Educational Society)"
COLLEGE_ACCREDITATION = "Autonomous, NAAC A+, UGC & AICTE Approved, JNTUH Affiliated"
COLLEGE_LOCATION = "Sanketika Nagar, Kothuru, Sathupally, Khammam District, Telangana - 507303"
COLLEGE_PHONE = "+91 9494641251"
COLLEGE_EMAIL = "info@mist.edu.in"
COLLEGE_WEBSITE = "https://mist.edu.in"

# Vector Database & Embedding settings
CHROMA_COLLECTION_NAME = "langchain"  # maintains compatibility with existing indexed store
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Text Splitting Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM & Retrieval Configuration
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b"
]
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.2
