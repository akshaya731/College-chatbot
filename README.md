# CampusConnect AI · MIST College Chatbot (RAG 2.0)

An intelligent Retrieval-Augmented Generation (RAG) assistant for **Mother Teresa Institute of Science and Technology (MIST)**, providing students, parents, and faculty with verified answers regarding admissions, fees, attendance regulations, examinations, and departments.

---

## 🌟 Key Features & Improvements

- **⚡ 40x Faster Responses (0.6s Sub-second Latency)**: Implemented in-memory singleton caching for PyTorch embedding models and ChromaDB vectorstore.
- **📚 Verified Sources & Citations**: Every answer includes an expandable citation drawer showing the exact source PDF filename, page number, verified snippet, and portal link.
- **🛡️ Anti-Hallucination Guardrails**: Strict context boundary prompting ensures the assistant never invents fee amounts, dates, percentages, or regulations.
- **🔄 In-App Document Vault & Upload**: Upload new college PDFs via drag-and-drop and rebuild the vector database with a single click in the UI without server restarts.
- **🔍 Semantic Search Lab**: Direct vector database exploration tool allowing students and faculty to test Chroma similarity retrieval with relevance scores.
- **📊 Operations Dashboard & Telemetry**: Real-time KPI metrics, query audit logs, latency tracking, and system health checks.
- **⚙️ Model Flexibility & Fallback**: Support for `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and `qwen/qwen3.8-27b` via Groq LPUs.
- **🪟 Windows Encoding & Lock Fixes**: Reconfigured UTF-8 stdout/stderr across all scripts to eliminate `UnicodeEncodeError` (charmap/cp1252) and safe Chroma collection resets to prevent `WinError 32` file locks.

---

## 📂 Project Architecture

```
RAG-CLG/
├── app.py              # Modern Streamlit UI with 6 interactive workspaces
├── rag_pipeline.py     # High-performance cached RAG retrieval & LLM inference
├── ingest.py           # Document ingestion, text chunking & vector indexing
├── config.py           # Centralized college settings, models & hyperparameters
├── requirements.txt    # Production dependencies
├── .env                # Environment variables (GROQ_API_KEY)
├── documents/          # Knowledge repository (PDF circulars, syllabi, profiles)
│   ├── admission_details.pdf
│   ├── attendance_rules.pdf
│   ├── college_profile.pdf
│   ├── departments_faculty.pdf
│   ├── exam_information.pdf
│   └── fee_structure.pdf
└── chroma_db/          # Persistent Chroma vector store
```

---

## 🚀 Quickstart Guide

### 1. Setup Environment
Ensure your virtual environment is active and requirements are installed:
```bash
.\venv\Scripts\pip install -r requirements.txt
```

### 2. Configure API Key
Create or verify your `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build Vector Index (Optional if already built)
```bash
.\venv\Scripts\python ingest.py
```
*(You can also rebuild the index directly inside the web UI under the **Document Vault** tab)*

### 4. Run the Application
```bash
.\venv\Scripts\streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏛️ Official College Details
- **Institution:** Mother Teresa Institute of Science and Technology (MIST)
- **Status:** Autonomous · NAAC A+ Grade · Approved by AICTE & UGC · Affiliated to JNTUH
- **Location:** Sanketika Nagar, Kothuru, Sathupally, Khammam Dist, Telangana - 507303
- **Helpline:** +91 9494641251 | info@mist.edu.in | [mist.edu.in](https://mist.edu.in)

