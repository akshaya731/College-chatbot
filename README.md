# MIST Connect AI

**An AI-powered college information assistant built with Retrieval-Augmented Generation (RAG).**

MIST Connect AI helps students and visitors find information about **Mother Teresa Institute of Science and Technology (MIST)** using a knowledge base of college PDF documents. It retrieves relevant passages, generates answers through Groq, and displays source document and page references.

<p align="center">
  <a href="https://college-chatbot-ggpsptmesybaxdfa98kwpm.streamlit.app/"><strong>Open the live application →</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/akshaya731/College-chatbot"><strong>GitHub repository</strong></a>
</p>

![Python](https://img.shields.io/badge/Python-3.x-blue) ![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![RAG](https://img.shields.io/badge/Architecture-RAG-6C63FF) ![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-2E8B57)

---

## Overview

Students often need to search multiple college documents to find answers about admissions, fees, attendance, examinations, departments and institutional information. MIST Connect AI provides a single chat interface for asking those questions in natural language.

The assistant is designed to ground college-specific answers in supplied PDF documents. When the knowledge base does not contain enough information, the prompt instructs the model to say so and direct the student to the college office rather than inventing college policies.

> **Important:** This is a student project and information assistant, not an official college policy or notification channel. Answers depend on the accuracy and freshness of the PDFs. Confirm important or time-sensitive matters with the college administration.

## Features

| Feature | Description |
|---|---|
| **AI Assistant** | Natural-language questions with streamed answers. |
| **Document-grounded responses** | Retrieves relevant passages from the college PDF knowledge base. |
| **Source references** | Shows source PDF names, page numbers and excerpts. |
| **Follow-up context** | Uses recent messages in the current chat session to interpret short follow-up questions. |
| **Greeting fast path** | Handles common greetings locally without a language-model call. |
| **College Dashboard** | Displays configured college information and quick links to academic questions. |
| **Document Vault** | Lists and previews bundled PDFs; uploads and re-indexing require administrator access. |
| **Semantic Search Lab** | Runs vector searches and displays retrieved passages and relevance scores. |
| **Usage Analytics** | Shows session-level query counts, response latency and query log. |
| **System Health** | Reports Groq configuration, PDF count and vector database status. |
| **Chat export and feedback** | Exports the current conversation and allows response ratings. |

## Technology stack

- **Frontend:** Streamlit
- **LLM integration:** LangChain and langchain-groq
- **Language model provider:** Groq
- **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** ChromaDB
- **PDF processing:** PyPDF and LangChain PDF loaders
- **Text chunking:** LangChain RecursiveCharacterTextSplitter
- **Configuration:** environment variables, `.env` locally, and Streamlit Secrets in the cloud

## Architecture

```text
College PDF files
      ↓
PDF loading and text extraction
      ↓
Text chunking with overlap
      ↓
Hugging Face embeddings
      ↓
ChromaDB vector index

Student question
      ↓
Similarity retrieval + recent session context
      ↓
Grounded prompt
      ↓
Groq language model
      ↓
Streamed answer + source references
```

## Repository structure

```text
College-chatbot/
├── .github/workflows/python-check.yml  # Syntax and TOML checks
├── .streamlit/config.toml              # Streamlit theme and configuration
├── documents/                          # College PDF knowledge base
│   ├── admission_details.pdf
│   ├── attendance_rules.pdf
│   ├── college_profile.pdf
│   ├── departments_faculty.pdf
│   ├── exam_information.pdf
│   └── fee_structure.pdf
├── app.py                              # Streamlit user interface
├── config.py                           # Paths, college metadata and model settings
├── ingest.py                           # PDF loading, chunking and vector indexing
├── rag_pipeline.py                     # Retrieval, prompts and LLM calls
├── requirements.txt                    # Python dependencies
├── README.md
└── run_app.bat                         # Windows launch helper
```

The `chroma_db/` directory is created at runtime and excluded from Git. The vector index can be rebuilt from PDFs in `documents/`.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/akshaya731/College-chatbot.git
cd College-chatbot
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
ADMIN_PASSWORD=choose_a_long_unique_password
```

Keep this file private. It is excluded by `.gitignore`. Never put real API keys or passwords in source code, screenshots, issues or commits.

### 5. Build the vector index

```bash
python ingest.py
```

The app can also initialize an empty ChromaDB collection from the bundled PDFs on startup. Running the ingestion script explicitly is useful after changing the PDF knowledge base.

### 6. Start the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. On Windows, `run_app.bat` can also be used after creating the virtual environment and installing dependencies.

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, then choose this repository, the `main` branch and `app.py`.
4. In **Advanced settings → Secrets** (or the deployed app's **Settings → Secrets**), add:

```toml
GROQ_API_KEY = "your_groq_api_key"
ADMIN_PASSWORD = "choose_a_long_unique_password"
```

5. Deploy the app and open the generated `streamlit.app` URL.

### Deployment notes

- Do not upload `.env`, `.streamlit/secrets.toml`, API keys or passwords to GitHub.
- The Document Vault is read-only until `ADMIN_PASSWORD` is configured. The password unlocks upload and re-indexing controls for the current session.
- Streamlit Community Cloud's local filesystem is not durable storage. Files uploaded through the app may be lost when the app restarts or rebuilds. For durable updates, commit approved PDFs to `documents/` or use persistent object storage.
- The ChromaDB index is generated in the app environment from the repository PDFs.
- Hosted apps may sleep after inactivity, so the first visit after a quiet period can take longer.

## Updating the knowledge base

1. Review and approve the new college PDF.
2. Add or replace the relevant file in `documents/`.
3. Run `python ingest.py` locally, or deploy the updated repository so the hosted app can rebuild its index.
4. Test representative questions and check the displayed source filename and page.

The ingestion script stops when a PDF cannot be read or no searchable chunks are produced. Review the console or app logs if indexing fails.

## Security and data handling

- Store `GROQ_API_KEY` and `ADMIN_PASSWORD` in local environment variables or Streamlit Secrets.
- Keep `.env`, `.streamlit/secrets.toml`, `.venv/`, `__pycache__/` and `chroma_db/` out of Git.
- Restrict administrator credentials to trusted maintainers.
- Do not upload confidential student records or personal information into the public knowledge base.
- Conversation history, feedback and analytics are session-level. They are not a persistent student account or records system.
- Use durable storage before treating uploaded PDFs as permanent cloud records.

## Testing

The repository includes a GitHub Actions workflow that checks Python module syntax and validates the Streamlit TOML configuration on pushes to `main` and pull requests.

These checks do not replace live testing. Before publishing an update, test:

- A greeting, such as `hello`.
- A factual question answered by a bundled PDF.
- A short follow-up question, such as `what about MBA?`.
- A question whose answer is not in the knowledge base.
- Source references and PDF download buttons.
- Groq connectivity and administrator-only Document Vault controls.

## Troubleshooting

| Problem | What to check |
|---|---|
| Groq key missing | Confirm `GROQ_API_KEY` is present in `.env` locally or Streamlit Secrets in the cloud. |
| No relevant documents found | Confirm PDFs exist in `documents/` and rebuild the index. |
| ChromaDB initialization fails | Check deployment logs and confirm all bundled PDFs can be read. |
| Document Vault is read-only | Configure `ADMIN_PASSWORD` in app Secrets. |
| First response is slow | The app may be waking from sleep or loading the embedding model for the first time. |
| A source download fails | Confirm the cited PDF is present in the deployed `documents/` directory. |

## College information configured in this project

**Mother Teresa Institute of Science and Technology (MIST)**  
Sanketika Nagar, Kothuru, Sathupally, Khammam District, Telangana - 507303  
Website: [mist.edu.in](https://mist.edu.in)  
Email: info@mist.edu.in  
Phone: +91 9494641251

---

## Author and links

**Repository:** [akshaya731/College-chatbot](https://github.com/akshaya731/College-chatbot)  
**Live demo:** [MIST Connect AI](https://college-chatbot-ggpsptmesybaxdfa98kwpm.streamlit.app/)
