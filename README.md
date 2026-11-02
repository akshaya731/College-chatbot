# MIST Connect AI

A Retrieval-Augmented Generation (RAG) college information assistant for **Mother Teresa Institute of Science and Technology (MIST)**. The app uses Streamlit, LangChain, Hugging Face sentence-transformer embeddings, ChromaDB and Groq.

## Features

- **AI Assistant:** Ask questions about college admissions, fees, attendance, examinations and departments.
- **Document-grounded answers:** Retrieves relevant PDF passages and displays source filenames, page numbers and excerpts alongside answers.
- **Conversational greetings:** Handles common greetings locally without calling the language model.
- **Document Vault:** View the bundled college PDFs. Uploading and rebuilding the index are restricted to an administrator.
- **Semantic Search Lab:** Inspect retrieved document passages and relevance scores.
- **College Dashboard:** College profile details and quick links to common academic questions.
- **Usage Analytics:** Session-level message counts and response-time information.
- **System Health:** Checks the Groq configuration, PDF count and vector index status.

> Answers depend on the content and freshness of the supplied college PDFs. The application should not be treated as an official policy notice. Students should confirm time-sensitive or consequential matters with the college office.

## Project structure

```text
College-chatbot/
├── app.py
├── config.py
├── ingest.py
├── rag_pipeline.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── documents/
    ├── admission_details.pdf
    ├── attendance_rules.pdf
    ├── college_profile.pdf
    ├── departments_faculty.pdf
    ├── exam_information.pdf
    └── fee_structure.pdf
```

The ChromaDB directory is generated locally at runtime and is intentionally excluded from Git.

## Run locally (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
ADMIN_PASSWORD=choose_a_long_unique_admin_password
```

4. Build the vector index:

```powershell
python ingest.py
```

5. Start Streamlit:

```powershell
streamlit run app.py
```

Open http://localhost:8501.

## Deploy to Streamlit Community Cloud

Connect the GitHub repository and choose `app.py` as the main file. In the app's **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
ADMIN_PASSWORD = "choose_a_long_unique_admin_password"
```

Do not commit `.env`, `.streamlit/secrets.toml`, API keys or passwords to GitHub. If a key was ever committed, revoke it and create a replacement.

The Document Vault requires the `ADMIN_PASSWORD` secret. Without it, the vault remains read-only. Uploaded files are stored on the running app instance; for durable cloud storage, add official PDFs to the repository's `documents/` folder or connect a persistent object-storage service. Streamlit Community Cloud can restart or rebuild an app, so local runtime files should not be treated as permanent storage.

## Rebuild the knowledge base

When the PDFs in `documents/` change, run:

```powershell
python ingest.py
```

Or sign in as an administrator and use **Document Vault → Re-index Knowledge Base**. If any PDF cannot be read, ingestion stops before replacing the existing index.

## Technology stack

- Python
- Streamlit
- LangChain
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB
- Groq API
- PyPDF

## College details

**Mother Teresa Institute of Science and Technology (MIST)**  
Sanketika Nagar, Kothuru, Sathupally, Khammam District, Telangana - 507303  
Website: https://mist.edu.in  
Email: info@mist.edu.in
