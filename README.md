<div align="center">

# 🚀 Day 24 — The Complete RAG API
### Final Mini Project (Phase 3)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)

*The ultimate Document Intelligence API. Upload PDFs or Text files on the fly, instantly index them into a Vector Database, and chat with them using Advanced AI.*

</div>

---

## 📖 Overview

This is the capstone project for Phase 3! We have combined everything learned so far into a single, scalable FastAPI backend. 
Unlike previous days where documents were hardcoded into a folder, this API starts with a blank slate and provides an `/upload` endpoint to dynamically ingest documents (including PDFs!). The FAISS vector database updates in real-time, allowing the AI to instantly answer questions about the files you just uploaded via the `/chat` endpoint.

## ✨ Key Features

- **📤 Dynamic File Uploads:** A completely functional `POST /upload` endpoint that parses `.pdf` and `.txt` files.
- **🔄 Real-time Indexing:** Documents are instantly chunked, embedded, and injected into the live FAISS vector database.
- **💬 Conversational RAG:** A `POST /chat` endpoint that maintains chat history and formulates context-aware queries.
- **🔐 API Security:** All endpoints are protected by `x-api-key` header authentication.
- **📜 Swagger UI:** Test both file uploads and chatting directly from your browser!

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **FastAPI & Uvicorn** | High-performance Web Framework |
| **PyPDF & python-multipart** | File Parsing and Upload Handling |
| **LangChain (LCEL)** | RAG Pipeline Orchestration |
| **FAISS** | Fast Vector Database Search |
| **Hugging Face** | Embeddings generation (`all-MiniLM-L6-v2`) |
| **Google Gemini API** | Advanced LLM Generation |

## 🚀 Setup & Usage

### 1. Install Dependencies
Navigate to the `day24` folder and install dependencies:
```bash
cd day24
pip install -U -r requirements.txt
```

### 2. Set the Gemini API Key
Provide your valid Google Gemini API key as an environment variable (Must start with `AIza`):

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY="AIza..."
```

### 3. Start the Server
Run the FastAPI application.
```bash
python main.py
```
*(The server runs locally on `http://127.0.0.1:8000`)*

---

## 🧪 Testing the API

The easiest way to test the API is using the built-in Swagger UI!
1. Open your browser to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Click **Authorize**, type `my-secret-key`, and close the dialog.

### Test 1: Upload a PDF
1. Expand the green **`POST /upload`** tab.
2. Click **Try it out**.
3. Choose a `.txt` or `.pdf` file from your computer.
4. Click **Execute**. You should get a response telling you how many text chunks were added to the AI's brain!

### Test 2: Chat with the PDF
1. Expand the green **`POST /chat`** tab.
2. Click **Try it out**.
3. Edit the JSON to ask a question about the PDF you just uploaded:
   ```json
   {
     "query": "Summarize the document I just uploaded.",
     "chat_history": []
   }
   ```
4. Click **Execute** and read the AI's response!

---
<div align="center">
<i>Built for the 100 Days of Code challenge. Phase 3 Capstone Project.</i>
</div>
