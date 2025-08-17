# AI-Powered PDF Context Retrieval Chatbot (RAG)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project provides a backend service built with **FastAPI** that ingests PDF documents, indexes their content for semantic search, and answers user queries using a **Retrieval-Augmented Generation (RAG)** pipeline.

## Table of Contents
- [Objective](#objective)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Objective

The primary goal is to create a robust system that can understand and answer questions based on the content of any uploaded PDF. It leverages a vector database for efficient semantic retrieval and a large language model (LLM) for generating context-aware, human-like answers.

---

## How It Works

The application follows a complete Retrieval-Augmented Generation (RAG) workflow:

1.  **PDF Ingestion**: A user uploads a PDF file via the `/upload/` endpoint.
2.  **Text Extraction & Chunking**: The system extracts all text from the PDF and splits it into smaller, overlapping chunks to preserve contextual integrity.
3.  **Embedding Generation**: Each text chunk is converted into a numerical vector (embedding) using Google's `embedding-001` model.
4.  **Vector Indexing**: These embeddings are stored and indexed in a **ChromaDB** vector database for efficient similarity searching.
5.  **User Query**: A user submits a question through the `/query/` endpoint.
6.  **Semantic Search**: The query is also converted into an embedding. The system then searches the vector database to find the text chunks with embeddings most similar to the query's embedding.
7.  **Context-Aware Generation**: The retrieved text chunks (the context) and the original query are passed to the **Gemini** LLM, which generates a final, coherent answer based *only* on the provided information.

---

## Features

* **PDF Ingestion**: Upload any PDF document through a REST API endpoint.
* **Text Processing**: Automatically extracts text, splits it into manageable chunks, and cleans it for processing.
* **Semantic Indexing**: Generates vector embeddings for each text chunk and stores them in a ChromaDB vector database.
* **Efficient Retrieval**: Implements a fast semantic search to find the most relevant context for a given query.
* **Context-Aware Generation**: Integrates with the Gemini LLM to generate answers grounded in the source document.
* **Source Referencing**: The API response includes the source text chunks from which the answer was derived.
* **Simple Frontend**: Includes a basic HTML/JavaScript frontend for easy interaction with the backend.

---

## Tech Stack

* **Backend Framework**: FastAPI
* **Vector Database**: ChromaDB
* **LLM & Embeddings**: Google Gemini API (`gemini-pro`, `embedding-001`)
* **PDF Parsing**: `pypdf`
* **API Validation**: Pydantic

---

## Project Structure

```
.
├── app
│   ├── __init__.py
│   ├── main.py         # FastAPI application, endpoints
│   └── services
│       ├── __init__.py
│       ├── chroma_db.py  # ChromaDB client configuration
│       ├── ingestion.py  # PDF processing and embedding
│       └── retrieval.py  # RAG pipeline and query logic
├── static
│   └── index.html      # Simple frontend for interaction
├── temp/               # Temporary storage for uploaded PDFs
├── chroma_store/       # Persistent storage for ChromaDB
└── requirements.txt    # Project dependencies
```

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/d-kavinraja/AI-Powered-PDF-Context-Retrieval-Chatbot-RAG
    
    cd AI-Powered-PDF-Context-Retrieval-Chatbot-RAG
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    You need to set up your Gemini API key. Create a `.env` file in the root directory and add your key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    The application will load this key automatically.

---

## Running the Application

To run the FastAPI server locally, use the following command from the root directory:

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`. You can access the simple frontend by navigating to this URL in your browser.

---

## API Endpoints

The application provides two main REST API endpoints:

### 1. Upload PDF

* **Endpoint**: `POST /upload/`
* **Description**: Uploads a PDF file for processing and indexing.
* **Request**: `multipart/form-data` with a `file` field containing the PDF.
* **`curl` Example**:
    ```bash
    curl -X POST -F "file=@/path/to/your/document.pdf" [http://127.0.0.1:8000/upload/](http://127.0.0.1:8000/upload/)
    ```
* **Success Response (200 OK)**:
    ```json
    {
      "status": "success",
      "filename": "document.pdf",
      "chunks_indexed": 50,
      "collection": "pdf_context"
    }
    ```

### 2. Query Document

* **Endpoint**: `POST /query/`
* **Description**: Accepts a user query and returns a context-based answer from the indexed PDF.
* **Request Body**:
    ```json
    {
      "query": "What is the main topic of the document?"
    }
    ```
* **`curl` Example**:
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{"query": "What is the main topic of the document?"}' \
         [http://127.0.0.1:8000/query/](http://127.0.0.1:8000/query/)
    ```
* **Success Response (200 OK)**:
    ```json
    {
      "answer": "The main topic of the document is the implementation of Retrieval-Augmented Generation pipelines.",
      "sources": [
        {
          "chunk_index": 12,
          "source": "document.pdf",
          "preview": "Retrieval-Augmented Generation (RAG) is a technique used to..."
        }
      ]
    }
    ```

---

## Future Improvements

* **Support for More Document Types**: Extend ingestion to support `.docx`, `.txt`, and other formats.
* **Chat History**: Implement conversation memory to allow for follow-up questions.
* **Asynchronous Processing**: Move the PDF ingestion and embedding process to a background task queue (e.g., Celery) to avoid blocking the API on large uploads.
* **Authentication**: Add an authentication layer to manage access to the API.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.