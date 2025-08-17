# app/services/ingestion.py
from pypdf import PdfReader
import google.generativeai as genai
from app.services.chroma_db import get_chroma_client
from typing import List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Get the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Make sure it's in your .env file.")

# Configure the Gemini client
genai.configure(api_key=api_key)

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt)
        except Exception as e:
            logger.error(f"Failed to read page {i}: {e}")
    full_text = "\n\n".join(pages).strip()
    logger.info(f"Extracted text length: {len(full_text)}")
    return full_text

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for i, text in enumerate(texts):
        try:
            embedding = genai.embed_content(model="models/embedding-001", content=text)
            embeddings.append(embedding["embedding"])
        except Exception as e:
            logger.error(f"Embedding error at chunk {i}: {e}")
            raise
    return embeddings

def ingest_pdf(file_path: str) -> Tuple[int, str]:
    text = extract_text_from_pdf(file_path)
    if not text:
        raise ValueError("No text extracted from PDF.")
    
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks produced from PDF text.")
    
    embeddings = generate_embeddings(chunks)
    client = get_chroma_client()
    
    collection = client.get_or_create_collection(
        name="pdf_context",
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(file_path), "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return len(chunks), "pdf_context"
