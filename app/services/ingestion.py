from pypdf import PdfReader
import google.generativeai as genai
from app.services.chroma_db import get_chroma_client
from typing import List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

genai.configure(api_key="AIzaSyA7y2jAw1WbuCkLKNCx-M7sPBa85ewGvNs")
model = genai.GenerativeModel("gemini-2.5-flash")

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
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
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
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"Ingested {len(chunks)} documents into Chroma DB.")
    return len(chunks), "pdf_context"
