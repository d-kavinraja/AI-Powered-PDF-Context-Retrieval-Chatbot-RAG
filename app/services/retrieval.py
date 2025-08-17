# app/services/retrieval.py
from app.services.chroma_db import get_chroma_client
import google.generativeai as genai
import logging
import os

logger = logging.getLogger(__name__)

# Get the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Make sure it's in your .env file.")

# Configure the Gemini client
genai.configure(api_key=api_key)

# Initialize the model once
model = genai.GenerativeModel("gemini-2.5-flash")

def query_rag(query: str, k: int = 5):
    """
    Queries the RAG pipeline.
    """
    client = get_chroma_client()

    try:
        collection = client.get_collection(name="pdf_context")
    except Exception as e:
        logger.error(f"Failed to get Chroma collection: {e}")
        return "No context found. Did you upload a PDF?", []

    try:
        query_embedding = genai.embed_content(model="models/embedding-001", content=query)["embedding"]
    except Exception as e:
        logger.error(f"Embedding query failed: {e}")
        return "Error embedding query.", []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    
    context_parts = []
    sources = []
    for i, doc in enumerate(documents):
        current_meta = metadatas[i] if i < len(metadatas) else {}
        context_parts.append(doc)
        sources.append({
            "chunk_index": current_meta.get("chunk_index", i),
            "source": current_meta.get("source", "PDF"),
            "preview": doc[:200] + ("..." if len(doc) > 200 else "")
        })

    context = "\n\n".join(context_parts)
    if not context.strip():
        return "No relevant context found in the PDF.", []

    prompt = f"""Answer the following query using only the provided context.
If the context does not contain the answer, state that the information is not available in the document.

Context:
{context}

Query:
{query}

Answer:"""

    try:
        answer = model.generate_content(prompt).text
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "Failed to generate an answer from the context.", sources
        
    return answer, sources
