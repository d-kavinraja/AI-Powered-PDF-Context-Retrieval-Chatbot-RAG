from app.services.chroma_db import get_chroma_client
import google.generativeai as genai
import logging
import os # Import os for basename, although not directly used in this snippet, good practice for full file

logger = logging.getLogger(__name__)

# It's better to load API key from environment for production
# For testing, we are directly configuring it as per previous conversation
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyA7y2jAw1WbuCkLKNCx-M7sPBa85ewGvNs"))
model = genai.GenerativeModel("gemini-2.5-flash")

def query_rag(query: str, k: int = 3):
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
        include=["documents", "metadatas"] # Only include documents and metadatas
    )

    docs = results.get("documents", [[]])
    # Ensure docs is not empty and get the first list of documents
    document_list = docs[0] if docs and len(docs) > 0 else []

    metadatas = results.get("metadatas", [[]])
    # Ensure metadatas is not empty and get the first list of metadatas
    metadata_list = metadatas if metadatas and len(metadatas) > 0 else []

    if not document_list:
        return "No relevant context found in the PDF.", []

    context_parts = []
    sources = []

    for i, doc in enumerate(document_list):
        current_meta = {}
        if i < len(metadata_list):
            meta_item = metadata_list[i]
            # Handle cases where meta_item might be a list or other unexpected type
            if isinstance(meta_item, dict):
                current_meta = meta_item
            elif isinstance(meta_item, list) and len(meta_item) > 0 and isinstance(meta_item, dict):
                # If it's a list containing dicts, take the first dict
                current_meta = meta_item
            # For other unexpected types, current_meta remains empty dict

        context_parts.append(doc) # Add document content to context

        sources.append({
            "chunk_index": current_meta.get("chunk_index", i), # Fallback to index if not found
            "source": current_meta.get("source", "PDF"), # Fallback to "PDF" if not found
            "preview": doc[:200] + ("..." if len(doc) > 200 else "")
        })

    context = "\n\n".join(context_parts)

    if not context.strip():
        return "No relevant context found in the PDF.", []

    prompt = f"""Answer the following query using only the provided context.
If the context does not contain the answer, say "I don't know".

Context:
{context}

Query:
{query}

Answer:"""

    try:
        answer = model.generate_content(prompt).text
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "Error generating answer. Please check the logs.", []

    return answer, sources
