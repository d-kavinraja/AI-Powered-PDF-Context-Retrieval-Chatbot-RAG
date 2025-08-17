import os
import chromadb
from chromadb.config import Settings

def get_chroma_client():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    os.makedirs(persist_dir, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client
