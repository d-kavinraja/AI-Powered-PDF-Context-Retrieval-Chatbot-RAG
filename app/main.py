# app/main.py
import os
import logging
from dotenv import load_dotenv

# --- IMPORTANT ---
# Load environment variables from .env file BEFORE other imports
load_dotenv()
# -----------------

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Now, we can safely import our services because the environment is loaded
from app.services.ingestion import ingest_pdf
from app.services.retrieval import query_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF RAG Chatbot", version="1.0.0")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return FileResponse("static/index.html")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", file.filename)
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file.")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        count, collection = ingest_pdf(file_path)
        return {"status": "success", "filename": file.filename, "chunks_indexed": count, "collection": collection}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(request: QueryRequest):
    try:
        answer, sources = query_rag(request.query)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get an answer.")