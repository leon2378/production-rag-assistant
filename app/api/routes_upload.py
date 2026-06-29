from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.rag.pipeline import ingest_document


router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = UPLOAD_DIR / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    result = ingest_document(
        file_path=str(file_path),
        source_file=file.filename,
    )

    return {
        "message": "Document uploaded and indexed successfully",
        **result,
    }