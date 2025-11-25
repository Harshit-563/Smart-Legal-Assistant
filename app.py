from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from src.smart_legal_assistant import (
    analyze_contract,
    extract_text_from_pdf
)

# app.py (replace contents with this)
from src.smart_legal_assistant import app  # use the app already defined there


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Smart Legal Assistant is running!"}

@app.post("/analyze-text")
async def analyze_contract_text(text: str = Form(...)):
    """
    Accepts raw contract text and analyzes it.
    """
    result = analyze_contract(text)
    return JSONResponse(content=result)

@app.post("/analyze-pdf")
async def analyze_contract_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts text, and analyzes it.
    """
    try:
        filename = file.filename or "uploaded_file"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Save uploaded PDF locally
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)

        # Run analysis
        result = analyze_contract(text)

        # Clean up (optional)
        os.remove(file_path)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
