"""
Smart Legal Assistant (single-file prototype) — FIXED

✔ Windows-safe PDF handling
✔ No file-lock PermissionError
✔ FastAPI + HuggingFace pipelines
"""

import os
import re
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdfminer.high_level import extract_text as pdf_extract_text
from transformers.pipelines import pipeline


# -------------------- FASTAPI APP --------------------

app = FastAPI(title="Smart Legal Assistant - Prototype")

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODELS --------------------

SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
NLI_MODEL = os.getenv("NLI_MODEL", "roberta-large-mnli")

_summarizer = None
_nli_classifier = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
    return _summarizer


def get_nli():
    global _nli_classifier
    if _nli_classifier is None:
        _nli_classifier = pipeline(
            "text-classification",
            model=NLI_MODEL,
            tokenizer=NLI_MODEL,
        )
    return _nli_classifier


# -------------------- FILE HANDLING (FIXED) --------------------

async def extract_text_from_upload(upload: UploadFile) -> str:
    """
    Windows-safe file extraction.
    """
    filename = (upload.filename or "").lower()
    contents = await upload.read()

    if filename.endswith(".pdf"):
        # 🔥 FIX: delete=False + manual cleanup
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.write(contents)
            tmp.close()  # IMPORTANT: release Windows lock
            return pdf_extract_text(tmp.name)
        finally:
            os.unlink(tmp.name)  # cleanup

    # Text file
    try:
        return contents.decode("utf-8")
    except Exception:
        return contents.decode("latin-1")


# -------------------- CLAUSE EXTRACTION --------------------

def extract_clauses(text: str, max_clauses: int = 20) -> List[str]:
    text = re.sub(r"\n{2,}", "\n\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    clauses: List[str] = []
    for p in paras:
        if len(p) > 40:
            clauses.append(p)
        if len(clauses) >= max_clauses:
            break

    return clauses


# -------------------- SUMMARIZATION --------------------

def chunk_text(text: str, max_words: int = 700) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_text(text: str, max_length: int = 200) -> str:
    summarizer = get_summarizer()

    if len(text.split()) < 800:
        return summarizer(
            text,
            max_length=max_length,
            min_length=30,
            truncation=True
        )[0]["summary_text"]

    chunks = chunk_text(text)
    partials = [
        summarizer(c, max_length=120, min_length=30, truncation=True)[0]["summary_text"]
        for c in chunks
    ]

    combined = " ".join(partials)
    return summarizer(
        combined,
        max_length=max_length,
        min_length=50,
        truncation=True
    )[0]["summary_text"]


# -------------------- RISK ANALYSIS (NLI) --------------------

def flag_risks_with_nli(clauses: List[str]) -> List[str]:
    nli = get_nli()
    hypotheses = [
        "This clause allows termination without notice.",
        "This clause removes or limits liability.",
        "This clause allows automatic renewal.",
        "This clause allows unilateral amendment.",
    ]

    flagged: List[str] = []

    for clause in clauses:
        for h in hypotheses:
            try:
                res = nli([{"text": clause, "text_pair": h}])
                top = res[0]
                if "ENTAIL" in top["label"].upper() and top["score"] > 0.6:
                    flagged.append(f"{h} → {clause[:200]}...")
            except Exception:
                pass

    return list(dict.fromkeys(flagged))


# -------------------- API RESPONSE MODEL --------------------

class AnalyzeResponse(BaseModel):
    summary: str
    clauses: List[str]
    flagged_risks: List[str]


# -------------------- API ENDPOINT --------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not text and not file:
        return {"summary": "", "clauses": [], "flagged_risks": []}

    doc_text = await extract_text_from_upload(file) if file else text or ""

    clauses = extract_clauses(doc_text)
    summary = summarize_text(doc_text)

    risks = flag_risks_with_nli(clauses)

    return {
        "summary": summary,
        "clauses": clauses,
        "flagged_risks": risks,
    }


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "models": {
            "summarizer": SUMMARIZATION_MODEL,
            "nli": NLI_MODEL,
        },
    }


# -------------------- LOCAL RUN --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.smart_legal_assistant:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
