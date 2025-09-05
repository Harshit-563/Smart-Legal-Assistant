"""
Smart Legal Assistant (single-file prototype)

Features:
- Clause extraction (heuristic + heading-based)
- Summarization using Hugging Face summarization pipeline
- Risk-flagging using an NLI pipeline (defaults to roberta-large-mnli)
- FastAPI server with an /analyze endpoint that accepts raw text or uploaded .txt/.pdf

Run:
1. python -m venv venv
2. source venv/bin/activate   (Windows: venv\\Scripts\\activate)
3. pip install -r requirements.txt
4. uvicorn src.smart_legal_assistant:app --reload --port 8000
"""

import os
import re
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

# PDF parsing
from pdfminer.high_level import extract_text

# Transformers
from transformers.pipelines import pipeline


# Initialize app
app = FastAPI(title="Smart Legal Assistant - Prototype")

# Configurable model names (change via ENV)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
NLI_MODEL = os.getenv("NLI_MODEL", "roberta-large-mnli")

# Load pipelines lazily (on first use)
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
        # Use sequence-classification pipeline for NLI
        _nli_classifier = pipeline("text-classification", model=NLI_MODEL, tokenizer=NLI_MODEL)
    return _nli_classifier


# Utility: extract text from uploaded file
async def extract_text_from_upload(upload: UploadFile) -> str:
    filename = (upload.filename or "").lower()
    contents = await upload.read()
    if filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(contents)
            tmp.flush()
            return extract_text(tmp.name)
    else:
        # treat as text file
        try:
            return contents.decode("utf-8")
        except Exception:
            return contents.decode("latin-1")


# Heuristic clause headings and risk patterns
CLAUSE_HEADINGS = [
    r"\bsection\b",
    r"\bclause\b",
    r"\barticle\b",
    r"\bsubsection\b",
    r"\bterms and conditions\b",
    r"\bdefinitions\b",
    r"\btermination\b",
    r"\blicense\b",
    r"\bpayment\b",
    r"\bconfidentiality\b",
    r"\bindemnif",
]

RISK_PATTERNS = [
    "without notice",
    "without prior notice",
    "no liability",
    "limitation of liability",
    "indemnif",
    "sole discretion",
    "automatic renewal",
    "termination for convenience",
    "penalt",
    "assignment without",
]


# Pydantic response model
class AnalyzeResponse(BaseModel):
    summary: str
    clauses: List[str]
    flagged_risks: List[str]

from pdfminer.high_level import extract_text
def extract_clauses(text: str, max_clauses: int = 20) -> List[str]:
    """
    Extract candidate clauses using headings + sentence split.
    This is heuristic — for best results pair with a clause classifier trained on annotated data.
    """
    # normalize multiple newlines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # find uppercase-ish headings followed by body (heuristic)
    pattern = re.compile(r"(^|\n)\s*([A-Z0-9][^\n]{0,120})\n", re.M)
    headings = [m.group(2).strip() for m in pattern.finditer(text)]

    # fallback: split into paragraphs and then into sentences
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    candidates: List[str] = []
    for p in paras:
        # If paragraph looks like a heading + body, break
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        if len(lines) > 1 and len(lines[0].split()) < 8 and lines[0].isupper():
            candidates.append("\n".join(lines))
        else:
            # split on sentence boundaries (heuristic)
            sent_end = re.split(r"(?<=[\.\?\!;])\s+(?=[A-Z0-9])", p)
            candidates.extend([s.strip() for s in sent_end if len(s.strip()) > 40])
        if len(candidates) >= max_clauses:
            break

    # ensure uniqueness and limit
    unique: List[str] = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
        if len(unique) >= max_clauses:
            break

    return unique


import re
from typing import List

def chunk_text(text: str, max_words: int = 700) -> List[str]:
    """
    Break text into chunks of ~max_words words.
    Ensures chunks are within model input size.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def summarize_chunks(chunks: List[str], summarizer, max_length: int = 120) -> List[str]:
    """
    Summarize each chunk individually.
    """
    summaries = []
    for c in chunks:
        try:
            s = summarizer(
                c,
                max_length=max_length,
                min_length=30,
                truncation=True
            )[0]["summary_text"]
            summaries.append(s)
        except Exception:
            summaries.append(c)  # fallback to original text
    return summaries


def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Hierarchical summarization:
    - Chunk large text
    - Summarize each chunk
    - Summarize all summaries into a final summary
    """
    summarizer = get_summarizer()

    # If small enough, just summarize directly
    if len(text.split()) < 800:
        out = summarizer(text, max_length=max_length, min_length=30, truncation=True)
        return out[0]["summary_text"]

    # Step 1: Chunk the text
    chunks = chunk_text(text, max_words=700)

    # Step 2: Summarize each chunk
    partial_summaries = summarize_chunks(chunks, summarizer, max_length=120)

    # Step 3: Combine all partial summaries
    combined = " ".join(partial_summaries)

    # Step 4: Summarize the combined summary (reduce step)
    final_summary = summarizer(
        combined,
        max_length=max_length,
        min_length=50,
        truncation=True
    )[0]["summary_text"]

    return final_summary



def flag_risks_with_nli(clauses: List[str]) -> List[str]:
    """
    Use an NLI model to check whether clauses entail common-risk hypotheses.
    """
    nli = get_nli()
    hypotheses = [
        "This clause allows the other party to terminate the agreement without notice.",
        "This clause limits or removes the company's liability in most cases.",
        "This clause allows automatic renewal of the contract without consent.",
        "This clause allows assignment without the other party's consent.",
        "This clause imposes heavy penalties on the counterparty.",
        "This clause allows unilateral amendment by one party.",
    ]

    flagged: List[str] = []
    for clause in clauses:
        for h in hypotheses:
            try:
                # many HF NLI pipelines accept tuple (premise, hypothesis)
                res = nli([{"text": clause, "text_pair": h}])

            except Exception:
                # fallback join
                res = nli(clause + " </s> " + h)

            if isinstance(res, list) and len(res) > 0:
                top = res[0]
                label = str(top.get("label", "")).upper()
                score = float(top.get("score", 0.0))
                # check for entailment-like labels
                if "ENTAIL" in label or "ENTAILMENT" in label:
                    if score > 0.6:
                        flagged.append(f"Risk: '{h}' — clause: {clause[:300]}...")
                else:
                    # handle label types like LABEL_0/LABEL_1 by heuristic:
                    # Some models map LABEL_0->ENTAILMENT etc. We skip complex mapping here.
                    pass

    # deduplicate
    uniq: List[str] = []
    for f in flagged:
        if f not in uniq:
            uniq.append(f)
    return uniq


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    """
    Analyze a legal document. Provide either `text` or upload a file (.txt or .pdf).
    Response contains: summary, extracted clauses, flagged risks (via NLI + heuristics).
    """
    if text is None and file is None:
        return {"summary": "", "clauses": [], "flagged_risks": []}

    if file is not None:
        doc_text = await extract_text_from_upload(file)
    else:
        doc_text = text or ""

    # Extract clauses
    clauses = extract_clauses(doc_text, max_clauses=25)

    # Summarize
    try:
        summary = summarize_text(doc_text, max_length=180)
    except Exception as e:
        summary = f"(summarization failed: {e})"

    # Quick regex risk scanning
    quick_flags: List[str] = []
    for p in RISK_PATTERNS:
        matches = re.findall(p, doc_text, flags=re.I)
        for m in matches:
            quick_flags.append(f"Pattern '{p}' matched: ...{m}...")

    # NLI-based risk flags (on clauses)
    try:
        nli_flags = flag_risks_with_nli(clauses)
    except Exception as e:
        nli_flags = [f"NLI check failed: {e}"]

    flagged = quick_flags + nli_flags

    return {"summary": summary, "clauses": clauses, "flagged_risks": flagged}


@app.get("/ping")
def ping():
    return {"status": "ok", "models": {"summarizer": SUMMARIZATION_MODEL, "nli": NLI_MODEL}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.smart_legal_assistant:app", host="0.0.0.0", port=8000, reload=True)

def analyze_contract(text: str):
    """
    Main pipeline: extract clauses, run NLI model, generate summary, and flag risky terms.
    Returns a dictionary with results.
    """
    clauses = extract_clauses(text)              # heuristic extraction
    summary = summarize_text(text)               # use summarizer
    risky = flag_risks_with_nli(clauses)         # use NLI risk checker

    return {
        "clauses": clauses,
        "summary": summary,
        "risky_clauses": risky
    }





from pdfminer.high_level import extract_text

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pdfminer.six.
    More robust than PyPDF2 for text-heavy PDFs.
    """
    try:
        text = extract_text(file_path)
        if not text.strip():
            return "No extractable text found in PDF (might be scanned images)."
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"




