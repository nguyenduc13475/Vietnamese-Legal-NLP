import os
import tempfile

import aspose.words as aw
from fastapi import APIRouter, File, UploadFile
from google import genai
from google.genai import types

from api.schemas import (
    ExtractRequest,
    ExtractResponse,
    IngestRequest,
    IngestResponse,
    QARequest,
    QAResponse,
)
from src.extraction.intent_classifier import classify_intent
from src.extraction.ner_engine import extract_entities
from src.extraction.srl_engine import extract_srl
from src.preprocessing.chunker import chunk_np
from src.preprocessing.parser import parse_dependency
from src.preprocessing.segmenter import segment_clauses
from src.qa.generator import RAGGenerator
from src.qa.retriever import LegalRetriever
from src.utils.document_parser import parse_and_clean_document
from src.utils.prompts import DOCUMENT_CLEANING_PROMPT

router = APIRouter()

google_api_key = os.getenv("GOOGLE_API_KEY")

retriever = LegalRetriever()
qa_system = RAGGenerator(api_key=google_api_key, retriever=retriever)


@router.post("/extract", response_model=ExtractResponse)
def extract_info(request: ExtractRequest):
    """
    Execute the full pipeline for Assignment 1 & 2.
    """
    clauses = segment_clauses(request.text)
    results = []

    for clause in clauses:
        ents = extract_entities(clause)
        deps = parse_dependency(clause)
        chunks = chunk_np(clause)
        results.append(
            {
                "clause": clause,
                "np_chunks": chunks,
                "dependencies": deps,
                "entities": ents,
                "srl": extract_srl(clause, ents, deps, chunks),
                "intent": classify_intent(clause),
            }
        )

    return ExtractResponse(results=results)


@router.post("/ask", response_model=QAResponse)
def ask_question(request: QARequest):
    """
    Execute the RAG pipeline with dynamic source filtering.
    """
    if not request.question.strip():
        return QAResponse(
            question="", answer="Please provide a valid query.", sources=[]
        )

    result = qa_system.ask(request.question, source_filter=request.source_filter)
    return QAResponse(**result)


@router.get("/sources")
def get_sources():
    """
    Retrieve a list of all processed contracts in data/processed
    and cross-reference them with their indexing status in the Vector DB.
    """
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Read physical files
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]
    indexed_sources = retriever.get_available_sources()

    sources_status = []
    for pf in processed_files:
        sources_status.append({"filename": pf, "indexed": pf in indexed_sources})

    # Catch edge case: files in Vector DB but deleted from the physical folder
    for src in indexed_sources:
        if src not in processed_files and src != "unknown":
            sources_status.append(
                {"filename": src, "indexed": True, "missing_file": True}
            )

    return {"sources": sources_status}


@router.get("/database/state")
def get_database_state(limit: int | None = None, offset: int = 0):
    """
    Retrieve the raw clauses and metadata currently stored in the Vector DB.
    Supports pagination via limit and offset. Pass limit=None to fetch all.
    """
    records = retriever.get_all_records(limit=limit, offset=offset)
    total = retriever.get_total_count()
    return {"total": total, "records": records}


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    """
    Dynamically process uploaded documents and ingest them directly into the Vector DB.
    """
    clauses = segment_clauses(request.text)
    if not clauses:
        return IngestResponse(
            message="No valid clauses found to ingest.", num_clauses=0
        )

    metadata = []

    for clause in clauses:
        ents = extract_entities(clause)
        deps = parse_dependency(clause)
        chunks = chunk_np(clause)
        srl = extract_srl(clause, ents, deps, chunks)
        intent = classify_intent(clause)

        meta = {
            "source": request.filename,
            "intent": intent,
            "entities": str([e["text"] for e in ents]),
            "predicate": str(srl.get("predicate", "")),
            "srl_roles": str(srl.get("roles", {})),
            "dependencies": str([f"{d['token']}({d['relation']})" for d in deps]),
        }
        metadata.append(meta)

    retriever.add_clauses(clauses, metadata)

    return IngestResponse(
        message=f"Successfully ingested {request.filename}", num_clauses=len(clauses)
    )


@router.post("/ingest_file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Process uploaded documents (TXT, PDF, DOCX) by converting to PDF if necessary,
    cleaning via Gemini LLM, and ingesting into the Vector Database.
    """
    content = await file.read()
    filename = file.filename

    # Save initial upload to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=f"_{filename}")
    with os.fdopen(temp_fd, "wb") as f:
        f.write(content)

    upload_filepath = temp_path
    temp_pdf_path = None
    uploaded_file = None

    try:
        # Step 1: Convert non-PDF files to PDF to ensure Gemini compatibility
        if not filename.lower().endswith(".pdf"):
            try:
                # Use Aspose.Words for conversion to maintain formatting
                doc = aw.Document(temp_path)
                temp_pdf_path = temp_path + ".pdf"
                doc.save(temp_pdf_path)
                upload_filepath = temp_pdf_path
            except Exception as e:
                print(
                    f"File conversion failed for {filename}: {e}. Attempting direct upload."
                )

        # Step 2: Process with Gemini API using the new google-genai SDK
        client = genai.Client(api_key=google_api_key)
        uploaded_file = client.files.upload(file=upload_filepath)

        prompt = DOCUMENT_CLEANING_PROMPT
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(temperature=0.4),
        )
        text = response.text.strip()

    except Exception as e:
        print(
            f"Error in Gemini ingestion pipeline for {filename}: {e}. Falling back to local parser."
        )
        # Fallback to the local parser if Gemini/Upload encounters an issue.
        text = parse_and_clean_document(content, filename)
    finally:
        # Step 3: Cleanup all temporary files and remote cloud files
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    # --- ENTERPRISE FIX: Save the cleaned text to data/processed ---
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Standardize filename to .txt
    base_name = os.path.splitext(file.filename)[0]
    processed_filename = f"{base_name}.txt"
    processed_filepath = os.path.join(processed_dir, processed_filename)

    with open(processed_filepath, "w", encoding="utf-8") as f:
        f.write(text)

    raw_clauses = segment_clauses(text)
    clauses = [c for c in raw_clauses if len(c.strip()) > 10]

    if not clauses:
        return IngestResponse(
            message="No valid clauses found to ingest.", num_clauses=0
        )

    # Update file.filename reference to match the saved .txt file for metadata consistency
    file.filename = processed_filename

    metadata = []

    for clause in clauses:
        ents = extract_entities(clause)
        deps = parse_dependency(clause)
        chunks = chunk_np(clause)
        srl = extract_srl(clause, ents, deps, chunks)
        intent = classify_intent(clause)

        meta = {
            "source": file.filename,
            "intent": intent,
            "entities": str([e["text"] for e in ents]),
            "predicate": str(srl.get("predicate", "")),
            "srl_roles": str(srl.get("roles", {})),
            "dependencies": str([f"{d['token']}({d['relation']})" for d in deps]),
        }
        metadata.append(meta)

    retriever.add_clauses(clauses, metadata)

    return IngestResponse(
        message=f"Successfully ingested {file.filename}", num_clauses=len(clauses)
    )
