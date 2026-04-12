import os

from fastapi import APIRouter, File, UploadFile
from google import genai

from api.schemas import (
    ExtractRequest,
    ExtractResponse,
    IngestRequest,
    IngestResponse,
    QARequest,
    QAResponse,
    RawFilesResponse,
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
    # segment_clauses returns a list of dicts: [{"text": "...", "context": "..."}]
    clauses_data = segment_clauses(request.text)
    results = []

    for item in clauses_data:
        clause_text = item["text"]

        # Ensure we pass the string 'text' to NLP engines, not the dictionary
        ents = extract_entities(clause_text)
        deps = parse_dependency(clause_text)
        chunks = chunk_np(clause_text)

        results.append(
            {
                "clause": clause_text,
                "np_chunks": chunks,
                "dependencies": deps,
                # Explicitly pass entity objects with labels
                "entities": [{"text": e["text"], "label": e["label"]} for e in ents],
                "srl": extract_srl(clause_text, ents, deps, chunks),
                "intent": classify_intent(clause_text),
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


@router.get("/documents/processed")
def list_processed_documents():
    """List all cleaned text files in data/processed independently."""
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]
    return {"files": files}


@router.get("/database/sources")
def get_indexed_sources():
    """List only sources currently existing in the Vector Database."""
    indexed_sources = retriever.get_available_sources()
    return {"sources": indexed_sources}


@router.get("/documents/raw", response_model=RawFilesResponse)
def list_raw_documents():
    """List all original files in data/raw."""
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    files = [
        f
        for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f)) and not f.startswith(".")
    ]
    return RawFilesResponse(files=files)


@router.post("/documents/{filename}/reprocess", response_model=IngestResponse)
async def reprocess_raw_document(filename: str):
    """
    Take an existing file from data/raw, clean it via Gemini,
    and update the processed/vector database (overriding existing data).
    """
    raw_path = os.path.join("data/raw", filename)
    if not os.path.exists(raw_path):
        return IngestResponse(
            message=f"Error: Raw file {filename} not found.", num_clauses=0
        )

    # 1. Clean/Anonymize using Gemini
    try:
        client = genai.Client(api_key=google_api_key)
        uploaded_file = client.files.upload(file=raw_path)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[DOCUMENT_CLEANING_PROMPT, uploaded_file],
        )
        text = response.text.strip()
        # Cleanup Gemini cloud file
        client.files.delete(name=uploaded_file.name)
    except Exception as e:
        print(f"Gemini Processing Error for {filename}: {e}")
        return IngestResponse(
            message=f"Gemini processing failed: {str(e)}", num_clauses=0
        )

    # 2. Overwrite processed txt
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    processed_filename = os.path.splitext(filename)[0] + ".txt"
    with open(
        os.path.join(processed_dir, processed_filename), "w", encoding="utf-8"
    ) as f:
        f.write(text)

    # 3. Synchronize Vector DB (Delete old if exists, then add new)
    retriever.delete_document(filename)

    clauses_with_ctx = segment_clauses(text)
    metadata = []
    texts = []
    for item in clauses_with_ctx:
        clause_text = item["text"]
        if len(clause_text.strip()) < 5:
            continue

        texts.append(clause_text)
        ents = extract_entities(clause_text)
        deps = parse_dependency(clause_text)
        chunks = chunk_np(clause_text)
        srl = extract_srl(clause_text, ents, deps, chunks)

        metadata.append(
            {
                "source": str(filename),
                "context": str(item.get("context", "General")),
                "intent": str(classify_intent(clause_text)),
                "entities": str([e["text"] for e in ents]),
                "predicate": str(srl.get("predicate", "N/A")),
                "srl_roles": str(srl.get("roles", {})),
                "dependencies": str([f"{d['token']}({d['relation']})" for d in deps]),
            }
        )

    retriever.add_clauses(texts, metadata)
    return IngestResponse(
        message=f"Successfully reprocessed {filename}", num_clauses=len(texts)
    )


@router.get("/database/state")
def get_database_state(limit: int | None = None, offset: int = 0):
    """
    Retrieve the raw clauses and metadata currently stored in the Vector DB.
    Supports pagination via limit and offset. Pass limit=None to fetch all.
    """
    records = retriever.get_all_records(limit=limit, offset=offset)
    total = retriever.get_total_count()
    return {"total": total, "records": records}


@router.delete("/documents/raw/{filename}")
def delete_raw_document(filename: str):
    """Delete only the raw source file. No impact on processed data."""
    try:
        raw_path = os.path.join("data/raw", filename)
        if os.path.exists(raw_path):
            os.remove(raw_path)
            return {"status": "success", "message": f"Deleted raw file: {filename}"}
        return {"status": "error", "message": "File not found"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.delete("/documents/processed/{filename}")
def delete_processed_document(filename: str):
    """Delete only the processed .txt file. No impact on raw files or Vector DB."""
    try:
        processed_path = os.path.join("data/processed", filename)
        if os.path.exists(processed_path):
            os.remove(processed_path)
            return {
                "status": "success",
                "message": f"Deleted processed file: {filename}",
            }
        return {"status": "error", "message": "File not found"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.delete("/database/source/{source_name}")
def delete_vector_source(source_name: str):
    """Delete vectors belonging to a specific source from the DB only."""
    try:
        success = retriever.delete_document(source_name)
        if success:
            return {
                "status": "success",
                "message": f"Deleted vectors for source: {source_name}",
            }
        return {"status": "error", "message": "Failed to delete vectors"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/documents/rename")
def rename_document(target_dir: str, old_name: str, new_name: str):
    """
    Rename a file and synchronize with Vector DB if target is data/processed.
    """
    try:
        valid_dirs = ["data/raw", "data/processed"]
        if target_dir not in valid_dirs:
            return {"status": "error", "message": "Invalid directory"}

        old_path = os.path.join(target_dir, old_name)
        new_path = os.path.join(target_dir, new_name)

        if not os.path.exists(old_path):
            return {"status": "error", "message": "Source file not found"}

        # Perform filesystem rename
        os.rename(old_path, new_path)

        # Synchronize Vector DB metadata if the processed file is renamed
        # Since processed filenames often map to the 'source' field in DB
        if target_dir == "data/processed":
            # Check if we need to map .txt back to original raw filename or just sync the string
            # Convention: source field in DB usually matches the RAW filename.
            # If your convention is processed filename, use this:
            retriever.update_source_name(old_name, new_name)

        return {
            "status": "success",
            "message": f"Renamed {old_name} to {new_name} and synced DB.",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


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
            # Fix: preserve labels for the vector DB
            "entities": str([{"text": e["text"], "label": e["label"]} for e in ents]),
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
    content = await file.read()
    filename = file.filename

    # Check for duplicates
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, filename)

    # Save to data/raw immediately
    with open(raw_path, "wb") as f:
        f.write(content)

    # Use Gemini to clean
    try:
        client = genai.Client(api_key=google_api_key)
        uploaded_file = client.files.upload(file=raw_path)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[DOCUMENT_CLEANING_PROMPT, uploaded_file],
        )
        text = response.text.strip()
    except Exception as e:
        print(
            f"Warning: Gemini API document cleaning failed ({e}). Falling back to local parser."
        )
        text = parse_and_clean_document(content, filename)

    # Save to data/processed
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    processed_filename = os.path.splitext(filename)[0] + ".txt"
    with open(
        os.path.join(processed_dir, processed_filename), "w", encoding="utf-8"
    ) as f:
        f.write(text)

    # Indexing with Context
    clauses_with_ctx = segment_clauses(text)
    metadata = []
    texts = []
    for item in clauses_with_ctx:
        clause_text = item["text"]
        if len(clause_text.strip()) < 5:  # Skip fragments
            continue

        texts.append(clause_text)
        ents = extract_entities(clause_text)
        deps = parse_dependency(clause_text)
        chunks = chunk_np(clause_text)
        srl = extract_srl(clause_text, ents, deps, chunks)

        # Ensure all metadata fields are present to prevent UI crashes in DB Explorer
        metadata.append(
            {
                "source": str(filename),
                "context": str(item.get("context", "General")),
                "intent": str(classify_intent(clause_text)),
                "np_chunks": str(chunks),
                "entities": str(
                    [{"text": e["text"], "label": e["label"]} for e in ents]
                ),
                "predicate": str(srl.get("predicate", "N/A")),
                "srl_roles": str(srl.get("roles", {})),
                "dependencies": str([f"{d['token']}({d['relation']})" for d in deps]),
            }
        )

    retriever.add_clauses(texts, metadata)
    return IngestResponse(message="Success", num_clauses=len(texts))
