import os
import tempfile

from fastapi import APIRouter, File, UploadFile
from google import genai
from google.genai import types
from langchain_community.llms.fake import FakeListLLM
from langchain_google_genai import ChatGoogleGenerativeAI

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

router = APIRouter()

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key and google_api_key != "your_google_gemini_api_key_here":
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.1, google_api_key=google_api_key
    )
else:
    llm = FakeListLLM(
        responses=[
            "[WARNING: GOOGLE_API_KEY not found in .env file]. Please provide a valid API Key to enable context-aware responses."
        ]
        * 1000
    )

retriever = LegalRetriever()
qa_system = RAGGenerator(llm_pipeline=llm, retriever=retriever)


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
    Execute the RAG pipeline for Assignment 3.
    """
    if not request.question.strip():
        return QAResponse(
            question="", answer="Please provide a valid query.", sources=[]
        )

    result = qa_system.ask(request.question)
    return QAResponse(**result)


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
        }
        metadata.append(meta)

    retriever.add_clauses(clauses, metadata)

    return IngestResponse(
        message=f"Successfully ingested {request.filename}", num_clauses=len(clauses)
    )


@router.post("/ingest_file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Send uploaded documents directly to the Gemini API for cleaning and data population,
    then ingest them into the Vector DB.
    """
    content = await file.read()

    temp_fd, temp_path = tempfile.mkstemp(suffix=f"_{file.filename}")
    with os.fdopen(temp_fd, "wb") as f:
        f.write(content)

    try:
        client = genai.Client(api_key=google_api_key)

        uploaded_file = client.files.upload(file=temp_path)
        prompt = (
            "Hãy làm sạch nội dung hợp đồng thô sau, sửa lỗi OCR. "
            "YÊU CẦU QUAN TRỌNG: MỖI DÒNG CHỈ ĐƯỢC CHỨA ĐÚNG MỘT MỆNH ĐỀ / CÂU ĐỘC LẬP. Hãy xuống dòng sau mỗi câu hoặc mục liệt kê. "
            "NẾU LÀ MẪU TRỐNG (có dấu ... hoặc ___), HÃY ĐIỀN THÔNG TIN THỰC TẾ ẢO VÀO TẤT CẢ CHỖ TRỐNG. "
            "CHỈ trả về văn bản đã được cấu trúc lại theo định dạng dòng, không giải thích hay bình luận thêm."
        )
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(temperature=0.4),
        )
        text = response.text.strip()
        client.files.delete(name=uploaded_file.name)
    except Exception as e:
        print(f"Error calling Gemini API in ingest_file: {e}")
        # Fallback to the local parser if Gemini encounters an issue.
        text = parse_and_clean_document(content, file.filename)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    raw_clauses = segment_clauses(text)
    clauses = [c for c in raw_clauses if len(c.strip()) > 10]

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
            "source": file.filename,
            "intent": intent,
            "entities": str([e["text"] for e in ents]),
            "predicate": str(srl.get("predicate", "")),
        }
        metadata.append(meta)

    retriever.add_clauses(clauses, metadata)

    return IngestResponse(
        message=f"Successfully ingested {file.filename}", num_clauses=len(clauses)
    )
