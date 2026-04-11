from typing import Any, Dict, List, Tuple

from pydantic import BaseModel


class ExtractRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    label: str
    span: Tuple[int, int]


class ClauseAnalysis(BaseModel):
    clause: str
    np_chunks: List[Tuple[str, str]]
    dependencies: List[Dict[str, Any]]
    entities: List[Entity]
    srl: Dict[str, Any]
    intent: str


class ExtractResponse(BaseModel):
    results: List[ClauseAnalysis]


class QARequest(BaseModel):
    question: str
    source_filter: str | None = None


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]


class IngestRequest(BaseModel):
    text: str
    filename: str = "uploaded_contract.txt"


class IngestResponse(BaseModel):
    message: str
    num_clauses: int
