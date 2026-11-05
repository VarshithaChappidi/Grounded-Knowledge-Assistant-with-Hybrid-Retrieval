from pydantic import BaseModel, Field

class DocumentIn(BaseModel):
    id: str
    title: str
    text: str

class IngestRequest(BaseModel):
    documents: list[DocumentIn]

class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)

class Evidence(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    abstained: bool
    evidence: list[Evidence]
    retrieval_mode: str
