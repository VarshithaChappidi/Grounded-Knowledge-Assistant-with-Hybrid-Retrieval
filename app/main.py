from fastapi import FastAPI
from app.schemas import IngestRequest, AskRequest, AskResponse
from app.retrieval import HybridRetriever

app = FastAPI(title="Grounded Knowledge Assistant", version="0.1.0")
retriever = HybridRetriever()

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(retriever.chunks)}

@app.post("/documents")
def ingest(payload: IngestRequest):
    count = retriever.add_documents([d.model_dump() for d in payload.documents])
    return {"chunks_added": count, "chunks_indexed": len(retriever.chunks)}

@app.post("/search")
def search(payload: AskRequest):
    return {"results": retriever.search(payload.question, payload.top_k)}

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    return retriever.answer(payload.question, payload.top_k)
