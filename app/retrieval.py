from __future__ import annotations
import os, re, math
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out

class HybridRetriever:
    def __init__(self, abstain_threshold: float | None = None):
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None
        self.abstain_threshold = abstain_threshold if abstain_threshold is not None else float(os.getenv("ABSTAIN_THRESHOLD", "0.05"))

    def add_documents(self, docs: list[dict]) -> int:
        added = 0
        for doc in docs:
            pieces = chunk_text(doc["text"])
            for idx, piece in enumerate(pieces):
                cid = f'{doc["id"]}:{idx}'
                if any(c.chunk_id == cid for c in self.chunks):
                    continue
                self.chunks.append(Chunk(cid, doc["id"], doc["title"], piece))
                added += 1
        self._fit()
        return added

    def _fit(self):
        if self.chunks:
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        else:
            self.matrix = None

    def _lexical_scores(self, query: str) -> list[float]:
        q = set(tokenize(query))
        scores = []
        for chunk in self.chunks:
            terms = tokenize(chunk.text)
            counts = {t: terms.count(t) for t in q}
            score = sum((1 + math.log(1 + n)) for n in counts.values() if n)
            scores.append(score)
        return scores

    @staticmethod
    def _rank(scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks or self.matrix is None:
            return []
        lexical = self._lexical_scores(query)
        qv = self.vectorizer.transform([query])
        dense = cosine_similarity(qv, self.matrix).ravel().tolist()
        lex_order, dense_order = self._rank(lexical), self._rank(dense)
        lex_rank = {idx: rank + 1 for rank, idx in enumerate(lex_order)}
        dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_order)}
        fused = []
        for idx, chunk in enumerate(self.chunks):
            score = 1 / (60 + lex_rank[idx]) + 1 / (60 + dense_rank[idx])
            fused.append((score, idx))
        fused.sort(reverse=True)
        max_score = fused[0][0] if fused else 1
        return [{
            "chunk_id": self.chunks[i].chunk_id,
            "document_id": self.chunks[i].document_id,
            "title": self.chunks[i].title,
            "text": self.chunks[i].text,
            "score": round(score / max_score, 6)
        } for score, i in fused[:top_k]]

    def answer(self, query: str, top_k: int = 5) -> dict:
        hits = self.search(query, top_k)
        if not hits or hits[0]["score"] < self.abstain_threshold:
            return {"answer": "I don't have enough supporting evidence to answer that question.", "abstained": True, "evidence": [], "retrieval_mode": "hybrid"}
        best = hits[0]
        sentence = re.split(r"(?<=[.!?])\s+", best["text"].strip())[0]
        answer = f'{sentence} [Source: {best["title"]}, chunk {best["chunk_id"]}]'
        return {"answer": answer, "abstained": False, "evidence": hits, "retrieval_mode": "hybrid"}
