import argparse, json, statistics, time
from pathlib import Path
from app.retrieval import HybridRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/sample_queries.jsonl")
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()
    docs = json.loads(Path("data/sample_documents.json").read_text())
    retriever = HybridRetriever()
    retriever.add_documents(docs)
    rows, latencies = [], []
    for line in Path(args.dataset).read_text().splitlines():
        if not line.strip(): continue
        item = json.loads(line)
        start = time.perf_counter()
        hits = retriever.search(item["question"], 5)
        latencies.append((time.perf_counter() - start) * 1000)
        relevant = set(item["relevant_chunk_ids"])
        ids = [h["chunk_id"] for h in hits]
        found = [i for i, cid in enumerate(ids, 1) if cid in relevant]
        rows.append({"question": item["question"], "recall_at_5": len(set(ids[:5]) & relevant) / max(1, len(relevant)), "reciprocal_rank": 1 / found[0] if found else 0, "top_hit_evidence_proxy": bool(found and hits[0]["chunk_id"] in relevant)})
    report = {"queries": len(rows), "recall_at_5": statistics.mean(r["recall_at_5"] for r in rows) if rows else 0, "mrr_at_5": statistics.mean(r["reciprocal_rank"] for r in rows) if rows else 0, "evidence_coverage_proxy": statistics.mean(r["top_hit_evidence_proxy"] for r in rows) if rows else 0, "mean_latency_ms": statistics.mean(latencies) if latencies else 0, "estimated_cost_per_query_usd": 0.0, "note": "Local TF-IDF/lexical baseline; no external model/API cost. Evidence proxy is not a human groundedness score.", "per_query": rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
