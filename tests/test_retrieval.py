from app.retrieval import HybridRetriever, chunk_text

def build():
    r = HybridRetriever()
    r.add_documents([
        {"id":"refund","title":"Refund Policy","text":"Customers may request a refund within 30 days of purchase."},
        {"id":"security","title":"Security Guide","text":"Passwords are stored using salted adaptive hashing."}
    ])
    return r

def test_chunk_text_empty_and_overlap():
    assert chunk_text("") == []
    assert len(chunk_text("a" * 1200, size=500, overlap=50)) == 3

def test_retrieves_relevant_document():
    hits = build().search("refund within 30 days", 2)
    assert hits[0]["document_id"] == "refund"

def test_answer_has_source_citation():
    answer = build().answer("When can a customer request a refund?")
    assert not answer["abstained"]
    assert "[Source:" in answer["answer"]
    assert answer["evidence"]

def test_empty_index_abstains():
    assert HybridRetriever().answer("What is the refund policy?")["abstained"]

def test_duplicate_ingestion_is_idempotent():
    r = build()
    count = r.add_documents([{"id":"refund","title":"Refund Policy","text":"Customers may request a refund within 30 days of purchase."}])
    assert count == 0
