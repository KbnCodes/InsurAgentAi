
from fastapi import FastAPI
from fastapi.testclient import TestClient
import day6_rag

app = FastAPI()
app.include_router(day6_rag.router)
client = TestClient(app)


def test_sensitive_claim_escalates(monkeypatch):
    def should_not_call(*args, **kwargs):
        raise AssertionError("Retrieval should not run")

    monkeypatch.setattr(
        day6_rag, "retrieve", should_not_call
    )

    result = client.post(
        "/api/rag/chat",
        json={"message": "What is my CLM-1001 status?"}
    )

    assert result.status_code == 200
    assert result.json()["requires_human"] is True
    assert result.json()["sources"] == []


def test_grounded_answer(monkeypatch):
    monkeypatch.setattr(
        day6_rag,
        "retrieve",
        lambda query: [{
            "id": "DEMO-002",
            "title": "Insurance Deductible",
            "text": "A deductible may apply.",
            "score": 0.85
        }]
    )

    monkeypatch.setattr(
        day6_rag,
        "generate_answer",
        lambda question, passage: "A deductible may apply."
    )

    result = client.post(
        "/api/rag/chat",
        json={"message": "What is a deductible?"}
    )

    assert result.status_code == 200
    body = result.json()
    assert body["requires_human"] is False
    assert body["sources"][0]["id"] == "DEMO-002"


def test_no_relevant_document(monkeypatch):
    monkeypatch.setattr(
        day6_rag, "retrieve", lambda query: []
    )

    result = client.post(
        "/api/rag/chat",
        json={"message": "What is quantum computing?"}
    )

    assert result.json()["requires_human"] is True


def test_model_abstains(monkeypatch):
    monkeypatch.setattr(
        day6_rag,
        "retrieve",
        lambda query: [{
            "id": "DEMO-001",
            "title": "Claim Number",
            "text": "A claim number is an identifier.",
            "score": 0.7
        }]
    )

    monkeypatch.setattr(
        day6_rag,
        "generate_answer",
        lambda question, passage: "UNSUPPORTED"
    )

    result = client.post(
        "/api/rag/chat",
        json={"message": "Explain claims processing"}
    )

    assert result.json()["requires_human"] is True
