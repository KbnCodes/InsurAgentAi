
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["version"] == "0.2.0"


def test_unsupported_question():
    response = client.post(
        "/api/chat",
        json={
            "message": "Approve my claim immediately.",
            "language": "en"
        }
    )

    assert response.status_code == 200
    assert response.json()["requires_human"] is True


def test_known_faq(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda definition, language: main.ModelAnswer(
            answer="A claim number identifies a claim.",
            language="en",
            needs_clarification=False
        )
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "What is a claim number?",
            "language": "en"
        }
    )

    assert response.status_code == 200
    assert response.json()["requires_human"] is False
    assert response.json()["source"] == "demo_faq:claim_number"


def test_conversation_memory(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda definition, language: main.ModelAnswer(
            answer="A deductible depends on policy terms.",
            language="en",
            needs_clarification=False
        )
    )

    first = client.post(
        "/api/chat",
        json={
            "message": "What is a deductible?",
            "language": "en"
        }
    ).json()

    second = client.post(
        "/api/chat",
        json={
            "session_id": first["session_id"],
            "message": "Explain that again",
            "language": "en"
        }
    ).json()

    assert second["session_id"] == first["session_id"]
    assert second["requires_human"] is False
    assert second["source"] == "demo_faq:deductible"


def test_invalid_language():
    response = client.post(
        "/api/chat",
        json={
            "message": "What is a deductible?",
            "language": "fr"
        }
    )

    assert response.status_code == 422


def test_model_failure_triggers_handoff(monkeypatch):
    def failing_model(definition, language):
        raise ValueError("Invalid model response")

    monkeypatch.setattr(
        main,
        "generate_answer",
        failing_model
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "What is motor insurance?",
            "language": "en"
        }
    )

    assert response.status_code == 200
    assert response.json()["requires_human"] is True
    assert response.json()["source"] is None
