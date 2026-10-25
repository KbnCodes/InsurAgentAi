from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_question_requires_human():
    response = client.post(
        "/api/chat",
        json={
            "message": "What is my claim status?"
        }
    )

    assert response.status_code == 200
    assert response.json()["requires_human"] is True
    assert response.json()["source"] is None


def test_known_question_uses_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda definition: "A claim number identifies a claim."
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "What is a claim number?"
        }
    )

    assert response.status_code == 200
    assert response.json()["requires_human"] is False
    assert response.json()["source"] == "demo_faq"


def test_empty_message_rejected():
    response = client.post(
        "/api/chat",
        json={"message": ""}
    )

    assert response.status_code == 422