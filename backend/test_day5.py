"""Run: python -m pytest test_day5.py -v"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import day5_workflow as workflow

app = FastAPI()
app.include_router(workflow.router)
client = TestClient(app)
HEADERS = {"X-Demo-Token": "local-test-token"}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setenv("DEMO_API_TOKEN", "local-test-token")
    with workflow._lock:
        workflow._sessions.clear()
    yield
    with workflow._lock:
        workflow._sessions.clear()


def post(data, headers=HEADERS):
    return client.post("/api/agent/v3/intake", json=data, headers=headers)


def test_authentication_required():
    assert post({}, headers={}).status_code == 401


def test_new_session_asks_about_injuries():
    r = post({})
    assert r.status_code == 200
    assert r.json()["state"] == "collecting"
    assert r.json()["next_field"] == "injury_status"
    assert r.json()["requires_human"] is False


def test_stepwise_collection_and_draft():
    first = post({}).json()
    sid = first["session_id"]
    second = post({"session_id": sid, "injury_status": "no"}).json()
    assert second["next_field"] == "incident_type"
    third = post({"session_id": sid, "incident_type": "collision"}).json()
    assert third["next_field"] == "city"
    fourth = post({"session_id": sid, "city": "Riyadh"}).json()
    assert fourth["state"] == "draft_ready"
    assert fourth["draft_id"].startswith("DRAFT-")
    assert fourth["requires_human"] is True
    assert fourth["collected"] == {
        "injury_status": "no", "incident_type": "collision", "city": "Riyadh"
    }


def test_all_fields_in_one_request():
    result = post({"injury_status": "no", "incident_type": "parking_damage", "city": "Jeddah"}).json()
    assert result["state"] == "draft_ready"


def test_injury_escalates_and_does_not_make_draft():
    result = post({"injury_status": "yes", "city": "Riyadh"}).json()
    assert result["state"] == "escalated"
    assert result["draft_id"] is None
    assert result["requires_human"] is True


def test_unknown_injury_escalates():
    assert post({"injury_status": "unknown"}).json()["state"] == "escalated"


def test_other_incident_escalates():
    assert post({"injury_status": "no", "incident_type": "other"}).json()["state"] == "escalated"


def test_explicit_human_request_escalates():
    assert post({"request_human": True}).json()["state"] == "escalated"


def test_terminal_state_cannot_be_overwritten():
    first = post({"injury_status": "yes"}).json()
    later = post({"session_id": first["session_id"], "injury_status": "no", "city": "Riyadh"}).json()
    assert later["state"] == "escalated"
    assert later["draft_id"] is None


def test_unknown_session_returns_404():
    assert post({"session_id": "nonexistent"}).status_code == 404


def test_invalid_fields_rejected():
    assert post({"injury_status": "maybe"}).status_code == 422
    assert post({"customer_phone": "1234567890"}).status_code == 422
    assert post({"city": "   "}).status_code == 422


def test_get_session_returns_saved_state():
    first = post({"injury_status": "no"}).json()
    response = client.get(
        f"/api/agent/v3/intake/{first['session_id']}", headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["next_field"] == "incident_type"
