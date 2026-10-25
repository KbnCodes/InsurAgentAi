"""Day 3: synthetic insurance claim status via LLM-requested tool calls."""
import hmac
import json
import os
import re

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

load_dotenv()
router = APIRouter(prefix="/api/agent", tags=["Day 3: Tool Calling"])

# Synthetic records only. Never replace with real records without real auth.
CLAIMS = {
    "CLM-1001": {"owner": "demo_account_1", "status": "Under review"},
    "CLM-1002": {"owner": "demo_account_1", "status": "Additional documents requested"},
    "CLM-2001": {"owner": "demo_account_2", "status": "Approved"},
    "CLM-1003": {
    "owner": "demo_account_1",
    "status": "Closed"
    }
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_claim_status",
        "description": "Retrieve the current status of a synthetic insurance claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "Claim ID such as CLM-1001"}
            },
            "required": ["claim_id"],
            "additionalProperties": False,
        },
    },
}]

class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)

class AgentResponse(BaseModel):
    answer: str
    requires_human: bool
    tool_used: bool
    source: str | None = None
    claim_id: str | None = None
    status: str | None = None


def request_tool_call(claim_id: str) -> tuple[str, dict]:
    """Ask the LLM for a tool request; never trust its arguments."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=20.0,
        max_retries=1,
    )
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {"role": "system", "content": (
                "You are a tool dispatcher for a synthetic insurance demo. "
                "Request get_claim_status for the exact claim ID supplied. "
                "Never guess a claim status."
            )},
            {"role": "user", "content": f"Check claim status for {claim_id}."},
        ],
        tools=TOOLS,
        tool_choice="required",
        parallel_tool_calls=False,
    )
    calls = completion.choices[0].message.tool_calls or []
    if len(calls) != 1:
        raise ValueError("Expected exactly one tool call")
    call = calls[0]
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid tool arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    return call.function.name, arguments


def get_claim_status(claim_id: str, account_id: str) -> dict | None:
    """Authorization happens in Python, not in the LLM."""
    record = CLAIMS.get(claim_id)
    if not record or record["owner"] != account_id:
        return None
    return {"claim_id": claim_id, "status": record["status"]}


def handoff(message: str) -> AgentResponse:
    return AgentResponse(answer=message, requires_human=True, tool_used=False)


@router.post("/chat", response_model=AgentResponse)
def agent_chat(
    request: AgentRequest,
    x_demo_token: str | None = Header(default=None, alias="X-Demo-Token"),
):
    expected = os.getenv("DEMO_API_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Demo token not configured")
    if not x_demo_token or not hmac.compare_digest(x_demo_token, expected):
        raise HTTPException(status_code=401, detail="Invalid demo token")

    # Only status requests are supported today. Never send raw input to LLM.
    if not re.search(r"\b(status|progress)\b", request.message, re.I):
        return handoff("This demo supports claim-status lookup only. Please contact a human representative for other requests.")

    ids = set(m.upper() for m in re.findall(r"\bCLM-\d{4}\b", request.message, re.I))
    if len(ids) != 1:
        return handoff("Provide exactly one demo claim ID, such as CLM-1001, or contact a human representative.")
    requested_id = ids.pop()

    try:
        tool_name, args = request_tool_call(requested_id)
    except (OpenAIError, ValueError):
        return handoff("The claim lookup service is unavailable. Please contact a human representative.")

    # Model output is untrusted, even when tool_choice='required'.
    if tool_name != "get_claim_status" or set(args) != {"claim_id"}:
        return handoff("Unable to validate the requested action. Please contact a human representative.")
    if not isinstance(args["claim_id"], str) or args["claim_id"] != requested_id:
        return handoff("Claim ID mismatch. Please contact a human representative.")

    # This identity is fixed by the development-only token, not the LLM.
    result = get_claim_status(requested_id, account_id="demo_account_1")
    if result is None:
        return handoff("Claim not found or not accessible. Please contact a human representative.")

    # Authoritative response assembled from tool data, never model prose.
    return AgentResponse(
        answer=f"Demo claim {result['claim_id']} status: {result['status']}.",
        requires_human=False,
        tool_used=True,
        source="synthetic_claims",
        claim_id=result["claim_id"],
        status=result["status"],
    )
