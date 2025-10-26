"""Real-estate workflow MCP server (OpenAI Agents SDK + FastMCP).

This module hosts the full workflow implementation that previously lived in
``real_estate_workflow.py``. It is organized as a standalone package so other
modules can import it without triggering application-level side effects.

Features
- Jailbreak Guardrail gate (OpenAI Guardrails if configured; fallback to Moderations)
- Router (buy | sell | disclosures | offer | general) with JSON schema
- Buyer Agent with sub-agents:
    * Buyer-Intake (stepwise, no repeat questions; persistent per session)
    * Search & Match (live MLS web search via Zillow/Redfin/MLSListings + payment estimate)
    * Tour Plan (open-house schedule optimizer)
    * Disclosure Q&A (file-based Q&A with sentence-level citations)
    * Offer Drafter (compliant draft + checklist)
    * Negotiation Coach (market-context driven offer logic)
- Enhanced workflow orchestration with conversation history and state management

Run
    uv run python real_estate_mcp_server.py  # or: python real_estate_mcp_server.py

Env
    OPENAI_API_KEY=...
    # Optional guardrails config directory (bundle)
    GUARDRAILS_BUNDLE_DIR=guardrails_bundle/

Notes
- This server exposes tools the host LLM/Agent can call. You can call a single
    orchestrator tool (run_workflow) or compose calls to the sub-agents.
- Replace the MLS and commute/payment stubs with your production integrations.
- All LLM calls use JSON-schema constrained responses for robust outputs.
- Enhanced workflow pattern supports conversation history tracking and conditional execution

Implementation note: the original ``real_estate_workflow`` module now serves as
a compatibility shim that re-exports these symbols so legacy imports continue to
function without change.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
import asyncio
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv
from agents.agent import Agent
from agents.model_settings import ModelSettings
from agents.run import Runner, RunConfig
from agents.tool import WebSearchTool
from agents.tracing import agent_span, get_current_trace, trace
from openai import OpenAI, AsyncOpenAI
from openai.types.responses.web_search_tool import Filters as WebSearchToolFilters
from openai.types.responses.web_search_tool_param import UserLocation
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field, ConfigDict, model_validator

import prompts
from schemas import (
    get_intake_extraction_schema,
    get_mls_web_search_schema,
    get_negotiation_coach_schema,
    get_offer_drafter_schema,
    get_required_intake_fields,
    get_router_json_schema,
    get_tour_plan_extraction_schema,
    get_tour_plan_generation_schema,
)

# Type alias for conversation history items
TResponseInputItem = Dict[str, Any]

# Load environment variables from .env file once at module import time.
load_dotenv()

# ---------------------------------------------------------------------------
# FastMCP server bootstrap
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import fallback for FastMCP server
    from fastmcp import FastMCP
except Exception:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("real_estate_mcp")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# OpenAI client & guardrails helpers
# ---------------------------------------------------------------------------
client = OpenAI()
async_client = AsyncOpenAI()
JSON_DECODER = json.JSONDecoder()

# Shared context for guardrails
ctx = SimpleNamespace(guardrail_llm=async_client)


def _can_set_temperature(model: str, temperature: Optional[float]) -> bool:
    """Return True if the model supports an explicit temperature override."""

    if temperature is None:
        return False
    model_lower = model.lower()
    if model_lower.startswith("gpt-5"):
        return False
    return True

try:  # Guardrails import is optional at runtime
    from openai_guardrails.runtime import (  # type: ignore
        instantiate_guardrails,
        load_config_bundle,
        run_guardrails,
    )

    _HAS_GUARDRAILS = True
except Exception:  # pragma: no cover
    _HAS_GUARDRAILS = False


# ---------------------------------------------------------------------------
# Enhanced Guardrails helpers (following workflow pattern)
# ---------------------------------------------------------------------------
def guardrails_has_tripwire(results):
    """Check if any guardrail result triggered a tripwire."""
    return any(getattr(r, "tripwire_triggered", False) is True for r in (results or []))


def get_guardrail_checked_text(results, fallback_text):
    """Extract checked text from guardrail results."""
    for r in (results or []):
        info = getattr(r, "info", None) or {}
        if isinstance(info, dict) and ("checked_text" in info):
            return info.get("checked_text") or fallback_text
    return fallback_text


def build_guardrail_fail_output(results):
    """Build failure output from guardrail results."""
    failures = []
    for r in (results or []):
        if getattr(r, "tripwire_triggered", False):
            info = getattr(r, "info", None) or {}
            failure = {
                "guardrail_name": info.get("guardrail_name"),
            }
            for key in ("flagged", "confidence", "threshold", "hallucination_type", "hallucinated_statements", "verified_statements"):
                if key in (info or {}):
                    failure[key] = info.get(key)
            failures.append(failure)
    return {"failed": len(failures) > 0, "failures": failures}


# Guardrails configuration
GUARDRAILS_CONFIG = {
    "guardrails": [
        {
            "name": "Jailbreak",
            "config": {
                "model": "gpt-4o-mini",
                "confidence_threshold": 0.7
            }
        }
    ]
}


# ---------------------------------------------------------------------------
# Session state (simple in-memory store for demo purposes)
# ---------------------------------------------------------------------------
@dataclass
class IntakeState:
    budget: Optional[str] = None
    locations: Optional[str] = None
    home_type: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    financing_status: Optional[str] = None
    timeline: Optional[str] = None
    must_haves: Optional[str] = None
    deal_breakers: Optional[str] = None
    consents: Optional[str] = None
    next_field: str = "budget"


@dataclass
class Session:
    created_ts: float
    last_ts: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    intake: IntakeState = field(default_factory=IntakeState)
    agent_items: List[Dict[str, Any]] = field(default_factory=list)
    intake_agent_items: List[Dict[str, Any]] = field(default_factory=list)
    # Enhanced workflow conversation history
    conversation_history: List[TResponseInputItem] = field(default_factory=list)


SESSIONS: Dict[str, Session] = {}


def _get_session(session_id: Optional[str]) -> Tuple[str, Session]:
    sid = session_id or str(uuid.uuid4())
    sess = SESSIONS.get(sid)
    now = time.time()
    if sess is None:
        sess = Session(created_ts=now, last_ts=now)
        SESSIONS[sid] = sess
    else:
        sess.last_ts = now
    return sid, sess


# ---------------------------------------------------------------------------
# LLM helper utilities
# ---------------------------------------------------------------------------

def llm_json(
    *,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = 0.2,
) -> Dict[str, Any]:
    """Call the Chat Completions API and coerce the response to JSON."""

    request_args: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": json_schema},
    }
    if _can_set_temperature(model, temperature):
        request_args["temperature"] = temperature

    resp = client.chat.completions.create(**request_args)
    try:
        text = resp.choices[0].message.content
        return json.loads(text)
    except Exception as exc:  # pragma: no cover
        log.exception("Failed to parse JSON response: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class GuardrailResult(BaseModel):
    allowed: bool
    reasons: List[str] = Field(default_factory=list)


def jailbreak_check(user_message: str) -> GuardrailResult:
    bundle_dir = os.getenv("GUARDRAILS_BUNDLE_DIR")
    if _HAS_GUARDRAILS and bundle_dir and os.path.isdir(bundle_dir):
        try:
            cfg = load_config_bundle(bundle_dir)
            rails = instantiate_guardrails(cfg)
            result = run_guardrails(rails, {"user_message": user_message})
            allowed = result.get("passed", True)
            reasons = [f"{k}: {v}" for k, v in result.items() if k != "passed"]
            return GuardrailResult(allowed=allowed, reasons=reasons)
        except Exception as exc:  # pragma: no cover
            log.warning("Guardrails execution failed, falling back to Moderations: %s", exc)
    try:
        moderation = client.moderations.create(model="omni-moderation-latest", input=user_message)
        flagged = any(cat for cat, val in moderation.results[0].categories.to_dict().items() if val)  # type: ignore
        if flagged:
            reasons = [
                key
                for key, score in moderation.results[0].category_scores.to_dict().items()  # type: ignore
                if score and score > 0.5
            ]
            return GuardrailResult(allowed=False, reasons=reasons)
        return GuardrailResult(allowed=True)
    except Exception as exc:  # pragma: no cover
        log.warning("Moderations failed open; allowing request. Error: %s", exc)
        return GuardrailResult(allowed=True)


# ---------------------------------------------------------------------------
# Agents SDK configuration
# ---------------------------------------------------------------------------


class RouterSlotsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget: Optional[str] = None
    areas: Optional[str] = None
    timing: Optional[str] = None
    property_ref: Optional[str] = None
    listing_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {key: (value if value not in ("", None) else None) for key, value in self.model_dump().items()}


class RouterOutputModel(BaseModel):
    intent: Literal["buy", "sell", "disclosures", "offer", "general"]
    confidence: float
    slots: RouterSlotsModel = Field(default_factory=RouterSlotsModel)

    @model_validator(mode="after")
    def _clamp_confidence(self) -> "RouterOutputModel":
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        return self

    def slots_as_dict(self) -> Dict[str, Optional[str]]:
        return self.slots.as_dict()


class SearchQueryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str
    query: str
    result_summary: str


class ListingSlotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str
    end: str


class ListingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    property_id: Optional[str] = None
    address: str
    list_price: Union[str, float, None] = None
    beds: Optional[Union[str, float]] = None
    baths: Optional[Union[str, float]] = None
    source_url: str
    source_name: Optional[str] = None
    fit_reasoning: str
    commute_notes: Optional[str] = None
    notes: Optional[str] = None
    open_house_slots: List[ListingSlotModel] = Field(default_factory=list)


class MLSWebSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_queries: List[SearchQueryModel]
    listings: List[ListingModel]


class IntakeSlotsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget: Optional[str] = None
    locations: Optional[str] = None
    home_type: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    financing_status: Optional[str] = None
    timeline: Optional[str] = None
    must_haves: Optional[str] = None
    deal_breakers: Optional[str] = None
    consents: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return self.model_dump()


class IntakeAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ask", "summary", "confirmed"]
    message: str
    field: Optional[str] = None
    next_field: Optional[str] = None
    slots: IntakeSlotsModel = Field(default_factory=IntakeSlotsModel)
    summary: Optional[IntakeSlotsModel] = None

    @model_validator(mode="after")
    def _validate_next_field(self) -> "IntakeAgentOutput":
        allowed = set(get_required_intake_fields()) | {"done", None}
        if self.next_field not in allowed:
            self.next_field = None
        if self.status == "ask" and not self.field:
            self.field = self.next_field
        if self.status in {"summary", "confirmed"} and self.summary is None:
            summary_payload = {
                key: value
                for key, value in self.slots.as_dict().items()
                if value not in (None, "")
            }
            self.summary = IntakeSlotsModel(**summary_payload)
        return self


def _default_reasoning_settings() -> ModelSettings:
    return ModelSettings(
        reasoning=Reasoning(effort="low", summary="auto"),
        store=True,
    )


ROUTER_AGENT = Agent(
    name="Router",
    instructions=prompts.router_system_prompt(),
    model="gpt-5",
    output_type=RouterOutputModel,
    model_settings=_default_reasoning_settings(),
)


# ---------------------------------------------------------------------------
# Router agent
# ---------------------------------------------------------------------------
INTAKE_QUESTIONS = prompts.intake_questions()


def router_route(user_message: str, session: Session) -> Dict[str, Any]:
    input_items: List[Dict[str, Any]] = []
    for item in session.agent_items:
        if isinstance(item, BaseModel):
            input_items.append(item.model_dump(mode="json", exclude_none=True))
        else:
            input_items.append(item)
    input_items.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_message,
                }
            ],
        }
    )

    try:
        with _agent_trace_context("Real Estate Router", ROUTER_AGENT.name):
            result = _run_agent(ROUTER_AGENT, input_items)
    except Exception as exc:  # pragma: no cover - fallback to legacy flow
        log.exception("Router agent run failed; falling back to legacy router prompt", exc_info=exc)
        history_snippets = [
            f"U:{h['user']}\nA:{h['agent']}" for h in session.history[-4:] if "user" in h and "agent" in h
        ]
        history_text = "\n\n".join(history_snippets)
        user_prompt = prompts.router_user_prompt(history_text=history_text, user_message=user_message)
        out = llm_json(
            system_prompt=prompts.router_system_prompt(),
            user_prompt=user_prompt,
            json_schema=get_router_json_schema(),
        )
        slots_obj: Dict[str, Any] = {}
        try:
            slots_obj = json.loads(out.get("slots") or "{}")
        except Exception:
            slots_obj = {}
        for key in ["budget", "areas", "timing", "property_ref", "listing_id", "email", "phone"]:
            slots_obj.setdefault(key, None)
        out["slots_obj"] = slots_obj

        fallback_history: List[Dict[str, Any]] = []
        for item in input_items:
            if isinstance(item, BaseModel):
                fallback_history.append(item.model_dump(mode="json", exclude_none=True))
            else:
                fallback_history.append(item)
        fallback_history.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(out),
                    }
                ],
            }
        )
        session.agent_items = fallback_history[-16:]

        session.history.append({
            "user": user_message,
            "agent": json.dumps(out),
        })
        if len(session.history) > 8:
            session.history = session.history[-8:]
        return out

    serialized_history: List[Dict[str, Any]] = []
    for item in result.to_input_list():
        if isinstance(item, BaseModel):
            serialized_history.append(item.model_dump(mode="json", exclude_none=True))
        else:
            serialized_history.append(item)
    session.agent_items = serialized_history[-16:]

    router_output = result.final_output_as(RouterOutputModel, raise_if_incorrect_type=True)
    slots_obj = router_output.slots_as_dict()

    session.history.append({
        "user": user_message,
        "agent": router_output.model_dump_json(),
    })
    if len(session.history) > 8:
        session.history = session.history[-8:]

    return {
        "intent": router_output.intent,
        "confidence": router_output.confidence,
        "slots": json.dumps(slots_obj),
        "slots_obj": slots_obj,
    }


# ---------------------------------------------------------------------------
# Buyer intake stepper
# ---------------------------------------------------------------------------

def _normalize_intake_value(field: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if field == "budget":
            return _format_currency(float(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    if field == "budget":
        price = _coerce_price(text)
        return _format_currency(price) if price is not None else text
    if field in {"bedrooms", "bathrooms"}:
        match = re.search(r"\d+(?:\.\d+)?", text)
        return match.group(0) if match else text
    if field == "consents":
        low = text.lower()
        if low in {"yes", "y", "yep", "yeah", "sure", "affirmative", "confirmed", "confirm", "i consent", "i agree"}:
            return "yes"
        if low in {"no", "n", "nope", "decline", "not yet", "i do not consent"}:
            return "no"
        return text
    return text


def _sync_next_field(intake: IntakeState) -> None:
    required_fields = get_required_intake_fields()
    for field in required_fields:
        if getattr(intake, field) in (None, ""):
            intake.next_field = field
            return
    intake.next_field = "done"


def _legacy_extract_intake_fields(message: str) -> Dict[str, Any]:
    if not message or not message.strip():
        return {}
    return llm_json(
        system_prompt=prompts.intake_extraction_system_prompt(),
        user_prompt=message,
        json_schema=get_intake_extraction_schema(),
    )


def _legacy_ingest_intake_message(
    session: Session,
    user_message: Optional[str],
    slot_hints: Optional[Dict[str, Any]] = None,
) -> None:
    intake = session.intake
    updated = False
    required_fields = get_required_intake_fields()

    def _set(field: str, value: Any, *, force: bool = False) -> None:
        nonlocal updated
        norm = _normalize_intake_value(field, value)
        if norm is None:
            return
        current = getattr(intake, field)
        if current in (None, "") or force:
            if current == norm:
                return
            setattr(intake, field, norm)
            updated = True

    if slot_hints:
        slot_map = {
            "budget": slot_hints.get("budget"),
            "locations": slot_hints.get("areas"),
            "timeline": slot_hints.get("timing"),
        }
        for field, value in slot_map.items():
            if value not in (None, ""):
                _set(field, value, force=True)

    if user_message and user_message.strip():
        try:
            extracted = _legacy_extract_intake_fields(user_message)
        except Exception as exc:  # pragma: no cover
            log.warning("Intake extraction failed; falling back to direct assignment: %s", exc)
            extracted = {}
        for field in required_fields:
            if field in extracted and extracted[field] not in (None, ""):
                _set(field, extracted[field], force=True)

    if not updated and user_message and user_message.strip():
        current = intake.next_field
        if current in INTAKE_QUESTIONS and getattr(intake, current) in (None, ""):
            _set(current, user_message.strip(), force=False)

    _sync_next_field(intake)


def _legacy_intake_step(
    session: Session,
    user_message: Optional[str],
    slot_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _legacy_ingest_intake_message(session, user_message, slot_hints)
    state = session.intake
    required_fields = get_required_intake_fields()

    if state.next_field == "done":
        summary = {key: getattr(state, key) for key in required_fields}
        return {
            "status": "summary",
            "message": "Here is a summary of your intake. Please confirm if everything is correct (yes/no) or indicate any corrections.",
            "summary": summary,
        }

    next_q = INTAKE_QUESTIONS[state.next_field]
    ctx_bits = []
    for key in required_fields:
        value = getattr(state, key)
        if value and key != state.next_field:
            ctx_bits.append(f"{key.replace('_', ' ').title()}: {value}")
    context = ("You already shared -> " + "; ".join(ctx_bits)) if ctx_bits else None

    return {
        "status": "ask",
        "field": state.next_field,
        "message": next_q if not context else f"{context}.\n{next_q}",
    }


def _set_intake_field(intake: IntakeState, field: str, value: Any, *, force: bool = False) -> bool:
    if field not in get_required_intake_fields():
        return False
    if value in (None, ""):
        if force and getattr(intake, field) not in (None, ""):
            setattr(intake, field, None)
            return True
        return False
    norm = _normalize_intake_value(field, value)
    if norm is None:
        if force and getattr(intake, field) not in (None, ""):
            setattr(intake, field, None)
            return True
        return False
    current = getattr(intake, field)
    if current == norm:
        return False
    if current in (None, "") or force or current != norm:
        setattr(intake, field, norm)
        return True
    return False


def _intake_state_snapshot(intake: IntakeState) -> Dict[str, Optional[str]]:
    return {key: getattr(intake, key) for key in get_required_intake_fields()}


def _apply_slot_hints(intake: IntakeState, slot_hints: Optional[Dict[str, Any]]) -> None:
    if not slot_hints:
        return
    slot_map = {
        "budget": slot_hints.get("budget"),
        "locations": slot_hints.get("areas"),
        "timeline": slot_hints.get("timing"),
    }
    updated = False
    for field, value in slot_map.items():
        if value not in (None, ""):
            if _set_intake_field(intake, field, value, force=True):
                updated = True
    if updated:
        _sync_next_field(intake)


def _build_intake_agent_input(
    session: Session,
    user_message: Optional[str],
    slot_hints: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    context_payload = {
        "current_state": _intake_state_snapshot(session.intake),
        "next_field": session.intake.next_field,
        "slot_hints": slot_hints or {},
    }
    context_text = json.dumps(context_payload, ensure_ascii=True)
    content_blocks: List[Dict[str, str]] = [{"type": "input_text", "text": f"CONTEXT:\n{context_text}"}]
    if user_message is not None:
        content_blocks.append({"type": "input_text", "text": f"USER_MESSAGE:\n{user_message}"})
    else:
        content_blocks.append({"type": "input_text", "text": "USER_MESSAGE:\n"})
    return [{"role": "user", "content": content_blocks}]


def _update_intake_state_from_slots(intake: IntakeState, slots: IntakeSlotsModel) -> None:
    updated = False
    for field, value in slots.as_dict().items():
        if _set_intake_field(intake, field, value, force=True):
            updated = True
    if updated:
        _sync_next_field(intake)


def intake_step(
    session: Session,
    user_message: Optional[str],
    slot_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    intake = session.intake
    _apply_slot_hints(intake, slot_hints)
    _sync_next_field(intake)
    agent_input = _build_intake_agent_input(session, user_message, slot_hints)

    try:
        with _agent_trace_context("Buyer Intake", INTAKE_AGENT.name):
            result = _run_agent(INTAKE_AGENT, agent_input)
    except Exception as exc:  # pragma: no cover - fallback to deterministic flow
        log.exception("Buyer intake agent run failed; falling back to legacy flow", exc_info=exc)
        return _legacy_intake_step(session, user_message, slot_hints)

    session.intake_agent_items = []

    try:
        agent_output = result.final_output_as(IntakeAgentOutput, raise_if_incorrect_type=True)
    except Exception as exc:  # pragma: no cover - fallback to deterministic flow
        log.exception("Buyer intake agent returned invalid output; falling back to legacy flow", exc_info=exc)
        return _legacy_intake_step(session, user_message, slot_hints)

    _update_intake_state_from_slots(intake, agent_output.slots)

    if agent_output.next_field:
        intake.next_field = agent_output.next_field
    else:
        _sync_next_field(intake)

    summary_model = agent_output.summary
    summary: Optional[Dict[str, Optional[str]]] = None
    if summary_model is not None:
        summary = summary_model.model_dump(exclude_none=True)
    if agent_output.status in {"summary", "confirmed"}:
        if summary is None:
            summary = _intake_state_snapshot(intake)
        else:
            snapshot = _intake_state_snapshot(intake)
            summary = {key: summary.get(key) or snapshot.get(key) for key in snapshot}

    response = {
        "status": agent_output.status,
        "field": agent_output.field,
        "message": agent_output.message,
        "summary": summary,
        "slots": agent_output.slots.as_dict(),
        "next_field": intake.next_field,
    }
    if response["field"] is None and agent_output.status == "ask":
        response["field"] = intake.next_field

    return response


def intake_is_complete(session: Session) -> bool:
    state = session.intake
    required_fields = get_required_intake_fields()
    return all(getattr(state, key) not in (None, "") for key in required_fields)


# ---------------------------------------------------------------------------
# Search & match helpers
# ---------------------------------------------------------------------------
class MatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    property_id: str
    address: str
    fit_rationale: str
    estimated_commute: str
    estimated_monthly_payment: str
    open_house_slots: List[Dict[str, str]] = Field(default_factory=list)
    list_price: Optional[float] = None
    list_price_display: Optional[str] = None
    beds: Optional[str] = None
    baths: Optional[str] = None
    source_url: Optional[str] = None
    source_name: Optional[str] = None


MLS_SEARCH_DOMAINS = ["zillow.com", "redfin.com", "mlslistings.com"]

MLS_WEB_SEARCH_TOOL = WebSearchTool(
    filters=WebSearchToolFilters(
        allowed_domains=[
            "www.mls.com",
            "www.zillow.com",
            "www.redfin.com",
            "www.mlslistings.com",
        ],
    ),
    search_context_size="medium",
    user_location=UserLocation(type="approximate"),
)

MLS_SEARCH_AGENT = Agent(
    name="Search & Match",
    instructions=prompts.mls_web_search_system_prompt(),
    model="gpt-5",
    tools=[MLS_WEB_SEARCH_TOOL],
    output_type=MLSWebSearchResult,
    model_settings=_default_reasoning_settings(),
)

# Buyer intake agent (LLM-managed stepper with structured JSON output)
INTAKE_AGENT = Agent(
    name="Buyer Intake",
    instructions=prompts.intake_agent_system_prompt(),
    model="gpt-5",
    output_type=IntakeAgentOutput,
    model_settings=_default_reasoning_settings(),
)

def _agent_trace_context(workflow_name: str, agent_name: str):
    current = get_current_trace()
    if current:
        return agent_span(agent_name, parent=current)
    return trace(workflow_name)

def _run_agent(agent: Agent[Any], agent_input: Union[str, List[Dict[str, Any]]]):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_queue: Queue[Tuple[bool, Any]] = Queue()

        def _target() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                run_result = new_loop.run_until_complete(
                    Runner.run(agent, input=agent_input)
                )
                result_queue.put((True, run_result))
            except Exception as exc:  # pragma: no cover - propagate back to caller
                result_queue.put((False, exc))
            finally:
                try:
                    new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                except Exception:
                    pass
                new_loop.close()
                asyncio.set_event_loop(None)

        thread = threading.Thread(target=_target, name=f"{agent.name}-runner", daemon=True)
        thread.start()
        thread.join()
        try:
            success, payload = result_queue.get_nowait()
        except Empty as missing:
            raise RuntimeError("Agent runner thread completed without returning a result") from missing
        if success:
            return payload
        raise payload

    return Runner.run_sync(agent, input=agent_input)


def _estimate_monthly_payment(price: float, down_pct: float = 0.2, rate: float = 0.065, years: int = 30) -> float:
    loan = price * (1 - down_pct)
    n = years * 12
    r = rate / 12.0
    if r == 0:
        return loan / n
    return loan * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def _format_currency(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.0f}"


def _coerce_price(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.]", "", value)
        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


def _extract_domain(url: Optional[str]) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc or None
    except Exception:  # pragma: no cover
        return None


def _derive_property_id(listing: Dict[str, Any]) -> str:
    existing = listing.get("property_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    url = listing.get("source_url")
    domain = _extract_domain(url)
    if url:
        slug = url.rstrip("/").split("/")[-1]
        slug = slug[:48] if slug else "listing"
    else:
        slug = "listing"
    return f"{(domain or 'property')}-{slug}-{uuid.uuid4().hex[:6]}"


def _response_output_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                candidate = getattr(content, "text", "")
                if candidate:
                    return candidate
    raise ValueError("No textual output in response")


def _schema_to_text_config(schema_def: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "type": schema_def.get("type", "json_schema"),
        "name": schema_def["name"],
        "schema": schema_def["schema"],
    }
    if "description" in schema_def:
        cfg["description"] = schema_def["description"]
    if "strict" in schema_def:
        cfg["strict"] = schema_def["strict"]
    return {"format": cfg}


def _parse_json_from_text(text: str) -> Any:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Empty response payload")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        obj, _ = JSON_DECODER.raw_decode(cleaned)
        return obj
    except json.JSONDecodeError:
        pass

    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = JSON_DECODER.raw_decode(cleaned, idx)
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Unable to parse JSON payload from response")


def _mls_search_live(criteria: Dict[str, Any]) -> Dict[str, Any]:
    criteria_lines = [
        f"Budget: {criteria.get('budget') or 'unspecified'}",
        f"Target locations: {criteria.get('locations') or 'unspecified'}",
        f"Home type: {criteria.get('home_type') or 'unspecified'}",
        f"Bedrooms: {criteria.get('bedrooms') or 'unspecified'}",
        f"Bathrooms: {criteria.get('bathrooms') or 'unspecified'}",
        f"Timeline: {criteria.get('timeline') or 'unspecified'}",
        f"Must haves: {criteria.get('must_haves') or 'unspecified'}",
        f"Deal breakers: {criteria.get('deal_breakers') or 'unspecified'}",
    ]
    user_prompt = prompts.mls_web_search_user_prompt(MLS_SEARCH_DOMAINS, criteria_lines)
    try:
        with _agent_trace_context("Real Estate Search & Match", MLS_SEARCH_AGENT.name):
            result = _run_agent(MLS_SEARCH_AGENT, user_prompt)
        payload = result.final_output_as(MLSWebSearchResult, raise_if_incorrect_type=True)
        return payload.model_dump(mode="json")
    except Exception as exc:  # pragma: no cover - fallback path
        log.exception("Search agent run failed; falling back to Responses API", exc_info=exc)

        model_name = "gpt-4o-mini"
        request_args: Dict[str, Any] = {
            "model": model_name,
            "input": [
                {"role": "system", "content": prompts.mls_web_search_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [{"type": "web_search"}],
            "text": _schema_to_text_config(get_mls_web_search_schema()),
        }
        if _can_set_temperature(model_name, 0.2):
            request_args["temperature"] = 0.2

        resp = client.responses.create(**request_args)
        raw_text = _response_output_text(resp)
        try:
            parsed = _parse_json_from_text(raw_text)
        except Exception as inner_exc:
            log.debug("Raw MLS search response text: %s", raw_text)
            raise ValueError("Failed to parse MLS search JSON payload") from inner_exc
        if not isinstance(parsed, dict):
            raise ValueError("MLS search response payload is not an object")
        return parsed


def search_and_match(session: Session) -> List[MatchItem]:
    state = session.intake
    required_fields = get_required_intake_fields()
    criteria = {key: getattr(state, key) for key in required_fields}
    listings: List[Dict[str, Any]] = []
    try:
        live_payload = _mls_search_live(criteria)
        listings = live_payload.get("listings", []) if isinstance(live_payload, dict) else []
    except Exception as exc:  # pragma: no cover
        log.exception("MLS web search call failed", exc_info=exc)
        raise RuntimeError("MLS web search failed; please retry later") from exc

    def _listing_to_match(listing: Dict[str, Any]) -> MatchItem:
        price_val = _coerce_price(listing.get("list_price"))
        monthly_val = (_estimate_monthly_payment(price_val) + 650) if price_val else None
        fit = listing.get("fit_reasoning") or listing.get("notes") or (
            f"Matches stated preferences for {criteria.get('locations') or 'target area'}."
        )
        commute = listing.get("commute_notes") or (
            f"Confirm commute from {listing.get('address', 'property address')} to {criteria.get('locations') or 'target area'}."
        )
        slots_clean: List[Dict[str, str]] = []
        raw_slots = listing.get("open_house_slots") or []
        if isinstance(raw_slots, dict):
            raw_slots = [raw_slots]
        if isinstance(raw_slots, list):
            for slot in raw_slots:
                if not isinstance(slot, dict):
                    continue
                start = _normalize_time_string(slot.get("start"))
                end = _normalize_time_string(slot.get("end"))
                if start and end:
                    slots_clean.append({"start": start, "end": end})
        beds_val = listing.get("beds")
        baths_val = listing.get("baths")
        source_url = listing.get("source_url")
        source_name = listing.get("source_name") or (_extract_domain(source_url) or "web_search")
        return MatchItem(
            property_id=_derive_property_id(listing),
            address=listing.get("address", "Unknown address"),
            fit_rationale=fit,
            estimated_commute=commute,
            estimated_monthly_payment=_format_currency(monthly_val),
            open_house_slots=slots_clean,
            list_price=price_val,
            list_price_display=_format_currency(price_val) if price_val is not None else None,
            beds=str(beds_val) if beds_val not in (None, "") else None,
            baths=str(baths_val) if baths_val not in (None, "") else None,
            source_url=source_url,
            source_name=source_name,
        )

    matches = [_listing_to_match(item) for item in listings[:7]]
    return matches


# ---------------------------------------------------------------------------
# Tour planning helpers
# ---------------------------------------------------------------------------
class TourInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    open_houses: List[Dict[str, Any]] = Field(
        description="List of {property_id, address, slots:[{'start':'HH:MM','end':'HH:MM'}]}"
    )
    preferred_windows: List[Dict[str, str]] = Field(
        description="User preferred windows: [{'start':'HH:MM','end':'HH:MM'}]"
    )


def _time_to_minutes(time_str: str) -> int:
    hours, minutes = time_str.split(":")
    return int(hours) * 60 + int(minutes)


def _minutes_to_time(total_minutes: int) -> str:
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _extract_tour_plan_inputs(message: str) -> Dict[str, Any]:
    return llm_json(
        system_prompt=prompts.tour_plan_extraction_system_prompt(),
        user_prompt=message,
        json_schema=get_tour_plan_extraction_schema(),
    )


def _normalize_time_string(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    text = text.replace(".", ":")
    text = text.replace("noon", "12:00pm")
    text = text.replace("midnight", "12:00am")
    text = text.replace("–", " ")
    text = text.replace("—", " ")
    text = re.sub(r"[,-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common day/month prefixes so formats like "Sat 1:00 PM" parse cleanly.
    day_month_tokens = {
        "sun",
        "mon",
        "tue",
        "wed",
        "thu",
        "thur",
        "thurs",
        "fri",
        "sat",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
    tokens: List[str] = []
    for part in text.split():
        if part in day_month_tokens:
            continue
        tokens.append(part)
    text = " ".join(tokens)

    match = re.search(r"(\d{1,2})(?:[:](\d{2}))?\s*(am|pm)", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        meridiem = match.group(3)
        if hour == 12:
            hour = 0 if meridiem == "am" else 12
        elif meridiem == "pm":
            hour += 12
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"
        return None

    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"
        return None

    digits_only = re.sub(r"\D", "", text)
    if digits_only.isdigit():
        text = digits_only
    if text.isdigit():
        hour = 0
        minute = 0
        if len(text) == 4:
            hour = int(text[:2])
            minute = int(text[2:])
        elif len(text) == 3:
            hour = int(text[:1])
            minute = int(text[1:])
        elif len(text) in (1, 2):
            hour = int(text)
            minute = 0
        else:
            return None
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"

    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"
    return None


def _sanitize_open_houses(entries: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    sanitized: List[Dict[str, Any]] = []
    issues: List[str] = []
    for item in entries or []:
        if not isinstance(item, dict):
            issues.append(f"Ignored open house entry; expected object, got {type(item).__name__}: {item!r}")
            continue
        address_raw = item.get("address")
        address = str(address_raw).strip() if address_raw is not None else ""
        raw_slots = item.get("slots")
        if raw_slots in (None, ""):
            raw_slots = []
        if not isinstance(raw_slots, list):
            issues.append(
                f"Ignored slots for {address or item!r}; expected list, got {type(raw_slots).__name__}"
            )
            raw_slots = []

        slots: List[Dict[str, str]] = []
        for slot in raw_slots:
            if not isinstance(slot, dict):
                issues.append(f"Ignored slot entry; expected object, got {type(slot).__name__}: {slot!r}")
                continue
            start = _normalize_time_string(slot.get("start"))
            end = _normalize_time_string(slot.get("end"))
            if start and end:
                slots.append({"start": start, "end": end})
            else:
                issues.append(
                    f"Ignored slot for {address or item!r}; unable to parse times {slot.get('start')!r}, {slot.get('end')!r}"
                )

        if not address:
            issues.append(f"Open house missing address: {item!r}")
            continue

        pid_raw = item.get("property_id")
        pid = str(pid_raw).strip() if pid_raw not in (None, "") else None
        sanitized.append({"property_id": pid, "address": address, "slots": slots})
        if not slots:
            issues.append(
                f"Open house for {address} has no usable time slots; keeping property with an empty slot list."
            )
    return sanitized, issues


def _sanitize_windows(entries: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, str]], List[str]]:
    windows: List[Dict[str, str]] = []
    issues: List[str] = []
    for item in entries or []:
        if not isinstance(item, dict):
            issues.append(
                f"Ignored preferred window; expected object, got {type(item).__name__}: {item!r}"
            )
            continue
        start = _normalize_time_string(item.get("start"))
        end = _normalize_time_string(item.get("end"))
        if start and end:
            windows.append({"start": start, "end": end})
        else:
            issues.append(
                f"Ignored preferred window; unable to parse times {item.get('start')!r}, {item.get('end')!r}"
            )
    return windows, issues


def _plan_tours_rule_based(
    inp: TourInput, *, visit_minutes: int = 30, travel_minutes: int = 20
) -> Dict[str, Any]:

    slot_entries: List[Dict[str, Any]] = []
    missing_slot_addresses: List[str] = []
    for open_house in inp.open_houses:
        prop_id = open_house.get("property_id") or f"addr:{open_house['address']}"
        if not open_house.get("slots"):
            missing_slot_addresses.append(open_house["address"])
        for slot in open_house.get("slots", []):
            slot_entries.append(
                {
                    "property_id": prop_id,
                    "address": open_house["address"],
                    "start": slot["start"],
                    "end": slot["end"],
                    "start_min": _time_to_minutes(slot["start"]),
                    "end_min": _time_to_minutes(slot["end"]),
                }
            )
    slot_entries.sort(key=lambda item: (item["start_min"], item["address"]))

    window_entries: List[Dict[str, Any]] = []
    for window in inp.preferred_windows:
        start_min = _time_to_minutes(window["start"])
        end_min = _time_to_minutes(window["end"])
        window_entries.append({"start": window["start"], "end": window["end"], "start_min": start_min, "end_min": end_min})
    window_entries.sort(key=lambda item: item["start_min"])

    coverage_minutes = sum(max(0, item["end_min"] - item["start_min"]) for item in window_entries)
    house_reasoning: List[str] = [
        f"Cataloged {len(inp.open_houses)} properties with {len(slot_entries)} normalized open house slots.",
        f"Mapped {len(window_entries)} preferred windows totaling {coverage_minutes} minutes of availability to align with 30-minute visits and {travel_minutes}-minute buffers.",
    ]
    if missing_slot_addresses:
        addressed = ", ".join(sorted(set(missing_slot_addresses)))
        house_reasoning.append(
            f"No published open house times for {addressed}; plan includes outreach steps for private showings or virtual walk-throughs."
        )

    property_index: Dict[str, Dict[str, Any]] = {}
    for open_house in inp.open_houses:
        prop_id = open_house.get("property_id") or f"addr:{open_house['address']}"
        property_index[prop_id] = {
            "address": open_house["address"],
            "scheduled": False,
            "feasible_windows": [],
            "has_slots": bool(open_house.get("slots")),
        }

    for window in window_entries:
        overlaps: List[str] = []
        for slot in slot_entries:
            overlap_start = max(window["start_min"], slot["start_min"])
            overlap_end = min(window["end_min"], slot["end_min"])
            if overlap_start + visit_minutes <= overlap_end:
                overlaps.append(f"{slot['address']} {slot['start']}-{slot['end']}")
                prop = property_index.get(slot["property_id"])
                if prop is not None:
                    prop["feasible_windows"].append(f"{window['start']}-{window['end']}")
        if overlaps:
            joined = "; ".join(sorted(set(overlaps)))
            house_reasoning.append(
                f"Window {window['start']}-{window['end']} supports {len(overlaps)} candidate visit block(s): {joined}."
            )
        else:
            house_reasoning.append(
                f"Window {window['start']}-{window['end']} has no slots wide enough to host a 30-minute visit plus travel buffer considerations."
            )

    schedule: List[Dict[str, Any]] = []
    used_slots: set = set()
    for window in window_entries:
        current_time = window["start_min"]
        while current_time + visit_minutes <= window["end_min"]:
            feasible_slots = [
                slot
                for slot in slot_entries
                if (slot["property_id"], slot["start"], slot["end"]) not in used_slots
                and slot["start_min"] <= current_time
                and current_time + visit_minutes <= slot["end_min"]
            ]
            if feasible_slots:
                chosen_slot = min(feasible_slots, key=lambda item: (item["end_min"], item["start_min"]))
                visit_block = {
                    "property_id": chosen_slot["property_id"],
                    "address": chosen_slot["address"],
                    "visit": {
                        "start": _minutes_to_time(current_time),
                        "end": _minutes_to_time(current_time + visit_minutes),
                    },
                    "travel_buffer": f"{travel_minutes}min",
                }
                schedule.append(visit_block)
                used_slots.add((chosen_slot["property_id"], chosen_slot["start"], chosen_slot["end"]))
                prop = property_index.get(chosen_slot["property_id"])
                if prop is not None:
                    prop["scheduled"] = True
                current_time += visit_minutes + travel_minutes
                continue

            upcoming_slots = [
                slot
                for slot in slot_entries
                if (slot["property_id"], slot["start"], slot["end"]) not in used_slots
                and slot["start_min"] > current_time
                and slot["start_min"] + visit_minutes <= window["end_min"]
            ]
            if not upcoming_slots:
                current_time += 10
                continue
            next_slot = min(upcoming_slots, key=lambda item: item["start_min"])
            current_time = max(current_time + 10, next_slot["start_min"])

    schedule.sort(key=lambda item: item["visit"]["start"])

    previous_end: Optional[int] = None
    for block in schedule:
        start_min = _time_to_minutes(block["visit"]["start"])
        end_min = _time_to_minutes(block["visit"]["end"])
        window_hit = next(
            (f"{window['start']}-{window['end']}" for window in window_entries if window["start_min"] <= start_min and end_min <= window["end_min"]),
            None,
        )
        if previous_end is None:
            house_reasoning.append(
                f"Scheduled {block['address']} from {block['visit']['start']} to {block['visit']['end']} within preferred window {window_hit} to anchor the day."
            )
        else:
            gap = start_min - previous_end
            if gap >= travel_minutes:
                house_reasoning.append(
                    f"Planned {block['address']} at {block['visit']['start']}-{block['visit']['end']} leaving {gap} minutes between visits for travel (needs {travel_minutes})."
                )
            else:
                house_reasoning.append(
                    f"WARNING: {block['address']} at {block['visit']['start']} compresses travel buffer to {gap} minutes; investigate ride-share or adjust slot."
                )
        previous_end = end_min

    unscheduled_details: List[str] = []
    for property_id, details in property_index.items():
        if not details["scheduled"]:
            feasible = sorted(set(details["feasible_windows"]))
            if not details["has_slots"]:
                unscheduled_details.append(
                    f"{details['address']} lacked published open house slots; coordinate directly with the listing agent for times or arrange a private showing/virtual tour."
                )
            elif feasible:
                unscheduled_details.append(
                    f"{details['address']} remained unscheduled because other commitments consumed matching windows ({', '.join(feasible)})."
                )
            else:
                unscheduled_details.append(
                    f"{details['address']} remained unscheduled because no preferred windows overlapped sufficiently with its slots."
                )
    if unscheduled_details:
        house_reasoning.append("Unscheduled properties: " + " ".join(unscheduled_details))
    else:
        house_reasoning.append("All candidate properties received confirmed visits within preferred windows while preserving travel buffers.")

    if not schedule:
        if missing_slot_addresses:
            house_reasoning.append(
                "No feasible visits landed because several properties lacked confirmed viewing slots; proposing outreach-driven follow-ups."
            )
        else:
            house_reasoning.append(
                "No feasible visits landed within the provided windows after enforcing duration and travel constraints."
            )

    travel_guidance = f"Maintain at least {travel_minutes} minutes for travel; extend buffers if traffic forecasts worsen."
    if missing_slot_addresses:
        travel_guidance += " Confirm availability with listing teams for properties without published slots and insert times once secured."

    if schedule:
        summary_text = f"Planned {len(schedule)} visit(s) across {len(inp.open_houses)} properties."
    elif missing_slot_addresses:
        summary_text = (
            "No visits scheduled yet because open house slots were unavailable; reach out for times, propose private previews, or schedule virtual tours."
        )
    else:
        summary_text = "No feasible schedule generated; consider expanding preferred windows or requesting alternate open house slots."

    final_schedule = {
        "visits": schedule,
        "travel_guidance": travel_guidance,
        "summary": summary_text,
    }

    calendar_reasoning = [
        "Map normalized tour data (addresses, slots, travel buffers) to calendar event payloads with explicit time zones.",
        "Select calendar providers (Google, Outlook) and register OAuth 2.0 client credentials to obtain user-granted tokens.",
        "Create events with structured descriptions (property details, contact info, travel guidance) and attach buffer blocks for commute segments.",
        "Subscribe to change notifications or use incremental sync to detect external edits and reconcile conflicts before overwriting user changes.",
        "Implement token refresh, retry with exponential backoff, and granular permissions revocation handling to keep integrations resilient.",
    ]

    route_reasoning = [
        "Accept bulk tour inputs via CSV/API containing property coordinates, slot windows, visit durations, and travel estimates.",
        "Geocode addresses and build distance matrices using mapping APIs, caching results to control quota usage.",
        "Model the problem as a vehicle routing problem with time windows (VRPTW) capturing visit durations and allowable arrival ranges.",
        "Run an optimization engine (e.g., OR-Tools CP-SAT) with constraints for maximum daily drive time, preferred start/end anchors, and optional priorities.",
        "Return optimized sequences with slack indicators so dispatchers can tweak manually and re-run scenarios quickly.",
    ]

    confirmation_reasoning = [
        "Consolidate participant preferences (SMS, email) along with backup contact channels for contingencies.",
        "Generate confirmation messages immediately after scheduling, embedding itinerary summaries, agent contacts, and property highlights.",
        "Schedule reminders at logical intervals (24 hours, 3 hours, and 30 minutes prior) while accounting for time zone shifts and daylight savings.",
        "Include dynamic update links so participants can acknowledge, request changes, or report delays in real time.",
        "Track delivery and engagement metrics to escalate follow-ups if acknowledgements are missing before critical milestones.",
    ]

    feedback_reasoning = [
        "Define concise surveys per property focusing on fit, condition, value, and qualitative notes.",
        "Trigger feedback requests within 12-24 hours of each visit via preferred channels to capture impressions while fresh.",
        "Provide mobile-friendly response formats and allow quick voice-to-text notes for on-the-go agents.",
        "Aggregate results into a centralized CRM or analytics dashboard with tagging by property and buyer sentiment trends.",
        "Loop insights back into search criteria refinement and follow-up communication with listing agents.",
    ]

    return {
        "house_visit_schedule": {
            "reasoning_steps": house_reasoning,
            "final_recommendation": final_schedule,
        },
        "calendar_integration": {
            "reasoning_steps": calendar_reasoning,
            "final_recommendation": "Integrate with Google and Outlook calendars via OAuth 2.0, pushing events with travel buffers and reconciling updates through webhook-backed sync pipelines.",
        },
        "route_optimization": {
            "reasoning_steps": route_reasoning,
            "final_recommendation": "Adopt a VRPTW solver (e.g., OR-Tools CP-SAT) fed by cached distance matrices to produce optimized multi-stop tour routes with manual override support.",
        },
        "confirmation_and_reminders": {
            "reasoning_steps": confirmation_reasoning,
            "final_recommendation": "Automate confirmations immediately after booking and dispatch reminders at 24h, 3h, and 30m intervals with actionable update links.",
        },
        "feedback_capture": {
            "reasoning_steps": feedback_reasoning,
            "final_recommendation": "Deliver post-tour micro-surveys within 24h and feed results into the CRM analytics layer to inform follow-ups and search refinement.",
        },
    }


def plan_tours(inp: TourInput) -> Dict[str, Any]:
    visit_minutes = 30
    travel_minutes = 20
    user_prompt = prompts.tour_plan_planner_user_prompt(
        inp.open_houses,
        inp.preferred_windows,
        visit_minutes=visit_minutes,
        travel_minutes=travel_minutes,
    )
    try:
        return llm_json(
            system_prompt=prompts.tour_plan_planner_system_prompt(),
            user_prompt=user_prompt,
            json_schema=get_tour_plan_generation_schema(),
            model="gpt-4o-mini",
            temperature=0.2,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("Tour plan LLM failed, falling back to rule-based planner: %s", exc)
        return _plan_tours_rule_based(
            inp,
            visit_minutes=visit_minutes,
            travel_minutes=travel_minutes,
        )


# ---------------------------------------------------------------------------
# Disclosure Q&A helper
# ---------------------------------------------------------------------------
class QAFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    content: str


def _find_sentences(text: str, needles: List[str]) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches: List[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(needle in lower for needle in needles):
            matches.append(sentence.strip())
    return matches[:6]


def disclosure_qa(files: List[QAFile], question: str) -> Dict[str, Any]:
    needles = [word.strip().lower() for word in question.split() if len(word) > 3]
    reasoning = [
        "Parsed question and selected keywords for lookup.",
        "Searched documents for matching sentences and compiled candidate evidence.",
    ]
    answers = []
    for item in files:
        hits = _find_sentences(item.content, needles)
        if not hits:
            continue
        citations = [{"source": item.name, "page": "?", "text": sentence[:280]} for sentence in hits[:3]]
        answers.append(
            {
                "question": question,
                "answer": "; ".join(hits[:2]) if hits else "No explicit disclosure found.",
                "citations": citations,
                "risk_flags": [
                    "Potential ambiguity: disclosures and inspection notes may conflict; verify recency and repair receipts."
                ],
                "follow_up_questions": [
                    "Please provide repair invoices or contractor reports, if any.",
                    "Confirm dates of any noted issues and whether they were remediated.",
                ],
            }
        )
    if not answers:
        answers.append(
            {
                "question": question,
                "answer": "The supplied files contain no clear statements matching the question.",
                "citations": [],
                "risk_flags": ["Missing disclosure details; request updated forms or third-party inspections."],
                "follow_up_questions": ["Can you share the specific section or page where this is discussed?"],
            }
        )
    return {"reasoning_steps": reasoning, "answers": answers}


# ---------------------------------------------------------------------------
# FastMCP tools
# ---------------------------------------------------------------------------
mcp = FastMCP("RealEstate Workflow MCP")


# ---------------------------------------------------------------------------
# Enhanced workflow execution (following workflow pattern)
# ---------------------------------------------------------------------------
class WorkflowInput(BaseModel):
    """Input model for workflow execution."""
    input_as_text: str
    session_id: Optional[str] = None


async def run_enhanced_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    """
    Enhanced workflow execution following the pattern with conversation history,
    guardrails, and conditional agent routing.
    """
    with trace("Real Estate Workflow Enhanced"):
        # Get or create session
        sid, session = _get_session(workflow_input.session_id)
        
        # Initialize conversation history for this workflow run
        user_message = workflow_input.input_as_text
        conversation_history: List[TResponseInputItem] = list(session.conversation_history)
        
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_message
                }
            ]
        })
        
        # Step 1: Run guardrails check
        guardrails_inputtext = user_message
        try:
            if _HAS_GUARDRAILS:
                guardrails_result = await run_guardrails(
                    ctx, 
                    guardrails_inputtext, 
                    "text/plain", 
                    instantiate_guardrails(load_config_bundle(GUARDRAILS_CONFIG)), 
                    suppress_tripwire=True, 
                    raise_guardrail_errors=True
                )
                guardrails_hastripwire = guardrails_has_tripwire(guardrails_result)
                guardrails_anonymizedtext = get_guardrail_checked_text(guardrails_result, guardrails_inputtext)
                guardrails_output = (guardrails_hastripwire and build_guardrail_fail_output(guardrails_result or [])) or (guardrails_anonymizedtext or guardrails_inputtext)
                
                if guardrails_hastripwire:
                    return {
                        "session_id": sid,
                        "status": "blocked",
                        "guardrails_output": guardrails_output
                    }
            else:
                # Fallback to basic moderation check
                guard = jailbreak_check(user_message)
                if not guard.allowed:
                    return {
                        "session_id": sid,
                        "status": "blocked",
                        "reason": "Guardrails/Jailbreak check failed",
                        "details": guard.reasons,
                    }
        except Exception as exc:
            log.warning("Guardrails check failed, proceeding without guardrails: %s", exc)
        
        # Step 2: Run router with conversation history context
        router_input = [
            *conversation_history,
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You need to remember chat history and use chat history to provide consistent analysis of routing decisions"
                    }
                ]
            }
        ]
        
        router_result_temp = await Runner.run(
            ROUTER_AGENT,
            input=router_input,
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": f"wf_{uuid.uuid4().hex}",
                "session_id": sid
            })
        )
        
        # Update conversation history with router response
        conversation_history.extend([item.to_input_item() for item in router_result_temp.new_items])
        
        router_output = router_result_temp.final_output_as(RouterOutputModel, raise_if_incorrect_type=True)
        router_result = {
            "output_text": router_output.model_dump_json(),
            "output_parsed": router_output.model_dump()
        }
        
        # Step 3: Route to appropriate agent based on intent
        intent = router_result["output_parsed"]["intent"]
        slots_obj = router_output.slots_as_dict()
        
        final_result = {
            "session_id": sid,
            "status": "ok",
            "router": router_result,
            "intent": intent,
            "slots": slots_obj,
        }
        
        # Route based on intent
        if intent == 'buy':
            # Run buyer intake workflow
            intake_result = intake_step(session, user_message, slots_obj)
            
            # Check if intake is complete
            if intake_is_complete(session) and intake_result.get("status") == "confirmed":
                # Auto-execute search & match
                try:
                    match_objs = search_and_match(session)
                    matches_payload = [match.model_dump() for match in match_objs]
                    
                    final_result["buyer_intake"] = intake_result
                    final_result["search_and_match"] = {
                        "matches": matches_payload,
                        "count": len(matches_payload),
                        "message": f"Found {len(matches_payload)} matching properties based on your criteria."
                    }
                    final_result["next_step"] = "tour_planning"
                except Exception as exc:
                    log.exception("Failed to auto-execute search_and_match: %s", exc)
                    final_result["buyer_intake"] = intake_result
                    final_result["error"] = str(exc)
                    final_result["next_step"] = "search_and_match"
            else:
                final_result["buyer_intake"] = intake_result
                final_result["next_step"] = "continue_intake"
                
        elif intent == 'sell':
            # For sell intent, provide seller workflow message
            final_result["message"] = "Seller workflow activated. Please provide your property details to begin the listing process."
            final_result["next_step"] = "seller_intake"
            
        elif intent == 'disclosures':
            # For disclosure requests
            final_result["message"] = "Please provide the disclosure documents and your specific question."
            final_result["next_step"] = "disclosure_qa"
            
        elif intent == 'offer':
            # For offer drafting
            final_result["message"] = "Please provide property details to draft an offer."
            final_result["next_step"] = "offer_drafter"
            
        else:
            # General intent
            final_result["message"] = "I can help you with buying, selling, property disclosures, or drafting offers. How can I assist you?"
            final_result["next_step"] = "router"
        
        # Update session conversation history
        session.conversation_history = conversation_history
        
        return final_result


@mcp.tool
def run_workflow(user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Main workflow orchestrator. Routes user messages through guardrails, intent detection,
    and appropriate sub-agents.
    
    Args:
        user_message: The user's input message
        session_id: Optional session identifier for conversation continuity
        
    Returns:
        Dictionary containing workflow results, intent, and next steps
    """
    workflow_input = WorkflowInput(input_as_text=user_message, session_id=session_id)
    
    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
        # If we have a running loop, we need to run in a new thread
        result_queue: Queue[Tuple[bool, Any]] = Queue()
        
        def _async_runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(run_enhanced_workflow(workflow_input))
                result_queue.put((True, result))
            except Exception as exc:
                result_queue.put((False, exc))
            finally:
                new_loop.close()
        
        thread = threading.Thread(target=_async_runner, daemon=True)
        thread.start()
        thread.join()
        
        success, payload = result_queue.get()
        if success:
            return payload
        raise payload
        
    except RuntimeError:
        # No running loop, we can create one
        return asyncio.run(run_enhanced_workflow(workflow_input))


@mcp.tool
def buyer_intake_step(session_id: Optional[str] = None, user_message: Optional[str] = None) -> Dict[str, Any]:
    sid, session = _get_session(session_id)
    result = intake_step(session, user_message)
    done = intake_is_complete(session)
    next_node = None
    next_step_result = None

    status = (result.get("status") or "").lower()
    user_confirmed = status == "confirmed"
    if not user_confirmed and user_message:
        low = user_message.strip().lower()
        user_confirmed = low in {"yes", "y", "confirmed", "correct", "confirm"}
        if user_confirmed and status != "confirmed" and done:
            result["status"] = "confirmed"
            status = "confirmed"

    if done and status in {"summary", "confirmed"}:
        if user_confirmed:
            if not result.get("summary"):
                result["summary"] = _intake_state_snapshot(session.intake)
            next_node = "tour_plan_tool"
            try:
                match_objs = search_and_match(session)
                matches_payload = [match.model_dump() for match in match_objs]
                open_house_payload = [
                    {
                        "property_id": match.property_id,
                        "address": match.address,
                        "slots": match.open_house_slots,
                    }
                    for match in match_objs
                ]
                match_count = len(matches_payload)

                if match_count == 0:
                    next_node = "search_and_match_tool"
                    next_step_result = {
                        "step": "search_and_match",
                        "matches": matches_payload,
                        "open_houses": open_house_payload,
                        "count": match_count,
                        "message": "No matching properties were found with the current criteria. Adjust your preferences or try again shortly.",
                        "next_node": "search_and_match_tool",
                    }
                    result["auto_executed_next_step"] = True
                else:
                    next_node = "tour_plan_tool"
                    next_step_result = {
                        "step": "search_and_match",
                        "matches": matches_payload,
                        "open_houses": open_house_payload,
                        "count": match_count,
                        "next_node": "tour_plan_tool",
                        "message": "Review the recommended properties and proceed with tour planning using available open house slots or queue creative follow-ups where times are missing.",
                    }
                    result["auto_executed_next_step"] = True
            except Exception as exc:
                log.exception("Failed to auto-execute search_and_match: %s", exc)
                next_step_result = {
                    "step": "search_and_match",
                    "error": str(exc),
                    "message": "Please call search_and_match_tool manually",
                }
                next_node = "search_and_match"
        else:
            next_node = "search_and_match"

    return {
        "session_id": sid,
        "result": result,
        "intake_complete": done,
        "next_node": next_node,
        "next_step_result": next_step_result,
    }


@mcp.tool
def search_and_match_tool(session_id: Optional[str] = None) -> Dict[str, Any]:
    sid, session = _get_session(session_id)
    if not intake_is_complete(session):
        return {
            "session_id": sid,
            "status": "error",
            "step": "search_and_match",
            "message": "Intake not complete. Call buyer_intake_step until summary confirmed.",
            "next_node": "buyer_intake_step",
        }
    try:
        match_objs = search_and_match(session)
        matches = [item.model_dump() for item in match_objs]
    except Exception as exc:
        return {
            "session_id": sid,
            "status": "error",
            "step": "search_and_match",
            "message": str(exc),
            "next_node": "search_and_match_tool",
        }

    open_houses = [
        {"property_id": item.property_id, "address": item.address, "slots": item.open_house_slots}
        for item in match_objs
    ]
    match_count = len(matches)

    if match_count == 0:
        return {
            "session_id": sid,
            "status": "ok",
            "step": "search_and_match",
            "matches": matches,
            "open_houses": open_houses,
            "count": match_count,
            "next_node": "search_and_match_tool",
            "message": "No matching properties were found with the current criteria. Update the preferences or retry in a bit.",
        }

    return {
        "session_id": sid,
        "status": "ok",
        "step": "search_and_match",
        "matches": matches,
        "open_houses": open_houses,
        "count": match_count,
        "next_node": "tour_plan_tool",
        "message": "Provide open house options and preferred windows to plan tours, noting any listings that need creative scheduling follow-ups due to missing slots.",
    }


@mcp.tool
def tour_plan_tool(
    open_houses: Optional[List[Dict[str, Any]]] = None,
    preferred_windows: Optional[List[Dict[str, str]]] = None,
    user_message: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    sid, _ = _get_session(session_id)

    extracted: Dict[str, Any] = {"open_houses": [], "preferred_windows": []}
    if user_message and user_message.strip():
        try:
            extracted = _extract_tour_plan_inputs(user_message)
        except Exception as exc:  # pragma: no cover
            log.warning("Tour plan extraction failed; expecting structured inputs: %s", exc)

    parsed_open_houses, open_house_issues = _sanitize_open_houses(open_houses or extracted.get("open_houses"))
    parsed_windows, window_issues = _sanitize_windows(preferred_windows or extracted.get("preferred_windows"))
    issues = open_house_issues + window_issues

    if not parsed_open_houses or not parsed_windows:
        missing = []
        if not parsed_open_houses:
            missing.append("open_houses")
        if not parsed_windows:
            missing.append("preferred_windows")
        error_payload: Dict[str, Any] = {
            "session_id": sid,
            "status": "error",
            "step": "tour_plan",
            "message": "Missing required inputs for tour planning.",
            "missing_inputs": missing,
            "expected_format": {
                "open_houses": [
                    {
                        "property_id": "id",
                        "address": "string",
                        "slots": [{"start": "HH:MM", "end": "HH:MM"}],
                    }
                ],
                "preferred_windows": [{"start": "HH:MM", "end": "HH:MM"}],
            },
            "hint": "Provide open house slots and preferred windows via fields or natural language in user_message.",
            "next_node": "tour_plan_tool",
        }
        if issues:
            error_payload["invalid_inputs"] = issues
        return error_payload

    plan = plan_tours(TourInput(open_houses=parsed_open_houses, preferred_windows=parsed_windows))
    response: Dict[str, Any] = {
        "session_id": sid,
        "status": "ok",
        "step": "tour_plan",
        "plan": plan,
        "message": "Review the proposed tour schedule and adjust inputs if needed.",
        "next_node": "negotiation_coach_tool",
    }
    if issues:
        response["warnings"] = issues
    return response


@mcp.tool
def disclosure_qa_tool(question: str, files: List[Dict[str, str]]) -> Dict[str, Any]:
    qa_files = [QAFile(**file_info) for file_info in files]
    return disclosure_qa(qa_files, question)


@mcp.tool
def offer_drafter_tool(prompt_context: str) -> Dict[str, Any]:
    return llm_json(
        system_prompt=prompts.offer_drafter_system_prompt(),
        user_prompt=prompt_context,
        json_schema=get_offer_drafter_schema(),
    )


@mcp.tool
def negotiation_coach_tool(prompt_context: str) -> Dict[str, Any]:
    return llm_json(
        system_prompt=prompts.negotiation_coach_system_prompt(),
        user_prompt=prompt_context,
        json_schema=get_negotiation_coach_schema(),
    )


@mcp.tool
def health() -> Dict[str, Any]:
    return {"status": "ok", "sessions": len(SESSIONS)}


@mcp.tool
def reset_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    sid, _ = _get_session(session_id)
    SESSIONS.pop(sid, None)
    return {"status": "reset", "session_id": sid}


__all__ = [
    "AsyncOpenAI",
    "FastMCP",
    "GuardrailResult",
    "IntakeState",
    "MatchItem",
    "QAFile",
    "RunConfig",
    "Session",
    "TResponseInputItem",
    "WorkflowInput",
    "build_guardrail_fail_output",
    "buyer_intake_step",
    "disclosure_qa",
    "disclosure_qa_tool",
    "get_guardrail_checked_text",
    "guardrails_has_tripwire",
    "health",
    "intake_is_complete",
    "intake_step",
    "jailbreak_check",
    "llm_json",
    "mcp",
    "negotiation_coach_tool",
    "offer_drafter_tool",
    "plan_tours",
    "reset_session",
    "router_route",
    "run_enhanced_workflow",
    "run_workflow",
    "search_and_match",
    "search_and_match_tool",
    "tour_plan_tool",
]


if __name__ == "__main__":  # pragma: no cover - entry point for local runs
    log.info("Starting MCP server ...")
    mcp.run()
