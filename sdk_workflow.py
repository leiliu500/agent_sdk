"""
Real‑estate workflow MCP server (OpenAI Agents SDK + FastMCP)

Features
- Jailbreak Guardrail gate (OpenAI Guardrails if configured; fallback to Moderations)
- Router (buy | sell | disclosures | offer | general) with JSON schema
- Buyer Agent with sub‑agents:
  * Buyer‑Intake (stepwise, no repeat questions; persistent per session)
  * Search & Match (stub MLS search + payment estimate; replace with real MLSSearchTool)
  * Tour Plan (open‑house schedule optimizer)
  * Disclosure Q&A (file‑based Q&A with sentence‑level citations)
  * Offer Drafter (compliant draft + checklist)
  * Negotiation Coach (market‑context driven offer logic)

Run
  uv run python real_estate_mcp_server.py  # or: python real_estate_mcp_server.py

Env
  OPENAI_API_KEY=...
  # Optional guardrails config directory (bundle)
  GUARDRAILS_BUNDLE_DIR=guardrails_bundle/

Notes
- This server exposes tools the host LLM/Agent can call. You can call a single
  orchestrator tool (run_workflow) or compose calls to the sub‑agents.
- Replace the MLS and commute/payment stubs with your production integrations.
- All LLM calls use JSON‑schema constrained responses for robust outputs.
"""
from __future__ import annotations

import os
import json
import time
import math
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# MCP server (FastMCP)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # FastMCP 2.x
    from fastmcp import FastMCP
except Exception:  # pragma: no cover
    # Fallback import path some distributions use
    from mcp.server.fastmcp import FastMCP  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client & Guardrails
# ──────────────────────────────────────────────────────────────────────────────
from openai import OpenAI

# Guardrails (optional, only if you provide a bundle dir)
try:
    from openai_guardrails.runtime import (
        load_config_bundle,  # type: ignore
        instantiate_guardrails,  # type: ignore
        run_guardrails,  # type: ignore
    )
    _HAS_GUARDRAILS = True
except Exception:
    _HAS_GUARDRAILS = False


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("real_estate_mcp")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Global state (simple in‑memory session store). Replace with Redis/Postgres
# for production.
# ──────────────────────────────────────────────────────────────────────────────
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
    consents: Optional[str] = None  # free‑text confirmation
    # track sequence position explicitly
    next_field: str = "budget"


@dataclass
class Session:
    created_ts: float
    last_ts: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    intake: IntakeState = field(default_factory=IntakeState)


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


# ──────────────────────────────────────────────────────────────────────────────
# JSON Schemas
# ──────────────────────────────────────────────────────────────────────────────
ROUTER_JSON_SCHEMA: Dict[str, Any] = {
    "name": "response_schema",
    "schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["buy", "sell", "disclosures", "offer", "general"]},
            "confidence": {"type": "number"},
            # NOTE: matches user's schema (string). We also return a parsed copy as slots_obj
            "slots": {"type": "string"},
        },
        "required": ["intent", "confidence", "slots"],
        "additionalProperties": False,
        "title": "response_schema",
    },
    "strict": True,
}

# Buyer‑Intake completion gate
REQUIRED_INTAKE_FIELDS = [
    "budget",
    "locations",
    "home_type",
    "bedrooms",
    "bathrooms",
    "financing_status",
    "timeline",
    "must_haves",
    "deal_breakers",
    "consents",
]


# ──────────────────────────────────────────────────────────────────────────────
# LLM Helpers
# ──────────────────────────────────────────────────────────────────────────────
client = OpenAI()


def llm_json(
    *,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    model: str = "gpt-4o-mini",  # adjust as needed
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """LLM call constrained to JSON schema using Chat Completions API."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": json_schema,
        },
    )
    try:
        # Extract the parsed JSON from the response
        text = resp.choices[0].message.content
        return json.loads(text)
    except Exception as e:  # pragma: no cover
        log.exception("Failed to parse JSON response: %s", e)
        raise
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Guardrails: Jailbreak validation
# ──────────────────────────────────────────────────────────────────────────────
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
        except Exception as e:  # pragma: no cover
            log.warning("Guardrails execution failed, falling back to Moderations: %s", e)
    # Fallback: Moderations
    try:
        m = client.moderations.create(model="omni-moderation-latest", input=user_message)
        flagged = any(cat for cat, val in m.results[0].categories.to_dict().items() if val)  # type: ignore
        if flagged:
            reasons = [
                k for k, v in m.results[0].category_scores.to_dict().items() if v and v > 0.5  # type: ignore
            ]
            return GuardrailResult(allowed=False, reasons=reasons)
        return GuardrailResult(allowed=True)
    except Exception as e:  # pragma: no cover
        log.warning("Moderations failed open; allowing request. Error: %s", e)
        return GuardrailResult(allowed=True)


# ──────────────────────────────────────────────────────────────────────────────
# Router Node
# ──────────────────────────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = (
    "You are an orchestration agent for a real-estate brokerage. Your task is to classify "
    "the user's intent strictly into one of these categories: {buy, sell, disclosures, offer}. "
    "Next, extract all relevant slots from the user's message, including budget, areas (geographical/locations of interest), "
    "timing, property_ref, listing_id, and contact information (email/phone) if it is explicitly provided. Do not infer any information related to Fair Housing protected categories or make assumptions not based on user statements.\n\n"
    "Your response should be a single JSON object containing: intent, confidence [0..1], and slots.\n"
    "If the user's request is ambiguous or falls outside the provided intent categories, use 'general'.\n"
    "Always reason step-by-step internally and only output the final JSON."
)


def router_route(user_message: str, session: Session) -> Dict[str, Any]:
    # include light chat history for consistency
    history_snippets = [
        f"U:{h['user']}\nA:{h['agent']}" for h in session.history[-4:] if 'user' in h and 'agent' in h
    ]
    history_text = "\n\n".join(history_snippets)

    user_prompt = (
        "Use prior messages for consistency if relevant (but only extract explicit facts).\n\n"
        f"Chat History:\n{history_text}\n\n"
        f"User message:\n{user_message}\n\n"
        "Return only the JSON."
    )

    out = llm_json(system_prompt=ROUTER_SYSTEM_PROMPT, user_prompt=user_prompt, json_schema=ROUTER_JSON_SCHEMA)

    # Safely parse slots string → dict
    slots_obj: Dict[str, Any] = {}
    try:
        slots_obj = json.loads(out.get("slots") or "{}")
    except Exception:
        slots_obj = {}
    # normalize slot keys
    for key in ["budget", "areas", "timing", "property_ref", "listing_id", "email", "phone"]:
        slots_obj.setdefault(key, None)

    out["slots_obj"] = slots_obj
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Buyer‑Intake Sub‑Agent (deterministic stepper, no repeats)
# ──────────────────────────────────────────────────────────────────────────────
INTAKE_QUESTIONS = {
    "budget": "What is your budget range for purchasing your new home?",
    "locations": "Which city, area, or neighborhoods are you considering?",
    "home_type": "What type of home would you like? (e.g., single-family, condo, townhouse)",
    "bedrooms": "How many bedrooms do you need?",
    "bathrooms": "How many bathrooms do you need?",
    "financing_status": "What is your financing status? (cash or mortgage; pre-approved?)",
    "timeline": "What is your timeline to move or purchase?",
    "must_haves": "What are your must-have features?",
    "deal_breakers": "Any deal-breakers we should avoid?",
    "consents": "Do you consent to us storing and using your information to search for properties and contact you with matches? (yes/no)",
}


def intake_step(session: Session, user_message: Optional[str]) -> Dict[str, Any]:
    s = session.intake
    # Update the previously asked field with user's latest answer when appropriate
    if user_message and s.next_field in INTAKE_QUESTIONS:
        current = s.next_field
        # naive de-dupe: if already filled, don't overwrite (respect no‑repeat rule)
        if getattr(s, current) is None:
            setattr(s, current, user_message.strip())
            # advance pointer
            i = REQUIRED_INTAKE_FIELDS.index(current)
            if i + 1 < len(REQUIRED_INTAKE_FIELDS):
                s.next_field = REQUIRED_INTAKE_FIELDS[i + 1]
            else:
                s.next_field = "done"

    # Decide what to ask next
    if s.next_field == "done":
        summary = {k: getattr(s, k) for k in REQUIRED_INTAKE_FIELDS}
        return {
            "status": "summary",
            "message": (
                "Here is a summary of your intake. Please confirm if everything is correct (yes/no) or indicate any corrections."
            ),
            "summary": summary,
        }

    next_q = INTAKE_QUESTIONS[s.next_field]
    # reference previously filled info as context (no repetition)
    ctx_bits = []
    for k in REQUIRED_INTAKE_FIELDS:
        v = getattr(s, k)
        if v and k != s.next_field:
            ctx_bits.append(f"{k.replace('_', ' ').title()}: {v}")
    context = ("You already shared → " + "; ".join(ctx_bits)) if ctx_bits else None

    return {
        "status": "ask",
        "field": s.next_field,
        "message": next_q if not context else f"{context}.\n{next_q}",
    }


def intake_is_complete(session: Session) -> bool:
    s = session.intake
    return all(getattr(s, k) not in (None, "") for k in REQUIRED_INTAKE_FIELDS)


# ──────────────────────────────────────────────────────────────────────────────
# Search & Match Sub‑Agent (stub MLSSearchTool)
# ──────────────────────────────────────────────────────────────────────────────
class MatchItem(BaseModel):
    property_id: str
    address: str
    fit_rationale: str
    estimated_commute: str
    estimated_monthly_payment: str


def _mls_search_stub(criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Replace with real MLS search. Returns a few mocked properties."""
    budget = criteria.get("budget", "?")
    loc = criteria.get("locations", "Preferred Area")
    base = [
        {"property_id": "MLS1001", "address": f"123 Maple St, {loc}", "price": 950000, "beds": 3, "baths": 2},
        {"property_id": "MLS1002", "address": f"456 Oak Ave, {loc}", "price": 880000, "beds": 3, "baths": 2.5},
        {"property_id": "MLS1003", "address": f"789 Pine Rd, {loc}", "price": 1025000, "beds": 4, "baths": 3},
        {"property_id": "MLS1004", "address": f"15 Cedar Ct, {loc}", "price": 825000, "beds": 2, "baths": 2},
    ]
    return base


def _estimate_monthly_payment(price: float, down_pct: float = 0.2, rate: float = 0.065, years: int = 30) -> float:
    loan = price * (1 - down_pct)
    n = years * 12
    r = rate / 12.0
    if r == 0:
        return loan / n
    return loan * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def search_and_match(session: Session) -> List[MatchItem]:
    s = session.intake
    criteria = {k: getattr(s, k) for k in REQUIRED_INTAKE_FIELDS}
    raw = _mls_search_stub(criteria)
    out: List[MatchItem] = []
    for p in raw[:7]:
        pay = _estimate_monthly_payment(p["price"]) + 650  # rough taxes+ins HOA pad
        rationale = (
            f"Within or near budget; matches bedrooms/bathrooms where possible; in {s.locations or 'target area'}."
        )
        commute = "25–35 min drive to downtown; 10–15 min to nearest school district (est.)."
        out.append(
            MatchItem(
                property_id=p["property_id"],
                address=p["address"],
                fit_rationale=rationale,
                estimated_commute=commute,
                estimated_monthly_payment=f"${pay:,.0f}",
            )
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tour Plan Sub‑Agent (optimizer)
# ──────────────────────────────────────────────────────────────────────────────
class TourInput(BaseModel):
    open_houses: List[Dict[str, Any]] = Field(
        description="List of {property_id, address, slots:[{'start':'HH:MM','end':'HH:MM'}]}"
    )
    preferred_windows: List[Dict[str, str]] = Field(
        description="User preferred windows: [{'start':'HH:MM','end':'HH:MM'}]"
    )


def _time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_time(x: int) -> str:
    return f"{x // 60:02d}:{x % 60:02d}"


def plan_tours(inp: TourInput) -> Dict[str, Any]:
    # Greedy schedule inside preferred windows with 30‑min visits + 20‑min travel
    VISIT = 30
    TRAVEL = 20
    reasoning: List[str] = [
        "Collected open house slots and user preferred windows.",
        "Use greedy selection to fit 30‑minute visits with 20‑minute travel buffers.",
    ]
    schedule: List[Dict[str, Any]] = []

    # flatten slots with property info
    slots = []
    for oh in inp.open_houses:
        for s in oh.get("slots", []):
            slots.append({"property_id": oh["property_id"], "address": oh["address"], **s})
    # iterate preferred windows
    for win in inp.preferred_windows:
        start = _time_to_minutes(win["start"])
        end = _time_to_minutes(win["end"])
        t = start
        # pick compatible slots that overlap the [t, t+VISIT]
        while t + VISIT <= end:
            # find first slot that can host [t, t+VISIT]
            chosen = None
            for s in slots:
                a = _time_to_minutes(s["start"])
                b = _time_to_minutes(s["end"])
                if a <= t and t + VISIT <= b:
                    chosen = s
                    break
            if not chosen:
                t += 10  # slide forward 10 minutes
                continue
            schedule.append(
                {
                    "property_id": chosen["property_id"],
                    "address": chosen["address"],
                    "visit": {"start": _minutes_to_time(t), "end": _minutes_to_time(t + VISIT)},
                    "travel_buffer": f"{TRAVEL}min",
                }
            )
            # advance time by visit + travel
            t += VISIT + TRAVEL
    reasoning.append("Sequenced tours to minimize gaps while respecting windows and slot bounds.")

    final = {
        "house_visit_schedule": {
            "reasoning_steps": reasoning,
            "final_recommendation": schedule,
        },
        "calendar_integration": {
            "reasoning_steps": [
                "Use Google/Outlook Calendar APIs with OAuth2 (user grant).",
                "Create events with travel buffers; set reminders at 24h and 1h.",
                "Handle edits via webhook push (watch) to resync changes.",
            ],
            "final_recommendation": "Implement bi‑directional sync with conflict detection and retry on 409s.",
        },
        "route_optimization": {
            "reasoning_steps": [
                "Treat as VRP with time windows (visit duration 30, travel 20).",
                "For more than ~8 stops, use OR‑Tools (CP‑SAT) or external service.",
            ],
            "final_recommendation": "Use OR‑Tools with time‑window constraints; fall back to greedy for small N.",
        },
        "confirmation_and_reminders": {
            "reasoning_steps": [
                "Send booking confirmation immediately (email+SMS).",
                "Reminders at T‑24h and T‑1h; include route link and parking notes.",
            ],
            "final_recommendation": "Automate via your comms service (SES/Twilio) with iCal attachments.",
        },
            "feedback_capture": {
            "reasoning_steps": [
                "Send post‑tour survey per house within 24h.",
                "Aggregate ratings (1–5) for fit, condition, value; collect notes.",
            ],
            "final_recommendation": "Store feedback in your CRM and surface a summary to agents/clients.",
        },
    }
    return final


# ──────────────────────────────────────────────────────────────────────────────
# Disclosure Q&A Sub‑Agent (files + citations)
# ──────────────────────────────────────────────────────────────────────────────
class QAFile(BaseModel):
    name: str
    content: str


def _find_sentences(text: str, needles: List[str]) -> List[str]:
    import re

    sents = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sents:
        low = s.lower()
        if any(n in low for n in needles):
            out.append(s.strip())
    return out[:6]


def disclosure_qa(files: List[QAFile], question: str) -> Dict[str, Any]:
    needles = [w.strip().lower() for w in question.split() if len(w) > 3]
    reasoning = [
        "Parsed question and selected keywords for lookup.",
        "Searched documents for matching sentences and compiled candidate evidence.",
    ]
    answers = []
    for f in files:
        hits = _find_sentences(f.content, needles)
        if not hits:
            continue
        citations = [
            {"source": f.name, "page": "?", "text": s[:280]} for s in hits[:3]
        ]
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
                "risk_flags": ["Missing disclosure details; request updated forms or third‑party inspections."],
                "follow_up_questions": ["Can you share the specific section or page where this is discussed?"],
            }
        )
    return {"reasoning_steps": reasoning, "answers": answers}


# ──────────────────────────────────────────────────────────────────────────────
# Offer Drafter & Negotiation Coach — use LLM with strict JSON output
# ──────────────────────────────────────────────────────────────────────────────
OFFER_DRAFTER_SCHEMA = {
    "name": "offer_draft_schema",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning_steps": {"type": "array", "items": {"type": "string"}},
            "offer_draft": {"type": "object"},
            "alternate_scenarios": {"type": "array"},
            "handoff": {"type": "object"},
        },
        "required": ["reasoning_steps", "offer_draft", "alternate_scenarios", "handoff"],
        "additionalProperties": True,
    },
}

NEGOTIATION_COACH_SCHEMA = {
    "name": "negotiation_coach_schema",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning_steps": {"type": "array", "items": {"type": "string"}},
            "market_context": {"type": "object"},
            "offer_draft": {"type": "object"},
            "alternate_scenarios": {"type": "array"},
            "scenario_talking_points": {"type": "array"},
            "handoff": {"type": "object"},
        },
        "required": [
            "reasoning_steps",
            "market_context",
            "offer_draft",
            "alternate_scenarios",
            "scenario_talking_points",
            "handoff",
        ],
        "additionalProperties": True,
    },
}

OFFER_DRAFTER_SYSTEM = (
    "You are an expert real estate Offer Drafter sub-agent. Generate compliant, ready-to-sign drafts "
    "following the procedural checklist, compliance guardrails, auto-filled terms, alternate scenarios, and e-sign handoff. "
    "Always reason step-by-step before conclusions. Use the exact JSON structure described in the user's spec."
)

NEGOTIATION_COACH_SYSTEM = (
    "You are an expert real estate Offer Drafter and Negotiation Coach. Incorporate market context—comps, DOM, and seasonality—into the draft and guidance. "
    "Provide stepwise reasoning before any conclusions and return JSON exactly per the user's spec."
)


# ──────────────────────────────────────────────────────────────────────────────
# MCP Server and Tools
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP("RealEstate Workflow MCP")


@mcp.tool
def run_workflow(user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end entry point. Applies Guardrails, then routes request. Returns the router result and a suggested next_node."""
    sid, session = _get_session(session_id)

    # 1) Guardrails
    guard = jailbreak_check(user_message)
    if not guard.allowed:
        return {
            "session_id": sid,
            "status": "blocked",
            "reason": "Guardrails/Jailbreak check failed",
            "details": guard.reasons,
        }

    # 2) Router
    routed = router_route(user_message, session)
    intent = routed.get("intent", "general")

    # keep minimal history
    session.history.append({"user": user_message, "agent": f"intent={intent}"})

    # 3) If/Else → Buyer Agent when buy
    next_node = None
    if intent == "buy":
        next_node = "buyer_intake_step"
    elif intent == "disclosures":
        next_node = "disclosure_qa"
    elif intent == "offer":
        next_node = "offer_drafter"
    elif intent == "sell":
        next_node = "general_sell_agent"  # placeholder for future
    else:
        next_node = "general"

    return {
        "session_id": sid,
        "status": "ok",
        "router": routed,
        "slots_obj": routed.get("slots_obj", {}),
        "next_node": next_node,
    }


@mcp.tool
def buyer_intake_step(session_id: Optional[str] = None, user_message: Optional[str] = None) -> Dict[str, Any]:
    """Ask only the next unanswered intake question; never repeat earlier fields. Returns either an ask or a summary for confirmation."""
    sid, session = _get_session(session_id)
    result = intake_step(session, user_message)
    done = intake_is_complete(session)
    next_node = None
    if done:
        next_node = "search_and_match"
    return {"session_id": sid, "result": result, "intake_complete": done, "next_node": next_node}


@mcp.tool
def search_and_match_tool(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run MLS search (stub) and generate 3–7 recommendations with rationale, commute, and payment estimates."""
    sid, session = _get_session(session_id)
    if not intake_is_complete(session):
        return [{"error": "Intake not complete. Call buyer_intake_step until summary confirmed."}]
    items = [x.model_dump() for x in search_and_match(session)]
    return items


@mcp.tool
def tour_plan_tool(open_houses: List[Dict[str, Any]], preferred_windows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Recommend a visit schedule based on open house times and user time preferences."""
    inp = TourInput(open_houses=open_houses, preferred_windows=preferred_windows)
    return plan_tours(inp)


@mcp.tool
def disclosure_qa_tool(question: str, files: List[Dict[str, str]]) -> Dict[str, Any]:
    """Answer disclosure questions strictly from provided files, with sentence-level citations and risk flags."""
    qa_files = [QAFile(**f) for f in files]
    return disclosure_qa(qa_files, question)


@mcp.tool
def offer_drafter_tool(prompt_context: str) -> Dict[str, Any]:
    """LLM-backed Offer Drafter. Provide the user/context details in prompt_context; returns structured JSON per spec."""
    return llm_json(system_prompt=OFFER_DRAFTER_SYSTEM, user_prompt=prompt_context, json_schema=OFFER_DRAFTER_SCHEMA)


@mcp.tool
def negotiation_coach_tool(prompt_context: str) -> Dict[str, Any]:
    """LLM-backed Negotiation Coach. Include comps, DOM, seasonality in prompt_context; returns structured JSON per spec."""
    return llm_json(system_prompt=NEGOTIATION_COACH_SYSTEM, user_prompt=prompt_context, json_schema=NEGOTIATION_COACH_SCHEMA)


# Convenience: simple health check and session reset
@mcp.tool
def health() -> Dict[str, Any]:
    return {"status": "ok", "sessions": len(SESSIONS)}


@mcp.tool
def reset_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    sid, _ = _get_session(session_id)
    SESSIONS.pop(sid, None)
    return {"status": "reset", "session_id": sid}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # FastMCP will default to stdio server; you can also run with SSE by setting env FASTMCP_SSE=1
    # Examples:
    #   python real_estate_mcp_server.py                 # stdio
    #   FASTMCP_SSE=1 python real_estate_mcp_server.py   # SSE on :8000
    #   FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8000 FASTMCP_SSE=1 python real_estate_mcp_server.py
    log.info("Starting MCP server ...")
    mcp.run()
