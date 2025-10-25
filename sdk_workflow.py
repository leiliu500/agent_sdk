"""
Real‑estate workflow MCP server (OpenAI Agents SDK + FastMCP)

Features
- Jailbreak Guardrail gate (OpenAI Guardrails if configured; fallback to Moderations)
- Router (buy | sell | disclosures | offer | general) with JSON schema
- Buyer Agent with sub‑agents:
  * Buyer‑Intake (stepwise, no repeat questions; persistent per session)
    * Search & Match (live MLS web search via Zillow/Redfin/MLSListings + payment estimate)
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
import re
import json
import time
import math
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

import prompts
from schemas import (
    get_intake_extraction_schema,
    get_negotiation_coach_schema,
    get_offer_drafter_schema,
    get_required_intake_fields,
    get_router_json_schema,
    get_mls_web_search_schema,
    get_tour_plan_extraction_schema,
)

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
# ──────────────────────────────────────────────────────────────────────────────
# LLM Helpers
# ──────────────────────────────────────────────────────────────────────────────
client = OpenAI()
JSON_DECODER = json.JSONDecoder()


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
def router_route(user_message: str, session: Session) -> Dict[str, Any]:
    # include light chat history for consistency
    history_snippets = [
        f"U:{h['user']}\nA:{h['agent']}" for h in session.history[-4:] if 'user' in h and 'agent' in h
    ]
    history_text = "\n\n".join(history_snippets)

    user_prompt = prompts.router_user_prompt(history_text=history_text, user_message=user_message)

    out = llm_json(
        system_prompt=prompts.router_system_prompt(),
        user_prompt=user_prompt,
        json_schema=get_router_json_schema(),
    )

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
INTAKE_QUESTIONS = prompts.intake_questions()


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


def _extract_intake_fields(message: str) -> Dict[str, Any]:
    if not message or not message.strip():
        return {}
    return llm_json(
        system_prompt=prompts.intake_extraction_system_prompt(),
        user_prompt=message,
        json_schema=get_intake_extraction_schema(),
    )


def _ingest_intake_message(
    session: Session,
    user_message: Optional[str],
    slot_hints: Optional[Dict[str, Any]] = None,
) -> None:
    # Pre-fill intake fields using structured hints and lightweight extraction.
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
            extracted = _extract_intake_fields(user_message)
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


def intake_step(
    session: Session,
    user_message: Optional[str],
    slot_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ingest_intake_message(session, user_message, slot_hints)
    s = session.intake
    required_fields = get_required_intake_fields()

    # Decide what to ask next
    if s.next_field == "done":
        summary = {k: getattr(s, k) for k in required_fields}
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
    for k in required_fields:
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
    required_fields = get_required_intake_fields()
    return all(getattr(s, k) not in (None, "") for k in required_fields)


# ──────────────────────────────────────────────────────────────────────────────
# Search & Match Sub‑Agent (live MLS web search)
# ──────────────────────────────────────────────────────────────────────────────
class MatchItem(BaseModel):
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

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": prompts.mls_web_search_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "web_search"}],
        temperature=0.2,
        text=_schema_to_text_config(get_mls_web_search_schema()),
    )
    raw_text = _response_output_text(resp)
    try:
        parsed = _parse_json_from_text(raw_text)
    except Exception as exc:
        log.debug("Raw MLS search response text: %s", raw_text)
        raise ValueError("Failed to parse MLS search JSON payload") from exc
    if not isinstance(parsed, dict):
        raise ValueError("MLS search response payload is not an object")
    return parsed


def search_and_match(session: Session) -> List[MatchItem]:
    s = session.intake
    required_fields = get_required_intake_fields()
    criteria = {k: getattr(s, k) for k in required_fields}
    listings: List[Dict[str, Any]] = []
    try:
        live_payload = _mls_search_live(criteria)
        listings = live_payload.get("listings", []) if isinstance(live_payload, dict) else []
    except Exception as exc:  # pragma: no cover - network/tool failure fallback
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


def _extract_tour_plan_inputs(message: str) -> Dict[str, Any]:
    return llm_json(
        system_prompt=prompts.tour_plan_extraction_system_prompt(),
        user_prompt=message,
        json_schema=get_tour_plan_extraction_schema(),
    )


def _normalize_time_string(raw: Any) -> Optional[str]:
    """Normalize flexible time strings (10am, 1:30 PM, 1330) → HH:MM 24h."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    text = text.replace(".", ":")
    text = re.sub(r"\s+", "", text)

    match = re.match(r"^(\d{1,2})(?:[:]?(\d{2}))?(am|pm)$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        if hour == 12:
            hour = 0 if match.group(3) == "am" else 12
        elif match.group(3) == "pm":
            hour += 12
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"
        return None

    match = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"
        return None

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
    # Fallback: extract the first HH:MM substring (useful for ISO timestamps)
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
        slots_raw = item.get("slots") or []
        if not isinstance(slots_raw, list):
            issues.append(f"Ignored slots for {address or item!r}; expected list, got {type(slots_raw).__name__}")
            slots_raw = []
        slots: List[Dict[str, str]] = []
        for slot in slots_raw:
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
        if address and slots:
            pid_raw = item.get("property_id")
            pid = str(pid_raw).strip() if pid_raw not in (None, "") else None
            sanitized.append({"property_id": pid, "address": address, "slots": slots})
        else:
            if not address:
                issues.append(f"Open house missing address: {item!r}")
            if not slots:
                issues.append(f"Open house missing valid slots for {address or item!r}")
    return sanitized, issues


def _sanitize_windows(entries: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, str]], List[str]]:
    windows: List[Dict[str, str]] = []
    issues: List[str] = []
    for item in entries or []:
        if not isinstance(item, dict):
            issues.append(f"Ignored preferred window; expected object, got {type(item).__name__}: {item!r}")
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
# ──────────────────────────────────────────────────────────────────────────────
# MCP Server and Tools
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP("RealEstate Workflow MCP")


@mcp.tool
def run_workflow(user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end entry point. Applies Guardrails, then routes request, and automatically executes the appropriate next step."""
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

    # 3) If/Else → Buyer Agent when buy - AUTO-EXECUTE next step
    next_node = None
    next_step_result = None
    
    if intent == "buy":
        next_node = "buyer_intake_step"
        # Auto-execute buyer intake with the user's message
        intake_result = intake_step(session, user_message, routed.get("slots_obj"))
        next_step_result = {
            "step": "buyer_intake_step",
            "result": intake_result,
            "intake_complete": intake_is_complete(session)
        }
    elif intent == "disclosures":
        next_node = "disclosure_qa"
        next_step_result = {
            "step": "disclosure_qa",
            "message": "Please provide the disclosure documents and your specific question."
        }
    elif intent == "offer":
        next_node = "offer_drafter"
        next_step_result = {
            "step": "offer_drafter",
            "message": "Please provide property details to draft an offer."
        }
    elif intent == "sell":
        next_node = "general_sell_agent"
        next_step_result = {
            "step": "general_sell_agent",
            "message": "Selling agent workflow - please provide property details."
        }
    else:
        next_node = "general"
        next_step_result = {
            "step": "general",
            "message": "I can help you with buying, selling, or property disclosures. How can I assist you?"
        }

    return {
        "session_id": sid,
        "status": "ok",
        "router": routed,
        "slots_obj": routed.get("slots_obj", {}),
        "next_node": next_node,
        "next_step_result": next_step_result,
    }


@mcp.tool
def buyer_intake_step(session_id: Optional[str] = None, user_message: Optional[str] = None) -> Dict[str, Any]:
    """Ask only the next unanswered intake question; never repeat earlier fields. Returns either an ask or a summary for confirmation."""
    sid, session = _get_session(session_id)
    result = intake_step(session, user_message)
    done = intake_is_complete(session)
    next_node = None
    next_step_result = None
    
    # If intake is complete and user confirmed, automatically execute search_and_match
    if done and result.get("status") == "summary":
        # Check if user confirmed (looking for affirmative response)
        if user_message and user_message.strip().lower() in ["yes", "y", "confirmed", "correct", "confirm"]:
            next_node = "tour_plan_tool"
            # Auto-execute search and match
            try:
                match_objs = search_and_match(session)
                next_step_result = {
                    "step": "search_and_match",
                    "matches": [m.model_dump() for m in match_objs],
                    "open_houses": [
                        {"property_id": m.property_id, "address": m.address, "slots": m.open_house_slots}
                        for m in match_objs
                        if m.open_house_slots
                    ],
                    "count": len(match_objs),
                    "next_node": "tour_plan_tool",
                    "message": "Review the recommended properties and proceed with tour planning using the provided open house slots."
                }
                result["auto_executed_next_step"] = True
            except Exception as e:
                log.exception("Failed to auto-execute search_and_match: %s", e)
                next_step_result = {
                    "step": "search_and_match",
                    "error": str(e),
                    "message": "Please call search_and_match_tool manually"
                }
                next_node = "search_and_match"
        else:
            # Still showing summary, waiting for confirmation
            next_node = "search_and_match"
    
    return {
        "session_id": sid, 
        "result": result, 
        "intake_complete": done, 
        "next_node": next_node,
        "next_step_result": next_step_result
    }


@mcp.tool
def search_and_match_tool(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Run live MLS web search and return a workflow envelope with property recommendations."""
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
        matches = [x.model_dump() for x in match_objs]
    except Exception as exc:
        return {
            "session_id": sid,
            "status": "error",
            "step": "search_and_match",
            "message": str(exc),
            "next_node": "search_and_match_tool",
        }

    return {
        "session_id": sid,
        "status": "ok",
        "step": "search_and_match",
        "matches": matches,
        "open_houses": [
            {"property_id": m.property_id, "address": m.address, "slots": m.open_house_slots}
            for m in match_objs
            if m.open_house_slots
        ],
        "count": len(matches),
        "next_node": "tour_plan_tool",
        "message": "Provide open house options and preferred windows to plan tours.",
    }


@mcp.tool
def tour_plan_tool(
    open_houses: Optional[List[Dict[str, Any]]] = None,
    preferred_windows: Optional[List[Dict[str, str]]] = None,
    user_message: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend a visit schedule based on open house times and user time preferences."""
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
        error_payload = {
            "session_id": sid,
            "status": "error",
            "step": "tour_plan",
            "message": "Missing required inputs for tour planning.",
            "missing_inputs": missing,
            "expected_format": {
                "open_houses": [{"property_id": "id", "address": "string", "slots": [{"start": "HH:MM", "end": "HH:MM"}]}],
                "preferred_windows": [{"start": "HH:MM", "end": "HH:MM"}],
            },
            "hint": "Provide open house slots and preferred windows via fields or natural language in user_message.",
            "next_node": "tour_plan_tool",
        }
        if issues:
            error_payload["invalid_inputs"] = issues
        return error_payload

    plan = plan_tours(TourInput(open_houses=parsed_open_houses, preferred_windows=parsed_windows))
    response = {
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
    """Answer disclosure questions strictly from provided files, with sentence-level citations and risk flags."""
    qa_files = [QAFile(**f) for f in files]
    return disclosure_qa(qa_files, question)


@mcp.tool
def offer_drafter_tool(prompt_context: str) -> Dict[str, Any]:
    """LLM-backed Offer Drafter. Provide the user/context details in prompt_context; returns structured JSON per spec."""
    return llm_json(
        system_prompt=prompts.offer_drafter_system_prompt(),
        user_prompt=prompt_context,
        json_schema=get_offer_drafter_schema(),
    )


@mcp.tool
def negotiation_coach_tool(prompt_context: str) -> Dict[str, Any]:
    """LLM-backed Negotiation Coach. Include comps, DOM, seasonality in prompt_context; returns structured JSON per spec."""
    return llm_json(
        system_prompt=prompts.negotiation_coach_system_prompt(),
        user_prompt=prompt_context,
        json_schema=get_negotiation_coach_schema(),
    )


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
