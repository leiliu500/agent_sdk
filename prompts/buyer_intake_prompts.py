"""Prompt definitions and question text for the buyer intake step."""

from typing import Dict

# Ordered mapping of intake fields to the question text shown to the buyer.
_INTAKE_QUESTIONS: Dict[str, str] = {
    "budget": "What is your budget range for purchasing your new home?",
    "locations": "Which city, area, or neighborhoods are you considering?",
    "home_type": "What type of home would you like? (e.g., single-family, condo, townhouse)",
    "bedrooms": "How many bedrooms do you need?",
    "bathrooms": "How many bathrooms do you need?",
    "financing_status": "What is your financing status? (cash or mortgage; pre-approved?)",
    "timeline": "What is your timeline to move or purchase?",
    "must_haves": "What are your must-have features?",
    "deal_breakers": "Any deal-breakers we should avoid?",
    "other_concerns": "Any other concerns we should be aware of?",
    "consents": "Do you consent to us storing and using your information to search for properties and contact you with matches? (yes/no)",
}


def intake_questions() -> Dict[str, str]:
    """Return the canonical mapping of intake fields to their question prompts."""
    return dict(_INTAKE_QUESTIONS)


def intake_extraction_system_prompt() -> str:
    """Return the system prompt for buyer intake field extraction."""
    return (
        "You extract real-estate buyer intake details. Identify only the fields that are explicitly "
        "stated in the user's latest message. Return literal values or short phrases; do not infer or "
        "fabricate information. If a field is not stated, return null. Normalize numbers using digits "
        "(e.g., 3 bedrooms) and keep user phrasing for free-text fields."
    )


def intake_agent_system_prompt() -> str:
    """Return the system prompt for the buyer intake conversational agent."""
    ordered_fields = "\n".join(
        f"- {field}: {question}" for field, question in _INTAKE_QUESTIONS.items()
    )
    return (
        "You are the Buyer Intake Agent for a real-estate workflow. Collect the buyer's preferences "
        "and constraints conversationally while respecting the strict question order shown below. "
        "Maintain context across turns, incorporate any pre-filled slot hints, and never request "
        "information that has already been confirmed unless the user issues a correction. When a user "
        "provides new details, update the slot values to the user's latest explicit statements."
        "\n\nRequired intake order and canonical question text:\n"
        f"{ordered_fields}\n\n"
        "Always output a single JSON object with the fields: status, message, field, next_field, slots, "
        "and summary.\n"
        "- status must be one of: ask (requesting the next field), summary (all fields captured; asking "
        "the user to confirm), confirmed (user has affirmed the summary or explicitly stated that the "
        "intake is correct).\n"
        "- message contains the assistant's reply to the user.\n"
        "- field identifies the field currently being requested (only when status=ask).\n"
        "- next_field reflects the next missing field or 'done' when all required fields are complete.\n"
        "- slots is an object containing the latest value for every field listed in the required order.\n"
        "- summary provides a key-value map of collected answers only when status is summary or confirmed; "
        "otherwise summary must be null.\n\n"
        "Ask exactly one question per turn when status is 'ask'. When status transitions to 'summary', "
        "produce a concise recap of all captured fields and invite the user to confirm or correct details. "
        "If the user says the summary is correct (yes/confirm), switch to status 'confirmed' on the next "
        "turn while preserving the summary. If the user provides corrections, update the affected slots, "
        "return status 'ask', and focus on clarifying the corrected field before progressing."
    )
