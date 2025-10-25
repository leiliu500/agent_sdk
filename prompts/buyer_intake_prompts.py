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
