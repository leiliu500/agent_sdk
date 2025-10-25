"""Prompt definitions for the buyer intake step."""


def intake_extraction_system_prompt() -> str:
    """Return the system prompt for buyer intake field extraction."""
    return (
        "You extract real-estate buyer intake details. Identify only the fields that are explicitly "
        "stated in the user's latest message. Return literal values or short phrases; do not infer or "
        "fabricate information. If a field is not stated, return null. Normalize numbers using digits "
        "(e.g., 3 bedrooms) and keep user phrasing for free-text fields."
    )
