"""Prompt definitions for the tour planning step."""


def tour_plan_extraction_system_prompt() -> str:
    """Return the system prompt for tour plan extraction."""
    return (
        "You extract open house details and preferred visit windows from the latest message. "
        "Return only what is explicitly stated. Normalize time to HH:MM 24-hour format where possible."
    )
