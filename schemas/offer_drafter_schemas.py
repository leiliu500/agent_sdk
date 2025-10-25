"""JSON schema definitions for the offer drafter step."""

from typing import Any, Dict


def get_offer_drafter_schema() -> Dict[str, Any]:
    """Return the schema for the offer drafter response."""
    return {
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
