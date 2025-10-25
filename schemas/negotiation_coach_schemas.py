"""JSON schema definitions for the negotiation coach step."""

from typing import Any, Dict


def get_negotiation_coach_schema() -> Dict[str, Any]:
    """Return the negotiation coach schema."""
    return {
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
