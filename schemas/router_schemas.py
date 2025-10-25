"""JSON schema definitions for the router workflow step."""

from typing import Any, Dict


def get_router_json_schema() -> Dict[str, Any]:
    """Return the router schema used for intent classification."""
    return {
        "name": "response_schema",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["buy", "sell", "disclosures", "offer", "general"]},
                "confidence": {"type": "number"},
                "slots": {"type": "string"},
            },
            "required": ["intent", "confidence", "slots"],
            "additionalProperties": False,
            "title": "response_schema",
        },
        "strict": True,
    }
