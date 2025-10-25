"""JSON schema definitions for the buyer intake step."""

from typing import Any, Dict, List


def get_required_intake_fields() -> List[str]:
    """Return the ordered list of buyer-intake fields."""
    return [
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


def get_intake_extraction_schema() -> Dict[str, Any]:
    """Return the JSON schema used for buyer-intake extraction."""
    return {
        "name": "buyer_intake_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "budget": {"type": ["string", "number", "null"]},
                "locations": {"type": ["string", "null"]},
                "home_type": {"type": ["string", "null"]},
                "bedrooms": {"type": ["string", "number", "null"]},
                "bathrooms": {"type": ["string", "number", "null"]},
                "financing_status": {"type": ["string", "null"]},
                "timeline": {"type": ["string", "null"]},
                "must_haves": {"type": ["string", "null"]},
                "deal_breakers": {"type": ["string", "null"]},
                "consents": {"type": ["string", "null"]},
            },
            "required": get_required_intake_fields(),
        },
    }
