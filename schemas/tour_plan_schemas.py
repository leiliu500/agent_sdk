"""JSON schema definitions for the tour planning step."""

from typing import Any, Dict


def get_tour_plan_extraction_schema() -> Dict[str, Any]:
    """Return the schema for extracting tour plan details."""
    return {
        "name": "tour_plan_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "open_houses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "property_id": {"type": ["string", "null"]},
                            "address": {"type": "string"},
                            "slots": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "start": {"type": "string"},
                                        "end": {"type": "string"},
                                    },
                                    "required": ["start", "end"],
                                },
                            },
                        },
                        "required": ["address", "slots"],
                    },
                },
                "preferred_windows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                        },
                        "required": ["start", "end"],
                    },
                },
            },
            "required": ["open_houses", "preferred_windows"],
        },
    }
