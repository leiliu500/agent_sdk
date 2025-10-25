"""JSON schema definitions for the search and match step."""

from typing import Any, Dict


def get_mls_web_search_schema() -> Dict[str, Any]:
    """Return the schema for MLS web search responses."""
    return {
        "name": "property_listing_results",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "search_queries": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "domain": {"type": "string"},
                            "query": {"type": "string"},
                            "result_summary": {"type": "string"},
                        },
                        "required": ["domain", "query", "result_summary"],
                    },
                },
                "listings": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 7,
                    "items": {
                        "type": "object",
                        "properties": {
                            "property_id": {"type": ["string", "null"]},
                            "address": {"type": "string"},
                            "list_price": {"type": ["number", "string"]},
                            "beds": {"type": ["number", "string", "null"]},
                            "baths": {"type": ["number", "string", "null"]},
                            "source_url": {"type": "string"},
                            "source_name": {"type": "string"},
                            "fit_reasoning": {"type": "string"},
                            "commute_notes": {"type": ["string", "null"]},
                            "notes": {"type": ["string", "null"]},
                            "open_house_slots": {
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
                        "required": [
                            "property_id",
                            "address",
                            "list_price",
                            "beds",
                            "baths",
                            "source_url",
                            "source_name",
                            "fit_reasoning",
                            "commute_notes",
                            "notes",
                            "open_house_slots",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["search_queries", "listings"],
        },
    }
