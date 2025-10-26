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
                                "minItems": 0,
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


def get_tour_plan_generation_schema() -> Dict[str, Any]:
    """Return the schema for LLM-driven tour plan generation."""
    visit_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "property_id": {"type": ["string", "null"]},
            "address": {"type": "string"},
            "visit": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["start", "end"],
            },
            "travel_buffer": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": ["address", "visit"],
    }

    planning_block_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning_steps": {
                "type": "array",
                "items": {"type": "string"},
            },
            "final_recommendation": {"type": "string"},
        },
        "required": ["reasoning_steps", "final_recommendation"],
    }

    return {
        "name": "tour_plan_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "house_visit_schedule": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reasoning_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "final_recommendation": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "visits": {
                                    "type": "array",
                                    "items": visit_schema,
                                },
                                "travel_guidance": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["visits", "travel_guidance", "summary"],
                        },
                    },
                    "required": ["reasoning_steps", "final_recommendation"],
                },
                "calendar_integration": planning_block_schema,
                "route_optimization": planning_block_schema,
                "confirmation_and_reminders": planning_block_schema,
                "feedback_capture": planning_block_schema,
            },
            "required": [
                "house_visit_schedule",
                "calendar_integration",
                "route_optimization",
                "confirmation_and_reminders",
                "feedback_capture",
            ],
        },
    }
