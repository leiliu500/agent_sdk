"""Prompt definitions for the tour planning step."""

from typing import Any, Dict, List


def tour_plan_extraction_system_prompt() -> str:
    """Return the system prompt for tour plan extraction."""
    return (
        "You extract open house details and preferred visit windows from the latest message. "
        "Return only what is explicitly stated. Normalize time to HH:MM 24-hour format where possible."
    )


def tour_plan_planner_system_prompt() -> str:
    """Return the system prompt for LLM-driven tour planning."""
    return (
        "You are a meticulous real estate tour coordinator. Craft feasible itineraries using provided open houses "
        "and buyer availability windows. Assume 30-minute property visits with a default 20-minute travel buffer "
        "unless the input specifies otherwise. Respect the supplied JSON schema exactly, keep reasoning concise, "
        "and reflect uncertainties or conflicts explicitly. When properties are missing open house slots, propose "
        "creative next steps (private showings, virtual tours, outreach plans) while still delivering a structured "
        "plan that highlights follow-up actions."
    )


def tour_plan_planner_user_prompt(
    open_houses: List[Dict[str, Any]],
    preferred_windows: List[Dict[str, Any]],
    visit_minutes: int = 30,
    travel_minutes: int = 20,
) -> str:
    """Build the user prompt for the tour planning LLM call."""
    house_lines = [
        "Open Houses:",
        *[
            "- {idx}. {address} (id: {pid}) slots: {slots}".format(
                idx=idx,
                address=item.get("address", "Unknown"),
                pid=item.get("property_id") or "n/a",
                slots=(
                    ", ".join(f"{slot['start']}-{slot['end']}" for slot in item.get("slots", []))
                    if item.get("slots")
                    else "no slots provided"
                ),
            )
            for idx, item in enumerate(open_houses, start=1)
        ],
    ]

    window_lines = [
        "Preferred Windows:",
        *[
            "- {idx}. {start}-{end}".format(idx=idx, start=item.get("start", ""), end=item.get("end", ""))
            for idx, item in enumerate(preferred_windows, start=1)
        ],
    ]

    guidance_lines = [
        "Planning Guardrails:",
        f"- Default visit length: {visit_minutes} minutes (adjust only if inputs require)",
        f"- Default travel buffer: {travel_minutes} minutes between visits",
        "- Do not schedule outside preferred windows unless you flag it as an exception",
        "- If scheduling is impossible, explain why and suggest concrete next actions",
        "- When no slots are provided, outline creative alternatives (agent outreach, private tours, virtual options)",
    ]

    return "\n".join([*house_lines, "", *window_lines, "", *guidance_lines])
