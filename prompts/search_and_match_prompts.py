"""Prompt definitions for the search and match step."""

from typing import Iterable


def mls_web_search_system_prompt() -> str:
    """Return the system prompt for MLS web search."""
    return (
        "You are a licensed real-estate research assistant. Use the web_search tool to research "
        "only Zillow, Redfin, and MLSListings webpages. For every query you send, include a `site:` "
        "filter targeting one of these domains. Extract on-market residential properties that match the "
        "buyer criteria, capture open house start and end times in HH:MM 24-hour format when available, and "
        "record verifiable facts only. Return structured JSON that conforms exactly to the provided schema."
    )


def mls_web_search_user_prompt(domains: Iterable[str], criteria_lines: Iterable[str]) -> str:
    """Compose the user prompt for MLS web search with domain limits and intake summary."""
    joined_domains = "`, `".join(domains)
    joined_criteria = "\n- ".join(criteria_lines)
    return (
        "Search each site `"
        + joined_domains
        + "` using the web_search tool with explicit `site:` filters. Focus on residential listings that "
        "best match the buyer intake. For every listing include the verified price, headline address, "
        "beds/baths, direct URL, and any published open house time ranges formatted as HH:MM 24-hour "
        "open_house_slots. If information is missing or cannot be verified, omit the listing.\n\n"
        "Buyer intake summary:\n- "
        + joined_criteria
    )
