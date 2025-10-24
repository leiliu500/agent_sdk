import os, asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool, FileSearchTool, handoff

# Load environment variables from .env file
load_dotenv()

# ---- child agents -----------------------------------------------------------
researcher = Agent(
    name="researcher",
    instructions="Search the web and return 3 bullet findings with citations.",
    tools=[WebSearchTool()],
)

writer = Agent(
    name="writer",
    instructions="Turn notes into a concise 150-word executive summary."
)

# Optional: demonstrate a true handoff path parent -> child
router = Agent(
    name="router",
    instructions=(
        "If the user asks for research, HANDOFF to the 'researcher'. "
        "Otherwise answer directly. Do not loop."
    ),
    handoffs=[handoff(researcher)],  # true delegation
)

# ---- parent agent -----------------------------------------------------------
parent = Agent(
    name="orchestrator",
    instructions=(
        "Use tools or sub-agents as needed. "
        "Prefer calling 'research' first, then 'summarize'."
    ),
    tools=[
        # sub-agent as a callable tool (parent keeps control):
        researcher.as_tool(tool_name="research", tool_description="Do web research."),
        writer.as_tool(tool_name="summarize", tool_description="Summarize notes."),
        # If you have vector stores, enable file search:
        # FileSearchTool(vector_store_ids=[...]),
    ],
)

async def main():
    q = "Is glass railing a good choice for very windy hillside decks? Give pros/cons."
    result = await Runner.run(parent, q)
    print("\n=== FINAL ===\n", result.final_output)

    # handoff example:
    result2 = await Runner.run(router, "Research the latest sources on this topic.")
    print("\n=== HANDOFF RESULT ===\n", result2.final_output)

if __name__ == "__main__":
    asyncio.run(main())
