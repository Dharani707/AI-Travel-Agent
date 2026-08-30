from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# Import the actual working tools from the tools directory

# pyrefly: ignore [missing-import]
from core.tools.google_places import search_places
# pyrefly: ignore [missing-import]
from core.tools.google_routes import get_route

# Load environment variables
load_dotenv()
# GROQ_API_KEY is automatically loaded by load_dotenv() and used by ChatGroq

# Define Pydantic output schemas for the planner agent
class Activity(BaseModel):
    time: str = Field(description="Specific time or period of day, e.g., '09:00 AM', 'Afternoon', 'Evening'")
    activity: str = Field(description="Detailed name and description of the activity")
    location: str = Field(description="Name or address of the location")
    notes: str = Field(description="Helpful notes, transit/routing tips, ticketing, or dining advice")

class DayItinerary(BaseModel):
    day: int = Field(description="Day number, starting from 1")
    date: str = Field(description="Date of this itinerary day in YYYY-MM-DD format")
    activities: list[Activity] = Field(description="Scheduled activities for this day in sequential order")

class PlannerOutput(BaseModel):
    days: list[DayItinerary] = Field(description="Day-by-day itinerary details")
    estimated_budget_breakdown: dict = Field(description="Estimated cost breakdown for lodging, food, transport, and attractions")
    packing_suggestions: list[str] = Field(description="Recommended packing checklist based on weather, activities, and destination characteristics")

# Wrap standard tool functions into LangChain StructuredTool objects so schemas are properly generated
search_places_tool = StructuredTool.from_function(
    func=search_places,
    name="search_places",
    description="Call this with EXACTLY these two arguments: location (string, e.g. 'Rome, Italy') and place_type (string, e.g. 'tourist_attraction' or 'restaurant'). Do NOT use 'query', 'radius', or 'type' — these will cause an error."
)

get_route_tool = StructuredTool.from_function(
    func=get_route,
    name="get_route",
    description="Call this with EXACTLY these arguments: origin (string), destination (string), mode (string, optional, default 'driving'). Do NOT use any other argument names — only origin, destination, and mode are valid."
)


# SYSTEM PROMPT DESIGN DOCUMENTATION:
# 1. Role: We precisely define the role as a structured day-by-day travel itinerary builder.
# 2. Tool Coordination (Google Places + Routes): Explicit instructions to search for specific attractions matching traveler profile and use get_route to chain activities by proximity. This prevents backtracking and optimizes travel logistics.
# 3. Tone/Quality: Enforces specificity over general placeholders (recommends real, named locations rather than "a local cafe").
# 4. Constraints: Focuses the LLM on budget tier limits and traveler volumes.
# 5. Feedback Loop (HITL): Defines standard revision procedures when a REJECT or MODIFY is requested. Instructs the agent to only modify fields impacted by the feedback to preserve user decisions elsewhere.
SYSTEM_PROMPT = """You are an itinerary planning specialist that builds structured, realistic day-by-day travel plans.

You will receive a destination research report (which includes a summary, top attractions, safety notes, weather, and seasonal considerations) along with the traveler's specific request details (travelers, budget, interests).

Your goal is to build a complete, highly optimized day-by-day travel itinerary using the available tools:
1. Call `search_places` to find specific attractions, activities, and restaurants that match the user's interests and budget.
2. Call `get_route` to get transit times and distances between consecutive stops on a given day to ensure the activities are sequenced sensibly by proximity, minimizing travel time and preventing backtracking across the city.

Tone & Quality Bar:
Your generated plan must be highly practical, realistic, and specific. Do not use generic placeholders or filler suggestions. Suggest specific, existing attractions and dining options, and allocate realistic times for meals, sightseeing, and transit.

Constraints and Context:
- Respect the number of travelers and the budget range. Budget ranges are strictly enforced: choose activities, lodging estimates, and restaurants that fit the budget level (e.g., Budget, Mid-range, Luxury).
- Group activities that are physically near each other on the same day. Do not schedule activities on opposite sides of a city on the same afternoon unless transit options are fast and clearly detailed.

Feedback / Iteration:
- If a feedback string is provided, you must revise the previous itinerary. Focus only on changing what is relevant to the feedback, keeping the rest of the working plan intact to maintain continuity.

Tool Parameter Rules (STRICT):
- Before calling any tool, check the exact parameter names defined in its schema. Use ONLY those names — do not invent or substitute alternatives.
- For search_places: use ONLY `location` and `place_type`. Never use `query`, `radius`, or `type`.
- For get_route: use ONLY `origin`, `destination`, and `mode`. Never use any other field names.

Output Rules:
- Keep each activity's 'notes' field to ONE short sentence (under 15 words).
- Keep activity descriptions brief and factual — no long paragraphs.
"""

def run_planner_agent(research_data: dict, travel_request: dict, feedback: str = None) -> dict:
    """
    Creates a detailed day-by-day travel itinerary based on research, request data, and optional feedback.

    Args:
        research_data (dict): The output of run_research_agent containing summary, attractions, and weather.
        travel_request (dict): General request data, e.g., destination, dates, budget, travelers, interests.
        feedback (str, optional): Human feedback for revising a previous itinerary.

    Returns:
        dict: A dictionary matching the PlannerOutput schema.
    """
    # Initialize LLM for tool-calling phase.
    # Model: qwen/qwen3.8-27b — confirmed available on this API key and has
    # reliable standard tool-calling behavior (unlike gpt-oss "reasoning" models,
    # which emit malformed tool_call blocks with langchain-groq).
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=2000)

    # Bind both tools simultaneously so the model can call either one
    tools = [search_places_tool, get_route_tool]
    llm_with_tools = llm.bind_tools(tools)

    # Construct the user message for the tool-calling phase.
    # NOTE: we deliberately keep this phase's input light — full research_data
    # is fine here since this call has no output structuring overhead.
    user_prompt_content = (
        f"Research Data:\n{research_data}\n\n"
        f"Travel Request:\n{ {k: v for k, v in travel_request.items() if k != 'previous_itinerary'} }\n"
    )
    if feedback:
        user_prompt_content += f"\nHuman Feedback / Revision Request:\n{feedback}\n"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt_content)
    ]

    # Track tool results as compact plain-text snippets to avoid bloating the final prompt
    tool_results_summary: list[str] = []

    # Run execution loop (up to 5 iterations) to handle sequential place searches and routing checks
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If the model does not request any tool runs, break
        if not response.tool_calls:
            break

        # Execute each tool call requested by the model
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name == "search_places":
                tool_output = search_places(**tool_args)
            elif tool_name == "get_route":
                tool_output = get_route(**tool_args)
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."

            # Store a compact snippet for the final compilation step (capped per-result)
            tool_results_summary.append(
                f"[{tool_name}({tool_args})] => {str(tool_output)[:250]}"
            )

            # Append the result of the tool run as a ToolMessage
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id,
                name=tool_name
            ))

    # -----------------------------------------------------------------------
    # FINAL STEP: Structured output compilation.
    #
    # IMPORTANT TOKEN-BUDGET NOTES:
    # - Groq's free tier enforces an 8000 TPM (tokens-per-minute) ceiling on
    #   this model. We must keep this request's total input+output tokens
    #   comfortably below that.
    # - We do NOT reuse the full `messages` history here (it contains
    #   AIMessage/ToolMessage tool-call pairs that Groq rejects unless the
    #   same tools are re-declared on this call).
    # - We do NOT pass the full research_data or travel_request dicts here —
    #   only trimmed, essential fields — to avoid re-inflating token usage
    #   after the (expensive) tool-calling phase already consumed budget.
    # - previous_itinerary (if present, from a HITL revision) is EXCLUDED
    #   entirely from this condensed prompt since it can be very large;
    #   only the free-text `feedback` string is passed through.
    # -----------------------------------------------------------------------

    # Condensed destination summary (max 150 chars)
    destination_summary = str(research_data.get("destination_summary", ""))[:150]

    # Condensed travel request — only essential scalar fields, no nested dicts
    trimmed_request = (
        f"destination={travel_request.get('destination')}, "
        f"dates={travel_request.get('start_date')} to {travel_request.get('end_date')}, "
        f"budget={travel_request.get('budget')}, "
        f"travelers={travel_request.get('num_travelers')}, "
        f"interests={travel_request.get('interests')}"
    )

    # Condensed tool findings — capped at 800 characters total
    if tool_results_summary:
        condensed_context = "\n".join(tool_results_summary)[:800]
    else:
        condensed_context = "No tool calls were made. Use research data and travel request directly."

    compilation_prompt = (
        f"Compile a concise structured travel itinerary.\n\n"
        f"Destination summary: {destination_summary}\n\n"
        f"Trip details: {trimmed_request}\n\n"
        f"Tool findings:\n{condensed_context}\n"
    )
    if feedback:
        compilation_prompt += f"\nRevision feedback to apply: {feedback}\n"

    compilation_prompt += (
        "\nRules:\n"
        "- Keep each activity's 'notes' field to 1 short sentence (under 15 words).\n"
        "- Keep activity and location names brief and factual.\n"
        "- Populate all days, budget breakdown, and packing list.\n"
        "- Do not add extra commentary or prose outside the schema fields."
    )

    # Debug: print estimated token count (rough heuristic: 1 token ≈ 4 chars)
    estimated_tokens = len(SYSTEM_PROMPT + compilation_prompt) // 4
    print(f"[DEBUG] Estimated compilation request tokens: ~{estimated_tokens}")

    # Use a SEPARATE, lower-max_tokens LLM instance for the structured output
    # step so it can't runaway-generate past the TPM budget.
    structured_llm_base = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=1500)
    structured_llm = structured_llm_base.with_structured_output(PlannerOutput)

    final_output = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=compilation_prompt)
    ])
    return final_output.model_dump()
    