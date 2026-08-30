from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# Import the actual working tools from the tools directory
# pyrefly: ignore [missing-import]
from core.tools.tavily_search import search_web
# pyrefly: ignore [missing-import]
from core.tools.open_meteo import get_weather

# Load environment variables
load_dotenv()
# GROQ_API_KEY is automatically loaded by load_dotenv() and used by ChatGroq

# Define Pydantic output schema for the research agent
# This schema is used by .with_structured_output() to enforce valid structured output matching requirements.
class ResearchOutput(BaseModel):
    destination_summary: str = Field(description="A brief, high-level summary of the destination's appeal and character.")
    top_attractions: list[str] = Field(description="Top attractions/activities matching user interests with brief descriptions.")
    safety_notes: str = Field(description="Critical safety notes, local scams to avoid, and general traveler precautions.")
    weather_summary: str = Field(description="Summary of the typical or expected weather during the specified travel dates.")
    seasonal_considerations: str = Field(description="Seasonal/crowd considerations, clothing requirements, and local holidays.")

# Wrap standard tool functions into LangChain StructuredTool objects so schemas are properly generated.
# NOTE: descriptions are written to be STRICT about argument names, since smaller/faster
# models on Groq have been observed hallucinating plausible-but-wrong argument names
# (e.g. inventing "query"/"radius" instead of the tool's actual parameters).
search_web_tool = StructuredTool.from_function(
    func=search_web,
    name="search_web",
    description="Call this with EXACTLY one argument: query (string). Uses the Tavily API to search for destination info (attractions, safety, local tips, weather, seasonal considerations). Do NOT use any other argument names."
)

get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="Call this with EXACTLY these arguments: latitude (float), longitude (float), start_date (string, YYYY-MM-DD), end_date (string, YYYY-MM-DD). Uses Open-Meteo API to get weather forecast or historical climate averages. Do NOT use any other argument names."
)

# SYSTEM PROMPT DESIGN DOCUMENTATION:
# 1. Role: We precisely define the agent's role as a specialist. This shapes the context and framing of its decisions.
# 2. Scope: We explicitly define what information it needs to compile (attractions, weather, safety, considerations) so the LLM doesn't miss anything.
# 3. Tool Coordination: We instruct it to find coordinates first to satisfy get_weather constraints, and to use search_web for qualitative content.
# 4. Quality/Tone: Explicit instructions to avoid vague descriptions ensure that the returned text is high-value and concrete.
# 5. Reasoning Control: We ask the agent to plan and reason step-by-step internally (relying on its internal thoughts) before calling tools, but limit the final output to only the Pydantic schema structure.
# 6. Tool Parameter Rules (added): Explicit strict parameter naming to prevent hallucinated argument names during tool calls — a recurring failure mode observed with smaller open models on Groq.
SYSTEM_PROMPT = """You are a destination research specialist for a travel planning system.

Your goal is to gather detailed, practical, and highly specific information about the requested travel destination to help construct a tailored itinerary.
You must gather:
1. Top attractions and activities that match the traveler's stated interests.
2. Local safety information, warnings, and common traveler pitfalls/scams to avoid.
3. Weather forecasts or seasonal considerations for the given travel dates.

Tool Usage Instructions:
- You have access to `get_weather` and `search_web`.
- You must find the geographical coordinates (latitude and longitude) of the destination (using your internal knowledge) and then call `get_weather` with those coordinates and the travel dates.
- Use `search_web` to retrieve qualitative information, local tips, safety warnings, and top attractions matching the traveler's interests.
- Do not make assumptions about the weather; use the tools to retrieve accurate data.

Tone & Quality Bar:
Your recommendations and summaries must be highly practical, realistic, and specific to the destination. Avoid generic filler statements or vague descriptions (like 'great food' or 'enjoyable nightlife'); instead, provide concrete names, actual safety concerns, and specific weather details.

Reasoning and Output:
- You must reason step-by-step internally before calling tools. Plan your search queries and tool calls based on the traveler's destination, dates, and interests.
- However, your final response must contain ONLY the structured JSON output matching the requested schema. Do not include any of your internal planning or reasoning in the final response.

Tool Parameter Rules (STRICT):
- Before calling any tool, check the exact parameter names defined in its schema. Use ONLY those names — do not invent or substitute alternatives.
- For search_web: use ONLY `query`. Never use any other field name.
- For get_weather: use ONLY `latitude`, `longitude`, `start_date`, `end_date`. Never use any other field names.

Output Rules:
- Keep each field concise: 2-4 sentences per field maximum. Do not write long paragraphs.
"""

def run_research_agent(destination: str, start_date: str, end_date: str, interests: list[str]) -> dict:
    """
    Gathers qualitative destination info and weather data, then returns a structured summary.

    Args:
        destination (str): The travel destination (e.g., "Paris, France").
        start_date (str): Travel start date in YYYY-MM-DD format.
        end_date (str): Travel end date in YYYY-MM-DD format.
        interests (list[str]): List of traveler interests (e.g., ["museums", "art"]).

    Returns:
        dict: A dictionary matching the ResearchOutput schema.
    """
    # Model: qwen/qwen3.8-27b — confirmed available on this Groq API key and
    # has reliable standard tool-calling behavior. (llama3-70b-8192 is
    # decommissioned; gpt-oss "reasoning" models emit malformed tool_call
    # blocks with langchain-groq.)
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=2000)

    # Bind the tools
    tools = [search_web_tool, get_weather_tool]
    llm_with_tools = llm.bind_tools(tools)

    # Setup initial messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Destination: {destination}\nStart Date: {start_date}\nEnd Date: {end_date}\nInterests: {', '.join(interests)}")
    ]

    # Track tool results as compact plain-text snippets to avoid bloating the final prompt
    tool_results_summary: list[str] = []

    # Run execution loop (up to 5 iterations) to allow multiple tool-calling steps
    for _ in range(5):
        # Debug: estimate running token size of the message history each loop,
        # so we can see if truncation is keeping us under the TPM limit.
        running_tokens = sum(len(str(m.content)) for m in messages) // 4
        print(f"[DEBUG] Estimated tool-loop request tokens: ~{running_tokens}")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If the model does not call any tools, we break out
        if not response.tool_calls:
            break

        # Execute each tool call requested by the model
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name == "search_web":
                tool_output = search_web(**tool_args)
            elif tool_name == "get_weather":
                tool_output = get_weather(**tool_args)
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."

            # IMPORTANT: Tavily search results can be very large (full page
            # snippets, URLs, multiple results per call), and these accumulate
            # in `messages` across loop iterations, quickly exceeding Groq's
            # free-tier 8000 TPM limit. We must truncate BEFORE appending to
            # messages, not just in the final compilation summary.
            truncated_output = str(tool_output)[:600]

            # Store a compact snippet for the final compilation step (capped per-result)
            tool_results_summary.append(
                f"[{tool_name}({tool_args})] => {truncated_output[:250]}"
            )

            # Append the TRUNCATED result of the tool run as a ToolMessage
            messages.append(ToolMessage(
                content=truncated_output,
                tool_call_id=tool_id,
                name=tool_name
            ))

    # -----------------------------------------------------------------------
    # FINAL STEP: Structured output compilation.
    #
    # IMPORTANT TOKEN-BUDGET / TOOL-VALIDATION NOTES:
    # - We do NOT reuse the full `messages` history here. It contains
    #   AIMessage/ToolMessage tool-call pairs, and Groq's API rejects a
    #   request referencing those tool calls if the same tools aren't
    #   re-declared on this call (this caused repeated "tool not in
    #   request.tools" errors during development).
    # - We also cap total tool findings to keep this request comfortably
    #   under Groq's free-tier 8000 TPM (tokens-per-minute) rate limit.
    # -----------------------------------------------------------------------

    if tool_results_summary:
        condensed_context = "\n".join(tool_results_summary)[:1200]
    else:
        condensed_context = "No tool calls were made. Rely on general destination knowledge."

    compilation_prompt = (
        f"Destination: {destination}\n"
        f"Travel dates: {start_date} to {end_date}\n"
        f"Interests: {', '.join(interests)}\n\n"
        f"Tool findings:\n{condensed_context}\n\n"
        "Compile the gathered research information into the final structured output format. "
        "Ensure all schema fields are populated with concrete, specific details. "
        "Keep each field to 2-4 sentences maximum."
    )

    # Debug: print estimated token count (rough heuristic: 1 token ≈ 4 chars)
    estimated_tokens = len(SYSTEM_PROMPT + compilation_prompt) // 4
    print(f"[DEBUG] Estimated compilation request tokens: ~{estimated_tokens}")

    # Use a SEPARATE, lower-max_tokens LLM instance for the structured output
    # step so it can't runaway-generate past the TPM budget.
    structured_llm_base = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=1200)
    structured_llm = structured_llm_base.with_structured_output(ResearchOutput)

    final_output = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=compilation_prompt)
    ])

    # Return as standard dictionary
    if isinstance(final_output, ResearchOutput):
        return final_output.model_dump()
    return final_output
    