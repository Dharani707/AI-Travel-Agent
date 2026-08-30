# AGENTS.md

Take-home assignment (Express Analytics, AI/ML Engineer): an AI travel planner using a multi-agent system with human-in-the-loop (HITL) approval, served via FastAPI. Full spec: `AI_ML Engineer Take Home Assignment.docx.pdf` in the repo root. Stack: Python 3.12, LangGraph, FastAPI, Groq.

## Status: incomplete vs. assignment

Built so far: two agents + four tools + a LangGraph orchestrator (see Architecture). **Not yet built**:
- The FastAPI app and its 4 required endpoints: `POST /plan`, `GET /plan/{id}`, `POST /plan/{id}/review`, `GET /plan/{id}/final` (must call `core/orchestrator.py`'s `start_planning_session` / `get_session_state` / `submit_review`).
- A populated `README.md` (currently empty).

The LangGraph orchestrator (`core/orchestrator.py`) is complete: `TravelState` (TypedDict), research → planner → hitl → finalize nodes, conditional re-route of `rejected`/`modified` back to the planner, native `interrupt()` + resume via `Command(resume=...)`, `InMemorySaver` checkpointer, and the three public functions above. Verified against installed **langgraph 1.2.11** — note its API differences from older 0.x docs: `MemorySaver` is renamed `InMemorySaver` (`langgraph.checkpoint.memory`), and `SqliteSaver` is NOT installed (separate `langgraph-checkpoint-sqlite` package).

## Commands / setup

- Managed with `uv` (`uv.lock`, `uv_build` backend, `pyproject.toml`). Requires Python >=3.12 (`.python-version`). Use `uv sync` / `uv add <pkg>`.
- `.env` at repo root is required; every agent/tool calls `load_dotenv()` at import. Keys: `GROQ_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_PLACES_API_KEY`, `GOOGLE_ROUTES_API_KEY` (see `.env.example`).
- The committed `.venv` is a **Windows-created** venv (`Lib/Scripts`) — unusable on Linux. Recreate it with `uv sync` on the target OS.
- No lint/typecheck/test config or dependency exists (only `# pyrefly: ignore` comments in the agents).

## Layout gotcha

The real code lives in the **top-level `core/` package** (not under `src/`); `src/ai_travel_planner/__init__.py` is only a stub `main()` backing the `ai-travel-planner` console script. Because `core/` is top-level, **all imports like `from core...` only resolve when running from the repo root**. Run scripts/tests from the repo root, never via a `src/`-relative path.

## Architecture today

- `core/agents/research_agent.py` — `run_research_agent()`; tools: `search_web` (Tavily), `get_weather` (Open-Meteo). Output model `ResearchOutput`.
- `core/agents/planner_agent.py` — `run_planner_agent(research_data, travel_request, feedback=None)`; tools: `search_places` (Google Places), `get_route` (Google Directions). Output model `PlannerOutput`.
- `core/orchestrator.py` — `StateGraph` wiring the two agents (research → planner → HITL → finalize) with StateGraph-native `interrupt()` + `Command(resume=...)`. Public interface: `start_planning_session(travel_request)` / `get_session_state(thread_id)` / `submit_review(thread_id, decision, feedback=None)`. Each agent is still a tool-calling loop + separate structured-output LLM call; the orchestrator calls them as graph nodes.

## Tests (not pytest)

`tests/test_*.py` are plain scripts with a `main()` that call the **real agents and live APIs**, printing JSON. Run from the repo root: `.venv/bin/python tests/test_planner_agent.py`. They need a valid `.env` and consume Groq/Tavily/Google rate-limit budget. There is no unit-test suite.

## Model/API quirks (hard-earned)

- Groq model must be `qwen/qwen3.8-27b`. `llama3-70b-8192` is decommissioned; gpt-oss "reasoning" models emit malformed `tool_call` blocks with langchain-groq.
- Groq free tier caps at **8000 TPM**. Keep prompts lean; tool results are truncated (600 chars per result, capped to 800–1200 chars before the structured-output step).
- Do **not** reuse the tool-calling `messages` history in the final structured-output call — Groq rejects tool-call references unless the same tools are re-declared; use a fresh System+Human prompt with condensed context.
- Tool argument names are intentionally strict — small Groq models hallucinate wrong names (e.g. `query`/`radius` instead of `location`/`place_type`). Don't "fix" the strict structure in the tool descriptions.
- `get_weather` auto-falls back to historical (same dates, previous year) when the trip start is >14 days out.
