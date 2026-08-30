# AI Travel Planner

A multi-agent AI travel planner that researches a destination, drafts a
day-by-day itinerary, and hands it to a human for approval before returning
the final plan — served as a FastAPI application.

This is a take-home assignment for **Express Analytics**, focused on
architecture and workflow design: a LangGraph state machine with two LLM
agents, human-in-the-loop (HITL) approval, external tool integration, and a
thin HTTP API over the whole thing.

---

## 1. Project Overview

The system accepts a travel request (destination, dates, budget, traveler
count, interests) and runs it through a multi-agent LangGraph workflow:

1. A **Research Agent** gathers qualitative destination info (attractions,
   safety, food) and weather via Tavily and Open-Meteo.
2. An **Itinerary Planner Agent** builds a structured, day-by-day itinerary
   using Google Places (to find real venues) and Google Routes (to sequence
   them sensibly by proximity and transit time).
3. The workflow **pauses** ("interrupts") after producing a draft itinerary
   and waits for a human decision: **approve**, **reject**, or **modify**.
   Reject/modify loops back to the planner with free-text feedback; approve
   finalizes the plan.
4. A FastAPI layer exposes the whole flow as four endpoints
   (`POST /plan`, `GET /plan/{id}`, `POST /plan/{id}/review`,
   `GET /plan/{id}/final`) plus a root health check.

Both agents are powered by a single Groq-hosted LLM, **`qwen/qwen3.8-27b`**.

---

## 2. Architecture

```
USER
 │
 ▼
Orchestrator (LangGraph StateGraph)
 │
 ▼
Research Agent
 ├── Tool: Tavily Search
 └── Tool: Open-Meteo
 │
 ▼
Itinerary Planner Agent
 ├── Tool: Google Places API
 └── Tool: Google Routes API
 │
 ▼
Draft Itinerary
 │
 ▼
HITL Review (workflow pauses/interrupts here)
 ├── APPROVE → Final Itinerary
 ├── REJECT (with feedback) → back to Planner Agent
 └── MODIFY → back to Planner Agent
 │
 ▼
FINAL ITINERARY

LLM: Groq, model qwen/qwen3.8-27b (used by both agents)
```

A request enters via `POST /plan`, which calls
`core.orchestrator.start_planning_session()` and compiles a new LangGraph
`StateGraph` instance with a fresh `thread_id` (UUID). The graph runs
`research_node` → `planner_node` → `hitl_node`. The `hitl_node` uses
LangGraph's `interrupt()` primitive, so `graph.invoke()` returns as soon as
the draft is ready and the workflow is suspended mid-graph, awaiting human
input.

`submit_review()` resumes the graph with `Command(resume={...})`. If the
decision is `approve`, `route_after_review` sends execution to
`finalize_node`, which sets `final_output = draft_itinerary` and reaches
`END`. If the decision is `reject` or `modify`, the conditional edge routes
back to `planner_node`, which regenerates the itinerary using the stored
research data, the draft as `previous_itinerary`, and the free-text
`feedback` — then interrupts again for the next review. The graph is
compiled with an `InMemorySaver` checkpointer, which is what makes the
paused state addressable by `plan_id` across calls within a running process.

---

## 3. Setup Instructions

**Prerequisites**

- Python **3.12+** (the repo pins `3.12` via `.python-version`)
- **uv** package manager (https://docs.astral.sh/uv/)

**Install**

```bash
git clone <repo-url>
cd ai-travel-planner
uv sync
```

`uv sync` installs all dependencies from `pyproject.toml` / `uv.lock` into a
project `.venv`. All dependency versions are pinned, so this is
reproducible.

**Environment variables**

Create a `.env` file and fill in the keys:

Required keys:

| Variable                  | Where to get it                                                                 |
| ------------------------- | ------------------------------------------------------------------------------- |
| `GROQ_API_KEY`            | Groq console (https://console.groq.com) — create an API key, free tier |
| `TAVILY_API_KEY`          | Tavily dashboard (https://tavily.com) — API keys section                        |
| `GOOGLE_PLACES_API_KEY`   | Google Cloud Console — enable the **Places API**, create an API key            |
| `GOOGLE_ROUTES_API_KEY`   | Google Cloud Console — enable the **Routes/Directions API**, create an API key |

The app reads secrets from `.env` via `python-dotenv`; `.env` is
gitignored. **Never commit real keys.** Note that the Groq free tier caps
this model at **8000 TPM**, which matters (see §6).

---

## 4. How to Run

Start the API from the repo root (imports resolve `core` and `app` only when
you run from here):

```bash
.venv\Scripts\uvicorn app.app_factory:app --reload    # Windows
.venv/bin/uvicorn app.app_factory:app --reload        # macOS/Linux
```

or, with the venv activated:

```bash
uvicorn app.app_factory:app --reload
```

The app entrypoint is `app.app_factory:app` — `app_factory.py` builds the
FastAPI instance via `create_app()` and exposes it as `app`.

**Interactive API docs** (Swagger UI) are available at
`http://127.0.0.1:8000/docs` — you can exercise every endpoint there,
including the HITL review flow.

---

## 5. API Reference — Example Requests & Responses

All endpoints return JSON. Relevant status codes:

| Code | Meaning                                                                                          |
| ---- | ------------------------------------------------------------------------------------------------ |
| 200  | Success (all happy paths)                                                                        |
| 404  | `plan_id` doesn't exist                                                                          |
| 409  | Invalid state transition: reviewing an approved/non-reviewable plan, or fetching an unapproved final plan |
| 422  | Invalid input (Pydantic validation failed)                                                       |
| 500  | Unexpected internal/agent/API failure (clean message, no stack trace leaked)                     |

---

### Health check

**`GET /`**

**Method:** GET
**URL:** http://127.0.0.1:8000/
**Body:** None

Response:

```json
{ "name": "AI Travel Planner", "status": "ok" }
```

---

### 1. Create a plan

**`POST /plan`**

**Method:** POST
**URL:** http://127.0.0.1:8000/plan
**Body:** raw JSON, e.g.:

```json
{
  "destination": "Rome, Italy",
  "start_date": "2026-09-10",
  "end_date": "2026-09-12",
  "budget": "Mid-range",
  "num_travelers": 2,
  "interests": ["ancient history", "pizza", "art museums"]
}
```

Response `200` — `PlanCreatedResponse`. The workflow runs the two agents and
pauses at HITL, so you get a draft immediately:

```json
{
  "plan_id": "b6d5a3ef-1c9f-4b2a-8d4e-5f0a11c92d77",
  "status": "pending",
  "draft_itinerary": {
    "days": [
      {
        "day": 1,
        "date": "2026-09-10",
        "activities": [
          {
            "time": "09:00 AM",
            "activity": "Colosseum and Roman Forum guided tour",
            "location": "Piazza del Colosseo, 1, 00184 Roma",
            "notes": "Pre-book skip-the-line tickets."
          },
          {
            "time": "01:00 PM",
            "activity": "Lunch: authentic Roman pizza at a local pizzeria",
            "location": "Trastevere neighborhood",
            "notes": "Closest to Colosseum, ~20 min walk."
          }
        ]
      },
      {
        "day": 2,
        "date": "2026-09-11",
        "activities": [
          {
            "time": "10:00 AM",
            "activity": "Vatican Museums & Sistine Chapel",
            "location": "Viale Vaticano, 00165 Roma",
            "notes": "Late-afternoon slots are less crowded."
          }
        ]
      }
    ],
    "estimated_budget_breakdown": {
      "lodging": "€180 per night",
      "food": "€50 per day",
      "transport": "€15 per day",
      "attractions": "€60 total"
    },
    "packing_suggestions": ["Comfortable walking shoes", "Light layers for warm September days"]
  },
  "final_output": null
}
```

*Output is illustrative — exact content reflects live provider responses.*

---

### 2. Check plan status

**`GET /plan/{id}`**

**Method:** GET
**URL:** http://127.0.0.1:8000/plan/972d87f8-5a7d-4226-b336-be01cceb9c66
**Body:** None

Response `200` — `PlanStatusResponse` (adds `pending_review`):

```json
{
  "plan_id": "b6d5a3ef-1c9f-4b2a-8d4e-5f0a11c92d77",
  "status": "pending",
  "draft_itinerary": { "...": "same as above" },
  "final_output": null,
  "pending_review": true
}
```

Returns `404` if the `plan_id` doesn't exist:

```json
{ "detail": "Plan 'b6d5a3ef-1c9f-4b2a-8d4e-5f0a11c92d77' not found." }
```

---

### 3. Submit a review decision

**`POST /plan/{id}/review`**

**Method:** POST
**URL:** http://127.0.0.1:8000/plan/972d87f8-5a7d-4226-b336-be01cceb9c66/review
**Body:** raw JSON, e.g.:

```json
{
  "decision": "reject",
  "feedback": "Swap day 2's museum for a relaxed park walk and picnic at Villa Borghese."
}
```

Response `200` — `ReviewResponse`. A rejected plan re-plans with the
feedback and pauses again, so `status` is `rejected` and a fresh draft is
ready:

```json
{
  "plan_id": "b6d5a3ef-1c9f-4b2a-8d4e-5f0a11c92d77",
  "status": "rejected",
  "draft_itinerary": { "...": "revised draft honoring the feedback" },
  "final_output": null
}
```

Approving (possibly after iterations) finalizes the plan:

```json
{
  "decision": "approve"
}
```

```json
{
  "plan_id": "b6d5a3ef-1c9f-4b2a-8d4e-5f0a11c92d77",
  "status": "approved",
  "draft_itinerary": { "...": "final draft" },
  "final_output": { "...": "same as the draft itinerary" }
}
```

Validation errors (`422`):

```json
{
  "detail": [
    {
      "loc": ["decision"],
      "msg": "Input should be 'approve', 'reject' or 'modify'",
      "type": "literal_error"
    }
  ]
}
```

- `422` — invalid `decision`, or `reject`/`modify` without `feedback`
- `404` — unknown `plan_id`
- `409` — plan already approved (no further reviews allowed) or not in a
  reviewable state:

```json
{ "detail": "Plan 'b6d5a3ef-1c9f-4b2a-8d4e-5f0a11c92d77' is not in a reviewable state (current status: approved)." }
```

---

### 4. Get the final plan

**`GET /plan/{id}/final`**

**Method:** GET
**URL:** http://127.0.0.1:8000/plan/972d87f8-5a7d-4226-b336-be01cceb9c66/final
**Body:** None

Response `200` — `FinalPlanResponse` (contains `final_output` only):

```json
{
  "final_output": {
    "days": [ "...": "approved itinerary" ],
    "estimated_budget_breakdown": { "...": "..." },
    "packing_suggestions": ["..."]
  }
}
```

Errors:

- `404` — unknown `plan_id`
- `409` — plan not yet approved:

```json
{ "detail": "Plan not yet approved. Current status: pending" }
```

---

### Testing in Postman step-by-step

1) Start the server (use the `uvicorn` command from §4). 2) Open Postman and create a new request. 3) For each call below: set the method, paste the URL, and for POST requests go to the **Body** tab → select **raw** → select **JSON** → paste the JSON shown. 4) After step 2 (`POST /plan`) succeeds, copy the real `plan_id` from the response and substitute it into all subsequent URLs in place of the placeholder UUID.

### Full flow, narratively

1. **Submit a plan** → `POST /plan` → `200` with `plan_id`,
   `status: "pending"`, and a `draft_itinerary` ready for review.
2. **Check status** → `GET /plan/{id}` → `200` with `pending_review: true`.
3. **Reject with feedback** → `POST /plan/{id}/review` with
   `{"decision": "reject", "feedback": "..."}` → `200`, new draft generated
   around the feedback, `status: "rejected"`, still `pending_review`.
4. **Approve** → `POST /plan/{id}/review` with `{"decision": "approve"}` →
   `200`, `status: "approved"`, `final_output` populated.
5. **Get the final plan** → `GET /plan/{id}/final` → `200` with the approved
   itinerary.
6. **Try to fetch the final plan of an unapproved plan** → e.g. immediately
   after step 1, or for a `rejected` plan → `409`:

```json
{ "detail": "Plan not yet approved. Current status: rejected" }
```

You can try all of this interactively in Swagger UI at
`http://127.0.0.1:8000/docs`.

---

## 6. Design Decisions & Tradeoffs

**Why Groq, not Gemini or OpenAI.** The original intent was Google Gemini —
`langchain-google-genai` is still in the dependency list — but the Gemini
billing/credits were exhausted mid-build, blocking progress. Groq
(`langchain-groq`) was already in the dependency set, is fast, and its API
keys were readily available, so the implementation switched to Groq for both
agents. The downside is a hard reliance on one provider and its strict
free-tier rate limits (see below).

**Model selection journey.** This was the most painful part of the build:

- **`llama3-70b-8192`** was the first choice but was **decommissioned
  mid-build** on Groq, forcing a migration.
- The **`gpt-oss` "reasoning" models** emitted **malformed `tool_call`
  blocks** through `langchain-groq` — the tool-calling loop either failed or
  returned non-parseable output, which broke the whole agent loop.
- **`qwen/qwen3.8-27b`** was confirmed available on the API key and has
  reliable, standard tool-calling behavior, so it became the model for both
  the research and planner agents (`temperature=0.2`).

**Groq free-tier 8000 TPM constraint.** This ceiling is easy to blow: Tavily
and Google results are large, and tool responses accumulate in the message
history across loop iterations. The fixes were:

- **Truncate tool output to 600 characters before appending it as a
  `ToolMessage`** in the research agent (truncation *before* writing to
  message history, not just in the final summary).
- Use a **separate, lower-`max_tokens` LLM instance for the structured-output
  step** (1200/1500 tokens) so the final generation can't run away past the
  budget.
- **Strip verbose/redundant context from the final compilation prompt.**
  The planner agent only passes a 150-character destination summary and a
  trimmed scalar trip line — it deliberately **excludes the full
  `research_data` and the `previous_itinerary`** dicts (which can be very
  large on a HITL revision); only the free-text `feedback` string carries the
  revision intent. Tool findings are likewise condensed (research: 1200-char
  cap; planner: 800-char cap with per-result 250-char truncation).
- The tool-calling message history is **not reused** for the structured
  output call at all (Groq rejects a request that references earlier
  tool-call IDs unless the same tools are re-declared).

**Strict tool-argument naming.** Smaller open models hallucinate plausible
but wrong argument names (e.g. inventing `query`/`radius` instead of the
real parameters). Every `StructuredTool` description states the **exact**
argument names and explicitly forbids alternatives, and the system prompts
repeat the same constraint: `search_web` takes only `query`; `get_weather`
takes only `latitude`/`longitude`/`start_date`/`end_date`; `search_places`
takes only `location`/`place_type`; `get_route` takes only
`origin`/`destination`/`mode`.

**InMemorySaver over SqliteSaver.** Given the 1.5–2 day build timeframe,
the simplest checkpointer that supports paused/interrupted state was chosen.
The tradeoff is explicit and accepted: with `InMemorySaver`, **state does
NOT survive a server restart** — interrupted HITL sessions are gone on
process exit. This is a documented limitation, not a hidden one (§7 covers
the fix).

**Google Places/Routes key setup friction.** On GCP the **billing
activation, API enablement, and API-key restriction changes propagate with
delays**, causing otherwise-valid keys to transiently fail with quota or
"request from this Android/desktop application is not authorized"-style
errors. This is real-world API integration friction: standard practice
(billing on, APIs enabled, key restricted to the API + referrer/IP) was still
flaky for a while after setup.

**"Modify" is currently a full re-plan.** The graph's `route_after_review`
returns `planner_node` for *any* decision other than `approve`, so `modify`
follows the same path as `reject`: full re-planning guided by free-text
feedback, rather than granular field-level editing of specific days or
activities. This was a time-constraint decision; the feedback still reaches
the planner and produces a revised draft. The ideal version is described in
§7.

**No frontend / chatbot UI.** The assignment explicitly said to "focus on
architecture and workflow design over UI polish," so the deliverable is the
well-tested HTTP API (exercised via Swagger UI at `/docs` and Postman), not
a web UI.

---

## 7. What I'd Improve With More Time

### Reliability & Model Infrastructure

- **Replace fixed-character truncation with proper token-counting**
  (`tiktoken`) for dynamic, guaranteed-safe context trimming instead of a
  600/800/1200-char heuristic.
- **Add retry with exponential backoff** on transient API failures (Groq,
  Tavily, Google) instead of surfacing them directly.
- **Add a model fallback chain across providers** so a single-provider
  outage or model deprecation (like `llama3-70b-8192`) doesn't take the
  whole pipeline down.

### Persistence & State Management

- **Swap `InMemorySaver` for a durable checkpointer** (`SqliteSaver` /
  Postgres) so paused HITL sessions survive server restarts.
- **Add session TTL/cleanup** to prevent unbounded in-memory state growth
  from abandoned plans.

### Multi-Day Itinerary Scale

- **Chunked/paginated itinerary generation** for longer trips (7–14 days)
  to stay within token budgets.
- **Incremental day-by-day HITL review** instead of reviewing the whole trip
  at once.

### Tool & Data Quality

- **Domain filtering/scoring on Tavily results** to prioritize authoritative
  sources over SEO/blog content.
- **Caching repeated tool calls** (same destination + dates) to cut redundant
  API spend and latency.
- **Validate suggested places are currently operating** via Google Places'
  `business_status` field, and drop permanently closed venues from drafts.

### HITL & API Design

- **Implement true granular "modify"** (patch a specific day/activity) rather
  than routing it through a full re-plan like `reject`.
- **Async/polling-friendly status endpoint** so a frontend could show
  "planning in progress" during the slow research + planning phase.
- **Add authentication/authorization** so `plan_id` alone isn't sufficient
  to review or approve any session.

### Observability & Testing

- **Structured logging per session** (tool calls, token usage, latency per
  node).
- **Replace the plain-script tests with a real pytest suite using mocked API
  responses** so CI doesn't depend on live providers or rate limits.
- **Integration tests for error paths** (invalid `plan_id`, reviewing an
  already-approved plan, malformed requests) — not just the happy path.

### Cost & Rate Limiting

- **Per-user / per-session API rate limiting**, since each `/plan` call
  triggers multiple paid LLM + external API calls.
- **Parallelize independent tool calls** (e.g. weather + places lookups) to
  reduce end-to-end latency.

---

## 8. Production Concerns & Assumptions

### Production concerns

- **Secrets management:** `.env` files are fine for local dev, but not for
  production — use a real secrets manager (Vault, AWS/GCP/Azure secret
  stores) and never bake keys into containers or CI.
- **Horizontal scaling:** the `InMemorySaver` checkpointer lives in the
  process, so state is not shared across instances. Multi-instance
  deployment requires a durable, shared checkpointer.
- **Monitoring/alerting:** agent runs can fail on any upstream call; you'd
  want structured logging, and alerting on error rates and on the 500 path.
- **Cost controls:** each plan is several LLM + third-party API calls; at
  scale you'd need budgets, rate limits, and per-user quotas.

### Assumptions made

- **Budget tiers are qualitative strings** (`Budget` / `Mid-range` /
  `Luxury`), not numeric ranges — the planner picks activities and estimates
  accordingly. The API accepts any non-blank string but the prompts expect
  these tier words.
- **Single currency (EUR)** is assumed for budget estimates in the itinerary
  output.
- **Trips are assumed to be under ~7 days** so a single generation pass stays
  within token budgets; longer trips are out of scope for now (see §7).

---

*Built for the Express Analytics take-home assignment. API-first, tested via
Swagger UI and Postman, with a LangGraph HITL workflow at its core.*