# Galileo Agent Demo

You give it a weak starting prompt. It figures out what's wrong with it, generates better versions, tests them against a real eval dataset, and keeps the winner. The whole loop runs in about 10 minutes.

The task it's optimizing: given a description of an LLM application, recommend the right Galileo evaluation metrics.

## How to Run

### 1. Clone the repo

```bash
git clone <repo-url>
cd galileo-agent-demo
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv demo_venv
source demo_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GALILEO_API_KEY="your-galileo-api-key"
GALILEO_CONSOLE_URL="https://console.dev.galileo.ai"   # omit for prod
OPENAI_API_KEY="your-openai-api-key"
GALILEO_PROJECT="your-project-name"
GALILEO_LOG_STREAM="your-log-stream-name"
OPENAI_MODEL="gpt-4o"
```

Required: `GALILEO_PROJECT`, `OPENAI_MODEL`, plus the API keys.
Optional: `GALILEO_LOG_STREAM` (orchestrator traces are skipped if unset), `GALILEO_CONSOLE_URL` (defaults to prod), `INFERENCE_MODEL` (defaults to `gpt-4o-mini`).

### 5. Run

```bash
streamlit run app.py
```

The UI shows a text area pre-filled with the default starting prompt. You can edit it before clicking Run — the agent will use whatever you put there as the baseline to improve from. Good for trying different starting points or testing a prompt you already have.

Headless (no UI):

```python
from orchestrator import run
run()                                      # uses the default baseline
run(baseline_prompt="your prompt here")   # uses a custom baseline
```

## Configurable Settings

**The starting prompt** is the most interesting knob. Try:
- A very terse prompt ("Recommend Galileo metrics for this app.") — should score low and improve a lot
- A prompt that's already pretty good — watch how much headroom the agent finds
- A prompt with a specific blind spot (e.g. never mentions SQL metrics) — see if the agent catches it

**The models** are controlled by env vars: `OPENAI_MODEL` for the orchestrator and scorer, `INFERENCE_MODEL` for the per-row inference calls.

## File Structure

| File | What It Does |
|---|---|
| `agent.py` | Core library. Metric catalog, three inference tools (`check_app_type`, `list_galileo_metrics`, `get_metric_details`), 10-row eval dataset, and the main building blocks: `run_prompt()`, `self_evaluate()`, `generate_variants()`, `save_winner_prompt()`. |
| `orchestrator.py` | Outer optimization loop. Gives the orchestrator model higher-level tools (`run_experiment`, `run_experiments_batch`, `evaluate_weaknesses`, `generate_variants`, `save_prompt`) and drives the full run. Call `run()` to start. |
| `app.py` | Streamlit UI. Lets you edit the starting prompt, runs `orchestrator.run()` in a background thread, and renders baseline scores, experiment cards, weakness breakdowns, variant rationales, and a final prompt comparison. |
| `prompts.py` | The three system prompts: `BASELINE_PROMPT`, `META_PROMPT` (variant generation), `EVAL_PROMPT` (weakness identification). |
| `requirements.txt` | Pinned deps: `openai`, `galileo`, `python-dotenv`. Streamlit installed separately. |

## Why This Task

The metric recommender is a real internal use case. New FDEs and customers often don't know whether they need `context_adherence` vs `correctness`, or when `tool_selection_quality` applies. A well-optimized recommender prompt could be a lightweight internal tool.

The recursive angle — using Galileo to evaluate and improve a Galileo-facing prompt — is intentional. It shows the platform is useful for your own internal tooling, not just customer apps.

## Galileo Platform Coverage

- **Datasets** — `create_dataset()` registers a 10-row eval set on startup.
- **Experiments** — every prompt tested via `run_experiment()`, all visible in the UI.
- **LocalMetricConfig** — three custom LLM-as-judge scorers: correctness, completeness, precision.
- **Native Metrics** — `instruction_adherence` runs server-side on every experiment.
- **Spans** — each dataset row gets a workflow span, with LLM and tool spans nested inside. Metadata includes experiment name, round number, and `use_tools`.
- **Log Streams** — `GalileoLogger` traces every orchestrator turn, weakness analysis, and variant generation call in a single persistent session.
- **Prompt Management** — winner saved via `create_prompt()` under a stable name with an `{app_description}` template variable. Galileo handles versioning.

## Architecture: Two Agents, One Loop

**Outer agent (orchestrator)** drives the optimization. It decides what to run, reads scores, picks the next strategy, and decides when to stop. Its tools are high-level: `run_experiment`, `run_experiments_batch`, `evaluate_weaknesses`, `generate_variants`, `save_prompt`.

**Inner agent (inference model)** answers metric recommendation questions. It runs once per dataset row per experiment, with up to 5 tool-calling rounds using `check_app_type`, `list_galileo_metrics`, and `get_metric_details`.

The inference model is intentionally cheaper and faster than the orchestrator. The orchestrator handles the reasoning-heavy work (scoring, weakness analysis, variant generation).

## The Optimization Loop

```
1. Run baseline (no tools, 1 round) → record baseline scores
2. Evaluate weaknesses → 3 specific failure patterns
3. Generate 3 variants targeting those weaknesses
4. Run all 3 in parallel (ThreadPoolExecutor)
5. Pick best variant → evaluate its weaknesses → repeat
6. After 3 batch rounds, save the best prompt found
```

The baseline runs without tools on purpose — the inference model with full tool access is strong enough that even a weak prompt produces decent answers, leaving no room for improvement. Without tools it answers from model weights only: vaguer, more generic, lower scores.

A variant "beats" the baseline if it wins on ≥ 2 of the 3 dimensions and no single dimension drops more than 0.05.

## Scoring

Three `LocalMetricConfig` scorers, each making one LLM-as-judge call per dataset row:

| Scorer | What It Checks |
|---|---|
| correctness | Right primary metrics for the app type (RAG → context_adherence, agent → tool_selection_quality, SQL → sql_correctness) |
| completeness | Important metrics not missing (safety for high-stakes apps, agentic metrics for multi-step agents) |
| precision | No irrelevant metrics included (RAG metrics for a non-RAG app, etc.) |

Winner selection uses the sum of all three scores (equal weight). The 0.5/0.3/0.2 figures appear only in the judge rubric prompts to steer the LLM-as-judge — not as code multipliers. Empty answers score 0.0. `temperature=0` on all scorer calls for determinism.

## The 10-Row Dataset

Ten archetypes an FDE would encounter. Each experiment completes in ~60 seconds and a full 3-round run stays under 10 minutes.

| Row | App Type | What It Tests |
|---|---|---|
| 1 | RAG chatbot (product docs) | Core RAG metric selection |
| 2 | Customer support bot (no retrieval) | Correctly skipping RAG metrics |
| 3 | Multi-step research agent | Agentic metric coverage |
| 4 | NL-to-SQL generator | SQL-specific metrics |
| 5 | Customer service bot with injected context | Trap row: looks like RAG but isn't |
| 6 | Hybrid RAG + agent (research assistant) | Recommending both RAG and agentic metrics |
| 7 | Content moderation pipeline | Safety-first, no retrieval |
| 8 | Document summarization pipeline | Pure generation, no retrieval |
| 9 | IDE code assistant | Pure LLM, skipping irrelevant metric categories |
| 10 | Clinical decision support (medical) | High-stakes RAG, safety metrics non-negotiable |

Row 5 is the classic adversarial case — context injected into the prompt, not retrieved. Row 6 tests whether the recommender covers both halves of a hybrid app. Row 10 tests whether safety metrics get flagged as required rather than optional for high-stakes domains.
