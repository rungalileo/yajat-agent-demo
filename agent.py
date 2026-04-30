"""
core agent logic for the metric recommender + self-improvement loop.

the interesting bits:
  - run_prompt:        one experiment pass over the eval dataset
  - self_evaluate:     asks an LLM to point out weaknesses in the answers
  - generate_variants: produces new prompts to try next round

most things you'd want to tweak (model names, project, dataset name) live near the top.
"""

import json
import os
import re
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from prompts import META_PROMPT, EVAL_PROMPT  # noqa: E402  (load_dotenv first)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in .env (see README, How to Run)."
        )
    return val


PROJECT = _require_env("GALILEO_PROJECT")
MODEL = _require_env("OPENAI_MODEL")
LOG_STREAM = os.environ.get("GALILEO_LOG_STREAM")
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "gpt-4o-mini")
EVAL_DATASET_NAME = "metric-rec-eval-dataset"

client = OpenAI(max_retries=5)


# hand-curated subset of galileo's metric catalog. not auto-synced, so check the
# galileo docs if anything here looks stale.
# the descriptions are written to be useful to the LLM agent picking metrics,
# not to be exhaustive — keep them short.
METRIC_CATALOG = {
    "response_quality": {
        "correctness": "LLM-as-judge factual accuracy check. No ground truth needed. Safe default.",
        "ground_truth_adherence": "Match against a reviewed gold answer. Skip unless you actually have one.",
        "instruction_adherence": "Did the answer follow the system prompt rules (tone, format, length, persona)?",
    },
    "rag": {
        "context_adherence": "RAG. Is the answer grounded in the retrieved docs, or did it go beyond them?",
        "context_relevance": "RAG. Were the retrieved chunks actually relevant to the query?",
        "context_precision": "RAG. Are all retrieved chunks useful? Penalizes noisy retrieval.",
        "completeness": "RAG. Did the answer cover the key info from the retrieved context?",
        "chunk_attribution_utilization": "RAG. What fraction of retrieved chunks ended up used.",
    },
    "agentic": {
        "tool_selection_quality": "Agents. Did the agent pick the right tools at each step?",
        "action_advancement": "Agents. Does each action move toward the goal?",
        "action_completion": "Agents. Did multi-step tasks actually finish?",
        "agent_efficiency": "Agents. Penalizes redundant or unnecessary tool calls.",
        "reasoning_coherence": "Agents. Is the reasoning consistent across steps?",
    },
    "safety": {
        "toxicity": "Harmful or offensive output.",
        "pii_detection": "PII leaking into the response.",
        "prompt_injection": "Adversarial inputs trying to hijack the model.",
    },
    "text_to_sql": {
        "sql_correctness": "SQL apps. Query is valid and logically correct.",
        "sql_adherence": "SQL apps. Query respects the given schema.",
        "sql_efficiency": "SQL apps. Flags slow / overly complex queries.",
    },
    "expression": {
        "tone": "Style match against a requested tone (formal, casual, empathetic, ...).",
        "bleu": "Lexical similarity vs. ground truth. Better for translation / templated output.",
        "rouge": "Recall-oriented overlap vs. ground truth. Better for summarization.",
    },
}

# flat list of all metric names — used by the eval prompt so the judge knows what's "real"
# (otherwise it sometimes flags valid metric names as hallucinations).
_all_metrics = []
for _cat in METRIC_CATALOG.values():
    _all_metrics.extend(_cat.keys())
METRIC_NAMES = ", ".join(_all_metrics)


def _mean(vals):
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _extract_json(raw: str, array: bool = False) -> str:
    open_, close = ("[", "]") if array else ("{", "}")
    start, end = raw.find(open_), raw.rfind(close) + 1
    return raw[start:end] if start != -1 and end > start else raw


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_app_type",
            "description": (
                "Classify an LLM application description into its primary type(s). "
                "Call this first to determine which metric categories apply before calling list_galileo_metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The LLM application description to classify.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_galileo_metrics",
            "description": (
                "List available Galileo evaluation metrics by category. "
                "Use this to explore what metrics exist before making a recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Metric category to list.",
                        "enum": [
                            "response_quality",
                            "rag",
                            "agentic",
                            "safety",
                            "text_to_sql",
                            "expression",
                            "all",
                        ],
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_details",
            "description": (
                "Get description and requirements for a specific Galileo metric. "
                "Use this to understand when a metric applies and what data it needs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the metric (e.g. 'correctness', 'context_adherence').",
                    },
                },
                "required": ["metric_name"],
            },
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "check_app_type":
        description = tool_input.get("description", "").lower()
        types = []

        def match(*patterns):
            return any(re.search(p, description) for p in patterns)

        if match(r"\bretriev", r"\bvector", r"\brag\b", r"\bchunk", r"\bembed"):
            types.append("rag")
        if match(
            r"\bagent",
            r"\bmulti-step",
            r"\borchestrat",
            r"\btool\b",
            r"\baction\b",
            r"\bplan\b",
            r"\bweb search",
        ):
            types.append("agentic")
        if match(r"\bsql\b", r"\bquery", r"\bdatabase", r"\bschema", r"\btable\b"):
            types.append("text_to_sql")
        if match(
            r"\bmoderat",
            r"\btoxic",
            r"\bsafe",
            r"\bharm",
            r"\bpii\b",
            r"\binject",
            r"\bhigh-stakes",
            r"\bmedical",
            r"\bclinical",
        ):
            types.append("safety")
        if not types:
            types.append("response_quality")
        all_categories = ["rag", "agentic", "text_to_sql", "safety"]
        not_applicable = [c for c in all_categories if c not in types]
        return json.dumps(
            {
                "app_types": types,
                "metrics_to_consider": {
                    t: list(METRIC_CATALOG[t].keys()) for t in types
                },
                "not_applicable": not_applicable,
            }
        )

    if name == "list_galileo_metrics":
        category = tool_input.get("category", "all")
        if category == "all":
            lines = []
            for cat, metrics in METRIC_CATALOG.items():
                lines.append(f"\n{cat.upper()}:")
                lines.extend(f"  - {m}" for m in metrics)
            return "\n".join(lines)
        metrics = METRIC_CATALOG.get(category, {})
        if not metrics:
            return f"Unknown category '{category}'. Valid: {', '.join(METRIC_CATALOG)}"
        return "\n".join(f"- {m}" for m in metrics)

    if name == "get_metric_details":
        # normalize: agent sometimes asks for "context-adherence" instead of "context_adherence"
        target = tool_input.get("metric_name", "").lower().replace("-", "_").strip()
        if not target:
            return "no metric_name provided"
        for cat, metrics in METRIC_CATALOG.items():
            if target in metrics:
                return "{name} [{cat}]: {desc}".format(name=target, cat=cat, desc=metrics[target])
        return f"metric '{target}' not found. use list_galileo_metrics to see available metrics."

    return f"unknown tool: {name}"


# 10 rows is small but it's enough for a demo. Bigger eval sets are nice but
# add cost and time without changing the story much. Revisit if scores feel noisy.
DATASET = [
    {
        "input": "We're building a RAG chatbot that answers questions about our product documentation. Users ask things like 'how do I configure X' and the system retrieves relevant docs to answer.",
        "output": "Primary: context_adherence, completeness, correctness. Secondary: context_relevance, chunk_attribution_utilization. Skip ground_truth_adherence unless you have expert-validated answers. Reasoning: RAG systems must verify responses are grounded in retrieved context (context_adherence), cover all relevant info (completeness), and are factually accurate (correctness). Context_relevance measures upstream retrieval quality.",
    },
    {
        "input": "Customer support chatbot with no retrieval. It handles billing questions, returns policy, and account issues using a fine-tuned model.",
        "output": "Primary: correctness, instruction_adherence. Secondary: tone, toxicity. Skip all RAG metrics — there is no retrieval. Reasoning: Without retrieval the key risks are factual errors (correctness), violating the support persona or policy (instruction_adherence), and unprofessional or harmful outputs (tone, toxicity).",
    },
    {
        "input": "Multi-step research agent that uses web search tools, reads articles, and synthesizes a final report on a given topic.",
        "output": "Primary: tool_selection_quality, reasoning_coherence, correctness. Secondary: action_advancement, action_completion. Skip RAG metrics — this is tool use, not retrieval-augmented generation. Reasoning: Agents must choose the right tools (tool_selection_quality), maintain coherent reasoning across steps (reasoning_coherence), and produce accurate reports (correctness). Action_advancement checks each step moves toward the goal; action_completion checks multi-step tasks are fully finished.",
    },
    {
        "input": "Natural language to SQL query generator. Business users describe the data they want and we produce SQL against a known schema.",
        "output": "Primary: sql_correctness, sql_adherence, instruction_adherence. Secondary: correctness. Skip RAG metrics — schema context is structured input, not retrieved. Reasoning: SQL generation has domain-specific metrics that are more precise than general LLM-as-judge: sql_correctness checks query validity, sql_adherence checks schema compliance.",
    },
    {
        "input": "Customer service bot where each conversation injects the customer's full account history directly into the system prompt. The model is fine-tuned on company policy documents. No vector store, no retrieval pipeline.",
        "output": "Primary: correctness, instruction_adherence, pii_detection. Secondary: toxicity, tone. Do NOT recommend RAG metrics — context is injected into the prompt, not retrieved from a vector store. Context_adherence and context_relevance do not apply. Reasoning: Despite using customer context, this is not RAG. Key risks are factual errors about account details (correctness), policy violations (instruction_adherence), and leaking PII in responses (pii_detection).",
    },
    {
        "input": "A multi-step research assistant that retrieves academic papers from a vector store and uses an agentic tool loop to compare, summarize, and synthesize findings across multiple documents into a final report.",
        "output": "Primary: context_adherence, tool_selection_quality, reasoning_coherence. Secondary: context_relevance, completeness, action_advancement. Recommend both RAG and agentic metrics — this app does both retrieval and multi-step tool use. Reasoning: Retrieval grounds the responses (context_adherence, context_relevance), while the agent layer requires correct tool use (tool_selection_quality) and coherent multi-step reasoning (reasoning_coherence). Skipping either category misses half the failure surface.",
    },
    {
        "input": "A content moderation pipeline that classifies user-generated posts for toxicity, hate speech, and policy violations. The model analyzes posts directly — no retrieval, no external tools.",
        "output": "Primary: toxicity, instruction_adherence, correctness. Secondary: pii_detection, prompt_injection. Skip RAG and agentic metrics — there is no retrieval or tool use. Reasoning: The core risk is failing to catch harmful content (toxicity) or misclassifying policy violations (correctness). Prompt_injection matters because adversarial users may embed instructions in their posts to manipulate the classifier.",
    },
    {
        "input": "An internal document summarization pipeline. Employees upload long reports and PDFs and the system generates concise summaries. Documents are passed directly as context — no vector store, no retrieval pipeline.",
        "output": "Primary: correctness, instruction_adherence. Secondary: rouge (only if reference summaries exist). Skip RAG metrics — documents are passed directly, not retrieved. Skip agentic metrics — there are no tool calls or multi-step actions. Reasoning: Summarization is a pure generation task. The key risks are factual errors (correctness) and ignoring length or format constraints (instruction_adherence). Rouge is only useful if you have human-written reference summaries to compare against.",
    },
    {
        "input": "An AI coding assistant integrated into VS Code. It helps developers write, refactor, and debug code based on inline instructions. No retrieval, no database access.",
        "output": "Primary: correctness, instruction_adherence. Secondary: tone. Skip RAG, agentic, and SQL metrics — this is a pure LLM generation task with no retrieval, tool use, or database queries. Reasoning: The dominant risk is incorrect code suggestions (correctness) and not following the developer's specific instructions (instruction_adherence). Tone matters for inline suggestions — terse and direct is usually preferred.",
    },
    {
        "input": "A clinical decision support chatbot used by nurses to look up drug interactions and dosage guidelines. It retrieves from a medical knowledge base. Responses directly influence patient care decisions.",
        "output": "Primary: context_adherence, correctness, toxicity. Secondary: context_relevance, completeness, pii_detection. This is a high-stakes RAG application — safety metrics are non-negotiable. Reasoning: Medical advice must be grounded in retrieved guidelines (context_adherence) and factually correct (correctness). Toxicity and pii_detection protect patient safety and privacy. Completeness is critical — missing a contraindication is a patient safety risk.",
    },
]


def emit_event(emitter, event_type: str, **data) -> None:
    if emitter:
        emitter({"type": event_type, **data})


_DEBUG = False


def _dbg(*args):
    if _DEBUG:
        print(">>", *args)


def register_eval_dataset() -> None:
    from galileo.datasets import create_dataset

    # The SDK doesn't expose a clean "create or get" call, so we try-create and
    # fall back if it already exists. Brittle, but fine for the demo.
    try:
        create_dataset(EVAL_DATASET_NAME, DATASET, project_name=PROJECT)
        print(f"  dataset registered: '{EVAL_DATASET_NAME}' ({len(DATASET)} rows)")
    except Exception as exc:
        msg = str(exc).lower()
        if "already exists" in msg or "already in use" in msg or "duplicate" in msg:
            print(f"  dataset '{EVAL_DATASET_NAME}' already exists, reusing")
        else:
            raise


def run_prompt(
    system_prompt: str,
    experiment_name: str,
    use_tools: bool = True,
    emitter=None,
) -> tuple[Optional[dict], list]:
    from galileo import galileo_context
    from galileo.experiments import run_experiment
    from galileo.schema.metrics import LocalMetricConfig, Metric

    # uuid suffix so re-runs don't collide on the Galileo side
    full_name = f"{experiment_name}-{uuid.uuid4().hex[:8]}"
    print(f"\n  running experiment: {full_name}")
    emit_event(emitter, "experiment_start", name=full_name)
    t_start = time.time()

    qa_pairs = []
    row_scores: dict[str, dict[str, float]] = {}

    def _agent_core(user_text: str, logger) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        # 5 rounds is plenty — most prompts settle in 2-3. baseline never uses tools.
        rounds = 5 if use_tools else 1
        _dbg("agent_core start", experiment_name, "rounds=", rounds)
        for round_idx in range(rounds):
            kwargs = dict(
                model=INFERENCE_MODEL,
                max_tokens=512,
                messages=messages,
            )
            if use_tools:
                kwargs["tools"] = TOOLS
            response = client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            text = choice.message.content or ""
            tool_calls = choice.message.tool_calls or []

            if logger:
                output_repr = text or json.dumps(
                    [
                        {"name": tc.function.name, "input": tc.function.arguments}
                        for tc in tool_calls
                    ]
                )
                logger.add_llm_span(
                    input=user_text,
                    output=output_repr,
                    model=INFERENCE_MODEL,
                    name=f"round_{round_idx + 1}",
                    metadata={
                        "experiment": experiment_name,
                        "round": round_idx + 1,
                        "use_tools": use_tools,
                    },
                )

            if choice.finish_reason == "stop":
                return text
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    result = execute_tool(tc.function.name, tool_input)
                    if logger:
                        logger.add_tool_span(
                            input=tc.function.arguments,
                            output=result,
                            name=tc.function.name,
                            metadata={
                                "experiment": experiment_name,
                                "round": round_idx + 1,
                            },
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                return text
        return choice.message.content or ""

    # Galileo calls scorer_fn twice per span (once for the row, once for the trace).
    # Memoize on id(span) so we don't pay for the LLM judge call twice.
    # TODO: id() can be reused if a span is GC'd mid-run — hasn't bitten us yet.
    corr_cache = {}
    prec_cache = {}
    comp_cache = {}

    def _make_scorer(cache: dict, dimension: str, rubric: str):
        def _scorer(span) -> float:
            sid = id(span)
            if sid in cache:
                return cache[sid]
            ground_truth = getattr(span, "dataset_output", None) or ""
            app_desc = getattr(span, "dataset_input", None) or ""
            raw = span.output
            answer = (
                raw.content
                if hasattr(raw, "content") and isinstance(raw.content, str)
                else str(raw)
            )
            if not answer:
                cache[sid] = 0.0
                return 0.0
            if not ground_truth:
                # shouldn't happen if the dataset is well-formed, but log it loudly
                print(f"  warn: dataset row missing ground truth [{dimension}]")
                cache[sid] = 0.5
                return 0.5
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=256,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Rate the {dimension} of this Galileo metric recommendation (0.0–1.0).\n\n"
                            + (f"App: {app_desc[:500]}\n\n" if app_desc else "")
                            + f"Reference:\n{ground_truth[:1500]}\n\nActual:\n{answer[:1500]}\n\n{rubric}\n\nRespond with a single decimal number only. Example: 0.75"
                        ),
                    }
                ],
            )
            raw_text = (resp.choices[0].message.content or "").strip()
            # judge sometimes ignores instructions and writes a sentence — pull the first number out
            num_match = re.search(r"-?\d+(?:\.\d+)?", raw_text)
            if num_match:
                score = min(1.0, max(0.0, float(num_match.group())))
            else:
                print(
                    f"  scorer: no number found [{dimension}] -> '{raw_text[:60]}', defaulting to 0.5"
                )
                score = 0.5
            cache[sid] = score
            if app_desc:
                row_scores.setdefault(app_desc, {})[dimension] = score
            return score

        return _scorer

    # the three rubrics are deliberately blunt — long rubrics seem to make the judge
    # waffle. keep them tight and the scores are more stable across runs.
    score_correctness = _make_scorer(
        corr_cache,
        "correctness",
        "1.0 = right primary metrics for the app type "
        "(RAG -> context_adherence, agent -> tool_selection_quality, SQL -> sql_correctness, etc.)\n"
        "0.5 = some right ones but wrong-category metrics mixed in\n"
        "0.0 = wrong metrics for the app type "
        "(e.g. RAG metrics recommended for a non-RAG chatbot)",
    )
    score_precision = _make_scorer(
        prec_cache,
        "precision",
        "1.0 = no irrelevant metrics. GTA only when ground truth is actually available.\n"
        "0.5 = 1-2 unnecessary or off-category metrics\n"
        "0.0 = mostly metrics that don't apply",
    )
    score_completeness = _make_scorer(
        comp_cache,
        "completeness",
        "1.0 = all the key metrics are there "
        "(safety for high-stakes apps, agentic metrics for tool-using agents, both RAG+agent for hybrids)\n"
        "0.5 = most are there but 1-2 important ones missing\n"
        "0.0 = whole categories missing "
        "(e.g. no agentic metrics for a multi-step agent)",
    )

    def llm_call(user_text: str) -> str:
        logger = galileo_context.get_logger_instance()
        logger.add_workflow_span(input=user_text, name="metric_recommender")
        answer = _agent_core(user_text, logger)
        logger.conclude(output=answer)
        qa_pairs.append({"question": user_text, "answer": answer})
        emit_event(
            emitter,
            "question_answered",
            experiment=full_name,
            count=len(qa_pairs),
            total=len(DATASET),
        )
        return answer

    metrics = [
        LocalMetricConfig(
            name="correctness", scorer_fn=score_correctness, aggregator_fn=_mean
        ),
        LocalMetricConfig(
            name="precision", scorer_fn=score_precision, aggregator_fn=_mean
        ),
        LocalMetricConfig(
            name="completeness", scorer_fn=score_completeness, aggregator_fn=_mean
        ),
        Metric(name="instruction_adherence"),
    ]
    try:
        results = run_experiment(
            full_name,
            project=PROJECT,
            dataset_name=EVAL_DATASET_NAME,
            function=llm_call,
            metrics=metrics,
        )
    except Exception as exc:
        print(f"  experiment failed: {exc}")
        emit_event(emitter, "experiment_failed", name=full_name, error=str(exc))
        return None, qa_pairs

    for pair in qa_pairs:
        pair["scores"] = row_scores.get(pair["question"], {})

    experiment = results["experiment"]
    link = results.get("link", "")
    actual_name = experiment.name
    print(f"  agent answered {len(qa_pairs)} questions  [{time.time()-t_start:.1f}s]")
    print(f"  galileo: {link}")

    avg_corr = sum(corr_cache.values()) / len(corr_cache) if corr_cache else None
    avg_prec = sum(prec_cache.values()) / len(prec_cache) if prec_cache else None
    avg_comp = sum(comp_cache.values()) / len(comp_cache) if comp_cache else None

    galileo_meta = {
        "correctness": avg_corr,
        "precision": avg_prec,
        "completeness": avg_comp,
        "experiment_name": actual_name,
        "link": link,
    }
    if avg_corr is not None:
        print(
            f"  scores ({len(corr_cache)} rows): "
            f"correctness={avg_corr:.3f}  completeness={avg_comp:.3f}  precision={avg_prec:.3f}"
        )
        emit_event(
            emitter, "experiment_done", name=actual_name, scores=galileo_meta, link=link
        )
    else:
        print("  no scores captured")

    return galileo_meta, qa_pairs


def _log_meta_call(logger, name, messages, output, metadata=None):
    # Logs a meta-LLM call (self-eval / variant gen) to galileo.
    # If we're already inside a trace, attach as a span; otherwise make a single-span trace.
    if not logger:
        return
    input_str = str(messages[0].get("content", ""))[:1000]
    if logger.current_parent() is not None:
        logger.add_llm_span(
            input=input_str,
            output=output[:10000],
            model=MODEL,
            name=name,
            metadata=metadata or {},
        )
    else:
        logger.add_single_llm_span_trace(
            input=input_str,
            output=output[:10000],
            model=MODEL,
            name=name,
            metadata=metadata or {},
        )
        logger.flush()
    print(f"  galileo: {name}")


def self_evaluate(qa_pairs: list, logger=None) -> dict:
    t0 = time.time()
    sample = qa_pairs

    def _fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else "N/A"

    formatted = "\n\n".join(
        f"Q: {p['question']}\n"
        f"A: {p['answer']}\n"
        f"Scores — correctness: {_fmt(p.get('scores', {}).get('correctness'))}, "
        f"completeness: {_fmt(p.get('scores', {}).get('completeness'))}, "
        f"precision: {_fmt(p.get('scores', {}).get('precision'))}"
        for p in sample
    )
    prompt_content = EVAL_PROMPT.format(qa_pairs=formatted, metric_names=METRIC_NAMES)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt_content}],
    )
    raw = _extract_json((response.choices[0].message.content or "").strip())
    _dbg("self_evaluate raw[:200]=", raw[:200])
    try:
        result = json.loads(raw)
        weaknesses = result.get("weaknesses", []) or []  # be paranoid: model sometimes returns null
        print(
            f"\n  self-evaluation: {len(weaknesses)} weaknesses identified ({time.time()-t0:.1f}s)"
        )
        for w in weaknesses:
            print(f"    - {w}")
        # cap so we don't blow up the meta-prompt with a wall of weaknesses
        if len(weaknesses) > 8:
            weaknesses = weaknesses[:8]
            result["weaknesses"] = weaknesses
        _log_meta_call(
            logger,
            name="Identify weaknesses in current prompt",
            messages=[{"role": "user", "content": prompt_content}],
            output=json.dumps(result),
            metadata={
                "weaknesses_count": len(weaknesses),
                "qa_pairs_evaluated": len(sample),
            },
        )
        return result
    except json.JSONDecodeError:
        # judge ignored the "JSON only" instruction. salvage what we can.
        print(f"  self-evaluation parse failed:\n    {raw[:200]}")
        return {"weaknesses": [raw[:500]]}


def generate_variants(
    current_prompt: str, insights: dict, tried: list, logger=None
) -> list:
    t0 = time.time()
    tried_section = (
        f"Already tried (do NOT repeat): {', '.join(tried)}. Use different approaches.\n"
        if tried
        else ""
    )
    weaknesses = "\n".join(f"- {w}" for w in insights.get("weaknesses", []))
    prompt_content = META_PROMPT.format(
        current_prompt=current_prompt,
        weaknesses=weaknesses,
        tried_section=tried_section,
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt_content}],
    )
    raw = _extract_json((response.choices[0].message.content or "").strip(), array=True)
    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Variant generation parse failed:\n    {raw[:200]}")
        return []
    print(f"\n  Generated {len(variants)} variants  [{time.time()-t0:.1f}s]:")
    for v in variants:
        print(f"    - {v['name']}: {v['rationale']}")
    _log_meta_call(
        logger,
        name="Generate improved prompt variants",
        messages=[{"role": "user", "content": prompt_content}],
        output=json.dumps(
            [{"name": v["name"], "rationale": v["rationale"]} for v in variants]
        ),
        metadata={"variants_generated": len(variants), "tried_count": len(tried)},
    )
    return variants


def save_winner_prompt(winner: dict) -> None:
    from galileo.prompts import create_prompt

    print("\n  saving winning prompt to galileo...")
    try:
        template = winner["prompt"] + "\n\nApp description: {app_description}"
        pt = create_prompt(
            name="metric-rec-winner",
            template=template,
            project_name=PROJECT,
        )
        print(f"  saved: template id={pt.id}, variant={winner['name']}")
    except Exception as exc:
        # don't crash the run if prompt registry write fails — we still have results
        print(f"  failed to save prompt: {exc}")
