"""
Orchestrator: drives the whole self-improvement loop.

Flow at a glance:
  baseline -> evaluate weaknesses -> generate variants -> batch run -> repeat

The agent decides when to call which tool. We just hand it run_experiment,
evaluate_weaknesses, generate_variants, and save_prompt and let it plan.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent import (
    PROJECT,
    MODEL,
    INFERENCE_MODEL,
    DATASET,
    LOG_STREAM,
    client,
    register_eval_dataset,
    self_evaluate,
    generate_variants,
    save_winner_prompt,
    run_prompt,
    emit_event,
)
from prompts import BASELINE_PROMPT

# 15 is a soft budget — orchestrator typically uses ~10. If you up the rounds, raise this.
MAX_TOOL_CALLS = 15
DIMS = ("correctness", "completeness", "precision")


def _format_result(meta: dict, beats: bool) -> dict:
    return {
        "correctness": round(meta.get("correctness", 0), 3),
        "completeness": round(meta.get("completeness", 0), 3),
        "precision": round(meta.get("precision", 0), 3),
        "beats_baseline": beats,
        "experiment_link": meta.get("link", ""),
    }


ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": (
                "Run a Galileo experiment with a given system prompt against the 10-row evaluation dataset. "
                "Returns per-dimension scores (correctness, completeness, precision). "
                "Also returns the experiment link in Galileo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The system prompt to evaluate.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short snake_case label, e.g. 'baseline' or 'checklist_v2'.",
                    },
                },
                "required": ["prompt", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_weaknesses",
            "description": (
                "Analyze the responses from a previously run experiment to identify specific failure patterns. "
                "Call run_experiment first. Returns a list of concrete weaknesses found in the answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "The name used in the run_experiment call.",
                    },
                },
                "required": ["experiment_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_variants",
            "description": (
                "Generate 3 improved prompt variants that target identified weaknesses. "
                "Each variant has a name, prompt text, and rationale. "
                "Pass tried_strategies to avoid repeating approaches that didn't work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_prompt": {"type": "string"},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                    "tried_strategies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Strategy names already tested. Do not repeat.",
                    },
                },
                "required": ["current_prompt", "weaknesses"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiments_batch",
            "description": (
                "Run multiple experiments in parallel. "
                "Use this instead of calling run_experiment multiple times when testing variants. "
                "All experiments run simultaneously and results are returned together. "
                "Always use this when you have more than one variant to test."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiments": {
                        "type": "array",
                        "description": "List of experiments to run in parallel.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "name": {
                                    "type": "string",
                                    "description": "Short snake_case label.",
                                },
                            },
                            "required": ["prompt", "name"],
                        },
                    },
                },
                "required": ["experiments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_prompt",
            "description": "Save the winning prompt to Galileo prompt management.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["name", "prompt"],
            },
        },
    },
]

ORCHESTRATOR_SYSTEM = """
You are an autonomous prompt optimization agent integrated with Galileo.

Your task: improve a Galileo metric recommendation prompt. Each experiment returns three scores
(correctness, completeness, precision), each between 0.0 and 1.0. A variant "beats_baseline"
when it improves on at least 2 of 3 dimensions vs the baseline, with no dimension dropping more
than 0.05. Use the "beats_baseline" field — do not compute it yourself.

The metric recommender reads LLM app descriptions and recommends Galileo evaluation metrics.

Tools available:
- run_experiment: test any prompt against the dataset, get scores back
- run_experiments_batch: run multiple variants in parallel (always use this for variants)
- evaluate_weaknesses: analyze what a prompt got wrong (call after run_experiment)
- generate_variants: create 3 improved prompt variants targeting specific weaknesses
- save_prompt: save the winning prompt to Galileo when you're done

How to proceed:
1. Run the baseline to establish reference scores
2. Evaluate its weaknesses
3. Generate variants, then run ALL of them at once using run_experiments_batch. Never call run_experiment for variants one at a time.
4. After each batch, pick the best-scoring variant (highest sum of correctness + completeness + precision) as the new champion. Do NOT stop early.
5. Evaluate that champion's weaknesses, generate new variants targeting them, and batch-run again.
6. Repeat for exactly 3 rounds of run_experiments_batch. After round 3, save the best prompt found across all rounds.

After save_prompt returns, write a short final summary using ONLY the exact numbers in the save_prompt response (winner_scores, baseline_scores, deltas_vs_baseline). Rules for the summary:
- Quote exact scores, do not paraphrase
- Never claim "improvement across all dimensions" unless every delta is positive
- For each dimension, state the change with sign (e.g., "correctness -0.006, completeness +0.074, precision +0.128")
- Do not invent scores that aren't in tool outputs

Be decisive. Always run all 3 rounds — do not stop early even if a variant beats the baseline.
"""


def run(emitter=None, baseline_prompt: str | None = None) -> None:
    print("\n" + "=" * 60)
    print("self-improving metric recommendation agent")
    print("=" * 60)
    print(f"  project:      {PROJECT}")
    print(f"  orchestrator: {MODEL}")
    print(f"  inference:    {INFERENCE_MODEL}")
    print(f"  dataset:      {len(DATASET)} rows")
    print("  stop rule:    beats baseline on >=2 of 3 dims, no dim drops >0.05")
    print("=" * 60)

    emit_event(
        emitter,
        "config",
        project=PROJECT,
        model=MODEL,
        inference_model=INFERENCE_MODEL,
        max_attempts="agent-decided",
        dataset_rows=len(DATASET),
    )

    register_eval_dataset()

    meta_logger = None
    if LOG_STREAM:
        try:
            from galileo import GalileoLogger
            from galileo.schema.metrics import Metric
            from galileo.utils.metrics import create_metric_configs

            meta_logger = GalileoLogger(
                project=PROJECT, log_stream=LOG_STREAM, mode="distributed"
            )
            meta_logger.start_session(name="optimization-run")
            create_metric_configs(
                project_id=str(meta_logger.project_id),
                run_id=str(meta_logger.log_stream_id),
                metrics=[Metric(name="agentic_session_success")],
            )
            meta_logger.start_trace(
                input="Optimize a Galileo metric recommendation prompt across 3 rounds",
                name="metric-optimization-run",
            )
            print(
                "  Galileo: session started, action completion scorer registered, main trace open"
            )
        except Exception as exc:
            print(f"  Galileo log stream init failed: {exc}")

    qa_store = {}
    all_scores = {}
    baseline_scores = None
    baseline_link = None
    winner = {"name": None, "prompt": None, "scores": None}
    best_so_far = {"name": None, "score_sum": -1.0, "prompt": None, "meta": None}
    state_lock = threading.Lock()

    def _score_sum(meta: dict) -> float:
        return sum(meta.get(d, 0.0) for d in DIMS)

    def _record_result(exp_name, prompt, meta, is_baseline):
        # rule: must improve on >= 2 of 3 dims AND not regress any dim by >0.05.
        # picked these thresholds by feel — happy to revisit once we have more data.
        with state_lock:
            beats = False
            if not is_baseline and baseline_scores is not None:
                wins = sum(
                    1 for d in DIMS if meta.get(d, 0) > baseline_scores.get(d, 0)
                )
                max_regression = max(
                    0.0, max(baseline_scores.get(d, 0) - meta.get(d, 0) for d in DIMS)
                )
                beats = wins >= 2 and max_regression <= 0.05
            if not is_baseline:
                all_scores[exp_name] = meta
            if beats and (
                not winner.get("scores")
                or _score_sum(meta) > _score_sum(winner["scores"])
            ):
                winner["scores"] = meta
            if not is_baseline and _score_sum(meta) > best_so_far["score_sum"]:
                best_so_far["name"] = exp_name
                best_so_far["score_sum"] = _score_sum(meta)
                best_so_far["prompt"] = prompt
                best_so_far["meta"] = meta
        return beats

    def _execute(name: str, tool_input: dict) -> str:
        nonlocal baseline_scores, baseline_link
        if name == "run_experiment":
            prompt = tool_input["prompt"]
            exp_name = tool_input["name"]
            is_baseline = "baseline" in exp_name.lower()
            meta, qa = run_prompt(prompt, f"metric-rec-{exp_name}", use_tools=not is_baseline, emitter=emitter)
            if meta and meta.get("correctness") is not None:
                qa_store[exp_name] = qa
                with state_lock:
                    if is_baseline and baseline_scores is None:
                        baseline_scores = {d: meta.get(d, 0.0) for d in DIMS}
                        baseline_link = meta.get("link", "")
                        print(
                            f"  Baseline: correctness={baseline_scores['correctness']:.3f}  completeness={baseline_scores['completeness']:.3f}  precision={baseline_scores['precision']:.3f}"
                        )
                beats = _record_result(exp_name, prompt, meta, is_baseline)
                return json.dumps(_format_result(meta, beats))
            return json.dumps({"error": "Experiment failed or returned no scores."})

        if name == "evaluate_weaknesses":
            exp_name = tool_input["experiment_name"]
            qa = qa_store.get(exp_name)
            if not qa:
                return json.dumps(
                    {"error": f"No data for '{exp_name}'. Call run_experiment first."}
                )
            insights = self_evaluate(qa, logger=meta_logger)
            emit_event(emitter, "weaknesses", items=insights.get("weaknesses", []))
            return json.dumps({"weaknesses": insights.get("weaknesses", [])})

        if name == "generate_variants":
            insights = {"weaknesses": tool_input.get("weaknesses", [])}
            tried = tool_input.get("tried_strategies", [])
            variants = generate_variants(
                tool_input["current_prompt"], insights, tried, logger=meta_logger
            )
            emit_event(emitter, "variants_generated", variants=variants)
            return json.dumps(
                [
                    {
                        "name": v["name"],
                        "prompt": v["prompt"],
                        "rationale": v["rationale"],
                    }
                    for v in variants
                ]
            )

        if name == "run_experiments_batch":
            experiments = tool_input["experiments"]
            batch_results = {}

            def _run_one(exp):
                meta, qa = run_prompt(
                    exp["prompt"],
                    f"metric-rec-{exp['name']}",
                    emitter=emitter,
                )
                return exp["name"], exp["prompt"], meta, qa

            with ThreadPoolExecutor(max_workers=len(experiments)) as pool:
                futures = {
                    pool.submit(_run_one, exp): exp["name"] for exp in experiments
                }
                for future in as_completed(futures):
                    exp_name, exp_prompt, meta, qa = future.result()
                    if meta and meta.get("correctness") is not None:
                        qa_store[exp_name] = qa
                        beats = _record_result(
                            exp_name, exp_prompt, meta, is_baseline=False
                        )
                        batch_results[exp_name] = _format_result(meta, beats)
                    else:
                        batch_results[exp_name] = {
                            "error": "Experiment failed or returned no scores."
                        }
            return json.dumps(batch_results)

        if name == "save_prompt":
            saved_name = tool_input["name"]
            saved_prompt = tool_input["prompt"]
            winner["name"] = saved_name
            winner["prompt"] = saved_prompt
            winner_meta = all_scores.get(saved_name) or best_so_far.get("meta") or {}
            winner["scores"] = winner_meta
            save_winner_prompt({"name": saved_name, "prompt": saved_prompt})
            winner_scores = {d: round(winner_meta.get(d, 0.0), 3) for d in DIMS}
            base_scores = {
                d: round((baseline_scores or {}).get(d, 0.0), 3) for d in DIMS
            }
            deltas = {d: round(winner_scores[d] - base_scores[d], 3) for d in DIMS}
            return json.dumps(
                {
                    "saved": True,
                    "name": saved_name,
                    "winner_scores": winner_scores,
                    "baseline_scores": base_scores,
                    "deltas_vs_baseline": deltas,
                }
            )

        return json.dumps({"error": f"Unknown tool: {name}"})

    starting = baseline_prompt or BASELINE_PROMPT
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Begin optimization. The baseline prompt is:\n\n{starting}\n\n"
                f"Start by running the baseline to establish reference scores."
            ),
        },
    ]

    t_start = time.time()
    call_num = 0
    turn_num = 0

    while call_num < MAX_TOOL_CALLS:
        turn_num += 1
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=messages,
            tools=ORCHESTRATOR_TOOLS,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        if text:
            print(f"\n  Agent: {text}")
            emit_event(emitter, "agent_thought", text=text)

        tool_calls = choice.message.tool_calls or []
        tool_names = [tc.function.name for tc in tool_calls]

        last_user_input = ""
        for m in reversed(messages):
            role = m["role"] if isinstance(m, dict) else m.role
            if role in ("user", "tool"):
                content = m.get("content", "") if isinstance(m, dict) else m.content
                last_user_input = str(content or "")[:3000]
                break

        orch_output = (text or "") + (
            f"\n\nCalling: {', '.join(tool_names)}" if tool_names else ""
        )
        turn_metadata = {
            "turn": turn_num,
            "tools": tool_names,
            "stop_reason": choice.finish_reason,
        }

        turn_name = f"Orchestrator turn {turn_num}"
        if meta_logger:
            meta_logger.add_workflow_span(input=last_user_input[:1000], name=turn_name)
            meta_logger.add_llm_span(
                input=last_user_input[:5000],
                output=orch_output[:10000],
                model=MODEL,
                name=turn_name,
                metadata=turn_metadata,
            )

        if choice.finish_reason != "tool_calls" or not tool_calls:
            if meta_logger:
                meta_logger.conclude(output=orch_output[:1000])
            break

        messages.append(choice.message)

        for tc in tool_calls:
            call_num += 1
            tool_input = json.loads(tc.function.arguments)
            print(
                f"\n  [{call_num}] {tc.function.name}({json.dumps(tool_input)[:300]})"
            )
            emit_event(
                emitter,
                "agent_tool_call",
                tool=tc.function.name,
                input=tool_input,
                call_num=call_num,
            )

            result_str = _execute(tc.function.name, tool_input)
            result_data = json.loads(result_str)
            print(f"       result: {result_str}")
            emit_event(
                emitter,
                "agent_tool_result",
                tool=tc.function.name,
                result=result_data,
                call_num=call_num,
            )

            if meta_logger:
                meta_logger.add_tool_span(
                    input=tc.function.arguments[:5000],
                    output=result_str[:5000],
                    name=tc.function.name,
                    metadata={"call_num": call_num},
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )

        if meta_logger:
            meta_logger.conclude(output=orch_output[:1000])

    # Safety net: sometimes the orchestrator runs out of budget before calling save_prompt.
    # In that case, save the best variant we saw so the run still produces something.
    if not winner.get("name") and best_so_far["name"]:
        m = best_so_far["meta"] or {}
        print(
            f"\n  agent didn't call save_prompt. saving best across all rounds: {best_so_far['name']} "
            f"(correctness={m.get('correctness', 0):.3f}  "
            f"completeness={m.get('completeness', 0):.3f}  "
            f"precision={m.get('precision', 0):.3f})"
        )
        save_winner_prompt(
            {"name": best_so_far["name"], "prompt": best_so_far["prompt"]}
        )
        winner["name"] = best_so_far["name"]
        winner["prompt"] = best_so_far["prompt"]
        winner["scores"] = best_so_far["meta"]

    total = time.time() - t_start

    emit_event(
        emitter,
        "final_result",
        baseline=baseline_scores,
        winner_name=winner.get("name"),
        winner_meta=winner.get("scores"),
        baseline_prompt=starting,
        winner_prompt=winner.get("prompt"),
        total_time=total,
    )

    if meta_logger:
        try:
            final_summary = (
                f"Optimization complete. Winner: {winner.get('name', 'none')}. "
                f"Tool calls: {call_num}. Duration: {total:.0f}s."
            )
            meta_logger.conclude(output=final_summary, conclude_all=True)
            meta_logger.flush()
        except Exception as exc:
            print(f"  Galileo log stream flush failed: {exc}")

    w_link = (winner.get("scores") or {}).get("link", "")
    if baseline_link or w_link:
        print("\n  experiment comparison:")
        if baseline_link:
            print(f"    baseline: {baseline_link}")
        if w_link:
            print(f"    winner:   {w_link}")
    print(f"\n  done. {call_num} tool calls in {total:.0f}s")
    print("=" * 60)
