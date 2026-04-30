# Baseline is intentionally weak — that's the point. The loop should make it better.
BASELINE_PROMPT = (
    "You are a helpful AI assistant. Given a description of an LLM application, "
    "recommend the most relevant Galileo evaluation metrics."
)

META_PROMPT = """\
You are an expert prompt engineer improving a Galileo metric recommendation assistant.

The assistant has three tools:
- check_app_type(description): classifies the app and returns which metric categories apply
- list_galileo_metrics(category): lists metrics in a category
- get_metric_details(metric_name): gets details for a specific metric

The assistant can call multiple tools in parallel in a single step.

Current system prompt:
<current_prompt>
{current_prompt}
</current_prompt>

Weaknesses identified by evaluation:
<weaknesses>
{weaknesses}
</weaknesses>

{tried_section}
Write 3 improved system prompts that address these weaknesses WITHOUT losing what the current \
prompt already does well. Identify the strengths of the current prompt and preserve them. \
Each variant must take a different strategic approach to fixing the weaknesses. The prompts \
should guide the agent to use its tools effectively, call check_app_type first, batch detail \
lookups in parallel, and reason carefully about app type before recommending metrics. \
Be concrete and specific — name actual app types (e.g. non-retrieval chatbot, text-to-SQL, \
agentic pipeline), metric categories, and failure modes directly in the prompt. \
Avoid vague instructions like "avoid unrelated metrics" — say exactly what to exclude and when.

Keep each prompt under 120 words. Concise and directive, not exhaustive.

Return a JSON array. Each element must have:
- "name": short snake_case identifier
- "rationale": one sentence explaining the approach
- "prompt": the full improved system prompt (max 120 words)

Return ONLY the JSON array. No markdown, no explanation.\
"""

EVAL_PROMPT = """\
You are evaluating a Galileo metric recommendation assistant's responses.

Each entry shows an LLM app description, the assistant's answer, and per-dimension scores \
(0.0–1.0). Lower scores mean worse performance on that dimension.

<qa_pairs>
{qa_pairs}
</qa_pairs>

Verified Galileo metric names (do NOT flag these as hallucinated or invented):
{metric_names}

Focus on the lowest-scoring rows. For each weakness you identify, state:
- Which app type it applies to (from the examples above)
- Which score dimension it hurt (correctness, completeness, or precision)
- Exactly what was wrong or missing in the answer

Return a JSON object with:
- "weaknesses": list of 3 specific weaknesses grounded in the scores above

Return ONLY the JSON object. No markdown, no explanation.\
"""
