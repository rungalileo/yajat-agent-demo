# Streamlit UI for the optimization run.
# We push events from a worker thread into a queue; the main thread drains it on each rerun.
# Not the prettiest, but it keeps Streamlit happy and avoids long blocking calls in the UI loop.

import queue
import threading
import time

import streamlit as st

from prompts import BASELINE_PROMPT

st.set_page_config(page_title="Galileo Agent Demo", layout="wide")
st.title("Self-Improving Metric Recommendation Agent")

if "events" not in st.session_state:
    st.session_state.events = []
if "running" not in st.session_state:
    st.session_state.running = False
if "q" not in st.session_state:
    st.session_state.q = None

st.subheader("Starting prompt")
st.caption("This is the prompt the agent will start from and try to improve. Edit it to try different baselines.")
starting_prompt = st.text_area(
    label="starting_prompt",
    value=BASELINE_PROMPT,
    height=120,
    disabled=st.session_state.running,
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 6])
with col_btn:
    if not st.session_state.running:
        clicked = st.button("▶  Run", type="primary", use_container_width=True)
    else:
        clicked = False
        st.button("Running...", disabled=True, use_container_width=True)

if clicked:
    st.session_state.events = []
    st.session_state.running = True
    q = queue.Queue()
    st.session_state.q = q
    prompt_snapshot = starting_prompt

    def _thread():
        # Imported inside the thread so Streamlit doesn't try to re-import on every rerun.
        from orchestrator import run

        try:
            run(emitter=q.put, baseline_prompt=prompt_snapshot)
        except Exception as exc:
            q.put({"type": "error", "msg": str(exc)})
        q.put({"type": "_done"})

    threading.Thread(target=_thread, daemon=True).start()
    st.rerun()

if st.session_state.q:
    try:
        while True:
            ev = st.session_state.q.get_nowait()
            if ev["type"] == "_done":
                st.session_state.running = False
            else:
                st.session_state.events.append(ev)
    except queue.Empty:
        pass


def derive(events):
    s = {
        "config": None,
        "baseline": {"running": False, "exp_name": None, "scores": None, "link": None},
        "agent_log": [],
        "final": None,
        "error": None,
    }
    for ev in events:
        t = ev["type"]
        if t == "config":
            s["config"] = ev
        elif t == "experiment_start":
            if "baseline" in ev["name"]:
                s["baseline"]["running"] = True
                s["baseline"]["exp_name"] = ev["name"]
        elif t == "question_answered":
            if "baseline" in ev["experiment"]:
                s["baseline"]["answered"] = ev["count"]
                s["baseline"]["total"] = ev["total"]
        elif t == "experiment_done":
            if "baseline" in ev["name"]:
                s["baseline"]["running"] = False
                s["baseline"]["scores"] = ev["scores"]
                s["baseline"]["link"] = ev["link"]
            else:
                s["agent_log"].append({"type": "experiment_done", **ev})
        elif t in ("agent_thought", "agent_tool_call", "agent_tool_result"):
            s["agent_log"].append(ev)
        elif t == "final_result":
            s["final"] = ev
        elif t == "error":
            s["error"] = ev["msg"]
    return s


state = derive(st.session_state.events)

if state["error"]:
    st.error(f"Error: {state['error']}")

if state["config"]:
    cfg = state["config"]
    st.caption(
        f"project: **{cfg['project']}**, eval: {cfg['model']}, "
        f"inference: {cfg['inference_model']}, dataset: {cfg['dataset_rows']} rows"
    )
    st.divider()

bl = state["baseline"]
if bl["running"] or bl["scores"]:
    if bl["running"] and not bl["scores"]:
        answered = bl.get("answered", 0)
        total = bl.get("total", "?")
        st.caption(f"baseline running: {answered}/{total} questions answered")
    if bl["scores"]:
        s = bl["scores"]
        link_md = f" [link]({bl['link']})" if bl["link"] else ""
        st.caption(
            f"**baseline**: correctness {s['correctness']:.3f}, "
            f"completeness {s['completeness']:.3f}, "
            f"precision {s['precision']:.3f}{link_md}"
        )
    st.divider()

if state["agent_log"]:
    st.subheader("Agent Reasoning")
    entries = state["agent_log"]
    i = 0
    while i < len(entries):
        entry = entries[i]
        t = entry["type"]

        if t == "agent_thought":
            st.markdown(f"*{entry['text']}*")
            i += 1

        elif t == "agent_tool_call" and entry["tool"] == "run_experiment":
            name = entry["input"].get("name", "")
            result = None
            if (
                i + 1 < len(entries)
                and entries[i + 1]["type"] == "agent_tool_result"
                and entries[i + 1]["tool"] == "run_experiment"
            ):
                result = entries[i + 1]["result"]
                i += 1
            with st.container(border=True):
                if result and "correctness" in result:
                    beat = result.get("beats_baseline", False)
                    prefix = "WIN " if beat else ""
                    link = result.get("experiment_link", "")
                    link_md = f" [link]({link})" if link else ""
                    st.markdown(f"**{prefix}{name}**")
                    st.caption(
                        f"correctness {result['correctness']}, "
                        f"completeness {result['completeness']}, "
                        f"precision {result['precision']}{link_md}"
                    )
                elif result and "error" in result:
                    st.markdown(f"**{name}** failed: {result['error']}")
                else:
                    st.markdown(f"**{name}** running...")
            i += 1

        elif t == "agent_tool_call" and entry["tool"] == "run_experiments_batch":
            experiments = entry["input"].get("experiments", [])
            results = None
            if (
                i + 1 < len(entries)
                and entries[i + 1]["type"] == "agent_tool_result"
                and entries[i + 1]["tool"] == "run_experiments_batch"
            ):
                results = entries[i + 1]["result"]
                i += 1
            for exp in experiments:
                exp_name = exp["name"]
                exp_result = results.get(exp_name, {}) if results else {}
                with st.container(border=True):
                    if exp_result and "correctness" in exp_result:
                        beat = exp_result.get("beats_baseline", False)
                        prefix = "WIN " if beat else ""
                        link = exp_result.get("experiment_link", "")
                        link_md = f" [link]({link})" if link else ""
                        st.markdown(f"**{prefix}{exp_name}**")
                        st.caption(
                            f"correctness {exp_result['correctness']}, "
                            f"completeness {exp_result['completeness']}, "
                            f"precision {exp_result['precision']}{link_md}"
                        )
                    elif exp_result and "error" in exp_result:
                        st.markdown(f"**{exp_name}** failed: {exp_result['error']}")
                    else:
                        st.markdown(f"**{exp_name}** running...")
            i += 1

        elif t == "agent_tool_call" and entry["tool"] == "evaluate_weaknesses":
            result = None
            if (
                i + 1 < len(entries)
                and entries[i + 1]["type"] == "agent_tool_result"
                and entries[i + 1]["tool"] == "evaluate_weaknesses"
            ):
                result = entries[i + 1]["result"]
                i += 1
            if result and "weaknesses" in result:
                weaknesses = result["weaknesses"]
                with st.expander(f"Weaknesses identified ({len(weaknesses)})"):
                    for w in weaknesses:
                        st.markdown(f"• {w}")
            else:
                st.caption("Analyzing weaknesses...")
            i += 1

        elif t == "agent_tool_call" and entry["tool"] == "generate_variants":
            result = None
            if (
                i + 1 < len(entries)
                and entries[i + 1]["type"] == "agent_tool_result"
                and entries[i + 1]["tool"] == "generate_variants"
            ):
                result = entries[i + 1]["result"]
                i += 1
            if result and isinstance(result, list):
                with st.expander(f"Variants generated ({len(result)})"):
                    for v in result:
                        st.markdown(f"**{v['name']}**: {v['rationale']}")
            else:
                st.caption("Generating variants...")
            i += 1

        elif t == "agent_tool_call" and entry["tool"] == "save_prompt":
            result = None
            if (
                i + 1 < len(entries)
                and entries[i + 1]["type"] == "agent_tool_result"
                and entries[i + 1]["tool"] == "save_prompt"
            ):
                result = entries[i + 1]["result"]
                i += 1
            if result and result.get("saved"):
                st.success(f"Saved `{result.get('name', '')}`")
            i += 1

        elif t == "agent_tool_result":
            i += 1

        else:
            i += 1
    st.divider()

if state["final"]:
    f = state["final"]
    st.subheader("Results")
    if f["winner_name"] and f["winner_prompt"]:
        col1, col2 = st.columns(2)
        with col1:
            b = f["baseline"] or {}
            b_str = (
                f"correctness {b.get('correctness',0):.3f}, completeness {b.get('completeness',0):.3f}, precision {b.get('precision',0):.3f}"
                if b
                else ""
            )
            st.caption(f"**baseline**: {b_str}")
            st.code(f["baseline_prompt"])
        with col2:
            w = f["winner_meta"] or {}
            w_str = (
                f"correctness {w.get('correctness',0):.3f}, completeness {w.get('completeness',0):.3f}, precision {w.get('precision',0):.3f}"
                if w
                else ""
            )
            st.caption(f"**{f['winner_name']}**: {w_str}")
            st.code(f["winner_prompt"])
    else:
        st.warning("No prompt beat the baseline.")
    st.caption(f"Total time: {f['total_time']:.0f}s")

if st.session_state.running:
    time.sleep(1)
    st.rerun()
