"""Explain the agent's decisions with LIME.

Examples:
    # explain one request directly (no agent run needed)
    python explain.py --dataset anomalous_test --index 1
    python explain.py --dataset normal_test --index 7 --granularity param

    # explain the requests of an agent run, using what the agent actually decided
    python explain.py --trace runs/run_20261107_120000.jsonl --target action

Targets:  verdict (normal|attack) | action (allow|block) | first_tool (which tool first)
Outputs:  runs/lime/<request>_<target>_<granularity>.html  (+ .json for later analysis)
"""

import argparse
import asyncio
import json
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool

from agent_xai.agent import build_llm, open_mcp_tools
from agent_xai.config import LLM_MODEL, RUNS_DIR
from agent_xai.data_loader import get_request
from agent_xai.xai.lime import explain_request
from agent_xai.xai.probe import DecisionProbe
from agent_xai.xai.report import print_explanation, save_explanation

LIME_DIR = RUNS_DIR / "lime"


async def tool_schemas() -> list[dict]:
    """The exact tool schemas the agent sees (fetched from the running MCP servers)."""
    async with open_mcp_tools() as tools:
        return [convert_to_openai_tool(t) for t in tools]


def agent_choice(trace: dict, target: str) -> str | None:
    seq = trace.get("tool_sequence") or []
    if target == "verdict":
        return trace.get("predicted")
    if target == "action":
        acts = [t for t in seq if t in ("allow_request", "block_request")]
        return acts[-1] if acts else None
    return seq[0] if seq else None  # first_tool


def load_trace_items(path: Path, limit: int | None) -> list[tuple]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        ds, idx = t["request_id"].rsplit(":", 1)
        items.append((get_request(ds, int(idx)), t))
    return items[:limit] if limit else items


def explain_many(llm, schemas, items, target, granularity, num_samples, seed=0, batch_size=4):
    """items: list of (HttpRequest, trace-or-None). Reusable from a notebook."""
    probe = DecisionProbe(llm, schemas, target=target, batch_size=batch_size)
    results, agree, compared = [], 0, 0
    for req, trace in items:
        choice = agent_choice(trace, target) if trace else None
        e = explain_request(req, probe, granularity, num_samples, seed=seed, agent_choice=choice)
        path = save_explanation(e, req, LIME_DIR)
        print_explanation(e)
        print(f"  report: {path}")
        if choice in e.original_probs:
            compared += 1
            agree += max(e.original_probs, key=e.original_probs.get) == choice
        results.append(e)
    if compared:
        print(f"\nProbe agrees with the agent's actual {target} on {agree}/{compared} requests "
              f"(how faithful the probe is to the full agent).")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", type=Path, help="runs/*.jsonl produced by main.py")
    src.add_argument("--dataset", choices=["normal_train", "normal_test", "anomalous_test"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--limit", type=int, help="max requests from the trace")
    p.add_argument("--target", choices=["verdict", "action", "first_tool"], default="verdict")
    p.add_argument("--granularity", choices=["token", "param"], default="token")
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=LLM_MODEL)
    a = p.parse_args()

    items = (load_trace_items(a.trace, a.limit) if a.trace
             else [(get_request(a.dataset, a.index), None)])

    llm = build_llm(a.model)  # downloads + loads Llama on first start
    schemas = asyncio.run(tool_schemas())
    explain_many(llm, schemas, items, a.target, a.granularity, a.num_samples, a.seed, a.batch_size)


if __name__ == "__main__":
    main()
