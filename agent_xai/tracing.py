"""Turns an agent run into a structured trace (the raw material for XAI later).

One JSON record per analysed request is appended to runs/<run_name>.jsonl.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from .config import RUNS_DIR
from .data_loader import HttpRequest

_JSON_LINE = re.compile(r"\{.*\"verdict\".*\}", re.DOTALL)


def parse_verdict(text: str) -> dict | None:
    m = _JSON_LINE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_trace(req: HttpRequest, messages: list, model: str) -> dict:
    steps = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            steps.append({
                "type": "llm",
                "text": msg.text,
                "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls],
            })
        elif isinstance(msg, ToolMessage):
            steps.append({"type": "tool_result", "name": msg.name, "content": msg.text})

    final_text = next((s["text"] for s in reversed(steps) if s["type"] == "llm"), "")
    verdict = parse_verdict(final_text)
    predicted = (verdict or {}).get("verdict")
    return {
        "request_id": req.request_id,
        "model": model,
        "ground_truth": req.label,
        "predicted": predicted,
        "correct": (predicted == "attack") == (req.label == "anomalous") if predicted else None,
        "verdict": verdict,
        "tool_sequence": [tc["name"] for s in steps if s["type"] == "llm" for tc in s["tool_calls"]],
        "request": req.to_dict(),
        "steps": steps,
    }


class TraceWriter:
    def __init__(self, run_name: str | None = None):
        RUNS_DIR.mkdir(exist_ok=True)
        run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.path: Path = RUNS_DIR / f"{run_name}.jsonl"

    def write(self, trace: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
