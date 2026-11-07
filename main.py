"""Entry point: run the WAF agent on CSIC 2010 requests.

On start the LLM is prepared first: with the default meta-llama/Llama-3.1-8B-Instruct the
weights are downloaded from Hugging Face (first run only, then cached) and loaded on the GPU.

Examples:
    python main.py --check-tools                       # no LLM: start MCP servers, call every tool once
    python main.py --n 4                               # 4 random requests (balanced normal/attack)
    python main.py --dataset anomalous_test --index 0  # one specific request
"""

import argparse
import asyncio
import os
import sys

from langgraph.errors import GraphRecursionError

from agent_xai.agent import build_agent, build_llm, open_mcp_tools, request_prompt
from agent_xai.config import LLM_MODEL, RECURSION_LIMIT
from agent_xai.data_loader import get_request, sample_requests
from agent_xai.tracing import TraceWriter, build_trace


async def check_tools() -> None:
    """Smoke test of the MCP layer without any LLM."""
    async with open_mcp_tools() as tools:
        print(f"Loaded {len(tools)} MCP tools:")
        for t in tools:
            args = {k: "test" for k in t.args}
            if "severity" in args:
                args["severity"] = "low"
            result = await t.ainvoke(args)
            text = result[0]["text"] if isinstance(result, list) else result
            print(f"  - {t.name}: {text}")


async def run(requests, llm, model: str) -> None:
    async with open_mcp_tools() as tools:
        agent = build_agent(llm, tools)
        print(f"Model: {model} | tools: {[t.name for t in tools]}")
        writer = TraceWriter()

        for req in requests:
            print(f"\n=== {req.request_id}  (ground truth: {req.label}) ===")
            state = {"messages": [("user", request_prompt(req))]}
            messages = []
            try:
                # stream state snapshots so partial history survives a recursion-limit stop
                async for values in agent.astream(state, {"recursion_limit": RECURSION_LIMIT},
                                                  stream_mode="values"):
                    messages = values["messages"]
            except GraphRecursionError:
                print(f"!! hit recursion limit ({RECURSION_LIMIT} steps) - model kept calling tools")
            trace = build_trace(req, messages, model)
            writer.write(trace)
            print(f"tools called : {trace['tool_sequence']}")
            print(f"verdict      : {trace['verdict']}")
            print(f"correct      : {trace['correct']}")

        print(f"\nTrace saved to {writer.path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-tools", action="store_true", help="test MCP tools without an LLM")
    p.add_argument("--n", type=int, default=2, help="number of random requests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["normal_train", "normal_test", "anomalous_test"])
    p.add_argument("--index", type=int)
    p.add_argument("--model", default=LLM_MODEL,
                   help="HF repo id (local) or provider string like anthropic:claude-opus-5-5")
    a = p.parse_args()

    if a.check_tools:
        asyncio.run(check_tools())
        return

    if a.model.startswith("anthropic:") and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    if a.dataset is not None and a.index is not None:
        requests = [get_request(a.dataset, a.index)]
    else:
        requests = sample_requests(a.n, a.seed)

    llm = build_llm(a.model)  # downloads + loads Llama on first start
    asyncio.run(run(requests, llm, a.model))


if __name__ == "__main__":
    main()
