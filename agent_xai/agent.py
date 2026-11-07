"""Builds the LangChain agent wired to the MCP servers."""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack, asynccontextmanager

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import LLAMA_QUANTIZATION, LLM_MODEL, MAX_NEW_TOKENS, PROJECT_ROOT
from .data_loader import HttpRequest

SYSTEM_PROMPT = """You are a web application firewall analyst. You receive one raw HTTP request
and must decide whether it is NORMAL traffic or an ATTACK.

Workflow:
1. Read the request. URL-decode parameters mentally.
2. Call the relevant analysis tools (detect_sql_injection, detect_xss, detect_path_traversal,
   check_parameter_anomaly) on any fragment you find suspicious. Skip them if nothing is suspicious.
   Call one tool at a time.
3. Take exactly one action: allow_request OR block_request. Additionally call raise_alert
   if the attack looks serious.
4. Finish with a single line of JSON and nothing after it:
   {"verdict": "normal"|"attack", "attack_type": "<type or none>", "confidence": <0..1>,
    "evidence": ["<short quote from the request>", ...]}

The tools are stubs that only acknowledge they were triggered, so your own reasoning
determines the verdict."""


def mcp_connections() -> dict:
    """stdio connections: each MCP server runs as a subprocess of this Python."""
    def server(module: str) -> dict:
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", module],
            "cwd": str(PROJECT_ROOT),
        }

    return {
        "analysis": server("agent_xai.mcp_servers.analysis_server"),
        "response": server("agent_xai.mcp_servers.response_server"),
    }


@asynccontextmanager
async def open_mcp_tools():
    """Start every MCP server once and keep its session open while the block runs.

    (client.get_tools() without a session would spawn a new server process per tool call.)
    """
    client = MultiServerMCPClient(mcp_connections())
    async with AsyncExitStack() as stack:
        tools = []
        for name in client.connections:
            session = await stack.enter_async_context(client.session(name))
            tools += await load_mcp_tools(session, server_name=name)
        yield tools


def build_llm(model: str = LLM_MODEL):
    """'org/name' -> local Hugging Face model (downloaded on first start); else init_chat_model."""
    if "/" in model and ":" not in model:
        from .llama_model import load_llama  # lazy: torch/transformers only needed here

        return load_llama(model, quantization=LLAMA_QUANTIZATION, max_new_tokens=MAX_NEW_TOKENS)
    return init_chat_model(model)


def build_agent(llm, tools):
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def request_prompt(req: HttpRequest) -> str:
    return f"Request ID: {req.request_id}\n\n```http\n{req.to_raw()}```"
