"""MCP servers exposing stub tools to the agent.

Tools deliberately do no real work: they report that they were triggered and
echo what they received. The *agent's choice* of tool and arguments is the
behaviour we will later explain with XAI techniques.
"""

import json
import sys
from datetime import datetime, timezone


def stub_response(server: str, tool: str, **received) -> str:
    """Uniform stub payload returned by every tool."""
    payload = {
        "server": server,
        "tool": tool,
        "status": "triggered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "received": received,
        "message": f"[{server}] tool '{tool}' was triggered (stub - no real processing).",
    }
    # stdout is the MCP stdio channel, so log to stderr only.
    print(f"[{server}] {tool} triggered", file=sys.stderr, flush=True)
    return json.dumps(payload)
