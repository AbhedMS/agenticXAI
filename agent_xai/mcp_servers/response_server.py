"""MCP server #2 - 'response': actions the agent takes after deciding.

Run standalone:  python -m agent_xai.mcp_servers.response_server
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

from . import stub_response

SERVER = "response"
mcp = FastMCP(SERVER, log_level="WARNING")


@mcp.tool()
def allow_request(request_id: str, reason: str) -> str:
    """Let a request through because it looks like normal traffic.

    Args:
        request_id: ID of the HTTP request.
        reason: Short justification for allowing it.
    """
    return stub_response(SERVER, "allow_request", request_id=request_id, reason=reason)


@mcp.tool()
def block_request(request_id: str, attack_type: str, reason: str) -> str:
    """Block a request judged to be malicious.

    Args:
        request_id: ID of the HTTP request.
        attack_type: e.g. sql_injection, xss, path_traversal, parameter_tampering, other.
        reason: Short justification citing the evidence.
    """
    return stub_response(SERVER, "block_request",
                         request_id=request_id, attack_type=attack_type, reason=reason)


@mcp.tool()
def raise_alert(request_id: str, severity: Literal["low", "medium", "high", "critical"],
                summary: str) -> str:
    """Notify the security team (SOC) about a request.

    Args:
        request_id: ID of the HTTP request.
        severity: low | medium | high | critical.
        summary: One-sentence summary of the incident.
    """
    return stub_response(SERVER, "raise_alert",
                         request_id=request_id, severity=severity, summary=summary)


if __name__ == "__main__":
    mcp.run(transport="stdio")
