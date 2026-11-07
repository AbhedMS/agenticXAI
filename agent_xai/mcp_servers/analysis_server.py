"""MCP server #1 - 'analysis': inspection tools the agent can call on a payload.

Run standalone:  python -m agent_xai.mcp_servers.analysis_server
"""

from mcp.server.fastmcp import FastMCP

from . import stub_response

SERVER = "analysis"
mcp = FastMCP(SERVER, log_level="WARNING")


@mcp.tool()
def detect_sql_injection(request_id: str, suspicious_fragment: str) -> str:
    """Inspect a request fragment for SQL injection (e.g. quotes, UNION, DROP, OR 1=1).

    Args:
        request_id: ID of the HTTP request being analysed.
        suspicious_fragment: The decoded parameter value or snippet you think is SQLi.
    """
    return stub_response(SERVER, "detect_sql_injection",
                         request_id=request_id, suspicious_fragment=suspicious_fragment)


@mcp.tool()
def detect_xss(request_id: str, suspicious_fragment: str) -> str:
    """Inspect a request fragment for cross-site scripting (e.g. <script>, onerror=, javascript:).

    Args:
        request_id: ID of the HTTP request being analysed.
        suspicious_fragment: The decoded parameter value or snippet you think is XSS.
    """
    return stub_response(SERVER, "detect_xss",
                         request_id=request_id, suspicious_fragment=suspicious_fragment)


@mcp.tool()
def detect_path_traversal(request_id: str, suspicious_fragment: str) -> str:
    """Inspect a request fragment for path traversal / file inclusion (e.g. ../, /etc/passwd).

    Args:
        request_id: ID of the HTTP request being analysed.
        suspicious_fragment: The path or parameter value you think is traversal.
    """
    return stub_response(SERVER, "detect_path_traversal",
                         request_id=request_id, suspicious_fragment=suspicious_fragment)


@mcp.tool()
def check_parameter_anomaly(request_id: str, parameter: str, value: str) -> str:
    """Check whether a parameter value looks abnormal for its field (wrong type, odd length,
    unexpected characters) even if it is not a classic injection.

    Args:
        request_id: ID of the HTTP request being analysed.
        parameter: Parameter name.
        value: Decoded parameter value.
    """
    return stub_response(SERVER, "check_parameter_anomaly",
                         request_id=request_id, parameter=parameter, value=value)


if __name__ == "__main__":
    mcp.run(transport="stdio")
