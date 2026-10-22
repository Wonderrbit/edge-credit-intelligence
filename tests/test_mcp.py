from __future__ import annotations


def test_mcp_server_is_configured() -> None:
    from mlops.mcp.server import server

    assert server.name == "akropoli-credit"
    assert callable(server.run)
