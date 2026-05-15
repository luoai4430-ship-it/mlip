import os
import json


def get_mcp_transport() -> str:
    """
    与 server.py/test.py 保持一致：
    - 默认使用 streamable-http
    - 支持通过 MCP_TRANSPORT 环境变量覆盖
    """
    return os.getenv("MCP_TRANSPORT", "streamable-http")


def get_agent_runtime_info() -> str:
    """返回 Agent 侧当前 MCP 传输配置。"""
    return json.dumps(
        {
            "component": "agent",
            "transport": get_mcp_transport(),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    print(get_agent_runtime_info())
