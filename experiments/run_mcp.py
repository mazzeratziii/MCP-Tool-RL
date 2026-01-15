from tools.tool_fast_unstable import FastUnstableTool
from tools.tool_slow_stable import SlowStableTool
from tools.tool_medium import MediumTool

from mcp.registry import ToolRegistry
from mcp.server import MCPServer
from mcp.client import MCPClient

from selector.semantic_selector import SemanticSelector
from agent.agent import Agent

def main():
    registry = ToolRegistry()
    registry.register(FastUnstableTool())
    registry.register(SlowStableTool())
    registry.register(MediumTool())

    server = MCPServer(registry)
    client = MCPClient(server)

    selector = SemanticSelector()
    agent = Agent(
        tool_names=registry.list_tools(),
        selector=selector,
        mcp_client=client
    )

    for q in ["search", "status", "lookup", "docs"]:
        print(agent.handle_query(q))

if __name__ == "__main__":
    main()
