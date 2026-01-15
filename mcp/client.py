from mcp.types import MCPRequest

class MCPClient:
    def __init__(self, server):
        self.server = server

    def call(self, tool_name: str, query: str):
        request = MCPRequest(tool_name=tool_name, query=query)
        return self.server.handle(request)
