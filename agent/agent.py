class Agent:
    def __init__(self, tool_names, selector, mcp_client):
        self.tool_names = tool_names
        self.selector = selector
        self.mcp = mcp_client

    def handle_query(self, query: str):
        tool_name = self.selector.select(self.tool_names, query)
        response = self.mcp.call(tool_name, query)

        return {
            "query": query,
            "tool": tool_name,
            "success": response.success,
            "latency": response.latency,
            "result": response.result,
            "error": response.error,
        }
