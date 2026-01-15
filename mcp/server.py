import time
from mcp.types import MCPRequest, MCPResponse

class MCPServer:
    def __init__(self, registry):
        self.registry = registry

    def handle(self, request: MCPRequest) -> MCPResponse:
        tool = self.registry.get(request.tool_name)

        if tool is None:
            return MCPResponse(
                success=False,
                result=None,
                error="Tool not found",
                latency=0.0
            )

        start = time.time()
        try:
            result = tool.invoke(request.query)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        latency = time.time() - start

        return MCPResponse(
            success=success,
            result=result,
            error=error,
            latency=latency
        )
