import time
from tools.base_tool import BaseTool

class SlowStableTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="slow_stable",
            description="Slow but stable tool"
        )

    def invoke(self, query: str) -> str:
        time.sleep(0.5)
        return f"[SlowStable] Result for '{query}'"
