import random
import time
from tools.base_tool import BaseTool

class MediumTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="medium",
            description="Medium speed, medium stability"
        )

    def invoke(self, query: str) -> str:
        time.sleep(0.25)
        if random.random() < 0.1:
            raise RuntimeError("Medium tool failed")
        return f"[Medium] Result for '{query}'"
