import random
import time
from tools.base_tool import BaseTool
class FastUnstableTool(BaseTool):
    def __init__(self):
        super().__init__(
            name = "fast_unstable",
            description="Быстрый, но не стабильный инструмент"
        )
    def init(self, query: str) ->str:
        time.sleep(0.1)
        if random.random() < 0.1:
            raise RuntimeError("Быстрый инструмент упал")
        return f"[FastUnstable] Result for '{query}'"