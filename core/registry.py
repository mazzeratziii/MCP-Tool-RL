from typing import Dict, List, Optional
from core.tool import Tool
from core.toolbench_loader import ToolBenchLoader

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tools_by_category: Dict[str, List[Tool]] = {}
        self.loader = ToolBenchLoader()

    def add_tool(self, tool: Tool) -> Tool:
        self.tools[tool.id] = tool
        self.tools_by_category.setdefault(tool.category, []).append(tool)
        return tool

    def load_from_toolbench(self, limit: int = 100, categories: Optional[List[str]] = None) -> List[Tool]:
        tools = self.loader.load_tools(limit=limit, categories=categories)
        for t in tools:
            self.add_tool(t)
        return tools

    def get_all_tools(self) -> List[Tool]:
        return list(self.tools.values())

    def get_tool_by_id(self, tool_id: str) -> Optional[Tool]:
        return self.tools.get(tool_id)

    def get_tools_by_category(self, category: str) -> List[Tool]:
        return self.tools_by_category.get(category, [])

# Глобальный экземпляр
registry = ToolRegistry()