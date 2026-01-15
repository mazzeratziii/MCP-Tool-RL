from agent.agent import Agent
from selector.semantic_selector import SemanticSelector

from tools.tool_fast_unstable import FastUnstableTool
from tools.tool_slow_stable import SlowStableTool
from tools.tool_medium import MediumTool

def main():
    tools = [
        FastUnstableTool(),
        SlowStableTool(),
        MediumTool(),
    ]

    selector = SemanticSelector()
    agent = Agent(tools, selector)

    queries = [
        "search weather",
        "get news",
        "find documentation",
        "lookup api",
        "check status",
    ]

    for q in queries:
        result = agent.handle_query(q)
        print(result)

if __name__ == "__main__":
    main()
