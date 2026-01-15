import random

class SemanticSelector:
    def select(self, tool_names, query: str):
        return random.choice(tool_names)
