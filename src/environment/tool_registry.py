from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import Config


class ToolRegistry:
    def __init__(self, config: Config):
        self.config = config

        # Ensure tools are loaded
        if not config.tools:
            print("Tools not loaded yet, loading data...")
            config.load_data()

        self.tools = config.tools

        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully")

        self.tool_embeddings = self._create_tool_embeddings()
        self.semantic_cache = {}

    def _create_tool_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}
        for tool in self.tools:
            text = f"{tool['name']} - {tool.get('category', 'general')}: {tool.get('description', '')}"
            if tool.get('required_parameters'):
                params = ", ".join([p['name'] for p in tool['required_parameters']])
                text += f" Required parameters: {params}"
            embeddings[tool['name']] = self.encoder.encode(text)
        return embeddings

    def get_tools_by_category(self, category: str) -> List[Dict]:
        return [t for t in self.tools if t.get('category', '').lower() == category.lower()]

    def get_tool_by_name(self, name: str) -> Optional[Dict]:
        for tool in self.tools:
            if tool['name'] == name:
                return tool
        return None

    def semantic_similarity(self, query: str, tool_name: str) -> float:
        cache_key = f"{query}_{tool_name}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]

        query_embedding = self.encoder.encode(query)
        tool_embedding = self.tool_embeddings[tool_name]

        similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
        )

        result = float((similarity + 1) / 2)
        self.semantic_cache[cache_key] = result
        return result

    def get_top_k_tools(self, query: str, k: int = 5) -> List[Dict]:
        scores = []
        for tool in self.tools:
            score = self.semantic_similarity(query, tool['name'])
            scores.append((score, tool))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [tool for score, tool in scores[:k]]

    def format_tool_for_prompt(self, tool: Dict) -> str:
        lines = [
            f"Tool: {tool['name']}",
            f"Description: {tool.get('description', 'No description')}",
            f"Category: {tool.get('category', 'general')}",
            f"Method: {tool.get('method', 'GET')}"
        ]

        if tool.get('required_parameters'):
            params = ", ".join([p['name'] for p in tool['required_parameters']])
            lines.append(f"Required parameters: {params}")

        return "\n".join(lines)