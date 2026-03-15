from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import Config


class ToolRegistry:
    """Реестр инструментов на основе ToolBench"""

    def __init__(self, config: Config):
        self.config = config
        self.tools = config.tools  # инструменты из ToolBench

        # Загружаем модель для эмбеддингов
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Создаем эмбеддинги для описаний инструментов
        self.tool_embeddings = self._create_tool_embeddings()

        # Кэш для семантических оценок
        self.semantic_cache = {}

    def _create_tool_embeddings(self) -> Dict[str, np.ndarray]:
        """Создание эмбеддингов для каждого инструмента"""
        embeddings = {}
        for tool in self.tools:
            # Комбинируем название, категорию и описание
            text = f"{tool['name']} - {tool['category']}: {tool['description']}"

            # Добавляем информацию о параметрах
            if tool['required_parameters']:
                params = ", ".join([p['name'] for p in tool['required_parameters']])
                text += f" Required parameters: {params}"

            embeddings[tool['name']] = self.encoder.encode(text)

        return embeddings

    def get_tools_by_category(self, category: str) -> List[Dict]:
        """Получение инструментов по категории"""
        return [t for t in self.tools if t['category'].lower() == category.lower()]

    def get_tool_by_name(self, name: str) -> Optional[Dict]:
        """Получение инструмента по имени"""
        for tool in self.tools:
            if tool['name'] == name:
                return tool
        return None

    def semantic_similarity(self, query: str, tool_name: str) -> float:
        """Вычисление семантической близости запроса к инструменту"""
        cache_key = f"{query}_{tool_name}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]

        query_embedding = self.encoder.encode(query)
        tool_embedding = self.tool_embeddings[tool_name]

        # Косинусное сходство
        similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
        )

        # Нормализация в [0, 1]
        result = float((similarity + 1) / 2)

        # Кэшируем результат
        self.semantic_cache[cache_key] = result

        return result

    def get_top_k_tools(self, query: str, k: int = 5) -> List[Dict]:
        """Получение top-k семантически близких инструментов"""
        scores = []
        for tool in self.tools:
            score = self.semantic_similarity(query, tool['name'])
            scores.append((score, tool))

        # Сортируем по убыванию и берем top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [tool for score, tool in scores[:k]]

    def format_tool_for_prompt(self, tool: Dict) -> str:
        """Форматирование описания инструмента для промпта"""
        lines = [
            f"Tool: {tool['name']}",
            f"Description: {tool['description']}",
            f"Category: {tool['category']}",
            f"Method: {tool['method']}"
        ]

        if tool['required_parameters']:
            params = ", ".join([p['name'] for p in tool['required_parameters']])
            lines.append(f"Required parameters: {params}")

        return "\n".join(lines)