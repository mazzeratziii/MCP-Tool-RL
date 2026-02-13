"""
Улучшенный семантический поиск с автоматическим расширением
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import re
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from core.tool_registry import Tool

logger = logging.getLogger(__name__)

class SemanticSearcher:
    """Семантический поиск с автоматическим расширением"""

    def __init__(self):
        if not ST_AVAILABLE:
            raise ImportError("Установите sentence-transformers: pip install sentence-transformers")

        self.model = None
        self.tools: List[Tool] = []
        self.tool_embeddings: np.ndarray = None

        # Кэш для эмбеддингов запросов
        self.query_cache = {}

        logger.info("Инициализация семантического поиска с авто-расширением")

    def initialize(self):
        """Инициализирует модель"""
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"Модель загружена")

    def index_tools(self, tools: List[Tool]):
        """Индексирует инструменты с использованием их автоматической семантики"""
        self.tools = tools
        self.initialize()

        logger.info(f"Создание эмбеддингов для {len(tools)} инструментов...")

        # Используем улучшенный search_text с авто-семантикой
        texts = [tool.search_text for tool in tools]

        self.tool_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Эмбеддинги созданы")

    def _expand_query_automatically(self, query: str) -> List[str]:
        """
        Автоматически расширяет запрос на основе паттернов
        Оптимизированная версия - создает меньше эмбеддингов
        """
        expanded = [query]  # Только оригинальный запрос
        query_lower = query.lower()

        # Определяем категорию запроса
        is_currency = any(word in query_lower for word in ['доллар', 'евро', 'рубл', 'валю', 'конверт'])
        is_weather = any(word in query_lower for word in ['погод', 'температур', 'дожд', 'снег', 'ветер'])
        is_flight = any(word in query_lower for word in ['рейс', 'билет', 'самолет', 'авиа', 'полет'])

        # Добавляем только ОДИН расширенный вариант в зависимости от категории
        if is_currency:
            expanded.append("конвертация валют курс обмена")
        elif is_weather:
            expanded.append("прогноз погоды температура осадки")
        elif is_flight:
            expanded.append("поиск авиабилетов рейсы перелеты")

        return list(set(expanded))  # Убираем дубликаты

    def _expand_with_keywords(self, query: str, top_tools: List[Tool]) -> str:
        """
        Расширяет запрос ключевыми словами из найденных инструментов
        """
        if not top_tools:
            return query

        # Собираем ключевые слова из топ-инструментов
        all_keywords = []
        for tool in top_tools[:2]:  # Берем первые 2
            all_keywords.extend(list(tool.keywords)[:5])

        # Выбираем самые частотные
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(5)]

        # Расширяем запрос
        if top_keywords:
            return f"{query} {' '.join(top_keywords)}"

        return query

    def search(self, query: str, top_k: int = 5, auto_expand: bool = True) -> List[Dict[str, Any]]:
        """
        Поиск с автоматическим расширением семантики
        Улучшенная версия с весами
        """
        if not self.tools or self.tool_embeddings is None:
            raise ValueError("Сначала вызовите index_tools()")

        self.initialize()

        # Определяем тип запроса для повышения веса
        query_lower = query.lower()
        boost_category = None

        if any(word in query_lower for word in ['доллар', 'евро', 'рубл', 'валю', 'конверт']):
            boost_category = 'finance'
        elif any(word in query_lower for word in ['погод', 'температур', 'дожд', 'снег']):
            boost_category = 'weather'
        elif any(word in query_lower for word in ['рейс', 'билет', 'самолет', 'авиа']):
            boost_category = 'transportation'

        # Создаём эмбеддинг запроса
        if auto_expand:
            expanded_queries = self._expand_query_automatically(query)
            query_embeddings = []

            for eq in expanded_queries:
                emb = self.model.encode(eq, convert_to_numpy=True)
                query_embeddings.append(emb)

            # Усредняем эмбеддинги
            query_embedding = np.mean(query_embeddings, axis=0)
        else:
            query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Вычисляем сходство
        query_norm = np.linalg.norm(query_embedding)
        tool_norms = np.linalg.norm(self.tool_embeddings, axis=1)

        similarities = np.dot(self.tool_embeddings, query_embedding) / (tool_norms * query_norm)

        # Применяем буст для категорий
        if boost_category:
            for i, tool in enumerate(self.tools):
                if tool.category == boost_category:
                    similarities[i] *= 1.3  # Увеличиваем вес на 30%

        # Топ-K результатов
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            tool = self.tools[idx]

            results.append({
                "tool": tool,
                "similarity": similarity,
                "tool_id": tool.id,
                "name": tool.name,
                "category": tool.category,
                "keywords": list(tool.keywords)[:5]
            })

        return results

    def explain_search(self, query: str) -> Dict[str, Any]:
        """
        Объясняет, как был произведён поиск
        Полезно для отладки
        """
        results = self.search(query, top_k=5)

        # Анализируем расширение запроса
        expanded_queries = self._expand_query_automatically(query)

        explanation = {
            "original_query": query,
            "expanded_queries": expanded_queries[:5],
            "total_variants": len(expanded_queries),
            "results": []
        }

        for r in results:
            tool = r["tool"]
            explanation["results"].append({
                "name": tool.name,
                "category": tool.category,
                "similarity": r["similarity"],
                "matched_keywords": [
                    kw for kw in tool.keywords
                    if any(kw in q.lower() for q in expanded_queries)
                ][:10]
            })

        return explanation

# Глобальный экземпляр
searcher = SemanticSearcher()