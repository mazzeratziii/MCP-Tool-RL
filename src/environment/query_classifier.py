# Классификатор запросов
"""
Классификатор запросов для автоматического определения fallback инструментов
"""
import re
from typing import Optional, Tuple


class QueryClassifier:
    """
    Определяет тип запроса и подходящий fallback инструмент
    """

    def __init__(self):
        # Паттерны для математических выражений
        """?????????????? ?????? ? ????????? ??????????? ???????????."""
        self.math_patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # 2 + 2, 10 * 5
            r'calculate|compute|solve|evaluate',
            r'what is \d+',
            r'how much is',
            r'\d+\s*(plus|minus|times|divided by)\s*\d+',
        ]

        # Паттерны для простых вопросов (не требуют API)
        self.simple_patterns = [
            r'^(hi|hello|hey|thanks|thank you|bye|goodbye)',
            r'^(what|who|when|where|why|how)\s+(is|are|was|were)\s+(a|an|the)?\s*\w+\s*$',
            r'(explain|tell me about|describe|define)\s+\w+\s*$',
            r'(good morning|good evening|good night)',
        ]

        # Паттерны для запросов, требующих внешние данные
        self.external_data_patterns = [
            r'(weather|temperature|forecast)',
            r'(stock|price|market)',
            r'(news|article|headline)',
            r'(translate|translation)',
            r'(search|find|lookup)',
            r'(api|endpoint|service)',
        ]

    def classify(self, query: str) -> Tuple[Optional[str], float]:
        """
        Классифицирует запрос и возвращает (fallback_tool, confidence)

        Возвращает:
            (tool_name, confidence) или (None, 0.0) если fallback не нужен
        """
        query_lower = query.lower().strip()

        # Проверка на математику
        if self._is_math_query(query_lower):
            return ("Calculator.Evaluate", 0.95)

        # Проверка на простой вопрос
        if self._is_simple_query(query_lower):
            # Но если запрос требует внешних данных, не используем NoToolNeeded
            if self._requires_external_data(query_lower):
                return (None, 0.0)
            return ("General.NoToolNeeded", 0.85)

        return (None, 0.0)

    def _is_math_query(self, query: str) -> bool:
        """Проверяет, является ли запрос математическим"""
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in self.math_patterns)

    def _is_simple_query(self, query: str) -> bool:
        """Проверяет, является ли запрос простым (не требует API)"""
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in self.simple_patterns)

    def _requires_external_data(self, query: str) -> bool:
        """Проверяет, требует ли запрос внешних данных"""
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in self.external_data_patterns)

    def should_add_fallback_to_candidates(self, query: str, candidates: list) -> Optional[dict]:
        """
        Определяет, нужно ли добавить fallback инструмент к кандидатам

        Аргументы:
            query: запрос пользователя
            candidates: список инструментов-кандидатов

        Возвращает:
            fallback инструмент или None
        """
        fallback_tool, confidence = self.classify(query)

        if fallback_tool is None:
            return None

        # Проверяем, есть ли уже подходящий инструмент в кандидатах
        has_math_tool = any('math' in c.get('category', '').lower() or
                           'calculator' in c.get('name', '').lower()
                           for c in candidates)

        if fallback_tool == "Calculator.Evaluate" and has_math_tool:
            return None  # Уже есть математический инструмент

        # Возвращаем fallback инструмент с высоким приоритетом
        return {
            "name": fallback_tool,
            "confidence": confidence,
            "is_fallback": True
        }
