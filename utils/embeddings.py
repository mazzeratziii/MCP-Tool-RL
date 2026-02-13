"""
Система эмбеддингов для семантического поиска инструментов
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from config.settings import config
from core.hf_loader import ToolSpec

logger = logging.getLogger(__name__)


class ToolEmbedder:
    """
    Генерирует и управляет эмбеддингами для инструментов.
    Оптимизирован для работы с онлайн-загрузчиком.
    """

    def __init__(self):
        if not ST_AVAILABLE:
            raise ImportError(
                "Установите sentence-transformers: pip install sentence-transformers"
            )

        self.config = config.embeddings
        self.model = None
        self._tool_embeddings: Dict[str, np.ndarray] = {}
        self._tool_texts: Dict[str, str] = {}

        logger.info(f"Инициализация эмбеддера с моделью: {self.config.model_name}")

    def initialize(self):
        """Инициализирует модель эмбеддингов"""
        if self.model is None:
            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_dir,
                device=self.config.device
            )
            logger.info(f"Модель эмбеддингов загружена на {self.config.device}")

    def create_tool_embedding(self, tool: ToolSpec) -> np.ndarray:
        """
        Создаёт эмбеддинг для инструмента.

        Args:
            tool: Инструмент для векторизации

        Returns:
            Вектор эмбеддинга
        """
        self.initialize()

        # Создаём текстовое представление инструмента
        tool_text = self._create_tool_text(tool)
        self._tool_texts[tool.id] = tool_text

        # Генерируем эмбеддинг
        embedding = self.model.encode(
            tool_text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        self._tool_embeddings[tool.id] = embedding
        return embedding

    def _create_tool_text(self, tool: ToolSpec) -> str:
        """Создаёт текстовое представление инструмента для эмбеддинга"""
        parts = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
            f"Category: {tool.category}",
            f"API: {tool.api_name}",
        ]

        # Добавляем параметры
        if tool.parameters:
            param_descs = []
            for param in tool.parameters[:5]:  # Ограничиваем количество
                if isinstance(param, dict):
                    param_str = param.get('name', '')
                    if param.get('description'):
                        param_str += f" - {param['description']}"
                    param_descs.append(param_str)

            if param_descs:
                parts.append(f"Parameters: {', '.join(param_descs)}")

        return ". ".join(parts)

    def semantic_search(
            self,
            query: str,
            tools: List[ToolSpec],
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск инструментов по запросу.

        Args:
            query: Поисковый запрос
            tools: Список инструментов для поиска
            top_k: Количество результатов

        Returns:
            Список результатов с оценками
        """
        self.initialize()

        if not tools:
            return []

        # Создаём эмбеддинги для инструментов, если их ещё нет
        tool_embeddings = []
        valid_tools = []

        for tool in tools:
            if tool.id in self._tool_embeddings:
                embedding = self._tool_embeddings[tool.id]
            else:
                embedding = self.create_tool_embedding(tool)

            tool_embeddings.append(embedding)
            valid_tools.append(tool)

        # Эмбеддинг запроса
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Вычисляем косинусное сходство
        tool_embeddings_array = np.array(tool_embeddings)
        similarities = np.dot(tool_embeddings_array, query_embedding) / (
                np.linalg.norm(tool_embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )

        # Сортируем по убыванию сходства
        sorted_indices = np.argsort(similarities)[::-1]

        # Формируем результаты
        results = []
        for idx in sorted_indices[:top_k]:
            similarity = float(similarities[idx])
            if similarity < config.selection.min_similarity_threshold:
                break

            tool = valid_tools[idx]
            results.append({
                "tool": tool,
                "similarity": similarity,
                "tool_id": tool.id,
                "name": tool.name,
                "category": tool.category
            })

        return results

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Кэшированное получение эмбеддинга для текста.

        Args:
            text: Текст для векторизации

        Returns:
            Вектор эмбеддинга
        """
        self.initialize()
        return self.model.encode(text, convert_to_numpy=True)

    def batch_create_embeddings(self, tools: List[ToolSpec]):
        """
        Пакетное создание эмбеддингов для инструментов.

        Args:
            tools: Список инструментов
        """
        self.initialize()

        if not tools:
            return

        # Подготавливаем тексты
        texts = [self._create_tool_text(tool) for tool in tools]
        tool_ids = [tool.id for tool in tools]

        # Пакетное кодирование
        logger.info(f"Пакетное создание эмбеддингов для {len(tools)} инструментов")
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Сохраняем эмбеддинги
        for i, tool_id in enumerate(tool_ids):
            self._tool_embeddings[tool_id] = embeddings[i]
            self._tool_texts[tool_id] = texts[i]

        logger.info(f"Эмбеддинги созданы для {len(tools)} инструментов")

    def clear_cache(self):
        """Очищает кэш эмбеддингов"""
        self._tool_embeddings.clear()
        self._tool_texts.clear()
        self.get_embedding.cache_clear()
        logger.info("Кэш эмбеддингов очищен")


# Глобальный экземпляр эмбеддера
embedder = ToolEmbedder()