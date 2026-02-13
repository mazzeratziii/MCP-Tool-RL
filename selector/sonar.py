"""
Улучшенный алгоритм SONAR для работы с онлайн-инструментами.
Интегрирует семантический поиск и сетевые метрики.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

from config.settings import config
from core.hf_loader import ToolSpec, loader
from utils.embeddings import embedder

logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Метрики сети для инструмента"""
    tool_id: str
    latency_history: List[float]  # История задержек в мс
    success_history: List[bool]   # История успешных вызовов
    last_call: Optional[datetime]
    call_count: int = 0
    success_count: int = 0

    def __post_init__(self):
        if self.latency_history is None:
            self.latency_history = []
        if self.success_history is None:
            self.success_history = []

    @property
    def avg_latency(self) -> float:
        """Средняя задержка"""
        if not self.latency_history:
            return 100.0  # Значение по умолчанию
        return np.mean(self.latency_history[-100:])  # Последние 100 вызовов

    @property
    def success_rate(self) -> float:
        """Процент успешных вызовов"""
        if self.call_count == 0:
            return 0.8  # Предполагаемая успешность для новых инструментов
        return self.success_count / self.call_count

    @property
    def reliability_score(self) -> float:
        """Общая оценка надёжности (0-1)"""
        # Нормализуем задержку: меньше = лучше
        latency_score = max(0, 1 - (self.avg_latency / 5000))  # 5 секунд макс

        return (latency_score * 0.4) + (self.success_rate * 0.6)

    @property
    def is_recently_used(self) -> bool:
        """Использовался ли инструмент недавно"""
        if not self.last_call:
            return False
        return (datetime.now() - self.last_call) < timedelta(hours=1)

    def update(self, latency: float, success: bool):
        """Обновляет метрики после вызова"""
        self.latency_history.append(latency)
        self.success_history.append(success)
        self.call_count += 1
        self.last_call = datetime.now()

        if success:
            self.success_count += 1

        # Ограничиваем историю
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-500:]
            self.success_history = self.success_history[-500:]

class EnhancedSONAR:
    """
    Улучшенный алгоритм SONAR для онлайн-инструментов.
    Сочетает семантику, сетевые метрики и разнообразие.
    """

    def __init__(self):
        self.config = config.selection
        self.network_metrics: Dict[str, NetworkMetrics] = {}
        self.selection_history: List[Dict[str, Any]] = []

        # Загружаем инструменты при инициализации
        self.tools = loader.load_tools(limit=config.max_tools_in_memory)
        logger.info(f"SONAR инициализирован с {len(self.tools)} инструментами")

    async def select_tool(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_semantic: bool = True,
        use_network: bool = True
    ) -> Tuple[Optional[ToolSpec], Dict[str, Any]]:
        """
        Основной метод выбора инструмента.

        Args:
            query: Пользовательский запрос
            context: Дополнительный контекст
            use_semantic: Использовать семантический поиск
            use_network: Использовать сетевые метрики

        Returns:
            Кортеж (инструмент, информация о выборе)
        """
        selection_id = f"sel_{len(self.selection_history) + 1}"
        logger.info(f"[{selection_id}] Выбор инструмента для: '{query[:50]}...'")

        start_time = datetime.now()

        try:
            # Шаг 1: Фильтрация по контексту (если есть)
            candidate_tools = self._filter_by_context(context)

            # Шаг 2: Семантический поиск
            if use_semantic and candidate_tools:
                semantic_results = embedder.semantic_search(
                    query,
                    candidate_tools,
                    top_k=self.config.top_k_candidates
                )
                candidate_tools = [r["tool"] for r in semantic_results]
                semantic_scores = {r["tool_id"]: r["similarity"] for r in semantic_results}
            else:
                semantic_scores = {}

            # Шаг 3: Оценка кандидатов
            if not candidate_tools:
                logger.warning(f"[{selection_id}] Нет подходящих инструментов")
                return None, {"error": "No suitable tools found"}

            scored_tools = []
            for tool in candidate_tools[:self.config.top_k_candidates]:
                score = self._compute_tool_score(
                    tool,
                    query,
                    semantic_scores.get(tool.id, 0.5),
                    use_network
                )
                scored_tools.append((tool, score))

            # Шаг 4: Выбор лучшего инструмента
            scored_tools.sort(key=lambda x: x[1], reverse=True)
            selected_tool, final_score = scored_tools[0]

            # Шаг 5: Подготовка альтернатив
            alternatives = []
            for tool, score in scored_tools[1:4]:  # Топ-3 альтернативы
                alternatives.append({
                    "id": tool.id,
                    "name": tool.name,
                    "score": score,
                    "category": tool.category
                })

            # Шаг 6: Формирование результата
            selection_info = {
                "selection_id": selection_id,
                "selected_tool": {
                    "id": selected_tool.id,
                    "name": selected_tool.name,
                    "category": selected_tool.category,
                    "score": final_score
                },
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "candidates_evaluated": len(scored_tools),
                "alternatives": alternatives,
                "score_components": self._get_score_breakdown(
                    selected_tool, query, semantic_scores.get(selected_tool.id, 0.5), use_network
                )
            }

            # Сохраняем в историю
            self.selection_history.append(selection_info)

            logger.info(
                f"[{selection_id}] Выбран: {selected_tool.name} "
                f"(категория: {selected_tool.category}, оценка: {final_score:.3f})"
            )

            return selected_tool, selection_info

        except Exception as e:
            logger.error(f"[{selection_id}] Ошибка выбора инструмента: {e}")
            return None, {"error": str(e), "selection_id": selection_id}

    def _filter_by_context(self, context: Optional[Dict[str, Any]]) -> List[ToolSpec]:
        """Фильтрует инструменты по контексту"""
        if not context:
            return list(self.tools.values())

        filtered = []
        category = context.get("category")
        required_params = context.get("required_params", [])

        for tool in self.tools.values():
            # Фильтрация по категории
            if category and tool.category.lower() != category.lower():
                continue

            # Проверка поддержки требуемых параметров
            if required_params:
                tool_params = {p.get("name", "").lower() for p in tool.parameters}
                required_lower = {p.lower() for p in required_params}
                if not required_lower.issubset(tool_params):
                    continue

            filtered.append(tool)

        return filtered

    def _compute_tool_score(
        self,
        tool: ToolSpec,
        query: str,
        semantic_score: float,
        use_network: bool
    ) -> float:
        """
        Вычисляет общую оценку инструмента.

        Args:
            tool: Инструмент для оценки
            query: Исходный запрос
            semantic_score: Оценка семантического сходства
            use_network: Учитывать сеть

        Returns:
            Общая оценка (0-1)
        """
        scores = []
        weights = []

        # 1. Семантическая составляющая
        scores.append(semantic_score)
        weights.append(self.config.semantic_weight)

        # 2. Сетевая составляющая
        if use_network:
            network_score = self._get_network_score(tool.id)
            scores.append(network_score)
            weights.append(self.config.network_weight)

        # 3. Составляющая разнообразия (избегаем частого использования одного инструмента)
        diversity_score = self._get_diversity_score(tool.id)
        scores.append(diversity_score)
        weights.append(self.config.diversity_weight)

        # 4. Качество описания инструмента
        quality_score = self._evaluate_tool_quality(tool)
        scores.append(quality_score)
        weights.append(0.1)  # 10% веса для качества

        # Нормализуем веса
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Вычисляем взвешенную сумму
        final_score = sum(s * w for s, w in zip(scores, weights))

        return min(1.0, max(0.0, final_score))

    def _get_network_score(self, tool_id: str) -> float:
        """Получает сетевую оценку инструмента"""
        if tool_id in self.network_metrics:
            return self.network_metrics[tool_id].reliability_score

        # Значение по умолчанию для новых инструментов
        return 0.7

    def _get_diversity_score(self, tool_id: str) -> float:
        """Оценка разнообразия (поощряет использование разных инструментов)"""
        # Считаем, как часто инструмент использовался недавно
        recent_uses = 0
        for selection in self.selection_history[-20:]:  # Последние 20 выборов
            if selection.get("selected_tool", {}).get("id") == tool_id:
                recent_uses += 1

        # Чем реже использовался, тем выше оценка
        return max(0.1, 1.0 - (recent_uses / 20))

    def _evaluate_tool_quality(self, tool: ToolSpec) -> float:
        """Оценивает качество описания инструмента"""
        score = 0.5  # Базовый балл

        # Наличие описания
        if tool.description and len(tool.description) > 20:
            score += 0.2

        # Наличие примеров
        if tool.examples and len(tool.examples) > 0:
            score += 0.15

        # Полнота параметров
        if tool.parameters and len(tool.parameters) > 0:
            score += 0.1

            # Проверяем, есть ли описания параметров
            described_params = sum(
                1 for p in tool.parameters
                if isinstance(p, dict) and p.get('description')
            )
            if described_params > 0:
                score += 0.05

        return min(1.0, score)

    def _get_score_breakdown(
        self,
        tool: ToolSpec,
        query: str,
        semantic_score: float,
        use_network: bool
    ) -> Dict[str, float]:
        """Возвращает детализацию оценки"""
        breakdown = {
            "semantic": semantic_score,
            "diversity": self._get_diversity_score(tool.id),
            "quality": self._evaluate_tool_quality(tool)
        }

        if use_network:
            breakdown["network"] = self._get_network_score(tool.id)

        return breakdown

    def update_network_metrics(self, tool_id: str, latency: float, success: bool):
        """
        Обновляет сетевые метрики после вызова инструмента.

        Args:
            tool_id: ID инструмента
            latency: Задержка в мс
            success: Успешен ли вызов
        """
        if tool_id not in self.network_metrics:
            self.network_metrics[tool_id] = NetworkMetrics(
                tool_id=tool_id,
                latency_history=[],
                success_history=[],
                last_call=None
            )

        self.network_metrics[tool_id].update(latency, success)
        logger.debug(f"Обновлены метрики для {tool_id}: latency={latency:.1f}ms, success={success}")

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику селектора"""
        return {
            "total_tools": len(self.tools),
            "network_metrics_tracked": len(self.network_metrics),
            "selection_history_count": len(self.selection_history),
            "avg_selection_time_ms": (
                np.mean([s.get("processing_time_ms", 0) for s in self.selection_history[-100:]])
                if self.selection_history else 0
            ),
            "tool_usage_distribution": self._get_usage_distribution()
        }

    def _get_usage_distribution(self) -> Dict[str, int]:
        """Распределение использования инструментов"""
        usage = {}
        for selection in self.selection_history[-100:]:  # Последние 100 выборов
            tool_id = selection.get("selected_tool", {}).get("id")
            if tool_id:
                usage[tool_id] = usage.get(tool_id, 0) + 1

        return usage

# Глобальный экземпляр селектора
sonar = EnhancedSONAR()