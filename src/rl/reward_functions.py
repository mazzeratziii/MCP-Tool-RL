# Функции награды
from typing import List, Dict, Any
from src.config import Config


class GRPOToolReward:
    def __init__(self, config: Config):
        """Initialize the object."""
        self.config = config

    def compute_outcome_reward(self, success: bool, steps: int, is_relevant: bool = True,
                               latency: float = 0.0, semantic_score: float = 0.0,
                               available: bool = True, stability: float = 1.0,
                               avg_latency: float = None, availability_ratio: float = None) -> float:
        """
        Улучшенная функция награды с учётом:
        - успеха и релевантности
        - скорости решения (кол-во шагов)
        - задержки сети (абсолютной и относительной)
        - семантической близости
        - доступности и стабильности
        """
        reward = 0.0

        # Базовая награда за успех
        if success and is_relevant:
            base_reward = 3.0
            efficiency_bonus = max(0.0, 1.5 - (steps - 1) * 0.6)  # сильный бонус за быстрый успех
            reward = base_reward + efficiency_bonus

            # Абсолютный штраф за высокую задержку
            if latency > 0.4:
                reward -= (latency - 0.4) * 1.2

            # НОВОЕ: Относительный бонус за выбор быстрого инструмента
            if avg_latency is not None and avg_latency > 0:
                relative_speed = (avg_latency - latency) / avg_latency
                if relative_speed > 0.2:  # выбран инструмент на 20%+ быстрее среднего
                    reward += 0.8 * relative_speed

            # Семантический бонус/штраф
            if semantic_score > 0.75:
                reward += 0.4
            elif semantic_score > 0.6:
                reward += 0.15
            else:
                # Штраф за низкую семантическую релевантность
                reward -= 0.5 * (0.6 - semantic_score)

            # НОВОЕ: Бонус за стабильность
            if stability < 0.8:
                reward -= 0.3 * (0.8 - stability)

        elif success and not is_relevant:
            reward = 0.4 - steps * 0.25

        else:  # неудачное выполнение
            reward = -1.8 - steps * 0.2  # сильный штраф за провал

        # НОВОЕ: Штраф за выбор недоступного инструмента
        if not available:
            reward -= 1.5

        # НОВОЕ: Бонус за выбор доступного при проблемах с сетью
        if availability_ratio is not None and availability_ratio < 0.7 and available:
            reward += 0.8

        return reward

    def compute_step_penalty(self, step_num: int) -> float:
        """Compute compute step penalty."""
        return -0.05 * step_num

    def compute_validity_reward(self, tool_call_valid: bool) -> float:
        """Compute compute validity reward."""
        return self.config.reward.invalid_call_penalty if not tool_call_valid else 0.0


class NetMCPRewardFunction:
    """Оставил для совместимости, если где-то используется"""

    def __init__(self, config: Config):
        """Initialize the object."""
        self.config = config

    def __call__(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """Call call."""
        return [0.0] * len(trajectories)