# src/rl/reward_functions.py
from typing import List, Dict, Any
from src.config import Config


class GRPOToolReward:
    def __init__(self, config: Config):
        self.config = config

    def compute_outcome_reward(self, success: bool, steps: int, is_relevant: bool = True,
                               latency: float = 0.0, semantic_score: float = 0.0) -> float:
        """
        Улучшенная функция награды с учётом:
        - успеха и релевантности
        - скорости решения (кол-во шагов)
        - задержки сети
        - семантической близости
        """
        if success and is_relevant:
            base_reward = 3.0
            efficiency_bonus = max(0.0, 1.5 - (steps - 1) * 0.6)  # сильный бонус за быстрый успех
            latency_penalty = max(0.0, (latency - 0.4) * 1.2)
            semantic_bonus = 0.4 if semantic_score > 0.75 else 0.15 if semantic_score > 0.6 else 0.0

            return base_reward + efficiency_bonus - latency_penalty + semantic_bonus

        elif success and not is_relevant:
            return 0.4 - steps * 0.25

        else:  # failure
            return -1.8 - steps * 0.2  # сильный штраф за провал

    def compute_step_penalty(self, step_num: int) -> float:
        return -0.05 * step_num

    def compute_validity_reward(self, tool_call_valid: bool) -> float:
        return self.config.reward.invalid_call_penalty if not tool_call_valid else 0.0


class NetMCPRewardFunction:
    """Оставил для совместимости, если где-то используется"""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        return [0.0] * len(trajectories)