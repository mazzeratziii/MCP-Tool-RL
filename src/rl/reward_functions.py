from typing import List, Dict, Any
from src.config import Config


class NetMCPRewardFunction:
    """
    Функция награды для ToolBrain
    Вычисляет награду на основе траектории взаимодействия
    """

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """
        Вычисление наград для батча траекторий

        Args:
            trajectories: список траекторий, каждая содержит:
                - prompts: исходный запрос
                - responses: ответы модели (вызовы инструментов)
                - tool_calls: совершенные вызовы инструментов
                - metadata: дополнительная информация (латенси, успешность)

        Returns:
            rewards: список наград для каждой траектории
        """
        rewards = []

        for traj in trajectories:
            reward = self._compute_trajectory_reward(traj)
            rewards.append(reward)

        return rewards

    def _compute_trajectory_reward(self, trajectory: Dict[str, Any]) -> float:
        """Вычисление награды для одной траектории"""
        total_reward = 0.0
        steps = len(trajectory.get('tool_calls', []))

        for i, tool_call in enumerate(trajectory.get('tool_calls', [])):
            step_reward = 0.0

            # Базовая награда за успешность
            if tool_call.get('success', False):
                step_reward += self.config.reward.success_reward
            else:
                step_reward += self.config.reward.failure_penalty

            # Штраф за каждый шаг
            step_reward += self.config.reward.step_penalty

            # Штраф за высокую латенси
            latency = tool_call.get('latency', 0)
            if latency > self.config.reward.latency_threshold:
                step_reward -= 0.2

            # Семантический бонус
            semantic_score = tool_call.get('semantic_score', 0)
            if semantic_score > 0.7:
                step_reward += self.config.reward.semantic_bonus

            total_reward += step_reward

        # Бонус за решение задачи за малое количество шагов
        if steps < self.config.rl.max_steps:
            total_reward += 0.5

        return total_reward


class GRPOToolReward:
    """
    Специализированная функция награды для GRPO алгоритма
    Следует рекомендациям ToolBrain для multi-turn RL
    """

    def __init__(self, config: Config):
        self.config = config

    def compute_outcome_reward(self, success: bool, steps: int) -> float:
        """
        Главная награда за исход задачи
        Доминирующая часть награды
        """
        if success:
            # Базовая награда за успех + бонус за эффективность
            base_reward = self.config.reward.success_reward
            efficiency_bonus = max(0, (self.config.rl.max_steps - steps) * 0.1)
            return base_reward + efficiency_bonus
        else:
            return self.config.reward.failure_penalty

    def compute_step_penalty(self, step_num: int) -> float:
        """
        Небольшой отрицательный штраф за каждый шаг
        Мотивирует агента решать задачу быстро
        """
        return self.config.reward.step_penalty * step_num

    def compute_validity_reward(self, tool_call_valid: bool) -> float:
        """
        Штраф за невалидный вызов инструмента
        """
        if not tool_call_valid:
            return self.config.reward.invalid_call_penalty
        return 0.0

    def compute_semantic_bonus(self, semantic_score: float) -> float:
        """
        Бонус за семантическую близость
        Помогает на ранних этапах обучения
        """
        if semantic_score > 0.8:
            return 0.3
        elif semantic_score > 0.6:
            return 0.1
        return 0.0