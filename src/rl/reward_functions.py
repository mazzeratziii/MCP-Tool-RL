from typing import List, Dict, Any
from src.config import Config


class NetMCPRewardFunction:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        rewards = []
        for traj in trajectories:
            reward = self._compute_trajectory_reward(traj)
            rewards.append(reward)
        return rewards

    def _compute_trajectory_reward(self, trajectory: Dict[str, Any]) -> float:
        total_reward = 0.0
        steps = len(trajectory.get('tool_calls', []))

        for i, tool_call in enumerate(trajectory.get('tool_calls', [])):
            step_reward = 0.0

            if tool_call.get('success', False):
                step_reward += self.config.reward.success_reward
            else:
                step_reward += self.config.reward.failure_penalty

            step_reward += self.config.reward.step_penalty

            latency = tool_call.get('latency', 0)
            if latency > self.config.reward.latency_threshold:
                step_reward -= 0.2

            semantic_score = tool_call.get('semantic_score', 0)
            if semantic_score > 0.7:
                step_reward += self.config.reward.semantic_bonus

            total_reward += step_reward

        if steps < self.config.rl.max_steps:
            total_reward += 0.5

        return total_reward


class GRPOToolReward:
    def __init__(self, config: Config):
        self.config = config

    def compute_outcome_reward(self, success: bool, steps: int) -> float:
        if success:
            base_reward = self.config.reward.success_reward
            efficiency_bonus = max(0, (self.config.rl.max_steps - steps) * 0.1)
            return base_reward + efficiency_bonus
        else:
            return self.config.reward.failure_penalty

    def compute_step_penalty(self, step_num: int) -> float:
        return self.config.reward.step_penalty * step_num

    def compute_validity_reward(self, tool_call_valid: bool) -> float:
        if not tool_call_valid:
            return self.config.reward.invalid_call_penalty
        return 0.0

    def compute_semantic_bonus(self, semantic_score: float) -> float:
        if semantic_score > 0.8:
            return 0.3
        elif semantic_score > 0.6:
            return 0.1
        return 0.0