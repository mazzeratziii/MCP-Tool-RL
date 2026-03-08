from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from src.data.toolbench_loader import ToolBenchLoader


@dataclass
class NetworkConfig:
    """Конфигурация эмуляции сети"""
    base_latency_range: Tuple[float, float] = (0.05, 1.0)  # диапазон базовых задержек
    jitter_range: Tuple[float, float] = (0.01, 0.2)  # диапазон jitter
    failure_rate_range: Tuple[float, float] = (0.01, 0.2)  # диапазон вероятности отказов
    congestion_factor_range: Tuple[float, float] = (0.5, 2.0)  # диапазон загрузки сети

    # Добавляем отдельные значения для текущего состояния
    base_latency: float = 0.1
    jitter: float = 0.05
    failure_rate: float = 0.1  # Добавляем этот атрибут
    congestion_factor: float = 1.0


@dataclass
class ToolBenchConfig:
    """Конфигурация загрузки ToolBench"""
    split: str = "train"
    sample_size: Optional[int] = 100
    num_tools: int = 50


@dataclass
class RLConfig:
    """Конфигурация RL"""
    algorithm: str = "grpo"
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_steps: int = 3
    kl_coef: float = 0.1
    temperature: float = 0.7


@dataclass
class RewardConfig:
    """Конфигурация системы наград"""
    success_reward: float = 1.0
    failure_penalty: float = -1.0
    step_penalty: float = -0.1
    invalid_call_penalty: float = -0.5
    semantic_bonus: float = 0.2
    latency_threshold: float = 0.5
    wrong_tool_penalty: float = -0.3
    extra_step_penalty: float = -0.15


class Config:
    """Главный конфигурационный класс"""

    def __init__(self):
        self.network = NetworkConfig()
        self.toolbench = ToolBenchConfig()
        self.rl = RLConfig()
        self.reward = RewardConfig()

        # Модель для обучения
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        # Загружаем данные из ToolBench
        print("Инициализация ToolBench...")
        self.loader = ToolBenchLoader(
            split=self.toolbench.split,
            sample_size=self.toolbench.sample_size
        )

        # Берем инструменты для обучения
        self.tools = self.loader.sample_tools(self.toolbench.num_tools)
        print(f"Загружено {len(self.tools)} инструментов для обучения")

        # Берем промпты для обучения
        self.prompts = self.loader.get_training_prompts()