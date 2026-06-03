# Эмулятор сети
import random
import time
import numpy as np
from typing import Dict, Any, Optional
from enum import Enum
from src.config import Config


class NetworkMode(Enum):
    """Режимы работы симуляции сети"""
    DETERMINISTIC = "deterministic"  # Фиксированные значения (для обучения)
    CONTROLLED = "controlled"  # Управляемые вариации (для тестирования)
    STOCHASTIC = "stochastic"  # Случайные вариации (для стресс-тестов)


class NetworkEmulator:
    """Эмулятор сетевых условий с управляемыми параметрами"""

    def __init__(self, config: Config, mode: NetworkMode = NetworkMode.DETERMINISTIC):
        """Initialize the object."""
        self.config = config
        self.mode = mode
        self.server_states = {}
        self.latency_history = {}

        # Базовые фиксированные параметры (для режима DETERMINISTIC)
        self.base_latency = 0.15  # 150ms базовая задержка
        self.congestion = 1.0  # Нормальная загрузка
        self.packet_loss = 0.02  # 2% потерь пакетов
        self.jitter = 0.01  # 10ms джиттера

        # Управляемые параметры (можно менять во время работы)
        self.active_servers = {}  # Состояние серверов

        # НОВОЕ: Детерминированные сценарии для разнообразия
        self.scenarios = {
            "normal": {"base_latency": 0.15, "availability": 0.98, "congestion": 1.0},
            "peak_hours": {"base_latency": 0.35, "availability": 0.95, "congestion": 1.8},
            "network_issues": {"base_latency": 0.55, "availability": 0.70, "congestion": 2.2},
            "optimal": {"base_latency": 0.08, "availability": 0.99, "congestion": 0.6},
        }
        self.current_scenario = "normal"
        self._scenario_step = 0

    def set_mode(self, mode: NetworkMode):
        """Переключение режима работы"""
        self.mode = mode
        print(f"Network mode changed to: {mode.value}")

    def set_custom_params(self, base_latency: float = None, congestion: float = None,
                          packet_loss: float = None, jitter: float = None):
        """Установка пользовательских параметров (для режима CONTROLLED)"""
        if base_latency is not None:
            self.base_latency = base_latency
        if congestion is not None:
            self.congestion = congestion
        if packet_loss is not None:
            self.packet_loss = packet_loss
        if jitter is not None:
            self.jitter = jitter
        print(f"Custom params set: latency={self.base_latency}, congestion={self.congestion}, "
              f"loss={self.packet_loss}, jitter={self.jitter}")

    def reset_to_defaults(self):
        """Сброс к значениям по умолчанию"""
        self.base_latency = 0.15
        self.congestion = 1.0
        self.packet_loss = 0.02
        self.jitter = 0.01
        print("Network params reset to defaults")

    def set_scenario(self, scenario_name: str):
        """НОВОЕ: Установка сетевого сценария"""
        if scenario_name in self.scenarios:
            self.current_scenario = scenario_name
            scenario = self.scenarios[scenario_name]
            self.base_latency = scenario["base_latency"]
            self.congestion = scenario["congestion"]
            print(f"Network scenario set to: {scenario_name}")
        else:
            print(f"Unknown scenario: {scenario_name}")

    def rotate_scenario(self):
        """НОВОЕ: Циклическая смена сценариев для разнообразия"""
        scenario_names = list(self.scenarios.keys())
        current_idx = scenario_names.index(self.current_scenario)
        next_idx = (current_idx + 1) % len(scenario_names)
        self.set_scenario(scenario_names[next_idx])

    def update_network_state(self):
        """Обновление состояния сети в зависимости от режима"""
        if self.mode == NetworkMode.DETERMINISTIC:
            # Фиксированные значения - никаких изменений
            pass
        elif self.mode == NetworkMode.CONTROLLED:
            # Медленные, предсказуемые изменения
            self.congestion = max(0.5, min(1.5, self.congestion + random.uniform(-0.05, 0.05)))
            self.jitter = max(0.005, min(0.05, self.jitter + random.uniform(-0.003, 0.003)))
        elif self.mode == NetworkMode.STOCHASTIC:
            # Полностью случайные (исходное поведение)
            self.config.network.congestion_factor = random.uniform(
                self.config.network.congestion_factor_range[0],
                self.config.network.congestion_factor_range[1]
            )
            self.config.network.jitter = random.uniform(
                self.config.network.jitter_range[0],
                self.config.network.jitter_range[1]
            )
            self.config.network.failure_rate = random.uniform(
                self.config.network.failure_rate_range[0],
                self.config.network.failure_rate_range[1]
            )

    def get_server_state(self, server_name: str) -> Dict[str, Any]:
        """Получение текущего состояния сервера"""
        if server_name not in self.server_states:
            self.server_states[server_name] = {
                'available': True,
                'load': 0.5,
                'last_response': time.time(),
                'failure_count': 0,
                'success_count': 0,
                'avg_latency': self.base_latency
            }

        if self.mode == NetworkMode.DETERMINISTIC:
            # УЛУЧШЕНО: Используем сценарий для определения доступности
            scenario = self.scenarios[self.current_scenario]
            availability_threshold = scenario["availability"]

            # Детерминированная доступность на основе хеша имени сервера
            server_hash = hash(server_name) % 100
            self.server_states[server_name]['available'] = (server_hash / 100.0) < availability_threshold
            self.server_states[server_name]['load'] = 0.5

        elif self.mode == NetworkMode.CONTROLLED:
            # Управляемая доступность с памятью
            if server_name not in self.active_servers:
                self.active_servers[server_name] = True

            # Редкие, предсказуемые отказы (0.1% вероятность)
            if random.random() < 0.001:
                self.active_servers[server_name] = False
                self.server_states[server_name]['available'] = False
            else:
                self.active_servers[server_name] = True
                self.server_states[server_name]['available'] = True

        else:  # случайный режим
            # Случайные отказы (исходное поведение)
            if random.random() < self.config.network.failure_rate:
                self.server_states[server_name]['available'] = False
            else:
                self.server_states[server_name]['available'] = True

        return self.server_states[server_name]

    def get_current_latency(self, server_name: str, tool_config: Dict) -> float:
        """
        Вычисление текущей задержки для сервера
        """
        # Базовые значения в зависимости от режима
        if self.mode == NetworkMode.DETERMINISTIC:
            # УЛУЧШЕНО: Используем сценарий + детерминированное распределение
            scenario = self.scenarios[self.current_scenario]
            scenario_base = scenario["base_latency"]

            if server_name not in self.latency_history:
                self.latency_history[server_name] = []

            # Разные серверы имеют разную задержку относительно базовой сценария
            # Используем хеш для детерминированного распределения: ±30% от базовой
            server_hash = hash(server_name) % 100
            variation = (server_hash / 100.0 - 0.5) * 0.6  # от -0.3 до +0.3
            total_latency = scenario_base * (1.0 + variation)
            total_latency = max(0.05, total_latency)  # минимум 50ms

        elif self.mode == NetworkMode.CONTROLLED:
            # Предсказуемая задержка с историей
            base = tool_config.get('base_latency', self.base_latency)
            network_delay = base * self.congestion
            total_latency = network_delay

        else:  # случайный режим
            # Полностью случайная (исходное поведение)
            base = tool_config.get('base_latency', 0.1)
            network_delay = base * self.config.network.congestion_factor
            jitter = random.gauss(0, self.config.network.jitter)
            total_latency = max(0.01, network_delay + jitter)

        # Сохраняем историю
        if server_name not in self.latency_history:
            self.latency_history[server_name] = []
        self.latency_history[server_name].append(total_latency)
        # Ограничиваем историю 50 значениями
        self.latency_history[server_name] = self.latency_history[server_name][-50:]

        return total_latency

    def get_qos_metrics(self, server_name: str) -> Dict[str, float]:
        """Получение метрик QoS"""
        history = self.latency_history.get(server_name, [])

        if not history:
            return {
                'avg_latency': 0.15,
                'latency_variance': 0,
                'jitter': 0.0,
                'stability': 1.0
            }

        avg_latency = float(np.mean(history))
        latency_variance = float(np.var(history))
        jitter = float(np.std(history))
        stability = 1.0 / (1.0 + latency_variance * 10)

        return {
            'avg_latency': avg_latency,
            'latency_variance': latency_variance,
            'jitter': jitter,
            'stability': stability
        }

    def get_network_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сети"""
        return {
            'mode': self.mode.value,
            'base_latency': self.base_latency,
            'congestion': self.congestion,
            'packet_loss': self.packet_loss,
            'jitter': self.jitter,
            'active_servers': len([s for s in self.server_states.values() if s['available']]),
            'total_servers': len(self.server_states)
        }
