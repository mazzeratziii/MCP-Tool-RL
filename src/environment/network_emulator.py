import random
import time
import numpy as np
from typing import Dict, Any
from src.config import Config


class NetworkEmulator:
    """Эмулятор сетевых условий для MCP серверов"""

    def __init__(self, config: Config):
        self.config = config
        self.server_states = {}  # состояние каждого сервера
        self.latency_history = {}  # история задержек

    def update_network_state(self):
        """Обновление состояния сети (случайные флуктуации)"""
        # Имитируем изменение нагрузки на сеть
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
                'load': random.uniform(0.1, 0.9),
                'last_response': time.time()
            }

        # Случайные отказы
        if random.random() < self.config.network.failure_rate:
            self.server_states[server_name]['available'] = False
        else:
            self.server_states[server_name]['available'] = True

        return self.server_states[server_name]

    def get_current_latency(self, server_name: str, tool_config: Dict) -> float:
        """
        Вычисление текущей задержки для сервера
        Учитывает: базовую задержку инструмента + загрузку сети + jitter
        """
        base = tool_config.get('base_latency', 0.1)

        # Задержка зависит от загрузки сети и jitter
        network_delay = base * self.config.network.congestion_factor

        # Добавляем случайные вариации
        jitter = random.gauss(0, self.config.network.jitter)
        total_latency = max(0.01, network_delay + jitter)

        # Сохраняем в историю
        if server_name not in self.latency_history:
            self.latency_history[server_name] = []
        self.latency_history[server_name].append(total_latency)

        # Оставляем только последние 10 значений
        self.latency_history[server_name] = self.latency_history[server_name][-10:]

        return total_latency

    def get_qos_metrics(self, server_name: str) -> Dict[str, float]:
        """Получение метрик QoS для сервера"""
        history = self.latency_history.get(server_name, [])

        if not history:
            return {
                'avg_latency': 0.1,
                'latency_variance': 0,
                'stability': 1.0
            }

        avg_latency = float(np.mean(history))
        latency_variance = float(np.var(history))

        # Стабильность: 1 - нормализованная вариативность
        stability = 1.0 / (1.0 + latency_variance * 10)

        return {
            'avg_latency': avg_latency,
            'latency_variance': latency_variance,
            'stability': stability
        }