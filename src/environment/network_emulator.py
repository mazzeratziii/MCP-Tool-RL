import random
import time
import numpy as np
from typing import Dict, Any
from src.config import Config


class NetworkEmulator:
    def __init__(self, config: Config):
        self.config = config
        self.server_states = {}
        self.latency_history = {}

    def update_network_state(self):
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
        if server_name not in self.server_states:
            self.server_states[server_name] = {
                'available': True,
                'load': random.uniform(0.1, 0.9),
                'last_response': time.time()
            }

        if random.random() < self.config.network.failure_rate:
            self.server_states[server_name]['available'] = False
        else:
            self.server_states[server_name]['available'] = True

        return self.server_states[server_name]

    def get_current_latency(self, server_name: str, tool_config: Dict) -> float:
        base = tool_config.get('base_latency', 0.1)
        network_delay = base * self.config.network.congestion_factor
        jitter = random.gauss(0, self.config.network.jitter)
        total_latency = max(0.01, network_delay + jitter)

        if server_name not in self.latency_history:
            self.latency_history[server_name] = []
        self.latency_history[server_name].append(total_latency)
        self.latency_history[server_name] = self.latency_history[server_name][-10:]

        return total_latency

    def get_qos_metrics(self, server_name: str) -> Dict[str, float]:
        history = self.latency_history.get(server_name, [])

        if not history:
            return {
                'avg_latency': 0.1,
                'latency_variance': 0,
                'stability': 1.0
            }

        avg_latency = float(np.mean(history))
        latency_variance = float(np.var(history))
        stability = 1.0 / (1.0 + latency_variance * 10)

        return {
            'avg_latency': avg_latency,
            'latency_variance': latency_variance,
            'stability': stability
        }