import time
import random
from typing import Dict, Any, Tuple, Optional
from src.config import Config
from .network_emulator import NetworkEmulator
from .tool_registry import ToolRegistry


class MCPEnvironment:
    """Среда NetMCP с реальными данными из ToolBench"""

    def __init__(self, config: Config):
        self.config = config
        self.network = NetworkEmulator(config)
        self.tools = ToolRegistry(config)

        self.current_query = None
        self.current_query_data = None
        self.relevant_tools = []  # инструменты, которые реально подходят
        self.step_count = 0
        self.used_tools = []  # уже использованные инструменты

    def reset(self, query_data: Optional[Dict] = None):
        """Сброс среды с данными из ToolBench"""
        self.step_count = 0
        self.used_tools = []

        if query_data:
            self.current_query_data = query_data
            self.current_query = query_data['query']
            self.relevant_tools = query_data.get('relevant_tools', [])
        else:
            # Если данные не переданы, берем случайный из конфига
            self.current_query_data = random.choice(self.config.prompts)
            self.current_query = self.current_query_data['query']
            self.relevant_tools = self.current_query_data.get('relevant_tools', [])

        # Обновляем состояние сети
        self.network.update_network_state()

        return self._get_current_state()

    def _get_current_state(self) -> Dict[str, Any]:
        """Формирование текущего состояния для агента"""
        # Получаем top-10 семантически близких инструментов
        candidate_tools = self.tools.get_top_k_tools(self.current_query, k=10)

        tools_state = []
        for tool in candidate_tools:
            server_state = self.network.get_server_state(tool['name'])
            qos = self.network.get_qos_metrics(tool['name'])

            # Является ли этот инструмент релевантным для запроса?
            is_relevant = any(
                rt['name'] == tool['name'] for rt in self.relevant_tools
            )

            tool_state = {
                'name': tool['name'],
                'category': tool['category'],
                'description': tool['description'][:50] + "...",  # сокращаем для состояния
                'available': server_state['available'],
                'latency': qos['avg_latency'],
                'stability': qos['stability'],
                'semantic_score': self.tools.semantic_similarity(
                    self.current_query, tool['name']
                ),
                'is_relevant': is_relevant,  # помечаем действительно нужные инструменты
                'used': tool['name'] in self.used_tools
            }
            tools_state.append(tool_state)

        return {
            'query': self.current_query,
            'query_domain': self.current_query_data.get('domain', 'unknown'),
            'step': self.step_count,
            'tools': tools_state,
            'total_tools': len(self.config.tools)
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Выполнение действия (вызов инструмента)

        Args:
            action: имя выбранного инструмента в формате "ToolName.APIName"
        """
        self.step_count += 1
        self.used_tools.append(action)

        # Получаем информацию о выбранном инструменте
        tool = self.tools.get_tool_by_name(action)
        if not tool:
            return (
                self._get_current_state(),
                self.config.reward.invalid_call_penalty,
                True,
                {'error': f'Invalid tool: {action}'}
            )

        # Проверяем доступность сервера
        server_state = self.network.get_server_state(tool['name'])
        if not server_state['available']:
            return (
                self._get_current_state(),
                -0.2,
                False,
                {'error': 'server unavailable', 'tool': action}
            )

        # Симулируем выполнение запроса с задержкой
        latency = self.network.get_current_latency(
            tool['name'],
            {'base_latency': tool['base_latency']}
        )
        time.sleep(latency * 0.01)  # масштабируем для тестов

        # Определяем успешность выполнения
        success = random.random() > tool['failure_rate']

        # Проверяем, является ли выбранный инструмент релевантным
        is_relevant = any(rt['name'] == action for rt in self.relevant_tools)

        # Вычисляем награду
        reward = self._calculate_reward(
            tool=tool,
            latency=latency,
            success=success,
            is_relevant=is_relevant
        )

        # Завершаем, если:
        # 1. Превышен лимит шагов
        # 2. Задача решена (использован релевантный инструмент с успехом)
        # 3. Использовано слишком много инструментов
        done = (
                self.step_count >= self.config.rl.max_steps
                or (success and is_relevant)  # успешно использован нужный инструмент
                or len(self.used_tools) >= 3  # не больше 3 попыток
        )

        info = {
            'latency': latency,
            'success': success,
            'is_relevant': is_relevant,
            'tool_used': action,
            'tool_category': tool['category'],
            'step': self.step_count
        }

        return self._get_current_state(), reward, done, info

    def _calculate_reward(self, tool: Dict, latency: float, success: bool, is_relevant: bool) -> float:
        """Усовершенствованное вычисление награды"""
        reward = 0.0

        # Главная награда за успех с правильным инструментом
        if success and is_relevant:
            reward += self.config.reward.success_reward
            # Бонус за быстрое решение
            if self.step_count == 1:
                reward += 0.3
        elif success and not is_relevant:
            # Инструмент сработал, но не тот, что нужен
            reward += 0.2  # небольшой плюс за успешный вызов
            reward += self.config.reward.wrong_tool_penalty
        elif not success and is_relevant:
            # Правильный инструмент, но не сработал
            reward += self.config.reward.failure_penalty * 0.5
        else:
            # Всё плохо
            reward += self.config.reward.failure_penalty

        # Штраф за высокую задержку
        if latency > self.config.reward.latency_threshold:
            reward -= 0.2

        # Штраф за каждый шаг
        reward += self.config.reward.step_penalty * self.step_count

        # Семантический бонус
        semantic_score = self.tools.semantic_similarity(
            self.current_query, tool['name']
        )
        if semantic_score > 0.7:
            reward += self.config.reward.semantic_bonus

        # Штраф за повторное использование того же инструмента
        if self.used_tools.count(tool['name']) > 1:
            reward += self.config.reward.extra_step_penalty

        return reward