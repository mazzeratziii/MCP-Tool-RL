# Окружение MCP
import time
import random
import json
import re
from typing import Dict, Any, Tuple, Optional

from src.config import Config
from .network_emulator import NetworkEmulator, NetworkMode
from .tool_registry import ToolRegistry


class MCPEnvironment:
    def __init__(self, config: Config, llm_client=None, network_mode: NetworkMode = NetworkMode.DETERMINISTIC):
        """Initialize the object."""
        self.config = config
        self.network = NetworkEmulator(config, mode=network_mode)
        self.llm_client = llm_client

        # Убеждаемся, что данные загружены до создания ToolRegistry
        if not config.tools:
            print("Loading data before creating environment...")
            config.load_data()

        self.tools = ToolRegistry(config)

        self.current_query = None
        self.current_query_data = None
        self.relevant_tools = []
        self.step_count = 0
        self.used_tools = []

    def set_network_mode(self, mode: NetworkMode):
        """Переключение режима сети"""
        self.network.set_mode(mode)

    def get_network_stats(self) -> Dict[str, Any]:
        """Получение статистики сети"""
        return self.network.get_network_stats()

    def reset(self, query_data: Optional[Dict] = None):
        """Reset reset."""
        self.step_count = 0
        self.used_tools = []

        if query_data:
            self.current_query_data = query_data
            self.current_query = query_data['query']
            self.relevant_tools = query_data.get('relevant_tools', [])
        else:
            self.current_query_data = self._get_random_query()
            self.current_query = self.current_query_data['query']
            self.relevant_tools = self.current_query_data.get('relevant_tools', [])

        self.network.update_network_state()
        return self._get_current_state()

    def _get_random_query(self) -> Dict:
        """Return get random query."""
        if not self.config.tools:
            return {
                'query': 'What is the weather today?',
                'domain': 'general',
                'relevant_tools': []
            }
        random_tool = random.choice(self.config.tools)
        return {
            'query': f"How to use {random_tool['name']}?",
            'domain': random_tool.get('category', 'general'),
            'relevant_tools': [{'name': random_tool['name']}]
        }

    def _get_current_state(self) -> Dict[str, Any]:
        """Return get current state."""
        candidate_tools = self.tools.get_top_k_tools(self.current_query, k=10)

        tools_state = []
        for tool in candidate_tools:
            server_state = self.network.get_server_state(tool['name'])
            qos = self.network.get_qos_metrics(tool['name'])
            total_calls = server_state.get('success_count', 0) + server_state.get('failure_count', 0)
            observed_success_rate = (
                server_state.get('success_count', 0) / total_calls
                if total_calls > 0
                else 1.0 - tool.get('failure_rate', self.config.network.failure_rate)
            )

            is_relevant = any(rt['name'] == tool['name'] for rt in self.relevant_tools)

            tool_state = {
                'name': tool['name'],
                'category': tool.get('category', 'general'),
                'description': tool.get('description', '')[:50] + "...",
                'available': server_state['available'],
                'latency': qos['avg_latency'],
                'jitter': qos.get('jitter', 0.0),
                'stability': qos['stability'],
                'success_rate': observed_success_rate,
                'semantic_score': self.tools.semantic_similarity(self.current_query, tool['name']),
                'is_relevant': is_relevant,
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
        """Основной шаг окружения"""
        self.step_count += 1
        self.used_tools.append(action)

        tool = self.tools.get_tool_by_name(action)
        if not tool:
            return (
                self._get_current_state(),
                self.config.reward.invalid_call_penalty,
                True,
                {'error': f'Invalid tool: {action}'}
            )

        # Обработка fallback инструментов
        if action == "Calculator.Evaluate":
            return self._handle_calculator(tool)
        elif action == "General.NoToolNeeded":
            return self._handle_no_tool_needed(tool)

        server_state = self.network.get_server_state(tool['name'])
        if not server_state['available']:
            return (
                self._get_current_state(),
                -0.2,
                False,
                {'error': 'server unavailable', 'tool': action}
            )

        latency = self.network.get_current_latency(tool['name'], {'base_latency': tool.get('base_latency', 0.1)})
        # При необходимости можно включить задержку: time.sleep(latency * 0.01)

        success = random.random() > tool.get('failure_rate', 0.1)
        is_relevant = any(rt['name'] == action for rt in self.relevant_tools)
        if success:
            server_state['success_count'] = server_state.get('success_count', 0) + 1
        else:
            server_state['failure_count'] = server_state.get('failure_count', 0) + 1
        server_state['avg_latency'] = latency

        reward = self._calculate_reward(tool, latency, success, is_relevant)
        response = self._generate_response(tool, success, is_relevant)

        done = (
            self.step_count >= self.config.rl.max_steps
            or (success and is_relevant)
            or len(self.used_tools) >= 3
        )

        info = {
            'latency': latency,
            'success': success,
            'is_relevant': is_relevant,
            'tool_used': action,
            'tool_category': tool.get('category', 'general'),
            'step': self.step_count,
            'response': response,
            'result': response,
            'semantic_score': self.tools.semantic_similarity(self.current_query, tool.get('name', ''))
        }

        return self._get_current_state(), reward, done, info

    def _generate_response(self, tool: Dict, success: bool, is_relevant: bool) -> str:
        """Generate generate response."""
        if not success:
            return f"Tool '{tool.get('name', 'unknown')}' could not process the request due to network or server error."

        if not is_relevant:
            return f"Tool '{tool.get('name', 'unknown')}' was called but is not the most suitable for this query."

        tool_info = {
            "name": tool.get('name', 'unknown'),
            "category": tool.get('category', 'general'),
            "description": tool.get('description', 'No description available'),
            "required_parameters": tool.get('required_parameters', []),
            "optional_parameters": tool.get('optional_parameters', []),
            "examples": tool.get('examples', [])
        }

        if self.llm_client:
            try:
                prompt = f"""You are a helpful assistant. Answer the user query using ONLY the information from this tool.

User query: {self.current_query}

Tool Information:
Name: {tool_info['name']}
Category: {tool_info['category']}
Description: {tool_info['description'][:200]}

Rules:
- Do not invent information.
- Be concise and helpful."""

                response = self.llm_client.ask(prompt)
                if response and len(response.strip()) > 5:
                    return response.strip()
            except Exception as e:
                print(f"[Warning] LLM generation failed: {e}")

        # Резервный ответ
        return self._fallback_response(tool_info)

    def _fallback_response(self, tool_info: Dict) -> str:
        """Безопасная версия ответа"""
        lines = [
            f"Tool: {tool_info['name']}",
            f"Category: {tool_info['category']}",
            f"Description: {tool_info['description'][:150]}..."
        ]

        # Обязательные параметры
        req = tool_info.get('required_parameters', [])
        if req:
            param_names = [p.get('name', str(p)) if isinstance(p, dict) else str(p) for p in req[:5]]
            if param_names:
                lines.append(f"Required: {', '.join(param_names)}")

        # Необязательные параметры
        opt = tool_info.get('optional_parameters', [])
        if opt:
            param_names = [p.get('name', str(p)) if isinstance(p, dict) else str(p) for p in opt[:3]]
            if param_names:
                lines.append(f"Optional: {', '.join(param_names)}")

        if tool_info.get('examples'):
            lines.append(f"Example: {str(tool_info['examples'][0])[:100]}...")

        return "\n".join(lines)

    def _calculate_reward(self, tool: Dict, latency: float, success: bool, is_relevant: bool) -> float:
        """Улучшенная награда с акцентом на адаптацию к сети"""
        reward = 0.0

        if success and is_relevant:
            reward += 3.0
            if self.step_count == 1:
                reward += 1.0  # Большой бонус за быстрое решение
        elif success and not is_relevant:
            reward += 0.4
        else:
            reward -= 1.8

        # Штраф за высокую задержку
        if latency > 0.6:
            reward -= (latency - 0.6) * 0.8

        # Штраф за количество шагов
        reward -= 0.08 * self.step_count

        # Бонус за семантическую близость
        semantic_score = self.tools.semantic_similarity(self.current_query, tool.get('name', ''))
        if semantic_score > 0.75:
            reward += 0.6
        elif semantic_score > 0.6:
            reward += 0.25

        # Штраф за повторное использование инструмента
        if self.used_tools.count(tool.get('name', '')) > 1:
            reward -= 0.4

        return reward

    def _handle_calculator(self, tool: Dict) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Обработка Calculator.Evaluate"""
        import re

        # Проверяем, содержит ли запрос математическое выражение
        math_pattern = r'[\d\+\-\*/\(\)\.\s]+'
        has_math = bool(re.search(r'\d+\s*[\+\-\*/]\s*\d+', self.current_query))

        latency = 0.05  # Быстрая операция
        is_relevant = has_math
        success = True

        # Пытаемся вычислить результат
        result = "Calculator result"
        try:
            # Извлекаем математическое выражение
            expr_match = re.search(r'([\d\+\-\*/\(\)\.\s]+)', self.current_query)
            if expr_match:
                expr = expr_match.group(1).strip()
                # Безопасное вычисление (только базовые операции)
                if all(c in '0123456789+-*/(). ' for c in expr):
                    result = f"Result: {eval(expr)}"
        except:
            result = "Could not evaluate expression"
            success = False

        reward = 3.0 if (success and is_relevant) else -0.5

        info = {
            'latency': latency,
            'success': success,
            'is_relevant': is_relevant,
            'tool_used': 'Calculator.Evaluate',
            'tool_category': 'math',
            'step': self.step_count,
            'response': result,
            'result': result,
            'semantic_score': 0.95 if has_math else 0.3
        }

        return self._get_current_state(), reward, True, info

    def _handle_no_tool_needed(self, tool: Dict) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Обработка General.NoToolNeeded"""
        # Проверяем, действительно ли запрос не требует инструмента
        # Простые вопросы, приветствия и т.д.
        simple_patterns = [
            r'^(hi|hello|hey|thanks|thank you)',
            r'^(what|who|when|where|why|how)\s+(is|are|was|were)',
            r'(explain|tell me|describe)',
        ]

        is_simple = any(re.search(pattern, self.current_query.lower()) for pattern in simple_patterns)

        # Проверяем, что нет других релевантных инструментов
        has_relevant_tools = len(self.relevant_tools) > 0

        latency = 0.02
        is_relevant = is_simple and not has_relevant_tools
        success = True

        reward = 2.0 if is_relevant else -1.0

        info = {
            'latency': latency,
            'success': success,
            'is_relevant': is_relevant,
            'tool_used': 'General.NoToolNeeded',
            'tool_category': 'general',
            'step': self.step_count,
            'response': 'No external tool needed for this query',
            'result': 'No external tool needed for this query',
            'semantic_score': 0.8 if is_simple else 0.2
        }

        return self._get_current_state(), reward, True, info
