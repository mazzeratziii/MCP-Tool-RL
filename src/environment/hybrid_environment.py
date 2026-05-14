"""
Hybrid Environment: ToolBench emulation + Real MCP calls
"""
import json
import random
from typing import Dict, Any, Tuple, Optional
from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.mcp.simple_client import SimpleMCPClient


class HybridMCPEnvironment(MCPEnvironment):
    """
    Гибридное окружение:
    - Использует реальные MCP вызовы для зарегистрированных инструментов
    - Эмулирует остальные инструменты
    """

    def __init__(self, config: Config, mcp_config_path: str = "mcp_config.json"):
        super().__init__(config)

        # Инициализируем MCP клиент
        self.mcp_client = SimpleMCPClient()
        self.mcp_tools = set()  # Инструменты, доступные через MCP

        # Загружаем конфигурацию MCP
        self._load_mcp_config(mcp_config_path)

    def _load_mcp_config(self, config_path: str):
        """Загрузка конфигурации MCP и запуск серверов"""
        try:
            with open(config_path, 'r') as f:
                mcp_config = json.load(f)

            print("\n" + "=" * 60)
            print("INITIALIZING MCP SERVERS")
            print("=" * 60)

            tools_config = mcp_config.get("tools", {})
            enabled_count = 0

            for tool_name, tool_config in tools_config.items():
                if not tool_config.get("enabled", False):
                    continue

                # Проверяем, требуется ли API ключ
                if tool_config.get("requires_api_key", False):
                    print(f"⊘ Skipping {tool_name} (requires API key)")
                    continue

                # Определяем команду запуска сервера
                server_command = tool_config.get("mcp_server", "")

                # Для Python серверов
                if server_command.startswith("python -m"):
                    module = server_command.replace("python -m ", "")
                    command = ["python", "-m", module]

                    # Регистрируем сервер
                    server_name = module.split(".")[-1]  # например, "calculator"
                    if self.mcp_client.register_server(server_name, command):
                        self.mcp_tools.add(tool_name)
                        enabled_count += 1

            print(f"\n✓ Initialized {enabled_count} MCP tools")
            print(f"  MCP tools: {list(self.mcp_tools)[:5]}...")
            print(f"  Emulated tools: {len(self.config.tools) - enabled_count}")
            print("=" * 60 + "\n")

        except FileNotFoundError:
            print(f"Warning: MCP config not found at {config_path}")
            print("Running in full emulation mode")
        except Exception as e:
            print(f"Warning: Could not load MCP config: {e}")
            print("Running in full emulation mode")

    def _is_mcp_tool(self, tool_name: str) -> bool:
        """Проверка, доступен ли инструмент через MCP"""
        return tool_name in self.mcp_tools

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Шаг окружения с гибридным вызовом
        """
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

        # Проверяем доступность сервера (эмулированная)
        server_state = self.network.get_server_state(tool['name'])
        if not server_state['available']:
            return (
                self._get_current_state(),
                -0.2,
                False,
                {'error': 'server unavailable', 'tool': action}
            )

        # ГИБРИДНЫЙ ВЫЗОВ
        if self._is_mcp_tool(action):
            # ✅ РЕАЛЬНЫЙ вызов через MCP
            result = self._call_mcp_tool(action, tool)
            latency = result['latency']
            success = result['success']
            response = result.get('result', {})
        else:
            # ❌ ЭМУЛЯЦИЯ
            latency = self.network.get_current_latency(
                tool['name'],
                {'base_latency': tool.get('base_latency', 0.1)}
            )
            success = random.random() > tool.get('failure_rate', 0.1)
            response = self._generate_response(tool, success,
                                              any(rt['name'] == action for rt in self.relevant_tools))

        is_relevant = any(rt['name'] == action for rt in self.relevant_tools)
        reward = self._calculate_reward(tool, latency, success, is_relevant)

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
            'semantic_score': self.tools.semantic_similarity(
                self.current_query, tool.get('name', '')
            ),
            'is_mcp': self._is_mcp_tool(action)  # Флаг: реальный или эмулированный
        }

        return self._get_current_state(), reward, done, info

    def _call_mcp_tool(self, tool_name: str, tool: Dict) -> Dict[str, Any]:
        """Реальный вызов инструмента через MCP"""
        try:
            # Подготавливаем аргументы
            arguments = {
                "query": self.current_query
            }

            # Специфичные аргументы для разных инструментов
            if "calculator" in tool_name:
                # Извлекаем математическое выражение из запроса
                arguments = {"expression": self.current_query}
            elif "wikipedia" in tool_name:
                arguments = {"query": self.current_query}
            elif "http" in tool_name:
                # Для HTTP нужен URL
                arguments = {"url": self.current_query}

            # Вызываем через MCP
            result = self.mcp_client.call_tool(tool_name, arguments)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": 0.5,
                "result": None
            }

    def close(self):
        """Закрытие MCP клиента"""
        if hasattr(self, 'mcp_client'):
            self.mcp_client.close()

    def __del__(self):
        """Деструктор для гарантированного закрытия"""
        self.close()
