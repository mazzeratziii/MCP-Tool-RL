"""
Упрощённый MCP-клиент для гибридного режима
Работает с простыми MCP-серверами на Python без полного MCP SDK
"""
import json
import subprocess
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class MCPToolMetrics:
    """Метрики реального вызова инструмента"""
    latency: float
    success: bool
    error: Optional[str]
    timestamp: float


class SimpleMCPClient:
    """Упрощённый MCP клиент для Python серверов"""

    def __init__(self):
        """?????????????? ?????? ? ????????? ??????????? ???????????."""
        self.servers: Dict[str, subprocess.Popen] = {}
        self.metrics_history: Dict[str, List[MCPToolMetrics]] = {}
        self.tool_to_server: Dict[str, str] = {}

    def register_server(self, server_name: str, server_command: List[str]):
        """
        Регистрация MCP сервера

        Аргументы:
            server_name: имя сервера (например, "calculator")
            server_command: команда запуска (например, ["python", "-m", "mcp_servers.calculator"])
        """
        try:
            # Запускаем сервер как subprocess
            process = subprocess.Popen(
                server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self.servers[server_name] = process
            print(f"✓ MCP server '{server_name}' started")

            # Получаем список инструментов
            tools = self._list_tools(server_name)
            for tool in tools:
                self.tool_to_server[tool] = server_name
                self.metrics_history[tool] = []

            return True
        except Exception as e:
            print(f"✗ Failed to start MCP server '{server_name}': {e}")
            return False

    def _list_tools(self, server_name: str) -> List[str]:
        """Получить список инструментов от сервера"""
        try:
            process = self.servers.get(server_name)
            if not process:
                return []

            request = {
                "method": "tools/list",
                "params": {}
            }

            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()

            response_line = process.stdout.readline()
            response = json.loads(response_line)

            tools = response.get("tools", [])
            return [tool["name"] for tool in tools]
        except Exception as e:
            print(f"Warning: Could not list tools from {server_name}: {e}")
            return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Синхронный вызов инструмента

        Аргументы:
            tool_name: имя инструмента (например, "calculator.evaluate")
            arguments: аргументы для инструмента

        Возвращает:
            Dict с результатом: {success, error, latency, result}
        """
        server_name = self.tool_to_server.get(tool_name)

        if not server_name:
            return {
                "success": False,
                "error": f"Tool {tool_name} not registered",
                "latency": 0.0,
                "result": None
            }

        process = self.servers.get(server_name)
        if not process:
            return {
                "success": False,
                "error": f"Server {server_name} not running",
                "latency": 0.0,
                "result": None
            }

        start_time = time.time()

        try:
            # Отправляем запрос
            request = {
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()

            # Читаем ответ
            response_line = process.stdout.readline()
            latency = time.time() - start_time

            if not response_line:
                raise Exception("No response from server")

            response = json.loads(response_line)

            # Парсим результат
            if "error" in response:
                success = False
                error = response["error"]
                result = None
            else:
                content = response.get("content", [])
                if content and len(content) > 0:
                    text = content[0].get("text", "{}")
                    result = json.loads(text)
                    success = result.get("success", True)
                    error = result.get("error")
                else:
                    success = False
                    error = "Empty response"
                    result = None

            # Сохраняем метрики
            metrics = MCPToolMetrics(
                latency=latency,
                success=success,
                error=error,
                timestamp=time.time()
            )
            self.metrics_history[tool_name].append(metrics)

            return {
                "success": success,
                "error": error,
                "latency": latency,
                "result": result
            }

        except Exception as e:
            latency = time.time() - start_time

            metrics = MCPToolMetrics(
                latency=latency,
                success=False,
                error=str(e),
                timestamp=time.time()
            )
            self.metrics_history[tool_name].append(metrics)

            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "result": None
            }

    def get_tool_metrics(self, tool_name: str, window: int = 50) -> Dict[str, float]:
        """
        Получение агрегированных метрик инструмента
        """
        if tool_name not in self.metrics_history:
            return {
                "avg_latency": 0.15,
                "availability": 1.0,
                "error_rate": 0.0,
                "stability": 1.0
            }

        history = self.metrics_history[tool_name][-window:]

        if not history:
            return {
                "avg_latency": 0.15,
                "availability": 1.0,
                "error_rate": 0.0,
                "stability": 1.0
            }

        latencies = [m.latency for m in history]
        successes = [m.success for m in history]

        avg_latency = sum(latencies) / len(latencies)
        availability = sum(successes) / len(successes)
        error_rate = 1.0 - availability

        # Стабильность = обратная величина вариации latency
        if len(latencies) > 1:
            variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
            stability = 1.0 / (1.0 + variance * 10)
        else:
            stability = 1.0

        return {
            "avg_latency": avg_latency,
            "availability": availability,
            "error_rate": error_rate,
            "stability": stability
        }

    def close(self):
        """Закрытие всех серверов"""
        for server_name, process in self.servers.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✓ MCP server '{server_name}' stopped")
            except Exception as e:
                print(f"Warning: Could not stop server '{server_name}': {e}")
                try:
                    process.kill()
                except:
                    pass

        self.servers.clear()
        self.tool_to_server.clear()
