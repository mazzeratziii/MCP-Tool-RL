import asyncio
import json
import logging
from typing import Dict, List, Any, Callable
from datetime import datetime

from mcp.protocol import MessageType, PROTOCOL_VERSION, Status

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self, host: str = "localhost", port: int = 8765, max_buffer_size: int = 10 * 1024 * 1024):
        self.host = host
        self.port = port
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.clients = set()
        self.server = None
        self.max_buffer_size = max_buffer_size
        self.stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "tools_registered": 0}

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable) -> bool:
        """Регистрирует инструмент в MCP."""
        if name in self.tools:
            logger.warning(f"Инструмент {name} уже зарегистрирован")
            return False

        self.tools[name] = {
            "name": name,
            "description": description[:200] + "..." if len(description) > 200 else description,
            # Обрезаем длинные описания
            "inputSchema": input_schema
        }
        self.handlers[name] = handler
        self.stats["tools_registered"] += 1
        logger.info(f"✅ Инструмент '{name}' зарегистрирован в MCP")

        asyncio.create_task(self._notify_clients({
            "type": MessageType.TOOL_REGISTERED,
            "tool": self.tools[name],
            "timestamp": datetime.now().isoformat()
        }))
        return True

    def list_tools(self) -> List[Dict[str, Any]]:
        """Возвращает список всех инструментов (сокращённые описания)."""
        return list(self.tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Вызывает инструмент по имени."""
        self.stats["total_calls"] += 1
        start = datetime.now()

        if name not in self.handlers:
            self.stats["failed_calls"] += 1
            return {
                "status": Status.ERROR,
                "error": f"Инструмент '{name}' не найден",
                "timestamp": datetime.now().isoformat()
            }

        try:
            result = await self.handlers[name](arguments)
            self.stats["successful_calls"] += 1
            return {
                "status": Status.SUCCESS,
                "result": result,
                "tool": name,
                "execution_time_ms": (datetime.now() - start).total_seconds() * 1000,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"Ошибка при вызове {name}: {e}")
            return {
                "status": Status.ERROR,
                "error": str(e),
                "tool": name,
                "timestamp": datetime.now().isoformat()
            }

    async def handle_message(self, msg: Dict[str, Any], writer) -> Dict[str, Any]:
        """Обрабатывает входящее сообщение."""
        msg_type = msg.get("type")

        if msg_type == MessageType.LIST_TOOLS:
            return {
                "type": MessageType.TOOLS_LIST,
                "tools": self.list_tools(),
                "protocol_version": PROTOCOL_VERSION,
                "timestamp": datetime.now().isoformat()
            }
        elif msg_type == MessageType.CALL_TOOL:
            res = await self.call_tool(
                msg.get("tool"),
                msg.get("arguments", {})
            )
            res["type"] = MessageType.TOOL_RESULT
            return res
        else:
            return {
                "type": MessageType.ERROR,
                "error": f"Неизвестный тип {msg_type}",
                "timestamp": datetime.now().isoformat()
            }

    async def _send_response(self, writer, response: Dict[str, Any]):
        """Отправляет ответ с правильным размером буфера."""
        try:
            data = json.dumps(response) + "\n"
            writer.write(data.encode())
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            logger.debug("Клиент отключился при отправке")
        except Exception as e:
            logger.error(f"Ошибка отправки: {e}")

    async def _handle_client(self, reader, writer):
        """Обрабатывает подключение клиента."""
        cid = id(writer)
        self.clients.add(writer)
        logger.info(f"🔌 Клиент {cid} подключился")

        try:
            while True:
                try:
                    # Читаем с таймаутом
                    data = await asyncio.wait_for(
                        reader.readuntil(b'\n'),
                        timeout=60.0
                    )
                    msg = json.loads(data.decode())
                    response = await self.handle_message(msg, writer)
                    await self._send_response(writer, response)

                except asyncio.IncompleteReadError:
                    # Клиент закрыл соединение
                    break
                except asyncio.TimeoutError:
                    # Таймаут чтения - проверяем, жив ли клиент
                    continue
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка парсинга JSON от клиента {cid}: {e}")
                    await self._send_response(writer, {
                        "type": MessageType.ERROR,
                        "error": "Invalid JSON",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Ошибка обработки клиента {cid}: {e}")
                    break

        except Exception as e:
            logger.error(f"Критическая ошибка клиента {cid}: {e}")
        finally:
            self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()
            logger.info(f"🔌 Клиент {cid} отключился")

    async def _notify_clients(self, notification: Dict[str, Any]):
        """Отправляет уведомление всем клиентам."""
        if not self.clients:
            return

        data = json.dumps(notification) + "\n"
        for client in list(self.clients):
            try:
                client.write(data.encode())
                await client.drain()
            except:
                pass

    async def start(self):
        """Запускает MCP сервер."""
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            limit=self.max_buffer_size  # Увеличиваем лимит буфера
        )

        addr = self.server.sockets[0].getsockname()
        logger.info(f"🚀 MCP сервер запущен на {addr[0]}:{addr[1]}")

        async with self.server:
            await self.server.serve_forever()

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сервера."""
        success_rate = self.stats["successful_calls"] / self.stats["total_calls"] if self.stats[
                                                                                         "total_calls"] > 0 else 0
        return {
            "tools": self.stats["tools_registered"],
            "total_calls": self.stats["total_calls"],
            "successful_calls": self.stats["successful_calls"],
            "failed_calls": self.stats["failed_calls"],
            "clients": len(self.clients),
            "success_rate": success_rate
        }