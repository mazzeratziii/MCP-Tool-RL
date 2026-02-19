import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from mcp.protocol import MessageType

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, host: str = "localhost", port: int = 8765,
                 max_buffer_size: int = 10 * 1024 * 1024):  # 10MB буфер
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.connected = False
        self.tools_cache = []
        self.max_buffer_size = max_buffer_size
        self._buffer = b""

    async def connect(self):
        """Подключается к MCP серверу с увеличенным буфером."""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port,
                limit=self.max_buffer_size  # Увеличиваем лимит буфера
            )
            self.connected = True
            logger.info(f"✅ MCP клиент подключен к {self.host}:{self.port}")
            await self.list_tools()
        except Exception as e:
            logger.error(f"❌ Ошибка подключения: {e}")
            raise

    async def disconnect(self):
        """Отключается от сервера."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        logger.info("🔌 MCP клиент отключен")

    async def _read_message(self) -> str:
        """
        Читает полное сообщение, учитывая возможность больших JSON.
        Читает до символа новой строки, но с увеличенным буфером.
        """
        try:
            # Читаем до символа новой строки с таймаутом
            data = await asyncio.wait_for(
                self.reader.readuntil(b'\n'),
                timeout=30.0
            )
            return data.decode('utf-8').strip()
        except asyncio.IncompleteReadError as e:
            # Если соединение закрыто, но есть частичные данные
            if e.partial:
                return e.partial.decode('utf-8').strip()
            raise
        except asyncio.LimitOverrunError:
            # Если превышен лимит, читаем всё доступное
            data = await self.reader.read(self.max_buffer_size)
            return data.decode('utf-8').strip()
        except asyncio.TimeoutError:
            logger.error("Таймаут при чтении сообщения")
            raise

    async def send_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Отправляет сообщение и получает ответ с поддержкой больших JSON."""
        if not self.connected:
            raise ConnectionError("Клиент не подключен")

        # Отправляем сообщение
        data = json.dumps(msg) + "\n"
        self.writer.write(data.encode())
        await self.writer.drain()

        # Получаем ответ
        try:
            response_data = await self._read_message()
            if not response_data:
                raise ConnectionError("Сервер закрыл соединение")

            return json.loads(response_data)

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            logger.debug(f"Полученные данные: {response_data[:200]}...")
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Получает список доступных инструментов."""
        try:
            resp = await self.send_message({"type": MessageType.LIST_TOOLS})
            self.tools_cache = resp.get("tools", [])
            logger.info(f"📦 Получено {len(self.tools_cache)} инструментов от сервера")
            return self.tools_cache
        except Exception as e:
            logger.error(f"Ошибка при получении списка инструментов: {e}")
            return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Вызывает инструмент."""
        return await self.send_message({
            "type": MessageType.CALL_TOOL,
            "tool": name,
            "arguments": arguments
        })

    def format_tools_for_prompt(self) -> str:
        """Форматирует список инструментов для промпта."""
        if not self.tools_cache:
            return "Нет доступных инструментов"

        lines = ["Доступные инструменты:\n"]
        for i, t in enumerate(self.tools_cache[:20], 1):  # Показываем только первые 20
            lines.append(f"{i}. {t['name']}")
            if t.get('description'):
                lines.append(f"   Описание: {t['description'][:100]}...")

        if len(self.tools_cache) > 20:
            lines.append(f"\n... и ещё {len(self.tools_cache) - 20} инструментов")

        return "\n".join(lines)