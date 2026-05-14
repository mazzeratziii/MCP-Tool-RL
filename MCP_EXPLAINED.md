# MCP (Model Context Protocol) — Подробное объяснение

## Что такое MCP?

**Model Context Protocol (MCP)** — открытый протокол от Anthropic для стандартизированного подключения AI-моделей к:
- Внешним инструментам (tools)
- Источникам данных (data sources)
- API сервисам

### Аналогия
Представьте MCP как **USB для AI**:
- USB стандартизирует подключение устройств к компьютеру
- MCP стандартизирует подключение инструментов к AI-модели

---

## Архитектура MCP

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Application                          │
│                   (ваш RL агент)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client                              │
│  - Управление соединениями                                   │
│  - Вызов инструментов                                        │
│  - Получение метрик                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ MCP Protocol (JSON-RPC)
                         │
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────┐
│ MCP Server 1│  │ MCP Server 2│  │ MCP Srv 3│  │ MCP Srv N│
│  (файлы)    │  │  (поиск)    │  │  (БД)    │  │  (API)   │
└──────┬──────┘  └──────┬──────┘  └────┬─────┘  └────┬─────┘
       │                │               │             │
       ▼                ▼               ▼             ▼
┌──────────┐    ┌──────────┐    ┌──────────┐  ┌──────────┐
│Filesystem│    │Google API│    │PostgreSQL│  │Weather API│
└──────────┘    └──────────┘    └──────────┘  └──────────┘
```

---

## Официальные MCP серверы от Anthropic

### 1. **Filesystem Server**
```bash
npm install @modelcontextprotocol/server-filesystem
```

**Что делает:**
- Чтение/запись файлов
- Навигация по директориям
- Поиск файлов

**Пример использования:**
```python
# Через MCP можно:
result = await mcp_client.call_tool(
    "filesystem.read_file",
    {"path": "/home/user/document.txt"}
)
```

---

### 2. **GitHub Server**
```bash
npm install @modelcontextprotocol/server-github
```

**Что делает:**
- Работа с репозиториями
- Issues, Pull Requests
- Commits, branches
- Code search

**Пример:**
```python
result = await mcp_client.call_tool(
    "github.search_code",
    {"query": "function authenticate", "repo": "owner/repo"}
)
```

---

### 3. **Google Drive Server**
```bash
npm install @modelcontextprotocol/server-gdrive
```

**Что делает:**
- Чтение/запись документов
- Поиск файлов
- Управление доступом

---

### 4. **Slack Server**
```bash
npm install @modelcontextprotocol/server-slack
```

**Что делает:**
- Отправка сообщений
- Чтение каналов
- Управление workspace

---

### 5. **PostgreSQL Server**
```bash
npm install @modelcontextprotocol/server-postgres
```

**Что делает:**
- SQL запросы
- Схема БД
- Транзакции

---

### 6. **Puppeteer Server** (браузер)
```bash
npm install @modelcontextprotocol/server-puppeteer
```

**Что делает:**
- Автоматизация браузера
- Скриншоты
- Парсинг веб-страниц

---

## Community MCP серверы

### Поиск и данные
- **Google Search** — поиск в Google
- **Brave Search** — приватный поиск
- **Wikipedia** — доступ к Wikipedia
- **Arxiv** — научные статьи
- **YouTube** — поиск видео

### Разработка
- **Docker** — управление контейнерами
- **Kubernetes** — управление кластерами
- **AWS** — работа с AWS сервисами
- **Git** — операции с git

### Коммуникация
- **Email** — отправка писем
- **Discord** — работа с Discord
- **Telegram** — Telegram боты

### Аналитика
- **Pandas** — анализ данных
- **Matplotlib** — визуализация
- **SQL** — работа с БД

---

## Как это работает в вашем проекте

### Сейчас (эмуляция):
```python
# Фейковый вызов
def step(self, tool_name):
    # Эмулируем latency
    latency = random.uniform(0.1, 0.5)
    
    # Фейковый ответ
    response = f"Fake response from {tool_name}"
    
    return response, latency
```

### С MCP (реально):
```python
# Реальный вызов через MCP
async def step(self, tool_name):
    # Реальный вызов инструмента
    result = await mcp_client.call_tool(
        tool_name,
        {"query": self.current_query}
    )
    
    # Реальная latency из сети
    latency = result['latency']
    
    # Реальный ответ от API
    response = result['result']
    
    return response, latency
```

---

## Пример: Интеграция с реальными API

### Вариант 1: Использовать готовые MCP серверы

```json
// mcp_servers_config.json
{
  "servers": {
    "search.google": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-google-search"],
      "env": {
        "GOOGLE_API_KEY": "your-key",
        "GOOGLE_CX": "your-cx"
      }
    },
    "github.search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    },
    "database.postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_URL": "postgresql://localhost/mydb"
      }
    }
  }
}
```

### Вариант 2: Создать свои MCP серверы

```python
# custom_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="weather.get_current",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "weather.get_current":
        city = arguments["city"]
        
        # Реальный вызов OpenWeather API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": city,
                    "appid": "your-api-key",
                    "units": "metric"
                }
            )
            data = response.json()
        
        return [TextContent(
            type="text",
            text=f"Temperature in {city}: {data['main']['temp']}°C"
        )]

if __name__ == "__main__":
    server.run()
```

---

## Что получаем с MCP в вашем проекте

### 1. Реальные инструменты вместо эмуляции

**Было (эмуляция):**
```python
tools = [
    {"name": "search.google", "description": "Search Google"},
    {"name": "weather.api", "description": "Get weather"},
    # ... 15000 фейковых инструментов
]
```

**Стало (MCP):**
```python
# Реальные инструменты через MCP
mcp_tools = [
    # Официальные от Anthropic
    "filesystem.read_file",
    "filesystem.write_file",
    "github.search_code",
    "github.create_issue",
    "slack.send_message",
    "postgres.query",
    
    # Community
    "google.search",
    "wikipedia.search",
    "weather.get_current",
    "email.send",
    
    # Ваши кастомные
    "custom.tool_1",
    "custom.tool_2",
]
```

### 2. Реальные метрики

**Было:**
```python
# Эмулированная latency
latency = random.uniform(0.1, 0.5)
```

**Стало:**
```python
# Реальная latency из сети
start = time.time()
result = await mcp_client.call_tool("search.google", {"query": "..."})
latency = time.time() - start  # Реальная задержка!
```

### 3. Реальные ошибки

**Было:**
```python
# Фейковые ошибки
if random.random() < 0.1:
    return "Error: simulated failure"
```

**Стало:**
```python
# Реальные ошибки из API
try:
    result = await mcp_client.call_tool(...)
except TimeoutError:
    return "Error: API timeout"
except RateLimitError:
    return "Error: Rate limit exceeded"
except NetworkError:
    return "Error: Network unavailable"
```

---

## Практический пример для вашего проекта

### Шаг 1: Выбрать 10-20 реальных инструментов

```python
# mcp_tools_selection.py
REAL_MCP_TOOLS = {
    # Поиск (5 инструментов)
    "search.google": "@modelcontextprotocol/server-google-search",
    "search.brave": "mcp-server-brave-search",
    "search.wikipedia": "mcp-server-wikipedia",
    "search.arxiv": "mcp-server-arxiv",
    "search.youtube": "mcp-server-youtube",
    
    # Разработка (5 инструментов)
    "github.search": "@modelcontextprotocol/server-github",
    "github.issues": "@modelcontextprotocol/server-github",
    "git.operations": "mcp-server-git",
    "docker.manage": "mcp-server-docker",
    "filesystem.ops": "@modelcontextprotocol/server-filesystem",
    
    # Данные (5 инструментов)
    "postgres.query": "@modelcontextprotocol/server-postgres",
    "sqlite.query": "mcp-server-sqlite",
    "mongodb.query": "mcp-server-mongodb",
    "redis.ops": "mcp-server-redis",
    "pandas.analyze": "mcp-server-pandas",
    
    # Коммуникация (5 инструментов)
    "slack.message": "@modelcontextprotocol/server-slack",
    "email.send": "mcp-server-email",
    "discord.message": "mcp-server-discord",
    "telegram.send": "mcp-server-telegram",
    "sms.send": "mcp-server-twilio",
}
```

### Шаг 2: Настроить MCP клиент

```python
# src/mcp/real_client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class RealMCPClient:
    def __init__(self):
        self.sessions = {}
    
    async def connect_server(self, tool_name: str, server_command: str):
        """Подключение к MCP серверу"""
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", server_command]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.sessions[tool_name] = session
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Вызов инструмента через MCP"""
        session = self.sessions.get(tool_name)
        if not session:
            raise ValueError(f"Tool {tool_name} not connected")
        
        result = await session.call_tool(tool_name, arguments)
        return result
```

### Шаг 3: Интегрировать в обучение

```python
# main.py
import asyncio
from src.mcp.real_client import RealMCPClient
from src.mcp_tools_selection import REAL_MCP_TOOLS

async def main():
    # Инициализация MCP клиента
    mcp_client = RealMCPClient()
    
    # Подключение к серверам
    for tool_name, server_command in REAL_MCP_TOOLS.items():
        await mcp_client.connect_server(tool_name, server_command)
    
    # Обучение с реальными инструментами
    trainer = RealMCPTrainer(config, mcp_client)
    trainer.train()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Гибридный подход (рекомендуется)

Используйте **комбинацию эмуляции и реальных MCP**:

```python
class HybridEnvironment:
    def __init__(self, config, mcp_client=None):
        self.config = config
        self.mcp_client = mcp_client
        
        # Реальные инструменты через MCP
        self.real_tools = [
            "search.google",
            "github.search",
            "weather.api",
            # ... 20 реальных
        ]
        
        # Эмулированные инструменты (для разнообразия)
        self.emulated_tools = [
            "tool_1", "tool_2", ..., "tool_14980"
        ]
    
    async def step(self, tool_name):
        if tool_name in self.real_tools and self.mcp_client:
            # Реальный вызов через MCP
            return await self._real_call(tool_name)
        else:
            # Эмуляция
            return self._emulated_call(tool_name)
```

**Преимущества:**
- ✅ Реальные метрики от важных инструментов
- ✅ Быстрое обучение (не ждём 15000 API вызовов)
- ✅ Разнообразие (эмуляция даёт больше вариантов)

---

## Итог

### Что такое MCP API?
MCP — это **протокол**, а не конкретные API. Через MCP вы подключаете:
- Официальные серверы от Anthropic (GitHub, Slack, PostgreSQL, etc.)
- Community серверы (Google Search, Weather, etc.)
- Свои кастомные серверы (любые API)

### Что убираем?
- ❌ Эмуляцию latency (random.uniform)
- ❌ Фейковые ответы
- ❌ Искусственные ошибки

### Что добавляем?
- ✅ Реальные вызовы через MCP протокол
- ✅ Настоящие метрики из сети
- ✅ Реальные ответы от API

### Рекомендация
Начните с **гибридного подхода**:
1. 20 реальных инструментов через MCP (важные)
2. 14980 эмулированных (для разнообразия)
3. Постепенно заменяйте эмуляцию на реальные MCP серверы

Хотите, чтобы я показал конкретный пример интеграции с 5-10 реальными MCP серверами?
