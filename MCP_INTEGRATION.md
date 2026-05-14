# Интеграция реального MCP в MCP-Tool-RL

## Проблема текущей реализации

**Сейчас:**
- ❌ Эмулируем сеть искусственно
- ❌ Фейковые ответы от инструментов
- ❌ Нет реальных метрик latency/availability
- ❌ Не используем MCP протокол

**Нужно:**
- ✅ Реальные MCP серверы
- ✅ Настоящие вызовы инструментов
- ✅ Реальные метрики из сети
- ✅ Стандартизированный протокол

---

## Архитектура с реальным MCP

```
┌─────────────────────────────────────────────────────────────┐
│                         RL Agent                            │
│  (выбирает инструмент на основе query + network metrics)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client Layer                         │
│  - Управление соединениями с MCP серверами                  │
│  - Сбор метрик (latency, availability, errors)              │
│  - Retry logic, timeout handling                            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ MCP Server 1│  │ MCP Server 2│  │ MCP Server N│
│  (search)   │  │  (weather)  │  │  (database) │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Шаг 1: Установка MCP SDK

### Python MCP SDK
```bash
pip install mcp anthropic-mcp
```

### Альтернатива: использовать httpx для прямых вызовов
```bash
pip install httpx sse-starlette
```

---

## Шаг 2: Создание MCP Client

### Файл: `src/mcp/mcp_client.py`

```python
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx


@dataclass
class MCPToolMetrics:
    """Метрики реального вызова инструмента"""
    latency: float
    success: bool
    error: Optional[str]
    timestamp: float
    response_size: int


class MCPClient:
    """Клиент для работы с реальными MCP серверами"""
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.servers: Dict[str, str] = {}  # tool_name -> server_url
        self.metrics_history: Dict[str, List[MCPToolMetrics]] = {}
        self.client = httpx.AsyncClient(timeout=timeout)
    
    def register_server(self, tool_name: str, server_url: str):
        """Регистрация MCP сервера для инструмента"""
        self.servers[tool_name] = server_url
        self.metrics_history[tool_name] = []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Реальный вызов инструмента через MCP
        Возвращает результат + метрики
        """
        if tool_name not in self.servers:
            return {
                "success": False,
                "error": f"Tool {tool_name} not registered",
                "latency": 0.0,
                "result": None
            }
        
        server_url = self.servers[tool_name]
        start_time = time.time()
        
        try:
            # MCP protocol: POST /tools/{tool_name}/call
            response = await self.client.post(
                f"{server_url}/tools/{tool_name}/call",
                json={"arguments": arguments},
                timeout=self.timeout
            )
            
            latency = time.time() - start_time
            success = response.status_code == 200
            
            if success:
                result = response.json()
                error = None
            else:
                result = None
                error = f"HTTP {response.status_code}: {response.text}"
            
            # Сохраняем метрики
            metrics = MCPToolMetrics(
                latency=latency,
                success=success,
                error=error,
                timestamp=time.time(),
                response_size=len(response.content)
            )
            self.metrics_history[tool_name].append(metrics)
            
            return {
                "success": success,
                "error": error,
                "latency": latency,
                "result": result,
                "response_size": len(response.content)
            }
        
        except httpx.TimeoutException:
            latency = time.time() - start_time
            metrics = MCPToolMetrics(
                latency=latency,
                success=False,
                error="Timeout",
                timestamp=time.time(),
                response_size=0
            )
            self.metrics_history[tool_name].append(metrics)
            
            return {
                "success": False,
                "error": "Timeout",
                "latency": latency,
                "result": None
            }
        
        except Exception as e:
            latency = time.time() - start_time
            metrics = MCPToolMetrics(
                latency=latency,
                success=False,
                error=str(e),
                timestamp=time.time(),
                response_size=0
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
        (последние N вызовов)
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
    
    async def close(self):
        """Закрытие соединений"""
        await self.client.aclose()
```

---

## Шаг 3: Интеграция в Environment

### Файл: `src/environment/mcp_environment_real.py`

```python
import asyncio
from typing import Dict, Any, Tuple, Optional
from src.config import Config
from src.mcp.mcp_client import MCPClient
from .tool_registry import ToolRegistry


class RealMCPEnvironment:
    """Environment с реальными MCP вызовами"""
    
    def __init__(self, config: Config, mcp_servers: Dict[str, str]):
        """
        Args:
            config: конфигурация
            mcp_servers: маппинг tool_name -> server_url
                Например: {
                    "search.google": "http://localhost:3000",
                    "weather.api": "http://localhost:3001",
                }
        """
        self.config = config
        self.mcp_client = MCPClient(timeout=config.min_request_timeout)
        
        # Регистрируем серверы
        for tool_name, server_url in mcp_servers.items():
            self.mcp_client.register_server(tool_name, server_url)
        
        self.tools = ToolRegistry(config)
        self.current_query = None
        self.current_query_data = None
        self.relevant_tools = []
        self.step_count = 0
        self.used_tools = []
    
    def reset(self, query_data: Optional[Dict] = None):
        """Сброс окружения"""
        self.step_count = 0
        self.used_tools = []
        
        if query_data:
            self.current_query_data = query_data
            self.current_query = query_data['query']
            self.relevant_tools = query_data.get('relevant_tools', [])
        
        return self._get_current_state()
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Получение текущего состояния с РЕАЛЬНЫМИ метриками"""
        candidate_tools = self.tools.get_top_k_tools(self.current_query, k=20)
        
        tools_state = []
        for tool in candidate_tools:
            # РЕАЛЬНЫЕ метрики из истории вызовов
            metrics = self.mcp_client.get_tool_metrics(tool['name'])
            
            is_relevant = any(rt['name'] == tool['name'] for rt in self.relevant_tools)
            
            tool_state = {
                'name': tool['name'],
                'category': tool.get('category', 'general'),
                'description': tool.get('description', '')[:50] + "...",
                'available': metrics['availability'] > 0.8,  # доступен если >80% успеха
                'latency': metrics['avg_latency'],
                'stability': metrics['stability'],
                'semantic_score': self.tools.semantic_similarity(
                    self.current_query, tool['name']
                ),
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
    
    async def step_async(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Асинхронный шаг с РЕАЛЬНЫМ вызовом инструмента
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
        
        # РЕАЛЬНЫЙ вызов через MCP
        result = await self.mcp_client.call_tool(
            tool_name=action,
            arguments={"query": self.current_query}
        )
        
        success = result['success']
        latency = result['latency']
        is_relevant = any(rt['name'] == action for rt in self.relevant_tools)
        
        # Reward на основе реальных метрик
        reward = self._calculate_reward(tool, latency, success, is_relevant)
        
        done = (
            self.step_count >= self.config.rl.max_steps
            or (success and is_relevant)
        )
        
        info = {
            'latency': latency,
            'success': success,
            'is_relevant': is_relevant,
            'tool_used': action,
            'tool_category': tool.get('category', 'general'),
            'step': self.step_count,
            'response': result.get('result'),
            'error': result.get('error'),
            'semantic_score': self.tools.semantic_similarity(
                self.current_query, tool.get('name', '')
            )
        }
        
        return self._get_current_state(), reward, done, info
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Синхронная обёртка для совместимости"""
        return asyncio.run(self.step_async(action))
    
    def _calculate_reward(self, tool: Dict, latency: float, 
                         success: bool, is_relevant: bool) -> float:
        """Reward на основе реальных результатов"""
        reward = 0.0
        
        if success and is_relevant:
            reward += 3.0
            if self.step_count == 1:
                reward += 1.0
        elif success and not is_relevant:
            reward += 0.4
        else:
            reward -= 1.8
        
        # Штраф за высокую latency
        if latency > 0.6:
            reward -= (latency - 0.6) * 0.8
        
        # Штраф за количество шагов
        reward -= 0.08 * self.step_count
        
        return reward
    
    async def close(self):
        """Закрытие соединений"""
        await self.mcp_client.close()
```

---

## Шаг 4: Настройка MCP серверов

### Файл: `mcp_servers_config.json`

```json
{
  "servers": {
    "search.google": {
      "url": "http://localhost:3000",
      "description": "Google Search MCP Server"
    },
    "weather.openweathermap": {
      "url": "http://localhost:3001",
      "description": "Weather API MCP Server"
    },
    "database.postgresql": {
      "url": "http://localhost:3002",
      "description": "PostgreSQL MCP Server"
    }
  }
}
```

### Загрузка конфигурации

```python
# src/mcp/config_loader.py
import json
from typing import Dict

def load_mcp_servers(config_path: str = "mcp_servers_config.json") -> Dict[str, str]:
    """Загрузка конфигурации MCP серверов"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return {
        name: server['url']
        for name, server in config['servers'].items()
    }
```

---

## Шаг 5: Обновление Training Loop

### Файл: `src/rl/train_grpo_real.py`

```python
import asyncio
from src.rl.train_grpo import NetMCPTrainer
from src.environment.mcp_environment_real import RealMCPEnvironment
from src.mcp.config_loader import load_mcp_servers


class RealMCPTrainer(NetMCPTrainer):
    """Trainer с реальными MCP вызовами"""
    
    def __init__(self, config, mcp_servers_config: str = "mcp_servers_config.json"):
        super().__init__(config)
        
        # Заменяем эмулированное окружение на реальное
        mcp_servers = load_mcp_servers(mcp_servers_config)
        self.env = RealMCPEnvironment(config, mcp_servers)
    
    def _make_group(self, prompt: Dict) -> Dict:
        """Сбор роллаутов с РЕАЛЬНЫМИ вызовами"""
        state, tools, scores = self._get_tools_state(prompt)
        input_ids = self._encode_context(state)
        tool_embs = self._build_tool_embs(tools)
        
        rollouts = []
        
        # Асинхронный сбор роллаутов
        async def collect_rollout():
            with torch.inference_mode():
                logits = self._forward_with_embs(input_ids, tool_embs, scores)
                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1e-8)
                probs = probs / (probs.sum() + 1e-8)
                
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()
                
                tool_name = tools[idx.item()]
                self.env.reset(prompt)
                
                # РЕАЛЬНЫЙ вызов через MCP
                _, _, _, info = await self.env.step_async(tool_name)
                
                reward = self.reward_fn.compute_outcome_reward(
                    success=info.get("success", False),
                    steps=1,
                    is_relevant=info.get("is_relevant", False),
                    latency=info.get("latency", 0.0),
                    semantic_score=info.get("semantic_score", 0.0),
                )
                
                return {
                    "tool_idx": idx.item(),
                    "reward": reward,
                    "success": info.get("success", False),
                    "is_relevant": info.get("is_relevant", False),
                }
        
        # Собираем роллауты параллельно
        loop = asyncio.get_event_loop()
        tasks = [collect_rollout() for _ in range(self.p.grpo_group_size)]
        rollouts = loop.run_until_complete(asyncio.gather(*tasks))
        
        # GRPO advantage
        rews = [r["reward"] for r in rollouts]
        mean_r = sum(rews) / len(rews)
        std_r = (sum((x - mean_r) ** 2 for x in rews) / len(rews)) ** 0.5
        for r in rollouts:
            adv = r["reward"] - mean_r
            r["advantage"] = adv / (std_r + 1e-8) if std_r > 1e-8 else 0.0
        
        return {
            "input_ids": input_ids,
            "tool_embs": tool_embs,
            "semantic_scores": scores,
            "rollouts": rollouts,
            "adv_std": std_r,
        }
```

---

## Шаг 6: Запуск с реальным MCP

### Обновлённый main.py

```python
# main.py
import argparse
from src.config import Config
from src.rl.train_grpo_real import RealMCPTrainer  # новый импорт

def main():
    parser = argparse.ArgumentParser(description="NetMCP-RL with Real MCP")
    parser.add_argument("--mode", default="train", 
                       choices=["train", "evaluate", "interactive"])
    parser.add_argument("--use-real-mcp", action="store_true",
                       help="Use real MCP servers instead of emulation")
    parser.add_argument("--mcp-config", default="mcp_servers_config.json",
                       help="Path to MCP servers configuration")
    # ... остальные аргументы
    
    args = parser.parse_args()
    config = Config()
    config.load_data()
    
    if args.use_real_mcp:
        print("Using REAL MCP servers")
        trainer = RealMCPTrainer(config, mcp_servers_config=args.mcp_config)
    else:
        print("Using EMULATED network")
        trainer = NetMCPTrainer(config)
    
    if args.mode == "train":
        trainer.train()
    # ...
```

### Запуск

```bash
# С эмуляцией (как раньше)
python main.py --mode train --epochs 30

# С реальным MCP
python main.py --mode train --epochs 30 --use-real-mcp --mcp-config mcp_servers_config.json
```

---

## Преимущества реального MCP

### 1. Реальные метрики
- ✅ Настоящая latency из сети
- ✅ Реальные ошибки и таймауты
- ✅ Фактическая доступность серверов

### 2. Реалистичное обучение
- ✅ Модель учится на реальных условиях
- ✅ Адаптация к настоящим проблемам сети
- ✅ Нет разрыва между обучением и продакшеном

### 3. Стандартизация
- ✅ MCP — открытый протокол
- ✅ Совместимость с любыми MCP серверами
- ✅ Легко добавлять новые инструменты

### 4. Масштабируемость
- ✅ Можно использовать сотни реальных серверов
- ✅ Распределённая архитектура
- ✅ Горизонтальное масштабирование

---

## Пример MCP сервера (Node.js)

### Простой MCP сервер для поиска

```javascript
// mcp-server-search/index.js
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

// MCP endpoint
app.post('/tools/search.google/call', async (req, res) => {
  const { arguments: args } = req.body;
  const query = args.query;
  
  try {
    // Реальный вызов Google Custom Search API
    const response = await axios.get('https://www.googleapis.com/customsearch/v1', {
      params: {
        key: process.env.GOOGLE_API_KEY,
        cx: process.env.GOOGLE_CX,
        q: query
      }
    });
    
    res.json({
      success: true,
      result: response.data.items.slice(0, 5)
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.listen(3000, () => {
  console.log('MCP Search Server running on port 3000');
});
```

---

## Следующие шаги

### 1. Минимальная интеграция
- [ ] Создать `src/mcp/mcp_client.py`
- [ ] Создать `src/environment/mcp_environment_real.py`
- [ ] Настроить 2-3 тестовых MCP сервера
- [ ] Запустить обучение с `--use-real-mcp`

### 2. Полная интеграция
- [ ] Подключить все 15000 инструментов через MCP
- [ ] Настроить load balancing для серверов
- [ ] Добавить мониторинг метрик (Prometheus/Grafana)
- [ ] Реализовать retry logic и circuit breakers

### 3. Продакшен
- [ ] Деплой MCP серверов в облако
- [ ] Настроить автоскейлинг
- [ ] Добавить rate limiting
- [ ] Интеграция с реальными API (Google, OpenWeather, etc.)

---

## Гибридный подход (рекомендуется)

Для начала можно использовать **гибридный подход**:

1. **Обучение**: эмуляция (быстро, дёшево)
2. **Валидация**: реальный MCP (проверка на реальных данных)
3. **Продакшен**: реальный MCP

```python
# Обучение на эмуляции
python main.py --mode train --epochs 30

# Валидация на реальном MCP
python main.py --mode evaluate --checkpoint checkpoints/best --use-real-mcp

# Если метрики хорошие → деплой
```

---

## Итог

Интеграция реального MCP даст:
- ✅ Реальные метрики вместо эмуляции
- ✅ Стандартизированный протокол
- ✅ Готовность к продакшену
- ✅ Масштабируемость

Начните с минимальной интеграции (2-3 сервера) и постепенно расширяйте.
