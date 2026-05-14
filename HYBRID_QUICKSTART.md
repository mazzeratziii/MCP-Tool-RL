# Гибридный режим — Быстрый старт

## ✅ Что реализовано

Гибридный режим, который использует:
- **5 реальных MCP инструментов** (без API ключей)
- **~14,995 эмулированных инструментов** из ToolBench

### Реальные MCP инструменты:
1. `calculator.evaluate` — математические вычисления
2. `wikipedia.search` — поиск в Wikipedia
3. `wikipedia.get_article` — получение статей Wikipedia
4. `http.get` — HTTP GET запросы
5. `http.post` — HTTP POST запросы

---

## 🚀 Запуск

### 1. Тестовый запуск (1 эпоха)
```bash
cd C:\Users\allmute\PycharmProjects\MCP-Tool-RL-v2
python main.py --mode train --epochs 1 --use-hybrid
```

**Что произойдёт:**
- Запустятся 3 MCP сервера (calculator, wikipedia, http_client)
- 5 инструментов будут вызываться реально
- Остальные ~14,995 будут эмулироваться
- Обучение на 1 эпохе (~15-20 минут)

---

### 2. Полное обучение
```bash
# Desktop (8GB VRAM)
python main.py --mode train --epochs 30 --use-hybrid --profile desktop

# Laptop (4GB VRAM)
python main.py --mode train --epochs 30 --use-hybrid --profile laptop --batch-size 4
```

---

### 3. Оценка
```bash
python main.py --mode evaluate --use-hybrid --checkpoint checkpoints/best --eval-episodes 200
```

**Новые метрики:**
- Показывает, сколько раз использовались реальные MCP инструменты
- Реальная latency от MCP вызовов
- Сравнение эмулированных vs реальных метрик

---

### 4. Интерактивный режим
```bash
python main.py --mode interactive --use-hybrid --checkpoint checkpoints/best
```

**Попробуйте:**
```
>>> 2 + 2 * 5
# Должен вызвать calculator.evaluate (реальный MCP)

>>> what is Python programming language
# Должен вызвать wikipedia.search (реальный MCP)

>>> /exit
```

---

## 📊 Как проверить, что MCP работает

### Тест 1: Запуск MCP серверов
```bash
python -c "
from src.mcp.simple_client import SimpleMCPClient

client = SimpleMCPClient()
client.register_server('calculator', ['python', '-m', 'mcp_servers.calculator'])
print('✓ Calculator server started')

result = client.call_tool('calculator.evaluate', {'expression': '2+2'})
print(f'Result: {result}')

client.close()
"
```

**Ожидается:**
```
✓ MCP server 'calculator' started
Result: {'success': True, 'error': None, 'latency': 0.05, 'result': {'success': True, 'error': None, 'result': 4.0}}
✓ MCP server 'calculator' stopped
```

---

### Тест 2: Wikipedia
```bash
python -c "
from src.mcp.simple_client import SimpleMCPClient

client = SimpleMCPClient()
client.register_server('wikipedia', ['python', '-m', 'mcp_servers.wikipedia'])

result = client.call_tool('wikipedia.search', {'query': 'Python programming'})
print(f'Success: {result[\"success\"]}')
print(f'Latency: {result[\"latency\"]:.3f}s')

client.close()
"
```

---

### Тест 3: Гибридное окружение
```bash
python -c "
from src.config import Config
from src.environment.hybrid_environment import HybridMCPEnvironment

config = Config()
config.load_data()

env = HybridMCPEnvironment(config, 'mcp_config.json')
print(f'MCP tools: {len(env.mcp_tools)}')
print(f'Tools: {list(env.mcp_tools)}')

env.close()
"
```

**Ожидается:**
```
INITIALIZING MCP SERVERS
============================================================
✓ MCP server 'calculator' started
✓ MCP server 'wikipedia' started
✓ MCP server 'http_client' started

✓ Initialized 5 MCP tools
  MCP tools: ['calculator.evaluate', 'wikipedia.search', ...]
  Emulated tools: 14995
============================================================

MCP tools: 5
Tools: ['calculator.evaluate', 'wikipedia.search', 'wikipedia.get_article', 'http.get', 'http.post']
```

---

## 🔍 Отличия от эмуляции

### Эмуляция (--use-hybrid НЕ указан)
```bash
python main.py --mode train --epochs 30
```
- ❌ Все инструменты эмулируются
- ❌ Фейковая latency
- ❌ Фейковые ответы
- ✅ Быстро
- ✅ Не требует зависимостей

### Гибридный режим (--use-hybrid)
```bash
python main.py --mode train --epochs 30 --use-hybrid
```
- ✅ 5 инструментов реальные
- ✅ Реальная latency от MCP
- ✅ Реальные ответы (Wikipedia, HTTP, Calculator)
- ✅ 14,995 инструментов эмулируются (для разнообразия)
- ⚠️ Немного медленнее (реальные вызовы)
- ⚠️ Требует requests (уже в requirements.txt)

---

## 📈 Ожидаемые результаты

### Метрики обучения
При использовании гибридного режима вы увидите:

```
--- Epoch 1 [scenario: normal] ---
    [resample → 1500 queries]
  loss=1.234  reward=2.456  success=65%  relevance=58%
  MCP calls: 45 (3% of rollouts)  ← НОВОЕ
  MCP avg latency: 0.234s         ← НОВОЕ
  ...
```

### Evaluation метрики
```
EVALUATION [desktop] — 200 ep, mode=controlled
============================================================
  Episodes:           200
  Success rate:       68%
  Relevance@1:        62%
  
  Network Adaptation Metrics:
  Avg latency:        0.187s
  Fast tool choices:  74%
  Available choices:  97%
  
  MCP Statistics:                 ← НОВОЕ
  MCP tools used:     12 times    ← НОВОЕ
  MCP success rate:   100%        ← НОВОЕ
  MCP avg latency:    0.156s      ← НОВОЕ
```

---

## 🛠️ Troubleshooting

### Ошибка: "No module named 'mcp_servers'"
```bash
# Убедитесь, что вы в правильной директории
cd C:\Users\allmute\PycharmProjects\MCP-Tool-RL-v2

# Проверьте, что папка существует
ls mcp_servers/
```

### Ошибка: "MCP server failed to start"
```bash
# Проверьте Python
python --version  # Должно быть 3.10+

# Проверьте зависимости
pip install requests
```

### MCP серверы не отвечают
```bash
# Проверьте, что серверы запускаются вручную
python -m mcp_servers.calculator
# Введите: {"method": "tools/list", "params": {}}
# Должен вернуть список инструментов
```

### Медленное обучение
```bash
# Отключите MCP для быстрого обучения
python main.py --mode train --epochs 30
# (без --use-hybrid)

# Или уменьшите batch size
python main.py --mode train --epochs 30 --use-hybrid --batch-size 4
```

---

## 📝 Следующие шаги

### 1. Протестируйте гибридный режим
```bash
python main.py --mode train --epochs 1 --use-hybrid
```

### 2. Сравните с эмуляцией
```bash
# Эмуляция
python main.py --mode train --epochs 10
python main.py --mode evaluate --checkpoint checkpoints/best

# Гибрид
python main.py --mode train --epochs 10 --use-hybrid
python main.py --mode evaluate --checkpoint checkpoints/best --use-hybrid
```

### 3. Добавьте больше MCP инструментов
См. `API_KEYS_GUIDE.md` для инструментов с API ключами

---

## 🎯 Итог

Гибридный режим готов к использованию:
- ✅ 5 реальных MCP инструментов (без API ключей)
- ✅ Работает из коробки
- ✅ Совместим со всеми режимами (train/evaluate/interactive)
- ✅ Можно расширить до 20+ инструментов (с API ключами)

**Запустите:**
```bash
python main.py --mode train --epochs 1 --use-hybrid
```
