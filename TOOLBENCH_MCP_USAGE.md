# Использование ToolBench датасета: Сейчас vs С MCP

## 📊 Что такое ToolBench датасет

**ToolBench** — это датасет от HuggingFace с:
- **80,000 промптов** (запросов пользователей)
- **15,000+ инструментов** (API и функций)
- **Метаданные**: категории, описания, параметры
- **Ground truth**: какой инструмент правильный для каждого запроса

### Структура датасета

```python
{
    "query": "What's the weather in Moscow?",
    "query_id": "12345",
    "domain": "weather",
    "api_list": [
        {
            "tool_name": "openweathermap",
            "api_name": "get_current_weather",
            "api_description": "Get current weather for a city",
            "category_name": "Weather",
            "required_parameters": ["city"],
            "optional_parameters": ["units"],
            "method": "GET"
        },
        {
            "tool_name": "weatherapi",
            "api_name": "current",
            "api_description": "Current weather data",
            ...
        }
    ],
    "answer": {
        "tool_name": "openweathermap",
        "api_name": "get_current_weather"
    }
}
```

---

## 🔄 Как датасет используется СЕЙЧАС (эмуляция)

### 1. Загрузка данных
```python
# src/data/toolbench_loader.py
loader = ToolBenchLoader(split="train", sample_size=80000)

# Извлекаем инструменты
tools = loader.tools  # 15,000 инструментов
# [
#   {"name": "openweathermap.get_current_weather", "description": "...", ...},
#   {"name": "google.search", "description": "...", ...},
#   ...
# ]

# Извлекаем промпты
prompts = loader.get_training_prompts()  # 80,000 промптов
# [
#   {"query": "What's the weather?", "relevant_tools": [...], ...},
#   {"query": "Search for Python", "relevant_tools": [...], ...},
#   ...
# ]
```

### 2. Обучение (эмуляция)
```python
for epoch in range(num_epochs):
    for prompt in prompts:
        # 1. Модель выбирает инструмент
        state = env.reset(prompt)
        tools_list = state['tools']  # top-20 кандидатов
        
        # 2. Модель делает предсказание
        predicted_tool = model.predict(prompt['query'], tools_list)
        
        # 3. ЭМУЛЯЦИЯ вызова
        # ❌ НЕ вызываем реальный API
        # ❌ Генерируем фейковую latency
        latency = random.uniform(0.1, 0.5)
        
        # ❌ Фейковый ответ
        response = f"Fake response from {predicted_tool}"
        
        # 4. Проверяем правильность
        is_correct = (predicted_tool in prompt['relevant_tools'])
        
        # 5. Reward
        reward = compute_reward(is_correct, latency)
        
        # 6. Обновление модели
        model.update(reward)
```

**Проблема:** Модель учится выбирать правильные инструменты, но:
- ❌ Не видит реальные ответы
- ❌ Не получает реальные метрики
- ❌ Не адаптируется к реальным проблемам сети

---

## ✅ Как датасет будет использоваться С MCP

### Вариант 1: Гибридный подход (рекомендуется)

**Идея:** Используем ToolBench для обучения, но добавляем реальные MCP вызовы для важных инструментов.

```python
class HybridToolBenchEnvironment:
    def __init__(self, config, mcp_client=None):
        # Загружаем ToolBench
        self.loader = ToolBenchLoader(split="train", sample_size=80000)
        self.all_tools = self.loader.tools  # 15,000 инструментов
        
        # Маппинг: какие инструменты доступны через MCP
        self.mcp_available = {
            "google.search": True,
            "openweathermap.get_current_weather": True,
            "github.search_code": True,
            "slack.send_message": True,
            # ... 20-50 реальных инструментов
        }
        
        self.mcp_client = mcp_client
    
    async def step(self, tool_name, query):
        # Проверяем, доступен ли инструмент через MCP
        if tool_name in self.mcp_available and self.mcp_client:
            # ✅ РЕАЛЬНЫЙ вызов через MCP
            result = await self.mcp_client.call_tool(tool_name, {"query": query})
            latency = result['latency']  # реальная
            response = result['result']  # реальный ответ
            success = result['success']  # реальный статус
        else:
            # ❌ Эмуляция для остальных
            latency = random.uniform(0.1, 0.5)
            response = f"Emulated response from {tool_name}"
            success = random.random() > 0.1
        
        return response, latency, success
```

**Преимущества:**
- ✅ Реальные метрики от важных инструментов
- ✅ Быстрое обучение (не ждём 15k API вызовов)
- ✅ Разнообразие (эмуляция даёт больше вариантов)

---

### Вариант 2: Полная замена на MCP

**Идея:** Создаём MCP серверы для всех инструментов из ToolBench.

```python
# Шаг 1: Маппинг ToolBench → MCP серверы
toolbench_to_mcp = {
    # ToolBench name → MCP server
    "openweathermap.get_current_weather": {
        "server": "weather-mcp-server",
        "tool": "weather.get_current",
        "api_key_env": "OPENWEATHER_API_KEY"
    },
    "google.search": {
        "server": "google-search-mcp-server",
        "tool": "search.google",
        "api_key_env": "GOOGLE_API_KEY"
    },
    # ... маппинг для всех 15k инструментов
}

# Шаг 2: Создать MCP серверы
for tool_name, config in toolbench_to_mcp.items():
    mcp_client.register_server(
        tool_name=tool_name,
        server_url=config['server'],
        api_key=os.getenv(config['api_key_env'])
    )

# Шаг 3: Обучение с реальными вызовами
for prompt in prompts:
    predicted_tool = model.predict(prompt['query'])
    
    # ✅ РЕАЛЬНЫЙ вызов через MCP
    result = await mcp_client.call_tool(
        predicted_tool,
        {"query": prompt['query']}
    )
```

**Проблемы:**
- ❌ Дорого (15k API ключей)
- ❌ Медленно (реальные вызовы)
- ❌ Rate limits

---

### Вариант 3: ToolBench как ground truth + MCP для валидации

**Идея:** Обучаемся на ToolBench (эмуляция), валидируемся на MCP (реально).

```python
# Обучение (быстро, эмуляция)
for epoch in range(30):
    for prompt in train_prompts:
        # Эмуляция
        predicted_tool = model.predict(prompt['query'])
        is_correct = (predicted_tool in prompt['relevant_tools'])
        reward = compute_reward(is_correct, emulated_latency)
        model.update(reward)

# Валидация (медленно, реально)
for prompt in val_prompts:
    predicted_tool = model.predict(prompt['query'])
    
    # ✅ РЕАЛЬНЫЙ вызов через MCP
    if predicted_tool in mcp_available:
        result = await mcp_client.call_tool(predicted_tool, {"query": prompt['query']})
        real_latency = result['latency']
        real_success = result['success']
        
        # Сравниваем с ground truth
        is_correct = (predicted_tool in prompt['relevant_tools'])
        
        print(f"Tool: {predicted_tool}")
        print(f"  Correct: {is_correct}")
        print(f"  Real latency: {real_latency}")
        print(f"  Real success: {real_success}")
```

---

## 🎯 Рекомендуемый подход для вашего проекта

### Этап 1: Обучение на ToolBench (эмуляция)

```python
# Используем весь ToolBench для обучения
loader = ToolBenchLoader(split="train", sample_size=80000)
tools = loader.tools  # 15,000 инструментов
prompts = loader.get_training_prompts()  # 80,000 промптов

# Обучение с эмуляцией (быстро)
trainer = NetMCPTrainer(config)
trainer.train(epochs=30)
```

**Результат:** Модель научилась выбирать релевантные инструменты

---

### Этап 2: Добавляем 20-50 реальных MCP инструментов

```python
# Выбираем самые популярные инструменты из ToolBench
top_tools = [
    "google.search",
    "openweathermap.get_current_weather",
    "github.search_code",
    "slack.send_message",
    "postgres.query",
    # ... ещё 15-45
]

# Создаём MCP серверы для них
mcp_servers = {}
for tool in top_tools:
    mcp_servers[tool] = create_mcp_server(tool)

# Дообучение с реальными вызовами
trainer = HybridMCPTrainer(config, mcp_servers)
trainer.train(epochs=10)  # дообучение
```

**Результат:** Модель адаптировалась к реальным метрикам

---

### Этап 3: Валидация на реальных MCP

```python
# Оценка на реальных инструментах
evaluator = RealMCPEvaluator(config, mcp_servers)
results = evaluator.evaluate(num_episodes=200)

print(f"Relevance@1: {results['relevance']}")
print(f"Real avg latency: {results['avg_latency']}")
print(f"Real success rate: {results['success_rate']}")
```

---

## 📋 Конкретный план интеграции

### Шаг 1: Анализ ToolBench инструментов

```python
# Найти самые популярные инструменты
from collections import Counter

tool_usage = Counter()
for prompt in loader.get_training_prompts():
    for tool in prompt['relevant_tools']:
        tool_usage[tool['name']] += 1

# Топ-50 инструментов
top_50 = tool_usage.most_common(50)
print("Top 50 tools:")
for tool_name, count in top_50:
    print(f"  {tool_name}: {count} uses")
```

### Шаг 2: Создать маппинг ToolBench → MCP

```python
# toolbench_mcp_mapping.json
{
    "google.search": {
        "mcp_server": "@modelcontextprotocol/server-google-search",
        "available": true,
        "api_key_required": true
    },
    "openweathermap.get_current_weather": {
        "mcp_server": "custom-weather-server",
        "available": true,
        "api_key_required": true
    },
    "github.search_code": {
        "mcp_server": "@modelcontextprotocol/server-github",
        "available": true,
        "api_key_required": true
    },
    # ... для топ-50
}
```

### Шаг 3: Реализовать гибридное окружение

```python
# src/environment/hybrid_environment.py
class HybridToolBenchEnvironment:
    def __init__(self, config, mcp_mapping_path="toolbench_mcp_mapping.json"):
        # Загружаем ToolBench
        self.loader = ToolBenchLoader(split="train", sample_size=80000)
        
        # Загружаем маппинг
        with open(mcp_mapping_path) as f:
            self.mcp_mapping = json.load(f)
        
        # Инициализируем MCP клиент
        self.mcp_client = MCPClient()
        
        # Подключаем доступные MCP серверы
        for tool_name, config in self.mcp_mapping.items():
            if config['available']:
                self.mcp_client.register_server(
                    tool_name,
                    config['mcp_server']
                )
    
    async def step(self, tool_name, query):
        if tool_name in self.mcp_mapping and self.mcp_mapping[tool_name]['available']:
            # Реальный вызов
            return await self._real_call(tool_name, query)
        else:
            # Эмуляция
            return self._emulated_call(tool_name, query)
```

### Шаг 4: Обновить training loop

```python
# main.py
def main():
    parser.add_argument("--use-hybrid", action="store_true",
                       help="Use hybrid ToolBench + MCP")
    
    if args.use_hybrid:
        trainer = HybridMCPTrainer(config)
    else:
        trainer = NetMCPTrainer(config)
    
    trainer.train()
```

---

## 📊 Сравнение подходов

| Подход | Скорость | Реализм | Стоимость | Сложность |
|--------|----------|---------|-----------|-----------|
| **Только эмуляция** | ⚡⚡⚡ Быстро | ❌ Низкий | 💰 Бесплатно | ✅ Просто |
| **Гибрид (20-50 MCP)** | ⚡⚡ Средне | ✅ Высокий | 💰💰 Средне | ⚡ Средне |
| **Полный MCP (15k)** | ❌ Медленно | ✅✅ Максимум | 💰💰💰 Дорого | ❌ Сложно |

---

## 🎯 Итоговая рекомендация

### Для вашего проекта:

1. **Обучение (эпохи 1-30):** Только ToolBench эмуляция
   - Быстро
   - Модель учится выбирать релевантные инструменты

2. **Дообучение (эпохи 31-40):** Гибрид (20-50 реальных MCP)
   - Модель адаптируется к реальным метрикам
   - Учится работать с настоящими ошибками

3. **Валидация:** Только реальные MCP
   - Проверка на реальных данных
   - Измерение реальной производительности

### Команды:

```bash
# Этап 1: Обучение на эмуляции
python main.py --mode train --epochs 30

# Этап 2: Дообучение с MCP
python main.py --mode train --epochs 10 --use-hybrid --checkpoint checkpoints/30

# Этап 3: Валидация на MCP
python main.py --mode evaluate --use-hybrid --checkpoint checkpoints/best
```

**Результат:** Лучшее из двух миров — быстрое обучение + реальная адаптация!

Хотите, чтобы я реализовал гибридный подход с конкретными 20 MCP серверами?
