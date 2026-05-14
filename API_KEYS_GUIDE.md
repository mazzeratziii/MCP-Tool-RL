# Где получить API ключи для MCP инструментов

## 🔑 Инструменты, требующие API ключи

### 1. **Brave Search API** (для поиска)
**Зачем:** Замена Google Search, более приватный и без ограничений

**Где получить:**
1. Перейти на https://brave.com/search/api/
2. Зарегистрироваться (бесплатно)
3. Создать API ключ

**Бесплатный план:**
- 2,000 запросов в месяц бесплатно
- Потом $5 за 1,000 запросов

**В .env:**
```bash
BRAVE_API_KEY=your_brave_api_key_here
```

---

### 2. **OpenWeather API** (для погоды)
**Зачем:** Получение данных о погоде

**Где получить:**
1. Перейти на https://openweathermap.org/api
2. Зарегистрироваться (бесплатно)
3. Получить API ключ в разделе "API keys"

**Бесплатный план:**
- 1,000 запросов в день бесплатно
- 60 запросов в минуту

**В .env:**
```bash
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

---

### 3. **GitHub Token** (для GitHub)
**Зачем:** Поиск кода, репозиториев, работа с issues

**Где получить:**
1. Перейти на https://github.com/settings/tokens
2. Generate new token (classic)
3. Выбрать scopes: `repo`, `read:org`, `read:user`

**Бесплатный план:**
- 5,000 запросов в час (authenticated)
- Без токена: 60 запросов в час

**В .env:**
```bash
GITHUB_TOKEN=ghp_your_github_token_here
```

---

### 4. **Slack Bot Token** (опционально)
**Зачем:** Отправка сообщений в Slack

**Где получить:**
1. Перейти на https://api.slack.com/apps
2. Create New App
3. Install App to Workspace
4. Скопировать Bot User OAuth Token

**В .env:**
```bash
SLACK_BOT_TOKEN=xoxb-your-slack-token-here
```

---

## ✅ Инструменты БЕЗ API ключей (работают сразу)

### 1. **Filesystem** (чтение/запись файлов)
- Работает локально
- Не требует API ключей
- Доступ к файловой системе

### 2. **Memory** (хранение данных)
- Работает локально
- Хранит данные в памяти
- Не требует API ключей

### 3. **Calculator** (вычисления)
- Работает локально
- Математические операции
- Не требует API ключей

### 4. **Time** (время и дата)
- Работает локально
- Системное время
- Не требует API ключей

### 5. **Wikipedia** (через библиотеку)
- Использует публичный API Wikipedia
- Не требует регистрации
- Бесплатно

### 6. **HTTP Client** (HTTP запросы)
- Работает локально
- Делает HTTP запросы
- Не требует API ключей

### 7. **JSON Tools** (парсинг JSON)
- Работает локально
- Парсинг и форматирование JSON
- Не требует API ключей

---

## 🎯 Рекомендуемая конфигурация для старта

### Минимальная (без API ключей)
Используйте только инструменты, не требующие API:

```json
{
  "enabled_tools": [
    "filesystem.read_file",
    "filesystem.write_file",
    "filesystem.list_directory",
    "memory.store",
    "memory.retrieve",
    "time.current",
    "calculator.evaluate",
    "wikipedia.search",
    "wikipedia.get_article",
    "http.get",
    "http.post",
    "json.parse"
  ]
}
```

**Итого: 12 реальных MCP инструментов без API ключей**

---

### Базовая (с бесплатными API)
Добавьте бесплатные API:

```bash
# .env
BRAVE_API_KEY=your_brave_key  # 2,000 запросов/месяц бесплатно
OPENWEATHER_API_KEY=your_weather_key  # 1,000 запросов/день бесплатно
GITHUB_TOKEN=your_github_token  # 5,000 запросов/час бесплатно
```

**Итого: 17 реальных MCP инструментов**

---

### Полная (с платными API)
Добавьте платные сервисы:

```bash
# .env
SLACK_BOT_TOKEN=your_slack_token
POSTGRES_URL=postgresql://localhost/mydb
```

**Итого: 20 реальных MCP инструментов**

---

## 📝 Обновлённый .env файл

```bash
# ============================================================
# Существующие настройки (для LLM)
# ============================================================
MODEL_NAME=your-model-name
BASE_URL=your-api-url
API_TOKEN=your-api-token
SYSTEM_PROMPT=You are a helpful AI assistant.
USER_PROMPT=
MAX_CONCURRENT_REQUESTS=100
MIN_REQUEST_TIMEOUT=60.0

# ============================================================
# MCP API Keys (для гибридного режима)
# ============================================================

# Brave Search (бесплатно: 2,000 запросов/месяц)
# Получить: https://brave.com/search/api/
BRAVE_API_KEY=

# OpenWeather (бесплатно: 1,000 запросов/день)
# Получить: https://openweathermap.org/api
OPENWEATHER_API_KEY=

# GitHub (бесплатно: 5,000 запросов/час)
# Получить: https://github.com/settings/tokens
GITHUB_TOKEN=

# Slack (опционально)
# Получить: https://api.slack.com/apps
SLACK_BOT_TOKEN=

# PostgreSQL (опционально)
POSTGRES_URL=postgresql://localhost:5432/mydb
```

---

## 🚀 Быстрый старт БЕЗ API ключей

Если не хотите регистрироваться на сервисах, можно начать с инструментов, не требующих API:

### Шаг 1: Обновить mcp_config.json

```json
{
  "tools": {
    "filesystem.read_file": {"enabled": true},
    "filesystem.write_file": {"enabled": true},
    "memory.store": {"enabled": true},
    "memory.retrieve": {"enabled": true},
    "calculator.evaluate": {"enabled": true},
    "wikipedia.search": {"enabled": true},
    "http.get": {"enabled": true},
    "json.parse": {"enabled": true},
    
    "google.search": {"enabled": false},
    "weather.current": {"enabled": false},
    "github.search_code": {"enabled": false}
  }
}
```

### Шаг 2: Запустить обучение

```bash
python main.py --mode train --epochs 30 --use-hybrid
```

**Результат:** 8 реальных MCP инструментов + 14,992 эмулированных

---

## 💡 Альтернативы платным API

### Вместо Brave Search
- **DuckDuckGo API** (бесплатно, без ключа)
- **SerpAPI** (100 запросов/месяц бесплатно)
- **Bing Search API** (1,000 запросов/месяц бесплатно)

### Вместо OpenWeather
- **WeatherAPI.com** (1,000,000 запросов/месяц бесплатно)
- **Open-Meteo** (бесплатно, без ключа)

### Вместо GitHub
- **GitLab API** (бесплатно)
- **Bitbucket API** (бесплатно)

---

## 🎯 Рекомендация для вашего проекта

### Вариант 1: Начать без API ключей (самый простой)
```bash
# Используем только локальные инструменты
python main.py --mode train --epochs 30 --use-hybrid
```

**Плюсы:**
- ✅ Работает сразу
- ✅ Бесплатно
- ✅ Нет rate limits

**Минусы:**
- ❌ Меньше реальных инструментов (8 вместо 17)

---

### Вариант 2: Добавить бесплатные API (рекомендуется)
```bash
# 1. Получить API ключи (5 минут)
# - Brave Search: https://brave.com/search/api/
# - OpenWeather: https://openweathermap.org/api
# - GitHub: https://github.com/settings/tokens

# 2. Добавить в .env
echo "BRAVE_API_KEY=your_key" >> .env
echo "OPENWEATHER_API_KEY=your_key" >> .env
echo "GITHUB_TOKEN=your_token" >> .env

# 3. Запустить обучение
python main.py --mode train --epochs 30 --use-hybrid
```

**Плюсы:**
- ✅ 17 реальных инструментов
- ✅ Всё ещё бесплатно
- ✅ Реальные метрики от популярных API

**Минусы:**
- ⚠️ Нужно зарегистрироваться (5 минут)
- ⚠️ Rate limits (но щедрые)

---

## 📊 Сравнение вариантов

| Вариант | Реальных MCP | Эмулированных | API ключи | Время настройки |
|---------|--------------|---------------|-----------|-----------------|
| **Без API** | 8 | 14,992 | 0 | 0 минут |
| **Бесплатные API** | 17 | 14,983 | 3 | 5 минут |
| **Все API** | 20 | 14,980 | 5 | 15 минут |

---

## ✅ Что делать дальше?

### Если хотите начать СЕЙЧАС (без API):
```bash
# Просто запустите
python main.py --mode train --epochs 30 --use-hybrid
```

### Если хотите больше реализма (5 минут настройки):
1. Получите 3 бесплатных API ключа (ссылки выше)
2. Добавьте их в `.env`
3. Запустите обучение

### Если хотите максимум (15 минут настройки):
1. Получите все API ключи
2. Настройте Slack, PostgreSQL
3. Запустите обучение

Какой вариант выбираете?
