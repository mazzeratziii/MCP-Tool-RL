# Универсальная система Fallback инструментов

**Дата:** 2026-05-14  
**Цель:** Автоматическое определение и использование fallback инструментов для всех типов запросов

## Архитектура

### 1. QueryClassifier (`src/environment/query_classifier.py`)

Классификатор автоматически определяет тип запроса и подходящий fallback инструмент:

**Поддерживаемые категории:**
- **Математические запросы** → `Calculator.Evaluate`
  - Паттерны: `2 + 2`, `calculate 10 * 5`, `what is 100 / 4`
  - Confidence: 0.95

- **Простые вопросы** → `General.NoToolNeeded`
  - Паттерны: `hello`, `what is AI`, `explain machine learning`
  - Confidence: 0.85
  - Исключение: запросы, требующие внешних данных (погода, новости, цены)

### 2. Интеграция с ToolRegistry

`ToolRegistry.get_top_k_tools()` автоматически:
1. Получает top-k инструментов через семантический поиск
2. Классифицирует запрос через `QueryClassifier`
3. Добавляет fallback инструмент в начало списка, если он релевантен
4. Проверяет, нет ли уже похожего инструмента в кандидатах

**Преимущества:**
- ✅ Работает для всех существующих инструментов
- ✅ Автоматически применяется к новым инструментам
- ✅ Не требует изменений в коде при добавлении новых API
- ✅ Прозрачная интеграция с RL моделью

## Как добавить новый тип fallback

### Шаг 1: Добавить паттерны в QueryClassifier

```python
# В src/environment/query_classifier.py

class QueryClassifier:
    def __init__(self):
        # ... существующие паттерны ...

        # Новая категория: конвертация единиц
        self.unit_conversion_patterns = [
            r'\d+\s*(km|miles|meters|feet)\s+to\s+(km|miles|meters|feet)',
            r'convert \d+',
            r'how many \w+ in \d+',
        ]
```

### Шаг 2: Добавить метод классификации

```python
def _is_unit_conversion(self, query: str) -> bool:
    return any(re.search(pattern, query, re.IGNORECASE) 
               for pattern in self.unit_conversion_patterns)
```

### Шаг 3: Обновить метод classify

```python
def classify(self, query: str) -> Tuple[Optional[str], float]:
    query_lower = query.lower().strip()

    # Проверка на конвертацию единиц
    if self._is_unit_conversion(query_lower):
        return ("UnitConverter.Convert", 0.90)

    # ... остальные проверки ...
```

### Шаг 4: Добавить инструмент в config.py

```python
fallback_tools = [
    # ... существующие ...
    {
        "name": "UnitConverter.Convert",
        "category": "conversion",
        "description": "Convert between different units of measurement",
        "method": "GET",
        "required_parameters": [
            {"name": "value", "type": "number"},
            {"name": "from_unit", "type": "string"},
            {"name": "to_unit", "type": "string"}
        ],
    }
]
```

### Шаг 5: Добавить обработчик в mcp_environment.py

```python
def step(self, action: str):
    # ... существующий код ...

    if action == "UnitConverter.Convert":
        return self._handle_unit_converter(tool)

    # ... остальной код ...

def _handle_unit_converter(self, tool: Dict):
    # Логика конвертации
    # ...
    return state, reward, done, info
```

## Расширенные возможности

### Динамическое добавление паттернов

Можно загружать паттерны из конфигурационного файла:

```python
# config/fallback_patterns.json
{
  "Calculator.Evaluate": {
    "patterns": [
      "\\d+\\s*[\\+\\-\\*/]\\s*\\d+",
      "calculate|compute|solve"
    ],
    "confidence": 0.95
  },
  "UnitConverter.Convert": {
    "patterns": [
      "convert \\d+",
      "\\d+\\s*\\w+\\s+to\\s+\\w+"
    ],
    "confidence": 0.90
  }
}
```

```python
# В QueryClassifier.__init__
import json

with open('config/fallback_patterns.json') as f:
    self.patterns = json.load(f)
```

### ML-based классификация

Для более точной классификации можно использовать ML модель:

```python
from transformers import pipeline

class MLQueryClassifier(QueryClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def classify(self, query: str):
        # Сначала пробуем rule-based
        result = super().classify(query)
        if result[0] is not None:
            return result

        # Если не сработало, используем ML
        prediction = self.classifier(query)[0]
        # ... логика определения fallback инструмента ...
```

## Мониторинг и метрики

### Логирование использования fallback

```python
# В ToolRegistry.get_top_k_tools()
if fallback_info:
    print(f"[Fallback] Added {fallback_info['name']} "
          f"(confidence={fallback_info['confidence']:.2f}) "
          f"for query: {query[:50]}")
```

### Метрики качества

Отслеживайте:
- **Fallback usage rate** - как часто используются fallback инструменты
- **Fallback success rate** - насколько часто fallback выбор правильный
- **False positive rate** - когда fallback добавлен, но не нужен

```python
# В mcp_environment.py
def step(self, action: str):
    # ... существующий код ...

    if tool.get('is_fallback'):
        self.fallback_stats['used'] += 1
        if is_relevant:
            self.fallback_stats['success'] += 1
```

## Тестирование

### Unit тесты для классификатора

```python
# tests/test_query_classifier.py
import pytest
from src.environment.query_classifier import QueryClassifier

def test_math_classification():
    classifier = QueryClassifier()

    assert classifier.classify("2 + 2")[0] == "Calculator.Evaluate"
    assert classifier.classify("what is 10 * 5")[0] == "Calculator.Evaluate"
    assert classifier.classify("hello")[0] != "Calculator.Evaluate"

def test_simple_query_classification():
    classifier = QueryClassifier()

    assert classifier.classify("hello")[0] == "General.NoToolNeeded"
    assert classifier.classify("what is AI")[0] == "General.NoToolNeeded"

def test_external_data_exclusion():
    classifier = QueryClassifier()

    # Должен вернуть None, т.к. требует внешних данных
    assert classifier.classify("what is the weather")[0] is None
```

### Интеграционные тесты

```python
# tests/test_fallback_integration.py
def test_fallback_in_tool_selection():
    config = Config()
    config.load_data()
    registry = ToolRegistry(config)

    # Математический запрос должен включать Calculator
    tools = registry.get_top_k_tools("2 + 2", k=5)
    tool_names = [t['name'] for t in tools]
    assert "Calculator.Evaluate" in tool_names

    # Простой вопрос должен включать NoToolNeeded
    tools = registry.get_top_k_tools("hello", k=5)
    tool_names = [t['name'] for t in tools]
    assert "General.NoToolNeeded" in tool_names
```

## Преимущества универсального подхода

1. **Масштабируемость** - добавление новых инструментов не требует изменений в fallback логике
2. **Гибкость** - легко добавлять новые типы fallback через паттерны
3. **Прозрачность** - классификатор работает на уровне ToolRegistry, не затрагивая RL модель
4. **Производительность** - классификация выполняется один раз при получении кандидатов
5. **Тестируемость** - каждый компонент можно тестировать независимо

## Roadmap

- [ ] Добавить ML-based классификацию для сложных случаев
- [ ] Реализовать динамическую загрузку паттернов из конфига
- [ ] Добавить метрики и дашборд для мониторинга
- [ ] Создать UI для добавления новых fallback паттернов без кода
- [ ] Интегрировать с системой логирования для анализа ошибок
