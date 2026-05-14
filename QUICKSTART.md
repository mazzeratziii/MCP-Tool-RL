# MCP-Tool-RL v2 — Quick Start

## Что изменилось?

Модель теперь **видит и учитывает сетевые метрики** при выборе инструментов:
- ✅ Latency (задержка)
- ✅ Availability (доступность)
- ✅ Stability (стабильность)

## Быстрый старт

### 1. Обучение (Desktop с 8GB VRAM)
```bash
cd C:\Users\allmute\PycharmProjects\MCP-Tool-RL-v2
python main.py --mode train --epochs 30 --profile desktop
```

### 2. Обучение (Laptop с 4GB VRAM)
```bash
python main.py --mode train --epochs 30 --profile laptop --batch-size 4
```

### 3. Оценка модели
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200
```

### 4. Интерактивное тестирование
```bash
python main.py --mode interactive --checkpoint checkpoints/best
```

В интерактиве попробуйте:
```
>>> /network normal
>>> найди информацию про Python

>>> /network network_issues
>>> найди информацию про Python
# Модель должна выбрать доступный инструмент

>>> /network stats
>>> /eval 50
>>> /exit
```

## Ключевые улучшения

### 1. Промпт теперь содержит сетевые метрики
**Было:**
```
1. search.google
   Описание: Search the web
   Категория: search
```

**Стало:**
```
1. search.google
   Описание: Search the web
   Категория: search
   Статус: ✓ доступен | Задержка: 0.142s (быстрый) | Стабильность: 0.95
```

### 2. Reward учитывает адаптацию к сети
- **+0.8** за выбор инструмента на 20%+ быстрее среднего
- **-1.5** за выбор недоступного инструмента
- **+0.8** за выбор доступного при проблемах с сетью

### 3. Детерминированные сценарии
Автоматическая ротация каждые 5 эпох:
- **normal**: latency=0.15s, availability=98%
- **peak_hours**: latency=0.35s, availability=95%
- **network_issues**: latency=0.55s, availability=70%
- **optimal**: latency=0.08s, availability=99%

### 4. Новые метрики evaluation
- **Avg latency** — средняя задержка выбранных инструментов
- **Fast tool choices** — % выбора быстрых инструментов
- **Available choices** — % выбора доступных инструментов

## Сравнение с оригиналом

| Метрика | Оригинал | v2 (ожидается) |
|---------|----------|----------------|
| Relevance@1 | ~65% | ~65% (без потерь) |
| Fast tool choices | ~50% (случайно) | **~75%** (осознанно) |
| Available choices | ~95% | **~98%** |
| Avg latency | ~0.25s | **~0.18s** |

## Полная документация

- **CHANGES.md** — детальное описание изменений
- **USAGE_GUIDE.md** — все варианты запуска с примерами
- **README.md** — оригинальная документация

## Следующие шаги

1. **Обучите модель** на 30-40 эпохах
2. **Сравните метрики** с оригинальной версией (в `MCP-Tool-RL/`)
3. **Протестируйте** в интерактивном режиме с разными сценариями
4. **Оцените** на всех network modes (deterministic/controlled/stochastic)

## Проблемы?

### OOM (Out of Memory)
```bash
python main.py --mode train --batch-size 2 --profile laptop
```

### Медленное обучение
```bash
# Используйте меньше эпох для тестирования
python main.py --mode train --epochs 10
```

### Вопросы по использованию
Смотрите **USAGE_GUIDE.md** — там 100+ примеров команд с объяснениями
