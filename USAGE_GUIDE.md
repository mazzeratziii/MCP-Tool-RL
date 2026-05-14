# Полное руководство по запуску MCP-Tool-RL v2

## Содержание
1. [Train режим](#train-режим)
2. [Evaluate режим](#evaluate-режим)
3. [Interactive режим](#interactive-режим)
4. [Комбинации параметров](#комбинации-параметров)
5. [Типичные сценарии использования](#типичные-сценарии-использования)

---

## Train режим

### Базовые команды

#### 1. Обучение с настройками по умолчанию
```bash
python main.py --mode train
```
**Что происходит:**
- Эпохи: 30 (по умолчанию)
- Batch size: 8
- Network mode: deterministic
- Profile: автоопределение по VRAM
- Сценарии сети ротируются каждые 5 эпох (normal → peak_hours → network_issues → optimal)

**Когда использовать:** Быстрый старт для тестирования

---

#### 2. Обучение с указанием количества эпох
```bash
python main.py --mode train --epochs 40
```
**Что происходит:**
- Обучение на 40 эпохах вместо 30
- Больше времени для конвергенции
- Лучшее качество модели

**Когда использовать:** Для финального обучения, когда нужна максимальная точность

---

#### 3. Обучение с конкретным профилем железа

**Desktop (RTX 3070 8GB или аналог):**
```bash
python main.py --mode train --epochs 30 --profile desktop
```
**Параметры профиля:**
- LoRA rank: 16
- LoRA alpha: 32
- LoRA targets: q_proj, v_proj, k_proj, o_proj
- GRPO group size: 6
- Train set size: 1500 промптов
- Top-K tools: 20
- Max context: 512 токенов
- Entropy coeff: 0.02
- Sample temperature: 1.0

**Когда использовать:** Если у вас >= 7GB VRAM и нужна максимальная скорость + качество

---

**Laptop (4GB VRAM):**
```bash
python main.py --mode train --epochs 30 --profile laptop
```
**Параметры профиля:**
- LoRA rank: 4 (меньше параметров)
- LoRA alpha: 8
- LoRA targets: q_proj, v_proj (только 2 слоя)
- GRPO group size: 3 (меньше роллаутов)
- Train set size: 600 промптов
- Top-K tools: 15
- Max context: 256 токенов
- Entropy coeff: 0.02
- Sample temperature: 1.2 (больше exploration)
- Double quantization: включена

**Когда использовать:** Если у вас 4-6GB VRAM или нужна экономия памяти

---

#### 4. Обучение с изменённым batch size
```bash
python main.py --mode train --epochs 30 --batch-size 4
```
**Что происходит:**
- Batch size уменьшен до 4 (по умолчанию 8)
- Меньше использование VRAM
- Медленнее обучение, но стабильнее

**Когда использовать:** Если возникают OOM (Out of Memory) ошибки

```bash
python main.py --mode train --epochs 30 --batch-size 16
```
**Что происходит:**
- Batch size увеличен до 16
- Быстрее обучение
- Больше VRAM требуется

**Когда использовать:** Если у вас много VRAM (12GB+) и нужна скорость

---

#### 5. Продолжение обучения с чекпоинта
```bash
python main.py --mode train --epochs 20 --checkpoint checkpoints/10
```
**Что происходит:**
- Загружаются веса из `checkpoints/10/`
- Обучение продолжается ещё 20 эпох
- Полезно для дообучения

**Когда использовать:** 
- Модель недообучена, нужно продолжить
- Хотите дообучить на новых данных
- Прервали обучение и хотите возобновить

---

#### 6. Обучение с разными сетевыми режимами

**Deterministic (по умолчанию):**
```bash
python main.py --mode train --epochs 30 --network-mode deterministic
```
**Особенности:**
- Фиксированные, но разнообразные условия
- 4 сценария ротируются каждые 5 эпох
- Стабильное обучение, воспроизводимые результаты

**Когда использовать:** Основной режим для обучения

---

**Controlled:**
```bash
python main.py --mode train --epochs 30 --network-mode controlled
```
**Особенности:**
- Медленные, предсказуемые изменения сети
- Congestion меняется на ±5% каждый шаг
- Редкие отказы серверов (0.1%)

**Когда использовать:** Для обучения на более реалистичных, но контролируемых условиях

---

**Stochastic:**
```bash
python main.py --mode train --epochs 30 --network-mode stochastic
```
**Особенности:**
- Полностью случайные условия
- Высокая вариативность latency и availability
- Стресс-тест для модели

**Когда использовать:** 
- Для финального обучения на максимально разнообразных условиях
- Когда нужна устойчивость к любым сетевым проблемам

---

### Комбинированные команды для обучения

#### Быстрое обучение на laptop
```bash
python main.py --mode train --epochs 20 --batch-size 4 --profile laptop
```
**Результат:** Экономия памяти + быстрое обучение для тестирования

---

#### Максимальное качество на desktop
```bash
python main.py --mode train --epochs 50 --batch-size 8 --profile desktop --network-mode stochastic
```
**Результат:** Лучшая модель с максимальной устойчивостью к сетевым проблемам

---

#### Дообучение с изменением условий
```bash
python main.py --mode train --epochs 10 --checkpoint checkpoints/best --network-mode controlled
```
**Результат:** Адаптация уже обученной модели к более реалистичным условиям

---

## Evaluate режим

### Базовые команды

#### 1. Оценка с настройками по умолчанию
```bash
python main.py --mode evaluate
```
**Что происходит:**
- 200 эпизодов оценки (по умолчанию)
- Network mode: controlled
- Используется базовая модель (без чекпоинта)

**Метрики:**
- Success rate
- Relevance@1 (top-1 accuracy)
- Relevance@3 (top-3 accuracy)
- Avg latency
- Fast tool choices (% выбора быстрых инструментов)
- Available choices (% выбора доступных инструментов)

---

#### 2. Оценка обученной модели
```bash
python main.py --mode evaluate --checkpoint checkpoints/best
```
**Когда использовать:** После обучения для проверки качества

---

#### 3. Оценка с разным количеством эпизодов
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 500
```
**Что происходит:**
- Оценка на 500 промптах вместо 200
- Более точная статистика
- Дольше выполняется

**Когда использовать:** Для финальной оценки перед деплоем

---

#### 4. Оценка в разных сетевых режимах

**Deterministic:**
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode deterministic
```
**Что тестируется:** Поведение в фиксированных сценариях (normal/peak_hours/network_issues/optimal)

---

**Controlled:**
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode controlled
```
**Что тестируется:** Адаптация к медленным изменениям сети

---

**Stochastic:**
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode stochastic
```
**Что тестируется:** Устойчивость к случайным сетевым проблемам (стресс-тест)

---

### Комбинированные команды для оценки

#### Полная оценка модели
```bash
# 1. Deterministic
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode deterministic --eval-episodes 300

# 2. Controlled
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode controlled --eval-episodes 300

# 3. Stochastic
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode stochastic --eval-episodes 300
```
**Результат:** Полная картина поведения модели в разных условиях

---

#### Быстрая проверка
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 50
```
**Результат:** Быстрая оценка за ~1-2 минуты

---

## Interactive режим

### Базовые команды

#### 1. Интерактив с базовой моделью
```bash
python main.py --mode interactive
```
**Что происходит:**
- Загружается базовая модель (без дообучения)
- Можно вводить запросы и видеть выбор инструментов
- Доступны команды для управления сетью

**Когда использовать:** Для тестирования базовой модели

---

#### 2. Интерактив с обученной моделью
```bash
python main.py --mode interactive --checkpoint checkpoints/best
```
**Что происходит:**
- Загружается лучшая обученная модель
- Можно сравнить с базовой моделью

**Когда использовать:** Основной режим для демонстрации и тестирования

---

#### 3. Интерактив с конкретным чекпоинтом
```bash
python main.py --mode interactive --checkpoint checkpoints/40
```
**Когда использовать:** Для сравнения разных версий модели

---

### Команды внутри Interactive режима

После запуска интерактивного режима доступны следующие команды:

#### Управление сетью

**Переключение сценария:**
```
>>> /network normal
```
Устанавливает сценарий "normal" (latency=0.15s, availability=98%)

```
>>> /network peak_hours
```
Устанавливает сценарий "peak_hours" (latency=0.35s, availability=95%)

```
>>> /network network_issues
```
Устанавливает сценарий "network_issues" (latency=0.55s, availability=70%)

```
>>> /network optimal
```
Устанавливает сценарий "optimal" (latency=0.08s, availability=99%)

---

**Просмотр статистики сети:**
```
>>> /network stats
```
**Вывод:**
```
mode: deterministic
base_latency: 0.15
congestion: 1.0
packet_loss: 0.02
jitter: 0.01
active_servers: 145
total_servers: 150
```

---

#### Быстрая оценка

```
>>> /eval 50
```
Запускает оценку на 50 валидационных промптах с текущими настройками сети

```
>>> /eval 100
```
Оценка на 100 промптах (дольше, но точнее)

---

#### Тестирование запросов

**Простой запрос:**
```
>>> найди информацию про Python
```
**Вывод:**
```
Top tools for: 'найди информацию про Python'
--------------------------------------------------
  1. search.google
     model_prob=0.856  semantic=0.912
  2. search.stackoverflow
     model_prob=0.089  semantic=0.834
  3. docs.python.org
     model_prob=0.034  semantic=0.798
  4. github.search
     model_prob=0.015  semantic=0.756
  5. wikipedia.search
     model_prob=0.006  semantic=0.723

Executed: search.google
Success:  True
Latency:  0.142s
Response: [результат выполнения]
--------------------------------------------------
```

---

**Запрос с проблемной сетью:**
```
>>> /network network_issues
>>> найди информацию про Python
```
Модель должна выбрать быстрый и доступный инструмент, даже если он менее релевантен

---

#### Выход
```
>>> /exit
```
или
```
>>> exit
```
или `Ctrl+C`

---

### Сценарии использования Interactive режима

#### Сценарий 1: Демонстрация адаптации к сети
```bash
# Запуск
python main.py --mode interactive --checkpoint checkpoints/best

# В интерактиве:
>>> /network optimal
>>> найди погоду в Москве
# Модель выбирает оптимальный инструмент

>>> /network network_issues
>>> найди погоду в Москве
# Модель должна выбрать доступный инструмент, даже если он медленнее
```

---

#### Сценарий 2: Сравнение моделей
```bash
# Терминал 1: базовая модель
python main.py --mode interactive

# Терминал 2: обученная модель
python main.py --mode interactive --checkpoint checkpoints/best

# Одинаковые запросы в обоих терминалах для сравнения
```

---

#### Сценарий 3: Отладка
```bash
python main.py --mode interactive --checkpoint checkpoints/20

# Тестирование на проблемных запросах
>>> /network network_issues
>>> сложный запрос который модель плохо обрабатывает
>>> /network stats
>>> /eval 20
```

---

## Комбинации параметров

### Полный цикл разработки

#### 1. Начальное обучение
```bash
python main.py --mode train --epochs 30 --profile desktop
```

#### 2. Оценка
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200
```

#### 3. Интерактивное тестирование
```bash
python main.py --mode interactive --checkpoint checkpoints/best
```

#### 4. Дообучение (если нужно)
```bash
python main.py --mode train --epochs 10 --checkpoint checkpoints/best --network-mode stochastic
```

#### 5. Финальная оценка
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode stochastic --eval-episodes 500
```

---

## Типичные сценарии использования

### Сценарий 1: Быстрое тестирование на laptop
```bash
# Обучение
python main.py --mode train --epochs 20 --batch-size 4 --profile laptop

# Оценка
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 100

# Интерактив
python main.py --mode interactive --checkpoint checkpoints/best
```

---

### Сценарий 2: Максимальное качество на desktop
```bash
# Обучение
python main.py --mode train --epochs 50 --batch-size 8 --profile desktop --network-mode deterministic

# Дообучение на stochastic
python main.py --mode train --epochs 10 --checkpoint checkpoints/best --network-mode stochastic

# Полная оценка
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode deterministic --eval-episodes 300
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode controlled --eval-episodes 300
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode stochastic --eval-episodes 300
```

---

### Сценарий 3: Сравнение разных чекпоинтов
```bash
# Оценка чекпоинта эпохи 10
python main.py --mode evaluate --checkpoint checkpoints/10 --eval-episodes 200

# Оценка чекпоинта эпохи 20
python main.py --mode evaluate --checkpoint checkpoints/20 --eval-episodes 200

# Оценка чекпоинта эпохи 30
python main.py --mode evaluate --checkpoint checkpoints/30 --eval-episodes 200

# Оценка best
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200
```

---

### Сценарий 4: Отладка проблем с памятью
```bash
# Если возникает OOM
python main.py --mode train --epochs 20 --batch-size 2 --profile laptop

# Если всё ещё OOM
python main.py --mode train --epochs 20 --batch-size 1 --profile laptop
```

---

## Советы и рекомендации

### Обучение
1. **Начните с deterministic** — стабильнее и быстрее сходится
2. **Используйте profile auto-detection** — система сама определит оптимальные параметры
3. **Сохраняйте чекпоинты каждые 10 эпох** — можно вернуться к лучшей версии
4. **Мониторьте relevance метрику** — если падает, уменьшите learning rate

### Оценка
1. **Тестируйте на всех network modes** — модель должна работать везде
2. **Используйте >= 200 эпизодов** — меньше даёт нестабильную статистику
3. **Сравнивайте с baseline** — оцените улучшение относительно базовой модели

### Интерактив
1. **Тестируйте edge cases** — необычные запросы, проблемная сеть
2. **Используйте /network stats** — понимайте текущее состояние
3. **Сравнивайте разные сценарии** — модель должна адаптироваться

---

## Troubleshooting

### OOM (Out of Memory)
```bash
# Решение 1: Уменьшить batch size
python main.py --mode train --batch-size 2

# Решение 2: Использовать laptop profile
python main.py --mode train --profile laptop

# Решение 3: Комбинация
python main.py --mode train --batch-size 2 --profile laptop
```

### Медленное обучение
```bash
# Решение 1: Увеличить batch size (если есть VRAM)
python main.py --mode train --batch-size 16

# Решение 2: Уменьшить количество эпох для тестирования
python main.py --mode train --epochs 10
```

### Модель не сходится
```bash
# Решение 1: Больше эпох
python main.py --mode train --epochs 50

# Решение 2: Deterministic режим
python main.py --mode train --network-mode deterministic

# Решение 3: Дообучение с checkpoint
python main.py --mode train --epochs 20 --checkpoint checkpoints/best
```

### Чекпоинт не загружается
```bash
# Проверьте путь
ls checkpoints/best/

# Должны быть файлы:
# - adapter_model.safetensors или adapter_model.bin
# - adapter_config.json
# - tokenizer files

# Если файлов нет, переобучите модель
python main.py --mode train --epochs 30
```
