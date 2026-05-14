# Чеклист перед запуском MCP-Tool-RL v2

## ✅ Проверка установки

### 1. Виртуальное окружение
```bash
cd C:\Users\allmute\PycharmProjects\MCP-Tool-RL-v2
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# или
source venv/bin/activate  # Linux/Mac
```

### 2. Зависимости
```bash
pip install -r requirements.txt
```

### 3. Переменные окружения (.env)
```bash
# Проверьте наличие файла .env
ls .env

# Должны быть установлены:
# MODEL_NAME=<your-model>
# BASE_URL=<your-api-url>
# API_TOKEN=<your-token>
```

### 4. CUDA (если используете GPU)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

---

## ✅ Проверка изменений

### Файлы, которые должны быть изменены:

#### 1. src/prompts.py
```bash
grep -n "Статус:" src/prompts.py
```
**Ожидается:** Строки с добавлением сетевых метрик (около строки 21)

#### 2. src/rl/reward_functions.py
```bash
grep -n "avg_latency" src/rl/reward_functions.py
```
**Ожидается:** Новые параметры в compute_outcome_reward (около строки 10)

#### 3. src/environment/network_emulator.py
```bash
grep -n "scenarios" src/environment/network_emulator.py
```
**Ожидается:** Определение сценариев (около строки 36)

#### 4. src/rl/train_grpo.py
```bash
grep -n "rotate_scenario" src/rl/train_grpo.py
```
**Ожидается:** Вызов ротации сценариев (около строки 426)

---

## ✅ Быстрый тест

### 1. Проверка импортов
```bash
python -c "from src.config import Config; print('Config OK')"
python -c "from src.rl.train_grpo import NetMCPTrainer; print('Trainer OK')"
python -c "from src.environment.network_emulator import NetworkEmulator, NetworkMode; print('Network OK')"
```

### 2. Проверка загрузки данных
```bash
python -c "
from src.config import Config
config = Config()
config.load_data()
print(f'Tools: {len(config.tools)}')
print(f'Train prompts: {len(config.train_prompts)}')
print(f'Val prompts: {len(config.val_prompts)}')
"
```

**Ожидается:**
- Tools: 15000
- Train prompts: ~64000
- Val prompts: ~16000

### 3. Проверка сценариев
```bash
python -c "
from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.environment.network_emulator import NetworkMode

config = Config()
config.load_data()
env = MCPEnvironment(config, network_mode=NetworkMode.DETERMINISTIC)

print('Scenarios:', list(env.network.scenarios.keys()))
print('Current scenario:', env.network.current_scenario)

env.network.set_scenario('peak_hours')
print('After set_scenario:', env.network.current_scenario)
"
```

**Ожидается:**
```
Scenarios: ['normal', 'peak_hours', 'network_issues', 'optimal']
Current scenario: normal
After set_scenario: peak_hours
```

---

## ✅ Тестовый запуск

### 1. Короткое обучение (1 эпоха)
```bash
python main.py --mode train --epochs 1 --batch-size 4
```

**Что проверяем:**
- ✅ Нет ошибок импорта
- ✅ Данные загружаются
- ✅ Модель инициализируется
- ✅ Обучение запускается
- ✅ Чекпоинт сохраняется

**Ожидаемый вывод:**
```
============================================================
INITIALIZING CONFIGURATION
============================================================
Configuration loaded:
  Model: <your-model>
  ...

------------------------------
LOADING TOOLBENCH DATA
------------------------------

============================================================
CREATING TOOL SELECTOR
============================================================

============================================================
SELECTING 15000 TOOLS FOR TRAINING
============================================================

============================================================
PREPARING TRAINING PROMPTS
============================================================

Device: cuda
Profile: desktop (RTX 3070 8 GB)
VRAM: 8192 MB

Pre-caching tool embeddings...
Cached 15000 tool embeddings

============================================================
GRPO TRAINING [desktop (RTX 3070 8 GB)] — 1 epochs
============================================================

--- Epoch 1 [scenario: normal] ---
    [resample → 1500 queries]
  loss=X.XXXX  reward=X.XXX  success=XX%  relevance=XX%  ...
  ✓ Checkpoint: checkpoints/best
```

### 2. Быстрая оценка
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 10
```

**Что проверяем:**
- ✅ Чекпоинт загружается
- ✅ Evaluation запускается
- ✅ Новые метрики выводятся

**Ожидаемый вывод:**
```
============================================================
EVALUATION [desktop (RTX 3070 8 GB)] — 10 ep, mode=controlled
============================================================
  Episodes:           10
  Success rate:       XX%
  Relevance@1:        XX%
  Relevance@3:        XX%
  Top-3 gap:          +XX%

  Network Adaptation Metrics:
  Avg latency:        X.XXXs
  Fast tool choices:  XX%  (below avg latency)
  Available choices:  XX%  (chose available tools)
```

### 3. Интерактивный режим
```bash
python main.py --mode interactive --checkpoint checkpoints/best
```

**Что проверяем:**
- ✅ Интерактив запускается
- ✅ Команды работают
- ✅ Сетевые метрики отображаются

**Тестовые команды:**
```
>>> /network stats
>>> /network peak_hours
>>> найди информацию про Python
>>> /exit
```

---

## ✅ Проверка документации

### Файлы должны существовать:
```bash
ls -la *.md
```

**Ожидается:**
- ✅ README.md (оригинальный)
- ✅ QUICKSTART.md (быстрый старт)
- ✅ USAGE_GUIDE.md (полное руководство)
- ✅ CHANGES.md (описание изменений)
- ✅ COMPARISON.md (сравнение до/после)
- ✅ CHECKLIST.md (этот файл)

---

## ✅ Готовность к полному обучению

### Перед запуском 30+ эпох проверьте:

#### 1. Достаточно места на диске
```bash
df -h .  # Linux/Mac
# или
dir  # Windows
```
**Нужно:** ~5GB для чекпоинтов

#### 2. VRAM не переполняется
```bash
# Запустите 1 эпоху и проверьте пиковое использование
python main.py --mode train --epochs 1

# В другом терминале (Linux):
watch -n 1 nvidia-smi

# Windows:
nvidia-smi -l 1
```

**Если OOM:**
```bash
# Уменьшите batch size
python main.py --mode train --batch-size 4

# Или используйте laptop profile
python main.py --mode train --profile laptop
```

#### 3. Время выполнения
**Примерное время 1 эпохи:**
- Desktop (8GB VRAM): ~15-20 минут
- Laptop (4GB VRAM): ~30-40 минут

**Для 30 эпох:**
- Desktop: ~8-10 часов
- Laptop: ~15-20 часов

---

## ✅ Финальный чеклист

Перед запуском полного обучения убедитесь:

- [ ] Виртуальное окружение активировано
- [ ] Все зависимости установлены
- [ ] .env файл настроен
- [ ] CUDA доступна (если используете GPU)
- [ ] Все изменённые файлы на месте
- [ ] Тестовый запуск (1 эпоха) прошёл успешно
- [ ] Evaluation работает
- [ ] Interactive режим работает
- [ ] Достаточно места на диске (~5GB)
- [ ] VRAM не переполняется
- [ ] Документация прочитана

---

## 🚀 Запуск полного обучения

### Desktop (рекомендуется)
```bash
python main.py --mode train --epochs 30 --profile desktop
```

### Laptop
```bash
python main.py --mode train --epochs 30 --profile laptop --batch-size 4
```

### С автоматической оценкой после обучения
```bash
# Обучение
python main.py --mode train --epochs 30 --profile desktop

# Оценка
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200

# Интерактив
python main.py --mode interactive --checkpoint checkpoints/best
```

---

## 📊 Мониторинг обучения

### Во время обучения следите за:

1. **loss** — должен снижаться (от ~2.0 к ~0.5)
2. **reward** — должен расти (от ~1.0 к ~3.5)
3. **relevance** — должен расти (от ~40% к ~65%+)
4. **skipped** — должен быть низким (<5% от rollouts)

### Признаки проблем:

❌ **loss растёт** → уменьшите learning rate или batch size
❌ **reward не растёт** → проверьте reward function
❌ **relevance падает** → модель переобучается на сетевые метрики
❌ **много skipped** → увеличьте entropy_coeff или sample_temperature

---

## 🎯 После обучения

### 1. Сравните с оригиналом
```bash
# Оригинал
cd ../MCP-Tool-RL
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200

# v2
cd ../MCP-Tool-RL-v2
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200
```

### 2. Протестируйте на всех network modes
```bash
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode deterministic --eval-episodes 200
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode controlled --eval-episodes 200
python main.py --mode evaluate --checkpoint checkpoints/best --network-mode stochastic --eval-episodes 200
```

### 3. Интерактивное тестирование
```bash
python main.py --mode interactive --checkpoint checkpoints/best

# Попробуйте разные сценарии:
>>> /network normal
>>> <ваш запрос>

>>> /network network_issues
>>> <тот же запрос>

# Модель должна адаптироваться
```

---

## 📝 Отчёт о результатах

После обучения заполните:

| Метрика | Оригинал | v2 | Изменение |
|---------|----------|-----|-----------|
| Relevance@1 | __%  | __%  | __% |
| Success rate | __%  | __%  | __% |
| Avg latency | __s  | __s  | __% |
| Fast tool choices | __%  | __%  | __% |
| Available choices | __%  | __%  | __% |

**Вывод:** _______________

---

## ❓ Проблемы?

### OOM
```bash
python main.py --mode train --batch-size 2 --profile laptop
```

### Медленно
```bash
# Уменьшите train_set_size в src/rl/train_grpo.py
# Desktop: 1500 → 1000
# Laptop: 600 → 400
```

### Модель не сходится
```bash
# Больше эпох
python main.py --mode train --epochs 50

# Или дообучение
python main.py --mode train --epochs 20 --checkpoint checkpoints/best
```

### Другие проблемы
Смотрите **USAGE_GUIDE.md** раздел "Troubleshooting"
