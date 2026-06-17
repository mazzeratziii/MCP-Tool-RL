# MCP-Tool-RL

Исследовательский прототип для выбора MCP-инструмента в условиях нестабильной сети.

Проект решает задачу: по пользовательскому запросу выбрать не просто самый похожий по описанию инструмент, а инструмент, который подходит по смыслу и способен стабильно выполниться при текущих сетевых условиях.

## Коротко

MCP-агент получает текстовый запрос и выбирает внешний инструмент. В классическом варианте выбор часто строится только на semantic similarity между запросом и описанием API. В этом проекте к смысловому соответствию добавлены нефункциональные признаки выполнения:

- задержка ответа;
- доступность инструмента;
- вероятность успешного ответа;
- стабильность;
- jitter;
- штрафы за повторы и недоступность.

На этой основе сравниваются четыре подхода:

| Политика | Идея |
| --- | --- |
| `semantic` | выбор только по смысловой близости |
| `sonar` | semantic score + сетевой score |
| `adaptive` | semantic + intent + QoS + штрафы |
| `LLM/GRPO` | обучаемая LLM-политика выбора через reward среды |

## Зачем это нужно

В MCP-сценариях инструмент может быть формально подходящим, но плохим практически:

- API сейчас недоступен;
- задержка слишком высокая;
- сервис часто возвращает ошибку;
- похожий инструмент работает стабильнее;
- запрос требует не просто поиска, а конкретного действия.

Поэтому выбор инструмента становится задачей адаптивного ранжирования: нужно учитывать функциональные требования и условия выполнения.

## Данные

В проекте используются несколько блоков данных.

| Блок | Что содержит | Где используется |
| --- | --- | --- |
| ToolBench  | пользовательские запросы, релевантные инструменты, описания API, train/validation prompts | обучение, benchmark, evaluate |
| MCP tools | названия, описания, категории, параметры, примеры вызовов | формирование реестра инструментов |
| `mcp_config.json` | локальные MCP-инструменты и дополнительные описания | расширение реестра |
| `.env` | модель, API-параметры, пути к retriever, настройки запуска | конфигурация окружения |
| `runs/*.jsonl` | логи выбора инструментов | анализ ошибок и сравнение политик |
| `checkpoints/*` | LoRA-checkpoints обученной политики | evaluate и interactive |

Схема данных генерируется командой:

```powershell
python scripts/build_used_data_slide.py
```

Результат появится в:

```text
outputs/data_blocks/used_data_slide.png
```

## Архитектура

Основные модули проекта:

| Модуль | Назначение |
| --- | --- |
| `ToolRegistry` | хранит инструменты, строит semantic/lexical retrieval, возвращает top-k кандидатов |
| `NetworkEmulator` | моделирует latency, jitter, availability, stability и success rate |
| `MCPEnvironment` | формирует state, выполняет выбранный tool, считает reward |
| `AdaptiveSelector` | прозрачная baseline-политика с учётом intent, QoS и штрафов |
| `SonarSelector` | SONAR-style baseline: semantic score + network score |
| `NetMCPTrainer` | обучение LLM/LoRA-политики через GRPO |

Упрощённый pipeline:

```text
запрос пользователя
        ↓
нормализация и retrieval
        ↓
top-k MCP-инструментов
        ↓
состояние среды: semantic + QoS + relevance labels
        ↓
политика выбора: semantic / sonar / adaptive / LLM-GRPO
        ↓
выполнение tool, reward, метрики и лог
```

## Установка

Требуется Python 3.10+.

```powershell
git clone https://github.com/mazzeratziii/MCP-Tool-RL.git
cd MCP-Tool-RL
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Создайте `.env` по примеру `.env.example`:

```env
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
BASE_URL=
API_TOKEN=
NETWORK_MODE=deterministic
BATCH_SIZE=1
MAX_STEPS=3
LEARNING_RATE=1e-5
RETRIEVER_MODEL_PATH=models/retriever
```

`.env`, `runs/` и `checkpoints/` игнорируются Git и не должны попадать в GitHub при обычном `git add .`.

## Быстрый старт

Проверить окружение:

```powershell
python main.py --mode healthcheck
```

Выбрать инструмент для одного запроса:

```powershell
python main.py --mode select `
  --query "What is the weather in London?" `
  --network-mode controlled `
  --top-k 10
```

Сравнить baseline-политики:

```powershell
python main.py --mode benchmark `
  --benchmark-policy all `
  --eval-episodes 200 `
  --network-mode controlled
```

Запустить интерактивный режим:

```powershell
python main.py --mode interactive `
  --checkpoint checkpoints/best `
  --profile laptop
```

Оценить обученную LLM/GRPO-политику:

```powershell
python main.py --mode evaluate `
  --checkpoint checkpoints/best `
  --eval-episodes 200 `
  --network-mode controlled `
  --profile laptop
```

## Режимы CLI

| Режим | Команда | Назначение |
| --- | --- | --- |
| healthcheck | `--mode healthcheck` | проверка зависимостей, конфигурации и retriever |
| select | `--mode select --query "..."` | выбор инструмента для одного запроса |
| benchmark | `--mode benchmark` | сравнение политик выбора |
| analyze-log | `--mode analyze-log` | анализ JSONL-логов benchmark |
| train | `--mode train` | GRPO-обучение LLM/LoRA-политики |
| evaluate | `--mode evaluate` | оценка checkpoint |
| interactive | `--mode interactive` | ручная проверка запросов |

Основные параметры:

| Параметр | Значение |
| --- | --- |
| `--network-mode` | `deterministic`, `controlled`, `stochastic` |
| `--benchmark-policy` | `semantic`, `sonar`, `adaptive`, `both`, `all` |
| `--profile` | `laptop`, `desktop` |
| `--eval-episodes` | число эпизодов оценки |
| `--top-k` | число кандидатов retriever |
| `--checkpoint` | путь к checkpoint |
| `--verbose` | подробный вывод |

## Обучение

LLM/GRPO-обучение запускается так:

```powershell
python main.py --mode train --epochs 40 --profile laptop
```

В ходе обучения:

1. `ToolRegistry` получает top-k кандидатов.
2. `MCPEnvironment` формирует состояние среды.
3. LLM выбирает инструмент.
4. Среда возвращает reward.
5. GRPO обновляет LoRA-политику.
6. Метрики сохраняются в `runs/training_metrics.csv`.
7. График сохраняется в `runs/training_curve.png`.
8. Лучший checkpoint сохраняется в `checkpoints/best`.

## Эксперименты

Экспериментальная часть вынесена в отдельный файл:

[EXPERIMENTS.md](EXPERIMENTS.md)

В нём описаны:

- benchmark-политики;
- сетевые режимы;
- метрики;
- последние результаты;
- интерпретация ошибок;
- выводы по сравнению baseline и LLM/GRPO.

## Последние результаты

Benchmark, `controlled`, 200 episodes:

| Policy | Relevance@1 | Relevance@3 | Success rate | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 77.00% | 84.50% | 91.50% | 0.308s | 3.243 |
| SONAR | 78.00% | 85.00% | 92.50% | 0.297s | 3.337 |
| Adaptive | 70.50% | 84.50% | 93.50% | 0.294s | 3.120 |

LLM/GRPO evaluation, `controlled`, 200 episodes:

| Policy | Relevance@1 | Relevance@3 | Success rate | Avg latency |
| --- | ---: | ---: | ---: | ---: |
| LLM/GRPO | 73.00% | 80.00% | 90.50% | 0.393s |



## Структура проекта

```text
.
├── main.py
├── mcp_config.json
├── requirements.txt
├── src/
│   ├── data/
│   ├── environment/
│   ├── selection/
│   ├── rl/
│   ├── llm/
│   └── mcp/
├── mcp_servers/
├── models/retriever/
├── tests/
├── runs/          # локальные логи, игнорируются Git
└── checkpoints/   # локальные checkpoints, игнорируются Git
```

## Ограничения

- Качество выбора сильно зависит от retriever и качества описаний MCP-инструментов.
- LLM/GRPO-политика может переобучаться на top-k кандидаты, если retriever отдаёт слабую выдачу.
- Реальные API-вызовы заменены эмуляцией среды и сетевых условий.
- Для полноценного обучения нужна локальная или доступная HuggingFace-модель.

## Итог

Проект показывает, что выбор MCP-инструмента полезно рассматривать как задачу адаптивного ранжирования. Семантическая близость остаётся важной, но в условиях нестабильной сети выбор должен учитывать доступность, задержку, стабильность и вероятность успешного выполнения.
