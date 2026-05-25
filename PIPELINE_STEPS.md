# Поэтапное описание работы проекта

Этот файл описывает, что происходит в проекте на каждом этапе: от запуска команды до выбора инструмента, benchmark и анализа ошибок.

## 1. Проверка окружения

Команда:

```bash
python main.py --mode healthcheck
```

Что происходит:

1. Проверяется наличие базовых зависимостей: `numpy`, `tqdm`, `pyyaml`.
2. Проверяется наличие зависимостей для ToolBench: `datasets`.
3. Проверяется наличие зависимостей для обучения: `torch`, `transformers`, `peft`.
4. Проверяются опциональные зависимости: `dotenv`, `sentence_transformers`.
5. Проверяется наличие `mcp_config.json`.
6. Проверяется папка `models/retriever`.
7. Проверяется наличие весов retriever-модели.

Если веса retriever отсутствуют, проект не падает. Он использует lexical fallback.

## 2. Загрузка данных

Используется:

```text
src/config.py
src/data/toolbench_loader.py
```

Что происходит:

1. `Config` создаёт настройки сети, ToolBench, RL и reward.
2. `ToolBenchLoader` загружает датасет ToolBench.
3. Из датасета извлекаются:
   - пользовательские запросы;
   - API/tool descriptions;
   - relevant tools;
   - target tool, если он указан.
4. Инструменты группируются и выбираются для экспериментов.
5. Добавляются fallback-инструменты:
   - `Calculator.Evaluate`;
   - `General.NoToolNeeded`.
6. Prompts делятся на train и validation части.

## 3. Получение кандидатов

Используется:

```text
src/environment/tool_registry.py
```

Что происходит:

1. Для каждого инструмента собирается текстовое описание.
2. Если доступна embedding-модель retriever, строятся embeddings.
3. Если весов модели нет, включается lexical retrieval.
4. Для пользовательского запроса выбираются top-k кандидатов.
5. Для каждого кандидата считается semantic score.

## 4. Эмуляция сетевых условий

Используется:

```text
src/environment/network_emulator.py
```

Режимы:

- `deterministic` — фиксированные условия;
- `controlled` — плавно меняющиеся условия;
- `stochastic` — случайные нестабильные условия.

Что считается:

- latency;
- jitter;
- availability;
- stability;
- failure rate.

Эти признаки используются selector-ом как нефункциональные требования.

## 5. Формирование состояния среды

Используется:

```text
src/environment/mcp_environment.py
```

Для каждого кандидата формируется состояние:

```text
name
category
description
available
latency
jitter
stability
success_rate
semantic_score
is_relevant
used
```

Это состояние передаётся selector-у или RL-политике.

## 6. Adaptive Selector

Используется:

```text
src/selection/adaptive_selector.py
src/selection/intent.py
src/selection/tool_features.py
```

Что происходит:

1. Кандидаты фильтруются по `semantic_threshold`.
2. Определяется intent запроса.
3. Определяется intent инструмента.
4. Считается functional score:
   - semantic score;
   - intent match;
   - action/entity overlap;
   - generic tool penalty.
5. Считается QoS score:
   - success rate;
   - stability;
   - latency penalty;
   - retry penalty;
   - availability penalty.
6. Выбирается инструмент с максимальным итоговым score.

Рекомендуемые параметры:

```bash
--rerank-weight 0.35 --provider-group-weight 0.0
```

## 6.1. SONAR-style baseline

Для связи с NetMCP/SONAR добавлена отдельная baseline-политика:

```bash
python main.py --mode benchmark --benchmark-policy sonar --eval-episodes 100 --network-mode controlled
```

Она использует фиксированную формулу:

```text
final_score = alpha * semantic_score + beta * network_score
```

`network_score` строится из availability, success rate, stability и latency quality. В отличие от adaptive selector, SONAR-style baseline не использует intent matching, action/entity overlap и обучаемую обратную связь. Он нужен как прозрачная точка сравнения между semantic-only routing и GRPO-trained LLM policy.

## 7. Выбор одного инструмента

Команда:

```bash
python main.py --mode select --query "What is the weather in London?" --network-mode controlled
```

Что происходит:

1. Загружается ToolBench.
2. Создаётся environment.
3. Извлекаются top-k кандидатов.
4. Adaptive selector ранжирует кандидатов.
5. В консоль выводятся:
   - выбранный инструмент;
   - итоговый score;
   - причина выбора;
   - top ranked candidates.

## 8. Benchmark

Команда:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled
```

Что происходит:

1. Берутся validation prompts.
2. Для каждого prompt выбираются candidate tools.
3. Политика выбирает инструмент.
4. Environment выполняет шаг с выбранным инструментом.
5. Считаются метрики:
   - Relevance@1;
   - Relevance@3;
   - Success rate;
   - Avg latency;
   - Avg reward;
   - Avg retries.
6. Каждая попытка записывается в JSONL-лог.

## 9. JSONL-лог

Пример полей:

```text
policy
query_id
query
selected_tool
target_tool
relevant_tools
target_in_top3
is_relevant
success
latency
reward
semantic_score
functional_score
query_intent
tool_intent
top3
```

Лог нужен, чтобы не только видеть итоговую точность, но и понимать причины ошибок.

## 10. Анализ ошибок

Команда:

```bash
python main.py --mode analyze-log --log-path runs/selection_benchmark.jsonl
```

Что происходит:

1. Читается JSONL-лог.
2. Считаются итоговые метрики.
3. Ошибки группируются по причинам:
   - `retrieval_miss_or_label_gap`;
   - `right_tool_in_top3_not_top1`;
   - `selected_too_generic`;
   - `selected_too_specific`;
   - `intent_mismatch`;
   - `execution_failure`;
   - `near_duplicate_or_alias`.
4. Выводятся примеры ошибок для ручного анализа.

## 11. Обучение GRPO

Во время обучения дополнительно пишется история метрик и график:

- `runs/training_metrics.csv` — значения по эпохам;
- `runs/training_curve.png` — визуализация reward, loss, relevance, success и числа обновлений.

Команда:

```bash
python main.py --mode train --epochs 20 --profile laptop
```

Что происходит:

1. Загружается LLM-модель.
2. Создаётся `MCPEnvironment`.
3. Для train prompts собираются группы rollouts.
4. Для каждого выбранного инструмента считается reward.
5. GRPO обновляет LoRA-адаптер.
6. Периодически сохраняются checkpoints.

Этот режим тяжелее baseline selector-а и требует корректно установленного ML-окружения.

## 12. Интерактивный режим

Команда:

```bash
python main.py --mode interactive --checkpoint checkpoints/best
```

Что происходит:

1. Загружается обученная политика.
2. Пользователь вводит запрос.
3. Модель ранжирует инструменты.
4. Выводится top tools.
5. Выполняется выбранный инструмент в environment.

## 13. Hybrid MCP mode

Команда:

```bash
python main.py --mode train --use-hybrid --mcp-config mcp_config.json
```

Что происходит:

1. Часть инструментов подключается как реальные MCP tools.
2. Остальные инструменты остаются ToolBench-эмуляцией.
3. Selector или RL-политика работают с общим интерфейсом environment.

## 14. Текущая готовность

Проект можно считать завершённым как воспроизводимый исследовательский прототип.

Готово:

- adaptive selector;
- network-aware ranking;
- benchmark;
- log analysis;
- healthcheck;
- tests;
- русская документация;
- fallback при отсутствии retriever weights.

Ограничения:

- selector выбирает один инструмент;
- ToolBench содержит шумные labels и aliases;
- learned reranker пока не обучен;
- реальные MCP tools подключаются только в optional hybrid mode.
