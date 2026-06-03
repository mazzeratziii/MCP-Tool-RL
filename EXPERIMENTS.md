# Эксперименты MCP-Tool-RL

Файл описывает экспериментальную часть проекта: какие политики сравнивались, в каких сетевых условиях, по каким метрикам и какие выводы получены.

## Цель экспериментов

Проверить, как разные политики выбирают MCP-инструменты, если учитывать не только смысл запроса, но и сетевые условия выполнения.

Основные вопросы:

- достаточно ли semantic retrieval для выбора инструмента;
- улучшает ли SONAR-style учёт сетевых признаков результат;
- как adaptive selector балансирует смысл и QoS;
- насколько обучаемая LLM/GRPO-политика способна выбирать инструмент по reward;
- как меняется качество выбора при нестабильной сети.

## Данные

В экспериментах используются:

| Источник | Назначение |
| --- | --- |
| ToolBench / MAuRS | запросы пользователей и релевантные инструменты |
| MCP tools | описания доступных инструментов |
| `mcp_config.json` | локальные MCP-инструменты |
| `runs/*.jsonl` | логи выбора инструментов |
| `checkpoints/best` | лучший LoRA-checkpoint LLM/GRPO-политики |

Каждый эпизод содержит:

- `query`;
- top-k кандидатов;
- выбранный инструмент;
- релевантные инструменты;
- success/failure;
- latency;
- reward;
- признаки сети;
- ранжирование кандидатов.

## Сетевые режимы

`NetworkEmulator` поддерживает три режима:

| Режим | Для чего используется |
| --- | --- |
| `deterministic` | воспроизводимое обучение и отладка |
| `controlled` | управляемая проверка в меняющихся условиях |
| `stochastic` | стресс-тест со случайными сбоями |

Также используются сценарии:

- `normal`;
- `peak_hours`;
- `network_issues`;
- `optimal`.

Сетевые признаки:

- `available`;
- `latency`;
- `jitter`;
- `stability`;
- `success_rate`;
- `retries`.

## Политики

### Semantic

Базовая политика. Выбирает инструмент по смысловой близости:

```text
score = semantic_similarity(query, tool_description)
```

Плюс: простая и сильная baseline-модель.  
Минус: не учитывает сеть и может выбрать медленный или нестабильный инструмент.

### SONAR

SONAR-style baseline. Объединяет смысловую близость и сетевой score:

```text
final_score = 0.70 * semantic + 0.30 * network
```

`network` учитывает:

- success rate;
- stability;
- latency quality;
- availability.

Плюс: сильный и прозрачный baseline.  
Минус: не использует intent, action/entity overlap и историю retry так подробно, как adaptive selector.

### Adaptive

Адаптивная политика выбора. Учитывает смысл, функциональное соответствие и QoS:

```text
total = semantic_part + qos - latency_penalty - retry_penalty - unavailable_penalty
```

Используемые признаки:

- semantic score;
- intent match;
- action/entity overlap;
- provider overlap;
- generic penalty;
- success rate;
- stability;
- latency penalty;
- retry penalty;
- unavailable penalty.

Плюс: лучше отражает идею проекта: выбирать инструмент по смыслу и условиям выполнения.  
Минус: при слишком сильных штрафах может уступать semantic/SONAR по Relevance@1.

### LLM/GRPO

Обучаемая политика. LLM получает top-k кандидатов, выбирает инструмент, среда возвращает reward, после чего LoRA-policy обновляется через GRPO.

Схема:

```text
prompt + top-k tools
        ↓
LLM policy
        ↓
selected tool
        ↓
MCPEnvironment.step()
        ↓
reward
        ↓
GRPO update
```

Плюс: политика может обучаться на reward, а не только на ручной формуле.  
Минус: качество зависит от retriever, top-k кандидатов и устойчивости training setup.

## Метрики

| Метрика | Что показывает |
| --- | --- |
| `Relevance@1` | выбран ли top-1 инструмент из списка релевантных |
| `Relevance@3` | есть ли релевантный инструмент в top-3 |
| `Success rate` | доля успешных выполнений |
| `Avg latency` | средняя задержка выбранных инструментов |
| `Avg reward` | средняя награда среды |
| `Avg retries` | среднее число повторных попыток |
| `Top-3 gap` | разница между Relevance@3 и Relevance@1 |

## Команды запуска

Проверка окружения:

```powershell
python main.py --mode healthcheck
```

Benchmark всех baseline-политик:

```powershell
python main.py --mode benchmark `
  --benchmark-policy all `
  --eval-episodes 200 `
  --network-mode controlled
```

Оценка SONAR:

```powershell
python main.py --mode benchmark `
  --benchmark-policy sonar `
  --eval-episodes 200 `
  --network-mode controlled
```

Оценка LLM/GRPO:

```powershell
python main.py --mode evaluate `
  --checkpoint checkpoints/best `
  --eval-episodes 200 `
  --network-mode controlled `
  --profile laptop
```

Анализ логов:

```powershell
python main.py --mode analyze-log `
  --log-path runs/selection_benchmark_sonar.jsonl
```

Обучение:

```powershell
python main.py --mode train --epochs 40 --profile laptop
```

## Последние результаты

Benchmark, `controlled`, 200 episodes:

| Policy | Relevance@1 | Relevance@3 | Success rate | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 77.00% | 84.50% | 91.50% | 0.308s | 3.243 |
| SONAR | 78.00% | 85.00% | 92.50% | 0.297s | 3.337 |
| Adaptive | 70.50% | 84.50% | 93.50% | 0.294s | 3.120 |

LLM/GRPO evaluation, `controlled`, 200 episodes:

| Policy | Relevance@1 | Relevance@3 | Success rate | Avg latency | Fast choices | Available choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM/GRPO | 73.00% | 80.00% | 90.50% | 0.393s | 5.00% | 100.00% |

## Обучение LLM/GRPO

Во время обучения сохраняются:

```text
runs/training_metrics.csv
runs/training_curve.png
checkpoints/best
checkpoints/<epoch>
```

Лучший результат в ходе обучения:

```text
best relevance: 82.33%
```

Финальные эпохи показали, что модель обучается выбирать релевантные инструменты, но качество итогового evaluate зависит от конкретного checkpoint, режима сети и качества top-k выдачи retriever.

## Анализ ошибок

Основные типы ошибок:

| Тип ошибки | Смысл |
| --- | --- |
| `retrieval_miss_or_label_gap` | нужный инструмент не попал в выдачу или разметка ToolBench неоднозначна |
| `right_tool_in_top3_not_top1` | правильный инструмент есть в top-3, но не выбран первым |
| `intent_mismatch` | политика спутала тип действия |
| `selected_too_generic` | выбран слишком общий инструмент |
| `selected_too_specific` | выбран слишком узкий инструмент |
| `near_duplicate_or_alias` | выбран почти тот же инструмент, но с другим названием |
| `execution_failure` | инструмент выбран близко к правильному, но выполнение неуспешно |

## Выводы

1. Semantic retrieval остаётся сильной baseline-стратегией.
2. SONAR оказался лучшим практическим baseline в последних controlled benchmark.
3. Adaptive selector лучше отражает исследовательскую идею учёта QoS, но требует аккуратной настройки весов.
4. LLM/GRPO-политика показывает обучаемость, но не гарантирует превосходство над сильным baseline без улучшения retriever и training setup.
5. Главная зона дальнейшего улучшения — качество top-k retrieval и более строгая разметка релевантных инструментов.

## Что считать успешным результатом

Проект можно считать завершённым как исследовательский прототип, потому что:

- есть рабочая среда выбора MCP-инструментов;
- реализованы несколько политик;
- есть эмуляция нестабильной сети;
- есть LLM/GRPO-обучение;
- есть benchmark и evaluate;
- есть интерактивный режим;
- есть логи, графики и анализ ошибок;
- результаты воспроизводятся через CLI-команды.
