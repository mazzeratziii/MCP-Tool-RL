# MCP-Tool-RL

## Финальное состояние

Проект оформлен как исследовательский прототип RL-расширения NetMCP/SONAR.

Финальный отчёт: [FINAL_REPORT.md](FINAL_REPORT.md)

Финальная проверка:

```powershell
.\scripts\run_final_checks.ps1
```

Финальная проверка с evaluation обученной LLM:

```powershell
.\scripts\run_final_checks.ps1 -IncludeLLMEvaluate
```

## График обучения

Во время `python main.py --mode train` после каждой эпохи сохраняются `runs/training_metrics.csv` и `runs/training_curve.png`.

Прототип выбора MCP-инструментов в условиях нестабильной сети.

Проект объединяет:

- метаданные инструментов и запросы из ToolBench;
- семантический поиск кандидатов;
- проверку функционального соответствия запроса и возможностей инструмента;
- адаптивное ранжирование по задержке, джиттеру, доступности, успешности и ретраям;
- benchmark и анализ JSONL-логов для сравнения политик выбора;
- опциональный GRPO/LLM pipeline для обучения политики.

## Основные команды

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Выбор инструмента для одного запроса:

```bash
python main.py --mode healthcheck
```

```bash
python main.py --mode select --query "What is the weather in London?" --network-mode controlled
```

Benchmark адаптивного выбора:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled --log-path runs/selection_benchmark.jsonl
```

Анализ ошибок benchmark:

```bash
python main.py --mode analyze-log --log-path runs/selection_benchmark.jsonl
```

Обучение GRPO-политики:

```bash
python main.py --mode train --epochs 20 --profile laptop
```

Интерактивный режим обученной политики:

```bash
python main.py --mode interactive --checkpoint checkpoints/best
```

## Текущий лучший baseline

Рекомендуемый профиль: `adaptive conservative`.

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled --rerank-weight 0.35 --provider-group-weight 0.0 --log-path runs/selection_benchmark_conservative.jsonl
```

Результаты на 100 validation episodes в controlled network:

| Профиль | Relevance@1 | Relevance@3 | Success | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive conservative | 56% | 68% | 96% | 0.358s | 2.727 |
| adaptive aggressive provider | 41% | 68% | 95% | 0.352s | 2.134 |

Для baseline по статье NetMCP/SONAR доступна отдельная политика:

```bash
python main.py --mode benchmark --benchmark-policy sonar --eval-episodes 100 --network-mode controlled
```

Для одновременного сравнения `semantic-only`, `sonar` и `adaptive`:

```bash
python main.py --mode benchmark --benchmark-policy all --eval-episodes 100 --network-mode controlled --log-path runs/final_report/selection_benchmark.jsonl
```


## Метрики после GRPO-обучения

Обучение запускалось командой:

```bash
python main.py --mode train --epochs 40 --profile laptop
```

Лучший checkpoint: `checkpoints/best`, обновлён на 40-й эпохе.

Training-метрики на 40-й эпохе:

| Metric | Value |
| --- | ---: |
| Relevance | 80.56% |
| Success | 91.28% |
| Avg reward | 3.698 |
| Rollouts | 1800 |
| Updates | 409 |
| Skipped groups | 191 |
| Advantage std | 0.892 |

Validation/evaluation на 200 episodes в `controlled` network:

```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200 --network-mode controlled --profile laptop
```

| Metric | Value |
| --- | ---: |
| Success rate | 94.50% |
| Relevance@1 | 78.50% |
| Relevance@3 | 82.50% |
| Top-3 gap | +4.00% |
| Avg latency | 0.315s |
| Fast tool choices | 14.00% |
| Available choices | 100.00% |

Сравнение с эвристическим adaptive baseline:

| Policy | Relevance@1 | Relevance@3 | Success | Avg latency |
| --- | ---: | ---: | ---: | ---: |
| Adaptive baseline | 53-56% | 68-70% | 95-96% | 0.326-0.358s |
| GRPO-trained LLM | 78.50% | 82.50% | 94.50% | 0.315s |

Вывод: GRPO-политика заметно улучшила функциональную релевантность выбора инструмента и сохранила высокий success rate. Слабое место текущей версии — оптимизация задержки: `Fast tool choices` пока составляет 14%, поэтому latency-компонент reward можно усиливать в дальнейших экспериментах.

Дополнительный deterministic benchmark на 200 episodes:

| Policy | Relevance@1 | Relevance@3 | Success | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic-only | 77.00% | 84.50% | 89.00% | 0.142s | 3.114 |
| SONAR-style | 75.50% | 84.00% | 94.50% | 0.148s | 3.328 |
| adaptive | 67.00% | 83.00% | 93.50% | 0.144s | 3.053 |

В deterministic-сети semantic-only даёт лучший Relevance@1, а SONAR даёт более высокий success rate и лучший reward за счёт учёта QoS.
Агрессивный provider-group профиль оставлен только для ablation-экспериментов. По умолчанию он выключен, потому что переобучается на шумные названия провайдеров ToolBench.

Для воспроизводимого локального прогона:

```powershell
.\scripts\run_baselines.ps1
```

## Ограничения

- Текущий selector оптимизирует выбор первого инструмента. Multi-tool planning оставлен как будущая работа.
- В ToolBench есть шумные метки, дублирующиеся провайдеры и alias-инструменты.
- Если веса retriever отсутствуют, включается lexical fallback. Он устойчивый, но менее точный, чем полноценная embedding-модель.
- Provider-group reranking является экспериментальным и выключен по умолчанию.
- Сетевые условия пока эмулируются режимами `deterministic`, `controlled`, `stochastic`.

## Структура

- `src/selection/` — adaptive selector, intent matcher, reranker features, benchmark runner и log analyzer.
- `src/environment/` — ToolBench environment и эмуляция сети.
- `src/rl/` — GRPO training code.
- `models/retriever/` — retriever-файлы. Веса модели могут отсутствовать в Git; в этом случае используется lexical fallback.
- `runs/`, checkpoints, виртуальные окружения и большие веса моделей игнорируются Git.

См. также:

- [QUICKSTART.md](QUICKSTART.md) — быстрый запуск;
- [PIPELINE_STEPS.md](PIPELINE_STEPS.md) — поэтапное описание работы проекта;
- [ADAPTIVE_SELECTOR.md](ADAPTIVE_SELECTOR.md) — детали selector-а;
- [CHANGES.md](CHANGES.md) — список изменений.
