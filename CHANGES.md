# Изменения

## Adaptive Selection Baseline

Добавлен лёгкий библиотечный selector, независимый от тяжёлой GRPO-модели:

- семантическая фильтрация ToolBench-кандидатов;
- функциональное intent-сопоставление;
- QoS-aware scoring по latency, jitter/stability, availability, success rate и retry history;
- обновление статистики выполнения через exponential moving average.

Новые файлы:

- `src/selection/adaptive_selector.py`
- `src/selection/intent.py`
- `src/selection/tool_features.py`
- `src/selection/benchmark.py`
- `src/selection/log_analysis.py`
- `src/selection/__init__.py`

## Benchmark и анализ

Добавлено:

- `python main.py --mode benchmark`;
- `python main.py --mode analyze-log`;
- JSONL-логи benchmark;
- категории ошибок: retrieval miss / label gap, correct tool in top-3 but not top-1, selected too generic/specific, intent mismatch, execution failure, near duplicate or alias;
- soft-correct метрика для alias и near-duplicate инструментов ToolBench.

## Reranker Features

Добавлен консервативный текстовый reranker:

- разбор названия `Provider.Action`;
- action overlap;
- entity overlap;
- generic tool penalty;
- опциональный provider-group adjustment для ablation.

Профиль по умолчанию:

```bash
--rerank-weight 0.35 --provider-group-weight 0.0
```

Агрессивный provider-profile выключен по умолчанию, потому что снизил качество benchmark на шумных названиях провайдеров ToolBench.

## Надёжность environment и retriever

Обновлено:

- `src/environment/network_emulator.py` теперь отдаёт jitter;
- `src/environment/mcp_environment.py` отдаёт success rate и jitter в tool state;
- `src/environment/tool_registry.py` не падает, если в `models/retriever` есть конфиги, но нет весов;
- `src/config.py` считает `python-dotenv` опциональной зависимостью.

## CLI

`main.py` теперь поддерживает:

- `select`;
- `benchmark`;
- `analyze-log`;
- `healthcheck`;
- reranker ablation flags;
- provider-group ablation flags;
- `--verbose`, при этом обычные запуски скрывают шумные логи загрузки данных и моделей.

## Воспроизводимость

Добавлен:

- `scripts/run_baselines.ps1` для запуска semantic/adaptive baseline и анализа логов.

Текущий лучший baseline:

| Профиль | Relevance@1 | Relevance@3 | Success | Avg reward |
| --- | ---: | ---: | ---: | ---: |
| adaptive conservative | 56% | 68% | 96% | 2.727 |
| adaptive aggressive provider | 41% | 68% | 95% | 2.134 |

Проект можно считать воспроизводимым исследовательским прототипом. Главное оставшееся ограничение — выбор одного инструмента для запросов, которым естественно требуется несколько инструментов.

## Тесты

Добавлены тесты для:

- adaptive selector;
- intent matching;
- reranker features;
- log analysis и soft-correct metrics.

Запуск:

```bash
python -m unittest tests.test_adaptive_selector tests.test_log_analysis tests.test_tool_features
```
# Финальная доводка

- Добавлен SONAR-style baseline как отдельная benchmark policy: `--benchmark-policy sonar`.
- Добавлен общий режим сравнения: `--benchmark-policy all`.
- Добавлен финальный скрипт проверки: `scripts/run_final_checks.ps1`.
- Добавлен итоговый отчёт: `FINAL_REPORT.md`.
- README дополнен финальными GRPO-метриками и командами проверки.
