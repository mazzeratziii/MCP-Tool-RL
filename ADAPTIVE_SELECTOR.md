# Adaptive Selector

Adaptive selector — лёгкий baseline для выбора MCP-инструмента среди ToolBench-кандидатов.

## Pipeline

1. Получить top-k candidate tools.
2. Отфильтровать по semantic score.
3. Определить query intent и tool intent.
4. Применить консервативный functional reranking.
5. Добавить QoS scoring по состоянию сети.
6. Выбрать лучший инструмент и, при необходимости, обновить observed feedback.

## Компоненты score

- `semantic_score` — семантическая релевантность из retriever или lexical fallback.
- `functional_score` — semantic score плюс intent и reranker adjustments.
- `query_intent` / `tool_intent` — грубые функциональные классы.
- `rerank_adjustment` — поправка по action/entity/generic-tool признакам.
- `group_adjustment` — экспериментальная provider-name поправка, выключена по умолчанию.
- `qos` — success rate и stability.
- `latency_penalty` — штраф за задержку.
- `retry_penalty` — штраф по наблюдаемым ретраям.

## Рекомендуемый профиль

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled --rerank-weight 0.35 --provider-group-weight 0.0 --log-path runs/selection_benchmark_conservative.jsonl
```

## Ablation

Semantic-only:

```bash
python main.py --mode benchmark --benchmark-policy semantic --eval-episodes 100
```

Агрессивный экспериментальный профиль:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --rerank-weight 1.0 --provider-group-weight 1.0 --log-path runs/selection_benchmark_aggressive.jsonl
```

## Анализ логов

```bash
python main.py --mode analyze-log --log-path runs/selection_benchmark_conservative.jsonl
```

Анализатор выводит strict accuracy, soft accuracy, Relevance@3, success rate, average latency, average reward и сгруппированные примеры ошибок.
