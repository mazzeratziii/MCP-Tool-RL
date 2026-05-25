# Использование ToolBench

ToolBench даёт:

- пользовательские запросы;
- candidate API lists;
- названия и описания инструментов;
- relevant tool labels.

Loader:

```text
src/data/toolbench_loader.py
```

Конфигурация:

```text
src/config.py
```

Benchmark использует validation prompts, где есть хотя бы один relevant tool.

Известная проблема: в ToolBench встречаются aliases, дублирующиеся провайдеры и шумные labels. Log analyzer отдельно считает strict correct и soft correct для near-duplicate aliases.
