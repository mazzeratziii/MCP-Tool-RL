# Краткое описание

MCP-Tool-RL исследует выбор инструментов, когда важны и функциональные требования запроса, и условия выполнения.

Система умеет:

- загружать инструменты и prompts из ToolBench;
- искать семантически релевантные кандидаты;
- ранжировать их по функциональным и сетевым признакам;
- эмулировать сетевые условия;
- запускать benchmark selector-а;
- анализировать ошибки из JSONL-логов;
- использовать GRPO-код для обучения learned policy.

Рекомендуемый baseline:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --rerank-weight 0.35 --provider-group-weight 0.0
```

Главное ограничение: многие оставшиеся ошибки связаны с retrieval miss, alias/noisy labels в ToolBench и отсутствием multi-tool planning.
