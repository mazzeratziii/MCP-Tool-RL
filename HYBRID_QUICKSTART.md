# Быстрый старт Hybrid Mode

Hybrid mode объединяет реальные MCP tools и ToolBench emulation.

1. Настройте инструменты в `mcp_config.json`.
2. Убедитесь, что локальные MCP servers доступны.
3. Запустите:

```bash
python main.py --mode train --use-hybrid --mcp-config mcp_config.json
```

Для selector-only экспериментов сначала используйте обычный benchmark:

```bash
python main.py --mode benchmark --benchmark-policy adaptive
```
