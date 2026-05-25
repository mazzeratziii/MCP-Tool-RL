# Hybrid Mode Summary

Hybrid mode является опциональным.

Он нужен для экспериментов, где часть инструментов вызывается как реальные MCP calls, а остальные остаются emulated ToolBench tools.

Основные файлы:

- `src/environment/hybrid_environment.py`
- `src/mcp/simple_client.py`
- `mcp_config.json`
- `mcp_servers/`

Используйте hybrid mode после проверки поведения selector-а в emulation benchmarks.
