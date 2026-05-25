# MCP-интеграция

Hybrid MCP support находится в:

```text
src/environment/hybrid_environment.py
src/mcp/simple_client.py
mcp_config.json
mcp_servers/
```

Запуск hybrid mode:

```bash
python main.py --mode train --use-hybrid --mcp-config mcp_config.json
```

Большинство экспериментов лучше сначала запускать в emulation mode. Hybrid mode полезен для проверки selector-а на небольшом наборе реальных MCP-compatible tools.
