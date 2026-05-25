# API Keys

Секреты должны храниться в `.env` и не попадать в Git.

Используйте `.env.example` как шаблон.

Основные переменные:

```text
MODEL_NAME=
BASE_URL=
API_TOKEN=
SYSTEM_PROMPT=
USER_PROMPT=
```

Adaptive selector и benchmark modes могут работать без API keys, если используются emulated ToolBench tools.

