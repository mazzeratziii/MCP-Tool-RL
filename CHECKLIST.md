# Checklist

Перед публикацией или commit:

- [ ] Запустить тесты selector-а:

```bash
python -m unittest tests.test_adaptive_selector tests.test_log_analysis tests.test_tool_features
```

- [ ] Проверить компиляцию:

```bash
python -m compileall src main.py
```

- [ ] Прогнать benchmark:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled --log-path runs/selection_benchmark.jsonl
```

- [ ] Проанализировать лог:

```bash
python main.py --mode analyze-log --log-path runs/selection_benchmark.jsonl
```

- [ ] Не добавлять в Git `venv/`, `runs/`, `checkpoints/`, `.env`, большие веса моделей.
