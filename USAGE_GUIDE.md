# Руководство по использованию

Этот файл намеренно короткий. Основная документация:

- [README.md](README.md) — обзор проекта;
- [QUICKSTART.md](QUICKSTART.md) — команды запуска;
- [ADAPTIVE_SELECTOR.md](ADAPTIVE_SELECTOR.md) — selector, benchmark и анализ логов;
- [CHANGES.md](CHANGES.md) — последние изменения.

Основные сценарии:

```bash
python main.py --mode select --query "..." --network-mode controlled
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100
python main.py --mode analyze-log --log-path runs/selection_benchmark.jsonl
python main.py --mode train --epochs 20
```
