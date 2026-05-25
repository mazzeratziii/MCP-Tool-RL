# Сравнение политик

Поддерживаемые политики:

| Policy | Описание |
| --- | --- |
| `semantic` | Выбирает кандидата с максимальным semantic score. |
| `adaptive` | Добавляет intent matching, консервативный reranking и QoS scoring. |
| GRPO policy | Learned policy из `src/rl/`, запускается через train/evaluate/interactive modes. |

Рекомендуемое сравнение:

```bash
python main.py --mode benchmark --benchmark-policy both --eval-episodes 100 --network-mode controlled
```

Ablation:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --rerank-weight 0.35 --provider-group-weight 0.0
python main.py --mode benchmark --benchmark-policy adaptive --rerank-weight 1.0 --provider-group-weight 1.0
```

Для разбора ошибок используйте `analyze-log`.
