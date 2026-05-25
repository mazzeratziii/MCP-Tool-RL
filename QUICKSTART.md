# Быстрый старт

При запуске обучения автоматически создаются `runs/training_metrics.csv` и `runs/training_curve.png`, чтобы было видно динамику loss, reward, relevance и success.

## 1. Окружение

Windows:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Выбрать один инструмент

Проверить окружение:

```bash
python main.py --mode healthcheck
```

```bash
python main.py --mode select --query "What is the weather in London?" --network-mode controlled
```

## 3. Запустить benchmark

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled --log-path runs/selection_benchmark.jsonl
```

Сравнить semantic-only и adaptive policy:

```bash
python main.py --mode benchmark --benchmark-policy both --eval-episodes 100 --network-mode controlled
```

Сравнить semantic-only, SONAR-style и adaptive policy:

```bash
python main.py --mode benchmark --benchmark-policy all --eval-episodes 100 --network-mode controlled
```

## 4. Проанализировать ошибки

```bash
python main.py --mode analyze-log --log-path runs/selection_benchmark.jsonl --example-limit 3
```

## 5. Обучение / оценка GRPO

```bash
python main.py --mode train --epochs 20 --profile laptop
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200
python main.py --mode interactive --checkpoint checkpoints/best
```

## Полезные флаги

- `--network-mode deterministic|controlled|stochastic`
- `--top-k 10`
- `--semantic-threshold 0.55`
- `--rerank-weight 0.35`
- `--provider-group-weight 0.0`
- `--log-path runs/name.jsonl`
- `--verbose` — показать подробные логи загрузки данных и моделей

## Финальная проверка

```powershell
.\scripts\run_final_checks.ps1
.\scripts\run_final_checks.ps1 -IncludeLLMEvaluate
```
