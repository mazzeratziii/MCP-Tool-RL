# Финальный отчёт

## Цель проекта

Проект реализует прототип выбора MCP-инструментов в условиях нестабильной сети. Агент должен учитывать два типа требований:

- функциональные: инструмент должен подходить по смыслу к запросу пользователя;
- нефункциональные: инструмент должен быть доступным, устойчивым и достаточно быстрым.

В качестве данных используется ToolBench. Сетевые условия эмулируются через режимы `deterministic`, `controlled` и `stochastic`.

## Связь со статьёй NetMCP

Проект основан на идее NetMCP/SONAR: выбор инструмента должен учитывать не только semantic relevance, но и сетевые QoS-характеристики. В статье SONAR использует фиксированную взвешенную функцию:

```text
final_score = alpha * semantic_score + beta * network_score
```

В проекте эта идея расширена:

- добавлен SONAR-style baseline;
- добавлен adaptive selector с functional matching и QoS-признаками;
- добавлена GRPO-политика, которая обучается выбирать инструмент по reward-сигналу.

Таким образом, проект можно рассматривать как RL-расширение NetMCP/SONAR.

## Реализованные компоненты

- загрузка и фильтрация ToolBench;
- retriever для выбора candidate tools;
- fallback при отсутствии весов retriever;
- эмулятор сетевых условий;
- semantic-only baseline;
- SONAR-style baseline;
- adaptive selector;
- GRPO training pipeline;
- evaluation trained LLM policy;
- interactive mode;
- query rewrite для multilingual/noisy запросов;
- отказ от выполнения при ненадёжном top-1;
- JSONL-логи benchmark;
- анализ ошибок;
- график обучения;
- healthcheck;
- финальный скрипт проверки.

## Финальные метрики

### Adaptive baseline

Команда:

```bash
python main.py --mode benchmark --benchmark-policy adaptive --eval-episodes 100 --network-mode controlled
```

Результаты на разных прогонах controlled benchmark:

| Policy | Relevance@1 | Relevance@3 | Success | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive conservative | 53-56% | 68-70% | 95-96% | 0.326-0.358s | 2.621-2.727 |

### Deterministic benchmark

Команда:

```bash
python main.py --mode benchmark --benchmark-policy all
```

Результаты на 200 episodes в `deterministic` network:

| Policy | Relevance@1 | Relevance@3 | Success | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic-only | 77.00% | 84.50% | 89.00% | 0.142s | 3.114 |
| SONAR-style | 75.50% | 84.00% | 94.50% | 0.148s | 3.328 |
| adaptive | 67.00% | 83.00% | 93.50% | 0.144s | 3.053 |

Отдельный повторный запуск SONAR на 200 episodes:

| Policy | Relevance@1 | Relevance@3 | Success | Avg latency | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| SONAR-style | 75.50% | 84.50% | 93.50% | 0.145s | 3.342 |

В deterministic-сценарии semantic-only даёт лучший Relevance@1, потому что сеть почти не создаёт конфликт между смысловой релевантностью и QoS. SONAR при этом повышает success rate и reward за счёт учёта сетевых признаков. Adaptive selector в этой конфигурации уступает по Relevance@1, что показывает важность отдельного сравнения baseline-ов, а не только одного эвристического ранжировщика.

### GRPO-trained LLM

Обучение:

```bash
python main.py --mode train --epochs 40 --profile laptop
```

Training на 40-й эпохе:

| Metric | Value |
| --- | ---: |
| Relevance | 80.56% |
| Success | 91.28% |
| Avg reward | 3.698 |
| Rollouts | 1800 |
| Updates | 409 |
| Skipped groups | 191 |

Evaluation:

```bash
python main.py --mode evaluate --checkpoint checkpoints/best --eval-episodes 200 --network-mode controlled --profile laptop
```

| Metric | Value |
| --- | ---: |
| Success rate | 94.50% |
| Relevance@1 | 78.50% |
| Relevance@3 | 82.50% |
| Top-3 gap | +4.00% |
| Avg latency | 0.315s |
| Fast tool choices | 14.00% |
| Available choices | 100.00% |

## Выводы

GRPO-политика заметно улучшила функциональную релевантность выбора инструмента по сравнению с эвристическим adaptive baseline:

```text
Relevance@1: примерно 53-56% -> 78.50%
Relevance@3: примерно 68-70% -> 82.50%
```

При этом success rate остался высоким, а модель выбирала только доступные инструменты в evaluation:

```text
Success rate: 94.50%
Available choices: 100.00%
```

Это подтверждает, что обучаемая политика способна лучше учитывать функциональные требования к инструментам, чем статическая weighted-функция.

## Ограничения

- RL-политика ранжирует только уже найденных кандидатов; если retriever принёс плохой top-k, модель не может выбрать правильный инструмент.
- Multilingual и noisy queries требуют более сильного retriever или отдельного обученного query rewriter.
- Значительная часть выполнения инструментов эмулируется.
- ToolBench содержит шумные метки, aliases и дублирующиеся API.
- Latency-компонент после обучения слабее semantic relevance: `Fast tool choices` составляет 14%.
- Текущий прототип выбирает первый инструмент, а не строит полноценный multi-tool plan.

## Финальная проверка

Базовая финальная проверка:

```powershell
.\scripts\run_final_checks.ps1
```

Проверка с evaluation обученной LLM:

```powershell
.\scripts\run_final_checks.ps1 -IncludeLLMEvaluate
```

После выполнения результаты benchmark и JSONL-логи сохраняются в:

```text
runs/final_report/
```

## Статус

Проект можно считать завершённым как исследовательский прототип:

- задача сформулирована;
- базовая статья NetMCP/SONAR учтена;
- baseline реализован;
- RL-расширение реализовано;
- метрики получены;
- ограничения зафиксированы;
- финальный сценарий проверки добавлен.
