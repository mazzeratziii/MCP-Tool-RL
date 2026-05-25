# Fallback-поведение

В проекте есть несколько fallback-механизмов:

- если `python-dotenv` отсутствует, загрузка `.env` пропускается;
- если в `models/retriever` нет весов, semantic retrieval заменяется lexical scoring;
- `Calculator.Evaluate` обрабатывает простые математические выражения;
- `General.NoToolNeeded` доступен для простых запросов без внешнего инструмента.

Эти fallback-и позволяют запускать лёгкие режимы `select`, `benchmark` и `analyze-log` даже без полной training-среды или весов модели.
