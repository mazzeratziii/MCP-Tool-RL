# Улучшения

Уже реализовано:

- adaptive selector baseline;
- functional intent matcher;
- conservative text reranker;
- QoS-aware scoring;
- benchmark mode;
- log analyzer;
- lexical fallback при отсутствии весов retriever;
- тесты selector-а, reranker-а и анализа логов.

Следующие полезные улучшения:

- обучить learned reranker для top-k candidates;
- улучшить обработку ToolBench aliases;
- добавить multi-tool selection для запросов, требующих несколько инструментов;
- добавить retry simulation и retry-aware benchmark metrics;
- экспортировать benchmark summaries в CSV.
