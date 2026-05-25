# Retriever Model

Эта папка хранит файлы sentence-transformers retriever.

Большие веса вроде `model.safetensors` или `pytorch_model.bin` могут отсутствовать в Git, потому что они игнорируются как крупные артефакты.

Если весов нет, `ToolRegistry` автоматически использует lexical retrieval. Это позволяет запускать лёгкие команды:

```bash
python main.py --mode select --query "What is the weather in London?"
python main.py --mode benchmark --benchmark-policy adaptive
```

Для лучшего semantic retrieval положите полную sentence-transformers модель в эту папку или задайте:

```text
RETRIEVER_MODEL_PATH=path/to/model
```
