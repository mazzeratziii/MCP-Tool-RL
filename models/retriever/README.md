# Retriever model

Эта папка предназначена для локальной sentence-transformers модели, которая используется в `ToolRegistry` для semantic retrieval.

Большие веса модели, например `model.safetensors` или `pytorch_model.bin`, не должны храниться в GitHub. Они игнорируются как крупные артефакты.

Если веса отсутствуют, проект не падает: `ToolRegistry` автоматически использует lexical fallback. Это менее точно, но позволяет запускать базовые режимы:

```powershell
python main.py --mode healthcheck
python main.py --mode select --query "What is the weather in London?"
python main.py --mode benchmark --benchmark-policy adaptive
```

Чтобы использовать полноценный semantic retriever, положите модель в эту папку или задайте путь в `.env`:

```env
RETRIEVER_MODEL_PATH=models/retriever
```
