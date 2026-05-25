"""
Дообучает sentence-transformer на парах (query, relevant_tool) из ToolBench.

Использование:
    python src/retriever/finetune_retriever.py

Выход: models/retriever/
После этого ToolRegistry использует SentenceTransformer('models/retriever')

Время: примерно 20-40 минут на laptop, около 10 минут на desktop.
VRAM: около 2 GB.
"""

import sys, os, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from src.config import Config

OUTPUT_PATH  = "models/retriever"
BASE_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS       = 3
BATCH_SIZE   = 32
WARMUP_RATIO = 0.1
MAX_TRAIN    = 50_000


# Синтетические примеры для улучшения качества ретривера
SYNTHETIC_EXAMPLES = [
    # Математические запросы
    ("2 + 2", "Calculator.Evaluate"),
    ("what is 10 * 5", "Calculator.Evaluate"),
    ("calculate 100 / 4", "Calculator.Evaluate"),
    ("2 + 234", "Calculator.Evaluate"),
    ("solve 15 - 7", "Calculator.Evaluate"),
    ("compute 3 * 8", "Calculator.Evaluate"),
    ("what's 50 divided by 2", "Calculator.Evaluate"),
    ("add 123 and 456", "Calculator.Evaluate"),
    ("multiply 7 by 9", "Calculator.Evaluate"),
    ("subtract 20 from 100", "Calculator.Evaluate"),
    ("5 + 5 * 2", "Calculator.Evaluate"),
    ("(10 + 5) * 3", "Calculator.Evaluate"),
    ("square root of 16", "Calculator.Evaluate"),
    ("25 percent of 200", "Calculator.Evaluate"),
    ("what is 2 to the power of 8", "Calculator.Evaluate"),

    # Простые вопросы без инструментов
    ("hello", "General.NoToolNeeded"),
    ("hi there", "General.NoToolNeeded"),
    ("thanks", "General.NoToolNeeded"),
    ("thank you", "General.NoToolNeeded"),
    ("what is AI", "General.NoToolNeeded"),
    ("explain machine learning", "General.NoToolNeeded"),
    ("who are you", "General.NoToolNeeded"),
    ("tell me about yourself", "General.NoToolNeeded"),
    ("how are you", "General.NoToolNeeded"),
    ("good morning", "General.NoToolNeeded"),
]


def build_synthetic_examples(tools_set):
    """Создаёт синтетические примеры для fallback инструментов"""
    examples = []
    for query, tool_name in SYNTHETIC_EXAMPLES:
        if tool_name not in tools_set:
            continue
        t = tools_set[tool_name]
        anchor = f"{tool_name} [{t.get('category','')}]: {t.get('description','')[:150]}"
        examples.append(InputExample(texts=[query, anchor]))
    print(f"  synthetic: {len(examples)} pairs")
    return examples


def build_examples(prompts, tools_set, split="train"):
    """?????? ????????? ???? query-tool ??? fine-tuning retriever-?."""
    examples, skipped = [], 0
    for prompt in prompts:
        query = prompt.get("query", "").strip()
        if not query:
            continue
        for tool in prompt.get("relevant_tools", []):
            name = tool.get("name", "")
            if name not in tools_set:
                skipped += 1
                continue
            t = tools_set[name]
            anchor = f"{name} [{t.get('category','')}]: {t.get('description','')[:150]}"
            examples.append(InputExample(texts=[query, anchor]))
    print(f"  {split}: {len(examples)} pairs  ({skipped} skipped)")
    return examples


def main():
    """????? ?????: ????????? ????????? CLI ? ????????? ????????? ?????."""
    print("Loading config and data...")
    config = Config()
    config.load_data()

    tools_set = {t["name"]: t for t in config.tools}
    print(f"Tools in registry: {len(tools_set)}")

    train_prompts = config.train_prompts
    if len(train_prompts) > MAX_TRAIN:
        train_prompts = random.sample(train_prompts, MAX_TRAIN)

    print("\nBuilding training pairs...")
    train_ex = build_examples(train_prompts, tools_set, "train")
    val_ex   = build_examples(config.val_prompts[:5000], tools_set, "val")

    # Добавляем синтетические примеры
    synthetic_ex = build_synthetic_examples(tools_set)
    train_ex.extend(synthetic_ex)
    print(f"  total training pairs: {len(train_ex)} (including synthetic)")

    if not train_ex:
        print("No examples — check prompts have relevant_tools")
        return

    train_loader = DataLoader(train_ex, shuffle=True, batch_size=BATCH_SIZE)

    print(f"\nLoading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    loss  = losses.MultipleNegativesRankingLoss(model)

    # Оценщик на validation-парах
    queries, corpus, relevant_docs = {}, {}, {}
    for i, ex in enumerate(val_ex[:2000]):
        queries[f"q{i}"]       = ex.texts[0]
        corpus[f"c{i}"]        = ex.texts[1]
        relevant_docs[f"q{i}"] = {f"c{i}"}

    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name="toolbench-val", show_progress_bar=False,
    )

    warmup = int(len(train_loader) * EPOCHS * WARMUP_RATIO)
    print(f"Training: {len(train_ex)} pairs  epochs={EPOCHS}  "
          f"batch={BATCH_SIZE}  warmup={warmup}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.fit(
        train_objectives=[(train_loader, loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup,
        output_path=OUTPUT_PATH,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("\nUpdate ToolRegistry (src/environment/tool_registry.py):")
    print("  from: SentenceTransformer('all-MiniLM-L6-v2')")
    print("  to:   SentenceTransformer('models/retriever')")


if __name__ == "__main__":
    main()