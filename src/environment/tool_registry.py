# Реестр инструментов
import os
import re
from typing import List, Dict, Optional, Set
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
from src.config import Config
from src.environment.query_classifier import QueryClassifier


class ToolRegistry:
    """
    Управляет метаданными инструментов и семантическим поиском.

    Исправления относительно исходной версии:
    - Кодирует все инструменты одним batch при инициализации
    - Заранее считает нормализованную матрицу embeddings для быстрого top-k
    - Неизвестный tool_name возвращает 0.0 вместо ошибки
    - Использует словари name→tool и name→index с доступом O(1)
    """

    def __init__(self, config: Config):
        """Initialize the object."""
        self.config = config

        if not config.tools:
            print("Tools not loaded yet, loading data...")
            config.load_data()

        self.tools: List[Dict] = config.tools
        self._name_to_tool: Dict[str, Dict] = {t["name"]: t for t in self.tools}
        self._name_to_idx: Dict[str, int] = {t["name"]: i for i, t in enumerate(self.tools)}

        # Инициализируем классификатор запросов
        self.query_classifier = QueryClassifier()

        tool_texts = [self._tool_text(t) for t in self.tools]
        self._tool_tokens: List[Set[str]] = [self._tokenize(text) for text in tool_texts]
        self.encoder = self._load_encoder()
        self._tool_matrix: Optional[np.ndarray] = None

        if self.encoder is not None:
            print(f"Encoding {len(self.tools)} tools in batch...")
            raw = self.encoder.encode(
                tool_texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Форма: (n_tools, embed_dim), уже L2-нормализовано
            self._tool_matrix = raw.astype(np.float32)
            print("Tool embeddings ready")
        else:
            print("Using lexical fallback retriever (no embedding weights available)")

        self._query_cache: Dict[str, np.ndarray] = {}
        self.semantic_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _tool_text(self, tool: Dict) -> str:
        """Handle tool text."""
        text = f"{tool['name']} - {tool.get('category', 'general')}: {tool.get('description', '')}"
        req = tool.get("required_parameters", [])
        if req and isinstance(req, list):
            try:
                params = ", ".join(
                    p["name"] if isinstance(p, dict) and "name" in p else str(p)
                    for p in req[:6]
                )
                if params:
                    text += f" Required: {params}"
            except Exception:
                pass
        return text

    def _load_encoder(self):
        """Load load encoder."""
        if SentenceTransformer is None:
            print("sentence-transformers is not installed")
            return None

        local_model_path = os.getenv("RETRIEVER_MODEL_PATH", "models/retriever")
        fallback_model = os.getenv("RETRIEVER_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        for model_path in (local_model_path, fallback_model):
            if model_path == local_model_path and not self._has_model_weights(model_path):
                print(f"Retriever weights not found in {model_path}")
                continue
            try:
                print(f"Loading embedding model: {model_path}")
                encoder = SentenceTransformer(model_path)
                print("Embedding model loaded")
                return encoder
            except Exception as exc:
                print(f"Could not load embedding model '{model_path}': {exc}")
        return None

    def _has_model_weights(self, model_path: str) -> bool:
        """Handle has model weights."""
        if not os.path.isdir(model_path):
            return False
        weight_names = {
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        }
        for root, _, files in os.walk(model_path):
            if weight_names.intersection(files):
                return True
            if "model.safetensors.index.json" in files:
                return True
        return False

    def _tokenize(self, text: str) -> Set[str]:
        """Handle tokenize."""
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(token) > 1
        }

    def _lexical_similarity(self, query: str, tool_idx: int) -> float:
        """Handle lexical similarity."""
        q_tokens = self._tokenize(query)
        t_tokens = self._tool_tokens[tool_idx]
        if not q_tokens or not t_tokens:
            return 0.0

        overlap = len(q_tokens & t_tokens)
        precision = overlap / len(q_tokens)
        coverage = overlap / len(t_tokens)
        name_tokens = self._tokenize(self.tools[tool_idx]["name"])
        name_bonus = 0.15 if q_tokens & name_tokens else 0.0
        return min(1.0, 0.75 * precision + 0.25 * coverage + name_bonus)

    def _encode_query(self, query: str) -> np.ndarray:
        """Возвращает кешированный L2-нормализованный embedding запроса."""
        if self.encoder is None:
            raise RuntimeError("Embedding encoder is not available")
        if query not in self._query_cache:
            emb = self.encoder.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            self._query_cache[query] = emb.astype(np.float32)
        return self._query_cache[query]

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def get_tool_by_name(self, name: str) -> Optional[Dict]:
        """Return get tool by name."""
        return self._name_to_tool.get(name)

    def get_tools_by_category(self, category: str) -> List[Dict]:
        """Return get tools by category."""
        cat = category.lower()
        return [t for t in self.tools if t.get("category", "").lower() == cat]

    def semantic_similarity(self, query: str, tool_name: str) -> float:
        """
        Косинусная близость, приведённая к диапазону [0, 1].
        Для неизвестного tool_name возвращает 0.0 и не выбрасывает KeyError.
        """
        cache_key = f"{query}\x00{tool_name}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]

        idx = self._name_to_idx.get(tool_name)
        if idx is None:
            return 0.0

        if self._tool_matrix is None:
            result = self._lexical_similarity(query, idx)
            self.semantic_cache[cache_key] = result
            return result

        q_emb = self._encode_query(query)
        t_emb = self._tool_matrix[idx]
        # Оба вектора нормализованы: dot == cosine ∈ [-1, 1]
        raw_sim = float(np.dot(q_emb, t_emb))
        result = (raw_sim + 1.0) / 2.0

        self.semantic_cache[cache_key] = result
        return result

    def get_top_k_tools(self, query: str, k: int = 10) -> List[Dict]:
        """
        Быстрый top-k через одно умножение матрицы на вектор.
        Это быстрее, чем вызывать semantic_similarity для каждого инструмента.

        Автоматически добавляет fallback инструменты если они релевантны.
        """
        if self._tool_matrix is None:
            scores = np.array(
                [self._lexical_similarity(query, i) for i in range(len(self.tools))],
                dtype=np.float32,
            )
        else:
            q_emb = self._encode_query(query)                  # (d,)
            scores = self._tool_matrix @ q_emb                 # (n,)
        k = min(k, len(self.tools))
        top_indices = np.argpartition(scores, -k)[-k:]         # первые k кандидатов без сортировки
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]  # сортируем по убыванию

        results = [self.tools[i] for i in top_indices]

        # Проверяем, нужен ли fallback инструмент
        fallback_info = self.query_classifier.should_add_fallback_to_candidates(query, results)

        if fallback_info:
            fallback_tool_name = fallback_info["name"]
            fallback_tool = self._name_to_tool.get(fallback_tool_name)

            if fallback_tool and fallback_tool not in results:
                # Добавляем fallback инструмент в начало списка с высоким приоритетом
                results.insert(0, fallback_tool)
                # Удаляем последний инструмент, чтобы сохранить размер k
                if len(results) > k:
                    results = results[:k]

        return results

    def format_tool_for_prompt(self, tool: Dict) -> str:
        """Format format tool for prompt."""
        lines = [
            f"Tool: {tool['name']}",
            f"Description: {tool.get('description', 'No description')}",
            f"Category: {tool.get('category', 'general')}",
            f"Method: {tool.get('method', 'GET')}",
        ]
        req = tool.get("required_parameters", [])
        if req and isinstance(req, list):
            try:
                params = ", ".join(
                    p["name"] if isinstance(p, dict) else str(p) for p in req
                )
                lines.append(f"Required parameters: {params}")
            except Exception:
                pass
        return "\n".join(lines)
