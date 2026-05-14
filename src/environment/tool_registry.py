# src/environment/tool_registry.py
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import Config
from src.environment.query_classifier import QueryClassifier


class ToolRegistry:
    """
    Manages tool metadata and semantic similarity search.

    Fixes vs original:
    - Batch-encodes all tools once at init (no per-call encode slowdown)
    - Precomputes normalised embedding matrix → top-k is one matmul, O(n·d)
    - Safe KeyError: unknown tool_name returns 0.0, never raises
    - O(1) name→tool and name→index dicts
    """

    def __init__(self, config: Config):
        self.config = config

        if not config.tools:
            print("Tools not loaded yet, loading data...")
            config.load_data()

        self.tools: List[Dict] = config.tools
        self._name_to_tool: Dict[str, Dict] = {t["name"]: t for t in self.tools}
        self._name_to_idx: Dict[str, int] = {t["name"]: i for i, t in enumerate(self.tools)}

        # Инициализируем классификатор запросов
        self.query_classifier = QueryClassifier()

        print("Loading embedding model...")
        self.encoder = SentenceTransformer("models/retriever")
        print("Embedding model loaded")

        print(f"Encoding {len(self.tools)} tools in batch...")
        tool_texts = [self._tool_text(t) for t in self.tools]
        raw = self.encoder.encode(
            tool_texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # Shape: (n_tools, embed_dim), already L2-normalised
        self._tool_matrix: np.ndarray = raw.astype(np.float32)
        print("Tool embeddings ready")

        self._query_cache: Dict[str, np.ndarray] = {}
        self.semantic_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tool_text(self, tool: Dict) -> str:
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

    def _encode_query(self, query: str) -> np.ndarray:
        """Return cached, L2-normalised query embedding."""
        if query not in self._query_cache:
            emb = self.encoder.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            self._query_cache[query] = emb.astype(np.float32)
        return self._query_cache[query]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_by_name(self, name: str) -> Optional[Dict]:
        return self._name_to_tool.get(name)

    def get_tools_by_category(self, category: str) -> List[Dict]:
        cat = category.lower()
        return [t for t in self.tools if t.get("category", "").lower() == cat]

    def semantic_similarity(self, query: str, tool_name: str) -> float:
        """
        Cosine similarity mapped to [0, 1].
        Returns 0.0 for unknown tool_name — never raises KeyError.
        """
        cache_key = f"{query}\x00{tool_name}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]

        idx = self._name_to_idx.get(tool_name)
        if idx is None:
            return 0.0

        q_emb = self._encode_query(query)
        t_emb = self._tool_matrix[idx]
        # Both normalised → dot == cosine ∈ [-1, 1]
        raw_sim = float(np.dot(q_emb, t_emb))
        result = (raw_sim + 1.0) / 2.0

        self.semantic_cache[cache_key] = result
        return result

    def get_top_k_tools(self, query: str, k: int = 10) -> List[Dict]:
        """
        Fast top-k via a single matrix–vector multiply.
        Much faster than calling semantic_similarity N times.

        Автоматически добавляет fallback инструменты если они релевантны.
        """
        q_emb = self._encode_query(query)                      # (d,)
        scores = self._tool_matrix @ q_emb                     # (n,)
        k = min(k, len(self.tools))
        top_indices = np.argpartition(scores, -k)[-k:]         # unsorted top-k
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]  # sort desc

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