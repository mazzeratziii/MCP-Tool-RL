import numpy as np
from typing import List, Dict, Any
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Установите sentence-transformers: pip install sentence-transformers")

from core.tool import Tool

logger = logging.getLogger(__name__)

class EmbeddingSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.tools: List[Tool] = []
        self.tool_embeddings: np.ndarray = None

    def initialize(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Модель {self.model_name} загружена")

    def index_tools(self, tools: List[Tool]):
        self.tools = tools
        if not tools:
            self.tool_embeddings = np.array([])
            logger.warning("Нет инструментов для индексации")
            return

        self.initialize()
        texts = [t.text_for_embedding for t in tools]
        logger.info(f"Создание эмбеддингов для {len(tools)} инструментов...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.tool_embeddings = embeddings
        logger.info(f"Эмбеддинги созданы, форма: {self.tool_embeddings.shape}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.tool_embeddings is None or len(self.tools) == 0:
            logger.warning("Поиск невозможен: нет индексированных инструментов")
            return []

        self.initialize()
        q_emb = self.model.encode(query, convert_to_numpy=True)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []

        if self.tool_embeddings.ndim == 1:
            self.tool_embeddings = self.tool_embeddings.reshape(1, -1)

        t_norms = np.linalg.norm(self.tool_embeddings, axis=1)
        t_norms[t_norms == 0] = 1.0

        sim = np.dot(self.tool_embeddings, q_emb) / (t_norms * q_norm)

        k = min(top_k, len(self.tools))
        if k == 0:
            return []
        top_idx = np.argsort(sim)[-k:][::-1]

        results = []
        for idx in top_idx:
            results.append({
                "tool": self.tools[idx],
                "similarity": float(sim[idx]),
                "name": self.tools[idx].name,
                "category": self.tools[idx].category
            })
        return results

# Глобальный экземпляр
searcher = EmbeddingSearcher()