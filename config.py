"""
Конфигурация MCP-Tool-RL v2
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SemanticSearchConfig:
    """Конфигурация семантического поиска"""
    # Используем многоязычную модель для русского языка
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Или: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (лучше, но медленнее)

    cache_dir: str = "./data/embeddings"
    similarity_threshold: float = 0.5  # Повышаем порог
    top_k_results: int = 5
    use_faiss: bool = False

@dataclass
class ToolConfig:
    """Конфигурация инструментов"""
    categories: List[str] = field(default_factory=lambda: [
        "weather", "finance", "transportation", "shopping", "entertainment"
    ])

@dataclass
class Config:
    """Основная конфигурация"""
    semantic: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)

    log_level: str = "INFO"
    debug_mode: bool = False
    max_tools_in_memory: int = 100

    def setup(self):
        """Создаёт необходимые директории"""
        # Важно: используем self, а не cls
        os.makedirs(self.semantic.cache_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

# Создаём экземпляр конфигурации
config = Config()
config.setup()  # Вызываем setup на экземпляре