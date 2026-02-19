import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Tool:
    """Инструмент с минимальной информацией – всё для эмбеддингов."""
    id: str
    name: str
    description: str
    category: str
    api_name: str = ""
    endpoint: str = ""
    method: str = "GET"
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def text_for_embedding(self) -> str:
        """Текст для эмбеддингов – только базовая информация."""
        parts = [self.name, self.description, self.category]
        for ex in self.examples[:2]:
            if 'query' in ex:
                parts.append(ex['query'])
        return " ".join(parts)

    @staticmethod
    def create_id(name: str, category: str, tool_name: str = "", api_name: str = "") -> str:
        """
        Создаёт уникальный ID из комбинации полей.
        Теперь использует и tool_name, и api_name для уникальности.
        """
        unique_string = f"{category}_{tool_name}_{api_name}_{name}"
        h = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        return f"{category}_{h}"