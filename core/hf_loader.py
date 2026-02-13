"""
Онлайн-загрузчик инструментов ToolBench с HuggingFace
Не требует скачивания 20 ГБ на диск
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import logging
from datetime import datetime

try:
    from datasets import load_dataset, Dataset, IterableDataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("HuggingFace datasets не установлен. Используйте: pip install datasets")

from config.settings import config

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Спецификация инструмента из ToolBench"""
    id: str
    name: str
    description: str
    category: str
    api_name: str
    endpoint: str
    method: str = "GET"
    parameters: List[Dict[str, Any]] = None
    required_params: List[str] = None
    returns: Dict[str, Any] = None
    examples: List[Dict[str, Any]] = None
    is_simulated: bool = True  # По умолчанию симулируем вызовы

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.required_params is None:
            self.required_params = []
        if self.returns is None:
            self.returns = {"type": "string", "description": "Response data"}
        if self.examples is None:
            self.examples = []

    @property
    def signature(self) -> str:
        """Уникальная сигнатура инструмента"""
        return f"{self.category}:{self.api_name}:{self.endpoint}"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "api_name": self.api_name,
            "endpoint": self.endpoint,
            "method": self.method,
            "parameters": self.parameters,
            "required_params": self.required_params,
            "examples": self.examples,
            "signature": self.signature
        }


class OnlineToolLoader:
    """
    Загрузчик инструментов ToolBench с HuggingFace.
    Работает в режиме потоковой загрузки, не сохраняя данные на диск.
    """

    def __init__(self):
        if not HF_AVAILABLE:
            raise ImportError(
                "Установите библиотеку datasets: pip install datasets\n"
                "Или используйте альтернативный загрузчик"
            )

        self.config = config
        self.tools: Dict[str, ToolSpec] = {}
        self.categories: Dict[str, List[str]] = {}
        self.loaded = False

        # Кэш для быстрого доступа
        self._name_to_id: Dict[str, str] = {}
        self._category_to_ids: Dict[str, List[str]] = {}

        logger.info("Инициализирован онлайн-загрузчик ToolBench")

    def load_tools(
            self,
            categories: Optional[List[str]] = None,
            limit: Optional[int] = None,
            use_cache: bool = True
    ) -> Dict[str, ToolSpec]:
        """
        Загружает инструменты с HuggingFace.

        Args:
            categories: Список категорий для загрузки
            limit: Максимальное количество инструментов
            use_cache: Использовать локальный кэш

        Returns:
            Словарь инструментов {id: ToolSpec}
        """
        if self.loaded and use_cache:
            logger.info("Используются загруженные инструменты из памяти")
            return self.tools

        if categories is None:
            categories = self.config.hf.available_categories

        logger.info(f"Начинаю загрузку инструментов категорий: {categories}")

        all_tools = {}
        total_loaded = 0

        for category in categories:
            if limit and total_loaded >= limit:
                break

            try:
                category_tools = self._load_category(category, limit)
                all_tools.update(category_tools)
                total_loaded += len(category_tools)

                logger.info(f"Загружено {len(category_tools)} инструментов из категории '{category}'")

            except Exception as e:
                logger.error(f"Ошибка загрузки категории '{category}': {e}")
                continue

        self.tools = all_tools
        self._build_indices()
        self.loaded = True

        logger.info(f"Всего загружено {len(self.tools)} инструментов")
        return self.tools

    def _load_category(self, category: str, limit: Optional[int] = None) -> Dict[str, ToolSpec]:
        """
        Загружает инструменты конкретной категории.

        Args:
            category: Название категории
            limit: Лимит инструментов

        Returns:
            Словарь инструментов категории
        """
        tools = {}

        try:
            # Пытаемся загрузить конкретную категорию
            dataset = load_dataset(
                self.config.hf.dataset_name,
                category,
                split="train",
                streaming=self.config.hf.streaming,
                cache_dir=self.config.hf.cache_dir,
                trust_remote_code=self.config.hf.trust_remote_code
            )
        except Exception:
            # Если категория не найдена, используем основной датасет и фильтруем
            logger.warning(f"Категория '{category}' не найдена, фильтруем из основного датасета")
            dataset = load_dataset(
                self.config.hf.dataset_name,
                self.config.hf.config_name,
                split="train",
                streaming=self.config.hf.streaming,
                cache_dir=self.config.hf.cache_dir
            )
            # Фильтрация по категории будет в цикле

        count = 0
        for item in dataset:
            if limit and count >= limit:
                break

            # Проверяем категорию
            item_category = item.get('category', '').lower()
            if item_category != category.lower():
                # Если загрузили основной датасет, фильтруем
                if self.config.hf.config_name in dataset.config_name:
                    if category.lower() not in item_category:
                        continue
                else:
                    continue

            try:
                tool = self._create_tool_from_item(item, category)
                tools[tool.id] = tool
                count += 1

                # Для потоковой загрузки выводим прогресс
                if count % 100 == 0:
                    logger.debug(f"Обработано {count} инструментов категории '{category}'")

            except Exception as e:
                logger.warning(f"Ошибка создания инструмента: {e}")
                continue

        return tools

    def _create_tool_from_item(self, item: Dict[str, Any], category: str) -> ToolSpec:
        """
        Создаёт ToolSpec из элемента датасета.

        Args:
            item: Элемент датасета HuggingFace
            category: Категория инструмента

        Returns:
            ToolSpec объект
        """
        # Генерируем уникальный ID
        tool_hash = hashlib.md5(
            f"{item.get('tool_name', '')}:{item.get('api_name', '')}:{category}".encode()
        ).hexdigest()[:12]

        tool_id = f"{category}_{tool_hash}"

        # Извлекаем параметры
        parameters = item.get('parameters', [])
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except:
                parameters = []

        # Определяем обязательные параметры
        required_params = []
        for param in parameters if isinstance(parameters, list) else []:
            if isinstance(param, dict) and param.get('required', False):
                required_params.append(param.get('name', ''))

        # Извлекаем примеры
        examples = item.get('examples', [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except:
                examples = []

        return ToolSpec(
            id=tool_id,
            name=item.get('tool_name', f'Unnamed {category} Tool'),
            description=item.get('description', 'No description available'),
            category=category,
            api_name=item.get('api_name', item.get('tool_name', 'unknown')),
            endpoint=item.get('endpoint', f'/api/{category}'),
            method=item.get('method', 'GET'),
            parameters=parameters if isinstance(parameters, list) else [],
            required_params=required_params,
            returns=item.get('returns', {'type': 'string'}),
            examples=examples if isinstance(examples, list) else [],
            is_simulated=config.network.enable_simulation
        )

    def _build_indices(self):
        """Строит индексы для быстрого поиска"""
        self._name_to_id = {}
        self._category_to_ids = {}

        for tool_id, tool in self.tools.items():
            # Индекс по имени
            self._name_to_id[tool.name.lower()] = tool_id

            # Индекс по категории
            if tool.category not in self._category_to_ids:
                self._category_to_ids[tool.category] = []
            self._category_to_ids[tool.category].append(tool_id)

        # Сохраняем статистику
        self.categories = {
            cat: len(ids) for cat, ids in self._category_to_ids.items()
        }

    def get_tool(self, tool_id: str) -> Optional[ToolSpec]:
        """Получает инструмент по ID"""
        return self.tools.get(tool_id)

    def get_tool_by_name(self, name: str) -> Optional[ToolSpec]:
        """Получает инструмент по имени"""
        tool_id = self._name_to_id.get(name.lower())
        return self.tools.get(tool_id) if tool_id else None

    def get_tools_by_category(self, category: str) -> List[ToolSpec]:
        """Получает все инструменты категории"""
        tool_ids = self._category_to_ids.get(category, [])
        return [self.tools[tid] for tid in tool_ids]

    def search_tools(
            self,
            query: str,
            category: Optional[str] = None,
            limit: int = 10
    ) -> List[ToolSpec]:
        """
        Простой текстовый поиск инструментов.
        В production заменить на семантический поиск.

        Args:
            query: Поисковый запрос
            category: Ограничение по категории
            limit: Максимум результатов

        Returns:
            Список инструментов
        """
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            if category and tool.category.lower() != category.lower():
                continue

            # Простая проверка по тексту
            if (query_lower in tool.name.lower() or
                    query_lower in tool.description.lower() or
                    query_lower in tool.category.lower()):
                results.append(tool)

            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по загруженным инструментам"""
        return {
            "total_tools": len(self.tools),
            "categories": self.categories,
            "categories_count": len(self.categories),
            "avg_tools_per_category": sum(self.categories.values()) / len(self.categories) if self.categories else 0,
            "loaded_at": datetime.now().isoformat()
        }


# Глобальный экземпляр загрузчика
loader = OnlineToolLoader()