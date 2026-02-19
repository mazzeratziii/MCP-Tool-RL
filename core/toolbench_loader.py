import logging
import json
from typing import List, Optional, Dict, Any, Set
from datasets import load_dataset
from core.tool import Tool

logger = logging.getLogger(__name__)


class ToolBenchLoader:
    def __init__(self, dataset_name: str = "Maurus/ToolBench", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split

    def _get_name(self, data: Dict[str, Any]) -> str:
        """Извлекает имя инструмента."""
        tool_name = data.get("tool_name", "")
        api_name = data.get("api_name", "")
        if tool_name and api_name:
            return f"{tool_name} - {api_name}"
        elif tool_name:
            return tool_name
        elif api_name:
            return api_name
        return "unknown_tool"

    def _get_category(self, data: Dict[str, Any]) -> str:
        """Извлекает категорию."""
        return data.get("category_name", "general")

    def _parse_api_list(self, api_list_str: str) -> List[Dict[str, Any]]:
        """Парсит строку JSON в список инструментов."""
        try:
            fixed_str = api_list_str.replace("'", '"')
            return json.loads(fixed_str)
        except Exception as e:
            logger.warning(f"Ошибка парсинга api_list: {e}")
            return []

    def _get_tool_key(self, data: Dict[str, Any]) -> str:
        """Создаёт уникальный ключ для инструмента (для дедупликации)."""
        tool_name = data.get("tool_name", "")
        api_name = data.get("api_name", "")
        category = data.get("category_name", "")
        return f"{category}|{tool_name}|{api_name}"

    def load_tools(self, limit: Optional[int] = None, categories: Optional[List[str]] = None) -> List[Tool]:
        logger.info(f"Загрузка инструментов из {self.dataset_name} (лимит: {limit}, категории: {categories})...")
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)

        tools = []
        seen_tools: Set[str] = set()  # Множество для отслеживания уникальных инструментов
        count = 0
        debug_printed = False

        for sample in dataset:
            api_list_str = sample.get("api_list", "[]")
            api_list = self._parse_api_list(api_list_str)

            for tool_data in api_list:
                # Проверяем, не видели ли мы уже такой инструмент
                tool_key = self._get_tool_key(tool_data)
                if tool_key in seen_tools:
                    continue  # Пропускаем дубликат

                seen_tools.add(tool_key)

                if count == 0 and not debug_printed:
                    print("\n" + "=" * 80)
                    print("🔍 ПЕРВЫЙ УНИКАЛЬНЫЙ ИНСТРУМЕНТ:")
                    print(json.dumps(tool_data, indent=2, ensure_ascii=False))
                    print("=" * 80 + "\n")
                    debug_printed = True

                name = self._get_name(tool_data)
                category = self._get_category(tool_data)
                tool_name = tool_data.get("tool_name", "")
                api_name = tool_data.get("api_name", "")

                if categories and category not in categories:
                    continue

                description = tool_data.get("api_description", "")
                parameters = []

                # Обрабатываем обязательные параметры
                for param in tool_data.get("required_parameters", []):
                    if isinstance(param, dict):
                        param_name = str(param.get('name', ''))
                    else:
                        param_name = str(param)

                    if param_name:
                        parameters.append({
                            "name": param_name,
                            "type": "string",
                            "required": True,
                            "description": ""
                        })

                # Обрабатываем опциональные параметры
                for param in tool_data.get("optional_parameters", []):
                    if isinstance(param, dict):
                        param_name = str(param.get('name', ''))
                    else:
                        param_name = str(param)

                    if param_name:
                        parameters.append({
                            "name": param_name,
                            "type": "string",
                            "required": False,
                            "description": ""
                        })

                required_params = [p["name"] for p in parameters if p.get("required")]
                method = tool_data.get("method", "GET")

                # Генерируем уникальный ID
                tool_id = Tool.create_id(name, category, tool_name, api_name)

                tool = Tool(
                    id=tool_id,
                    name=name,
                    description=description,
                    category=category,
                    api_name=api_name,
                    endpoint="",
                    method=method,
                    parameters=parameters,
                    required_params=required_params,
                    examples=[]
                )
                tools.append(tool)
                count += 1

                if limit and count >= limit:
                    logger.info(f"Достигнут лимит {limit} уникальных инструментов, завершаем загрузку.")
                    return tools

        logger.info(f"Загружено {len(tools)} уникальных инструментов")
        return tools