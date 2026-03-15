# src/data/toolbench_loader.py
from datasets import load_dataset
import json
import random
import ast
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class ToolBenchLoader:
    """Загрузчик данных из ToolBench"""

    def __init__(self, split="train", sample_size=None):
        print(f"\n{'=' * 60}")
        print(f"ЗАГРУЗКА TOOLBENCH {split.upper()} SPLIT")
        print(f"{'=' * 60}")

        # Загружаем датасет
        print("📦 Загрузка датасета с HuggingFace...")
        self.dataset = load_dataset("Maurus/ToolBench", split=split)

        print(f"✅ Датасет загружен!")
        print(f"   - Размер: {len(self.dataset)} примеров")
        print(f"   - Колонки: {self.dataset.column_names}")

        # Ограничиваем размер если нужно
        if sample_size and sample_size < len(self.dataset):
            self.dataset = self.dataset.select(range(sample_size))
            print(f"   - Используем {sample_size} примеров")

        # Обрабатываем данные
        print("\n🔄 Обработка данных...")
        self.data = self._process_dataset()

        # Извлекаем инструменты
        print("\n🔧 Извлечение инструментов...")
        self.tools = self._extract_tools()

        print(f"\n{'=' * 60}")
        print(f"ИТОГИ ЗАГРУЗКИ:")
        print(f"   - Примеров: {len(self.data)}")
        print(f"   - Инструментов: {len(self.tools)}")

        if self.tools:
            print(f"\n📋 ПЕРВЫЕ 5 ИНСТРУМЕНТОВ:")
            for i, tool in enumerate(self.tools[:5]):
                print(f"   {i + 1}. {tool['name']}")
                print(f"      Категория: {tool['category']}")
                print(f"      Описание: {tool['description'][:100]}...")
        else:
            print("\n⚠️ ИНСТРУМЕНТЫ НЕ НАЙДЕНЫ!")
            print("   Проверьте структуру датасета...")
            self._debug_dataset_structure()

    def _safe_parse_json(self, data: Any) -> Any:
        """Безопасный парсинг JSON/строки в Python объект"""
        if isinstance(data, (dict, list)):
            return data

        if isinstance(data, str):
            # Пробуем разные методы парсинга
            try:
                return json.loads(data)
            except:
                try:
                    return ast.literal_eval(data)
                except:
                    # Если это строка с экранированными кавычками
                    try:
                        # Убираем экранирование
                        cleaned = data.replace('\\"', '"').replace("\\'", "'")
                        return json.loads(cleaned)
                    except:
                        pass
        return data

    def _process_dataset(self) -> List[Dict]:
        """Обработка датасета с прогресс-баром"""
        processed = []

        for idx, item in enumerate(tqdm(self.dataset, desc="Обработка примеров")):
            try:
                # Получаем запрос
                query = item.get('query', '')
                if not query:
                    continue

                # Получаем API лист и парсим
                api_list = item.get('api_list', [])
                api_list = self._safe_parse_json(api_list)

                # Убеждаемся, что это список
                if not isinstance(api_list, list):
                    api_list = []

                # Получаем домен
                domain = item.get('domain', '')
                if isinstance(domain, bytes):
                    domain = domain.decode('utf-8')

                # Получаем ответ и парсим
                answer = item.get('answer', {})
                answer = self._safe_parse_json(answer)
                if not isinstance(answer, dict):
                    answer = {}

                # Получаем embedding если есть
                embedding = item.get('embedding', [])
                embedding = self._safe_parse_json(embedding)
                if not isinstance(embedding, list):
                    embedding = []

                processed.append({
                    'query': query,
                    'api_list': api_list,
                    'domain': domain,
                    'answer': answer,
                    'embedding': embedding,
                    'query_id': str(item.get('query_id', idx))
                })

            except Exception as e:
                print(f"   ⚠️ Ошибка в примере {idx}: {e}")
                continue

        return processed

    def _extract_tools(self) -> List[Dict]:
        """Извлечение всех уникальных инструментов"""
        tools_dict = {}

        print("\n🔍 Ищем инструменты в датасете...")

        for idx, item in enumerate(tqdm(self.data, desc="Извлечение инструментов")):
            api_list = item['api_list']

            if not api_list or not isinstance(api_list, list):
                continue

            for api in api_list:
                if not isinstance(api, dict):
                    continue

                # Получаем имя инструмента
                tool_name = api.get('tool_name', '')
                api_name = api.get('api_name', '')

                if not tool_name or not api_name:
                    continue

                tool_key = f"{tool_name}.{api_name}"

                if tool_key not in tools_dict:
                    # Получаем описание
                    description = api.get('api_description', '')
                    if not description:
                        description = api.get('description', '')

                    # Получаем категорию
                    category = api.get('category_name', 'Unknown')

                    # Получаем параметры
                    required = api.get('required_parameters', [])
                    required = self._safe_parse_json(required)
                    if not isinstance(required, list):
                        required = []

                    optional = api.get('optional_parameters', [])
                    optional = self._safe_parse_json(optional)
                    if not isinstance(optional, list):
                        optional = []

                    tools_dict[tool_key] = {
                        'name': tool_key,
                        'tool_name': tool_name,
                        'api_name': api_name,
                        'description': description,
                        'category': category,
                        'required_parameters': required,
                        'optional_parameters': optional,
                        'method': api.get('method', 'GET'),
                        'base_latency': random.uniform(0.1, 0.5),
                        'failure_rate': random.uniform(0.01, 0.1)
                    }

        tools = list(tools_dict.values())
        print(f"\n📊 Найдено {len(tools)} уникальных инструментов")
        return tools

    def _debug_dataset_structure(self):
        """Отладка структуры датасета"""
        print("\n🔍 ОТЛАДКА СТРУКТУРЫ ДАТАСЕТА:")

        # Смотрим первый пример
        if len(self.dataset) > 0:
            first = self.dataset[0]
            print(f"\nПервый пример (ключи): {list(first.keys())}")

            # Проверяем api_list
            api_list = first.get('api_list', [])
            print(f"Тип api_list: {type(api_list)}")

            if isinstance(api_list, str):
                print(f"api_list (строка, первые 200 символов): {api_list[:200]}")
                try:
                    # Пробуем разные методы парсинга
                    print("\nПопытки парсинга:")

                    # Метод 1: json.loads
                    try:
                        parsed = json.loads(api_list)
                        print(
                            f"✓ json.loads успешен: тип {type(parsed)}, длина {len(parsed) if isinstance(parsed, list) else 'не список'}")
                    except Exception as e:
                        print(f"✗ json.loads: {e}")

                    # Метод 2: ast.literal_eval
                    try:
                        parsed = ast.literal_eval(api_list)
                        print(
                            f"✓ ast.literal_eval успешен: тип {type(parsed)}, длина {len(parsed) if isinstance(parsed, list) else 'не список'}")
                    except Exception as e:
                        print(f"✗ ast.literal_eval: {e}")

                    # Метод 3: с очисткой экранирования
                    try:
                        cleaned = api_list.replace('\\"', '"').replace("\\'", "'")
                        parsed = json.loads(cleaned)
                        print(f"✓ json.loads с очисткой успешен: тип {type(parsed)}")
                    except Exception as e:
                        print(f"✗ json.loads с очисткой: {e}")

                except:
                    print("Не удалось распарсить JSON")

            # Проверяем answer
            answer = first.get('answer', {})
            print(f"\nТип answer: {type(answer)}")
            if isinstance(answer, str):
                print(f"answer (строка, первые 200 символов): {answer[:200]}")

    def get_training_prompts(self) -> List[Dict]:
        """Получение промптов для обучения"""
        prompts = []

        for item in self.data:
            relevant_tools = []
            target_tool = None

            # Если есть ответ, извлекаем целевой инструмент
            answer = item.get('answer', {})
            if isinstance(answer, dict):
                tool_name = answer.get('tool_name')
                api_name = answer.get('api_name')
                if tool_name and api_name:
                    target_tool = f"{tool_name}.{api_name}"

            # Ищем инструменты из api_list в нашем словаре
            for api in item['api_list']:
                if isinstance(api, dict):
                    t_name = api.get('tool_name')
                    a_name = api.get('api_name')
                    if t_name and a_name:
                        tool_key = f"{t_name}.{a_name}"
                        for tool in self.tools:
                            if tool['name'] == tool_key:
                                relevant_tools.append(tool)
                                break

            prompts.append({
                'query': item['query'],
                'query_id': item['query_id'],
                'domain': item['domain'],
                'relevant_tools': relevant_tools,
                'target_tool': target_tool
            })

        return prompts

    def sample_tools(self, n: int = 10) -> List[Dict]:
        """Случайная выборка инструментов"""
        if not self.tools:
            print("⚠️ Нет инструментов для выборки!")
            return []

        n = min(n, len(self.tools))
        sampled = random.sample(self.tools, n)
        print(f"   Выбрано {n} инструментов для обучения")
        return sampled