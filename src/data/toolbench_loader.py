import json
import random
from typing import List, Dict, Any, Optional
import os
from datasets import load_dataset


class ToolBenchLoader:
    """Загрузчик и обработчик данных из ToolBench с исправленной обработкой"""

    def __init__(self, split="train", sample_size=None):
        """
        Загрузка датасета ToolBench

        Args:
            split: "train" или "test"
            sample_size: количество примеров для загрузки (None = все)
        """
        print(f"Загрузка ToolBench {split} split...")

        try:
            # Пытаемся загрузить через datasets с указанием trust_remote_code
            self.dataset = load_dataset(
                "Maurus/ToolBench",
                split=split,
                trust_remote_code=True
            )

            # Преобразуем в список словарей
            self.data = []
            for item in self.dataset:
                try:
                    processed_item = self._process_item(item)
                    if processed_item:
                        self.data.append(processed_item)
                except Exception as e:
                    print(f"Пропускаем элемент: {e}")
                    continue

        except Exception as e:
            print(f"Ошибка загрузки через datasets: {e}")
            print("Создаем тестовые данные...")
            # Создаем тестовые данные если загрузка не удалась
            self.data = self._create_test_data()

        if sample_size and sample_size < len(self.data):
            self.data = self.data[:sample_size]

        print(f"Загружено {len(self.data)} примеров")

        # Извлекаем все уникальные инструменты
        self.tools = self._extract_all_tools()
        print(f"Найдено {len(self.tools)} уникальных инструментов")

    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Безопасная обработка элемента датасета"""
        try:
            # Получаем api_list
            api_list = item.get('api_list', [])
            if isinstance(api_list, str):
                try:
                    api_list = json.loads(api_list)
                except:
                    api_list = []

            # Получаем query
            query = item.get('query', '')
            if isinstance(query, bytes):
                query = query.decode('utf-8')
            elif not isinstance(query, str):
                query = str(query)

            # Получаем domain
            domain = item.get('domain', '')
            if isinstance(domain, bytes):
                domain = domain.decode('utf-8')
            elif not isinstance(domain, str):
                domain = str(domain)

            return {
                'api_list': api_list,
                'query': query,
                'query_id': str(item.get('query_id', '')),
                'domain': domain,
            }
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            return None

    def _create_test_data(self) -> List[Dict]:
        """Создание тестовых данных если загрузка не удалась"""
        test_data = []

        # Тестовые запросы
        test_queries = [
            "Найди информацию о последних новостях в мире технологий",
            "Посчитай 234 * 567",
            "Какая погода в Москве сегодня?",
            "Найди в базе данных информацию о пользователе с id 12345",
            "Сколько будет 15% от 8500 рублей?",
            "Переведи текст 'Hello world' на русский язык",
            "Найди рецепт пиццы",
            "Какое сегодня число?",
            "Напиши код на Python для сортировки списка",
            "Сколько времени в Токио?"
        ]

        # Тестовые инструменты
        test_tools = [
            {
                'tool_name': 'WebSearch',
                'api_name': 'search',
                'api_description': 'Поиск информации в интернете',
                'category_name': 'Search',
                'required_parameters': [{'name': 'query', 'type': 'string'}],
                'optional_parameters': [],
                'method': 'GET'
            },
            {
                'tool_name': 'Calculator',
                'api_name': 'calculate',
                'api_description': 'Выполнение математических вычислений',
                'category_name': 'Math',
                'required_parameters': [{'name': 'expression', 'type': 'string'}],
                'optional_parameters': [],
                'method': 'POST'
            },
            {
                'tool_name': 'Weather',
                'api_name': 'get_weather',
                'api_description': 'Получение информации о погоде',
                'category_name': 'Weather',
                'required_parameters': [{'name': 'city', 'type': 'string'}],
                'optional_parameters': [],
                'method': 'GET'
            },
            {
                'tool_name': 'Database',
                'api_name': 'query',
                'api_description': 'Запрос к базе данных',
                'category_name': 'Data',
                'required_parameters': [{'name': 'sql', 'type': 'string'}],
                'optional_parameters': [],
                'method': 'POST'
            },
            {
                'tool_name': 'Translation',
                'api_name': 'translate',
                'api_description': 'Перевод текста',
                'category_name': 'Language',
                'required_parameters': [
                    {'name': 'text', 'type': 'string'},
                    {'name': 'target_lang', 'type': 'string'}
                ],
                'optional_parameters': [],
                'method': 'POST'
            }
        ]

        # Создаем тестовые элементы
        for i, query in enumerate(test_queries):
            # Выбираем случайный инструмент
            tool = random.choice(test_tools)
            domain = tool['category_name']

            item = {
                'api_list': [tool],
                'query': query,
                'query_id': str(i + 1000),
                'domain': domain
            }
            test_data.append(item)

        return test_data

    def _extract_all_tools(self) -> List[Dict]:
        """Извлечение всех уникальных инструментов из датасета"""
        tools_dict = {}

        for item in self.data:
            api_list = item.get('api_list', [])
            if not isinstance(api_list, list):
                continue

            for api in api_list:
                if not isinstance(api, dict):
                    continue

                tool_name = api.get('tool_name', 'unknown')
                api_name = api.get('api_name', 'unknown')
                tool_key = f"{tool_name}_{api_name}"

                if tool_key not in tools_dict:
                    # Обрабатываем параметры
                    required_params = api.get('required_parameters', [])
                    if isinstance(required_params, str):
                        try:
                            required_params = json.loads(required_params)
                        except:
                            required_params = []

                    optional_params = api.get('optional_parameters', [])
                    if isinstance(optional_params, str):
                        try:
                            optional_params = json.loads(optional_params)
                        except:
                            optional_params = []

                    tools_dict[tool_key] = {
                        'name': f"{tool_name}.{api_name}",
                        'tool_name': tool_name,
                        'api_name': api_name,
                        'description': str(api.get('api_description', '')),
                        'category': str(api.get('category_name', 'Unknown')),
                        'required_parameters': required_params,
                        'optional_parameters': optional_params,
                        'method': str(api.get('method', 'GET')),
                        'base_latency': random.uniform(0.1, 0.8),
                        'failure_rate': random.uniform(0.01, 0.15)
                    }

        return list(tools_dict.values())

    def get_training_prompts(self) -> List[Dict]:
        """Получение промптов для обучения"""
        prompts = []

        for item in self.data:
            query = item.get('query', '')
            if not query:
                continue

            # Инструменты, которые подходят для этого запроса
            relevant_tools = []
            api_list = item.get('api_list', [])

            if isinstance(api_list, list):
                for api in api_list:
                    if not isinstance(api, dict):
                        continue

                    tool_name = api.get('tool_name', 'unknown')
                    api_name = api.get('api_name', 'unknown')

                    # Находим полную информацию об инструменте
                    for tool in self.tools:
                        if tool['name'] == f"{tool_name}.{api_name}":
                            relevant_tools.append(tool)
                            break

            prompts.append({
                'query': query,
                'query_id': item.get('query_id', ''),
                'domain': item.get('domain', ''),
                'relevant_tools': relevant_tools
            })

        return prompts

    def get_tools_for_domain(self, domain: str) -> List[Dict]:
        """Получение инструментов для конкретного домена"""
        domain_tools = []
        for tool in self.tools:
            if domain.lower() in tool['category'].lower():
                domain_tools.append(tool)
        return domain_tools

    def sample_tools(self, n: int = 10) -> List[Dict]:
        """Случайная выборка инструментов"""
        if not self.tools:
            return []
        return random.sample(self.tools, min(n, len(self.tools)))