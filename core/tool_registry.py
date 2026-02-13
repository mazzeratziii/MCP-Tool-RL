"""
Улучшенный реестр инструментов с автоматической семантикой
Исправленная версия с YAKE
"""

import hashlib
import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

# YAKE с правильным импортом
try:
    import yake
    from yake import KeywordExtractor
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("⚠️  YAKE не установлен. Используем базовый экстрактор ключевых слов.")
    print("   Для улучшенной семантики: pip install yake")

@dataclass
class Tool:
    """Описание инструмента с автоматической семантикой"""
    id: str
    name: str
    description: str
    category: str
    api_name: str
    endpoint: str
    method: str
    parameters: List[Dict[str, Any]]
    required_params: List[str]
    examples: List[Dict[str, Any]]

    # Автоматически генерируемые поля
    keywords: Set[str] = field(default_factory=set)
    semantic_variants: List[str] = field(default_factory=list)
    weight: float = 1.0  # Вес инструмента при поиске

    def __post_init__(self):
        """Автоматически генерирует семантику при создании инструмента"""
        self.keywords = self._extract_keywords_advanced()
        self.semantic_variants = self._generate_semantic_variants()

    def _extract_keywords_advanced(self) -> Set[str]:
        """Умное извлечение ключевых слов с YAKE или без него"""
        keywords = set()

        # Текст для анализа
        full_text = f"{self.name} {self.description} {self.category} {self.api_name}"

        # Добавляем примеры запросов
        for ex in self.examples:
            if 'query' in ex:
                full_text += f" {ex['query']}"

        # Добавляем параметры
        for param in self.parameters:
            if isinstance(param, dict):
                full_text += f" {param.get('name', '')} {param.get('description', '')}"

        full_text = full_text.lower()

        # Используем YAKE если доступен
        if YAKE_AVAILABLE:
            try:
                extractor = KeywordExtractor(
                    lan="ru",  # Русский язык
                    n=3,       # Максимальная длина фразы
                    dedupLim=0.9,
                    dedupFunc='seqm',
                    windowsSize=2,
                    top=20,    # Количество ключевых слов
                    features=None
                )

                extracted = extractor.extract_keywords(full_text)

                # Добавляем извлеченные ключевые слова
                for kw, score in extracted:
                    if len(kw.split()) <= 2 and len(kw) > 2:  # Короткие фразы
                        keywords.add(kw.lower())

                print(f"   📊 YAKE извлек {len(keywords)} ключевых слов для {self.name}")

            except Exception as e:
                print(f"   ⚠️  Ошибка YAKE: {e}, используем базовый метод")
                keywords = self._extract_keywords_basic()
        else:
            keywords = self._extract_keywords_basic()

        # Добавляем обязательные ключевые слова на основе категории
        category_keywords = {
            'weather': ['погода', 'температура', 'дождь', 'снег', 'ветер', 'прогноз', 'град', 'осадки', 'климат'],
            'finance': ['валюта', 'доллар', 'евро', 'рубль', 'курс', 'конвертация', 'деньги', 'банк', 'обмен'],
            'transportation': ['рейс', 'билет', 'самолет', 'поезд', 'путешествие', 'авиа', 'перелет', 'аэропорт']
        }

        if self.category in category_keywords:
            keywords.update(category_keywords[self.category])

        return keywords

    def _extract_keywords_basic(self) -> Set[str]:
        """Базовое извлечение ключевых слов (без YAKE)"""
        keywords = set()

        # Извлекаем слова из названия
        name_words = re.findall(r'\w+', self.name.lower())
        keywords.update([w for w in name_words if len(w) > 2])

        # Извлекаем важные слова из описания
        desc_words = re.findall(r'\w+', self.description.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'get', 'from', 'api', 'your', 'you', 'this', 'that',
                     'как', 'для', 'это', 'что', 'когда', 'где', 'кто', 'почему'}

        for word in desc_words:
            if len(word) > 3 and word not in stop_words:
                keywords.add(word)

        return keywords

    def _generate_semantic_variants(self) -> List[str]:
        """Генерирует разные формулировки описания инструмента"""
        variants = []

        # Специальная обработка для валют
        if self.category == 'finance':
            variants.extend([
                f"Конвертация валют: доллар USD евро EUR рубль RUB фунт GBP",
                f"Курс обмена валют, конвертер денег, exchange rate, currency converter",
                f"Перевести деньги из одной валюты в другую по текущему курсу",
                f"Сколько будет долларов в рублях, евро в долларах, рубль в евро",
                f"Валютный калькулятор, обменник, forex, currency exchange"
            ])
        elif self.category == 'weather':
            variants.extend([
                f"Прогноз погоды, температура воздуха, осадки, ветер",
                f"Погода в городе, температура на сегодня, завтра",
                f"Метеорология, климат, атмосферное давление"
            ])
        elif self.category == 'transportation':
            variants.extend([
                f"Авиабилеты, рейсы самолетов, перелеты",
                f"Расписание рейсов, цены на билеты, авиакомпании",
                f"Путешествия, туризм, командировки"
            ])

        return variants

    @property
    def search_text(self) -> str:
        """Улучшенный текст для семантического поиска"""
        parts = [
            f"Tool: {self.name}",
            f"Category: {self.category}",
            f"Description: {self.description}",
            f"What it does: {self._generate_function_description()}"
        ]

        # Добавляем ключевые слова с весами
        if self.keywords:
            keywords_str = ' '.join(list(self.keywords)[:15])
            parts.append(f"Keywords: {keywords_str}")

        # Добавляем семантические варианты
        if self.semantic_variants:
            parts.extend(self.semantic_variants[:3])

        # Добавляем примеры
        if self.examples:
            examples_text = '; '.join([ex.get('query', '') for ex in self.examples[:2]])
            parts.append(f"Example requests: {examples_text}")

        return ". ".join(parts)

    def _generate_function_description(self) -> str:
        """Генерирует функциональное описание на основе категории"""
        descriptions = {
            'weather': f"This tool provides weather information for {', '.join([p.get('name', 'city') for p in self.parameters if p.get('name') == 'city'][:1])}",
            'finance': f"This tool converts {', '.join([p.get('name', 'currency') for p in self.parameters if 'currency' in p.get('name', '')][:2])}",
            'transportation': f"This tool searches for {self.category} between {', '.join([p.get('name', 'locations') for p in self.parameters if p.get('name') in ['origin', 'destination']][:2])}"
        }
        return descriptions.get(self.category, f"Tool for {self.category} operations")

class ToolRegistry:
    """Реестр инструментов с автоматической семантикой"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tools_by_category: Dict[str, List[Tool]] = {}

        if YAKE_AVAILABLE:
            print("✅ YAKE загружен - используется улучшенное извлечение ключевых слов")

    def add_tool(self, tool_data: Dict[str, Any]) -> Tool:
        """Добавляет инструмент с автоматической семантикой"""
        # Генерация ID
        tool_hash = hashlib.md5(
            f"{tool_data.get('name')}:{tool_data.get('category')}".encode()
        ).hexdigest()[:8]

        tool_id = f"{tool_data.get('category', 'general')}_{tool_hash}"

        # Создаём инструмент с автоматической семантикой
        tool = Tool(
            id=tool_id,
            name=tool_data.get('name', 'Unnamed Tool'),
            description=tool_data.get('description', 'No description'),
            category=tool_data.get('category', 'general'),
            api_name=tool_data.get('api_name', tool_data.get('name', 'unknown')),
            endpoint=tool_data.get('endpoint', f'/api/{tool_data.get("category", "general")}'),
            method=tool_data.get('method', 'GET'),
            parameters=tool_data.get('parameters', []),
            required_params=tool_data.get('required_params', []),
            examples=tool_data.get('examples', [])
        )

        # Сохраняем
        self.tools[tool_id] = tool

        # Индексируем по категории
        if tool.category not in self.tools_by_category:
            self.tools_by_category[tool.category] = []
        self.tools_by_category[tool.category].append(tool)

        print(f"   ✅ Инструмент '{tool.name}' добавлен с {len(tool.keywords)} ключевыми словами")
        return tool

    def get_sample_tools(self) -> List[Tool]:
        """Возвращает примерные инструменты"""
        print("\n📦 Загрузка инструментов с авто-семантикой:")
        print("-" * 60)

        sample_data = [
            {
                "name": "Текущая погода",
                "description": "Получить текущую погоду для указанного города. Возвращает температуру, влажность, скорость ветра, атмосферное давление, видимость и условия (ясно, облачно, дождь, снег).",
                "category": "weather",
                "api_name": "weather_current",
                "endpoint": "/api/weather/current",
                "method": "GET",
                "parameters": [
                    {"name": "city", "type": "string", "required": True, "description": "Название города на русском или английском"},
                    {"name": "units", "type": "string", "required": False, "description": "metric или imperial"},
                    {"name": "lang", "type": "string", "required": False, "description": "ru или en"}
                ],
                "required_params": ["city"],
                "examples": [
                    {"query": "Какая погода в Москве?", "parameters": {"city": "Москва"}},
                    {"query": "Температура в Санкт-Петербурге", "parameters": {"city": "Санкт-Петербург"}},
                    {"query": "Прогноз погоды на завтра", "parameters": {"city": "Москва"}}
                ]
            },
            {
                "name": "Конвертер валют",
                "description": "Конвертировать сумму из одной валюты в другую по актуальному курсу. Поддерживаются USD, EUR, RUB, GBP, JPY, CNY и другие валюты.",
                "category": "finance",
                "api_name": "currency_converter",
                "endpoint": "/api/finance/convert",
                "method": "GET",
                "parameters": [
                    {"name": "amount", "type": "number", "required": True, "description": "Сумма для конвертации"},
                    {"name": "from_currency", "type": "string", "required": True, "description": "Исходная валюта (USD, EUR, RUB, GBP, JPY, CNY)"},
                    {"name": "to_currency", "type": "string", "required": True, "description": "Целевая валюта"}
                ],
                "required_params": ["amount", "from_currency", "to_currency"],
                "examples": [
                    {"query": "Сколько будет 100 долларов в рублях?", "parameters": {"amount": 100, "from_currency": "USD", "to_currency": "RUB"}},
                    {"query": "Конвертируй 50 евро в доллары", "parameters": {"amount": 50, "from_currency": "EUR", "to_currency": "USD"}},
                    {"query": "Переведи 1000 рублей в евро", "parameters": {"amount": 1000, "from_currency": "RUB", "to_currency": "EUR"}}
                ]
            },
            {
                "name": "Поиск авиарейсов",
                "description": "Поиск доступных авиарейсов между городами. Информация о времени вылета и прилета, авиакомпаниях, ценах, наличии мест, длительности перелета.",
                "category": "transportation",
                "api_name": "flight_search",
                "endpoint": "/api/flights/search",
                "method": "GET",
                "parameters": [
                    {"name": "origin", "type": "string", "required": True, "description": "Город отправления"},
                    {"name": "destination", "type": "string", "required": True, "description": "Город назначения"},
                    {"name": "date", "type": "string", "required": False, "description": "Дата вылета (ГГГГ-ММ-ДД)"},
                    {"name": "passengers", "type": "integer", "required": False, "description": "Количество пассажиров"}
                ],
                "required_params": ["origin", "destination"],
                "examples": [
                    {"query": "Найди рейсы из Москвы в Лондон", "parameters": {"origin": "Москва", "destination": "Лондон"}},
                    {"query": "Авиабилеты в Париж на завтра", "parameters": {"origin": "Москва", "destination": "Париж", "date": "2024-01-20"}},
                    {"query": "Рейсы Санкт-Петербург - Берлин", "parameters": {"origin": "Санкт-Петербург", "destination": "Берлин"}}
                ]
            }
        ]

        tools = []
        for data in sample_data:
            tool = self.add_tool(data)
            tools.append(tool)

        print("-" * 60)
        print(f"✅ Загружено {len(tools)} инструментов с авто-семантикой\n")
        return tools

    def print_tool_keywords(self):
        """Показывает извлеченные ключевые слова"""
        print("\n" + "="*70)
        print("🔑 АВТОМАТИЧЕСКИ ИЗВЛЕЧЕННЫЕ КЛЮЧЕВЫЕ СЛОВА")
        print("="*70)

        for tool in self.tools.values():
            print(f"\n📌 {tool.name} [{tool.category}]")
            print(f"   Ключевых слов: {len(tool.keywords)}")

            # Показываем топ-20 ключевых слов
            keywords_list = list(tool.keywords)[:20]
            for i, kw in enumerate(keywords_list, 1):
                print(f"      {i:2d}. {kw}")

            print(f"\n   📝 Семантические варианты:")
            for i, variant in enumerate(tool.semantic_variants[:3], 1):
                print(f"      {i}. {variant[:100]}...")

# Глобальный экземпляр
registry = ToolRegistry()