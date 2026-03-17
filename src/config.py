# config.py (исправленный)
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from src.data.toolbench_loader import ToolBenchLoader
from src.tools.tool_selector import ToolSelector
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NetworkConfig:
    # ... (оставить как есть)
    base_latency_range: Tuple[float, float] = (0.05, 1.0)
    jitter_range: Tuple[float, float] = (0.01, 0.2)
    failure_rate_range: Tuple[float, float] = (0.01, 0.2)
    congestion_factor_range: Tuple[float, float] = (0.5, 2.0)
    base_latency: float = 0.1
    jitter: float = 0.05
    failure_rate: float = 0.1
    congestion_factor: float = 1.0


@dataclass
class ToolBenchConfig:
    split: str = "train"
    sample_size: Optional[int] = 50000
    num_tools: int = 9000


@dataclass
class RLConfig:
    algorithm: str = "grpo"
    learning_rate: float = 2e-5
    batch_size: int = 2
    num_epochs: int = 20
    max_steps: int = 3
    kl_coef: float = 0.1
    temperature: float = 0.7
    gradient_accumulation_steps: int = 2


@dataclass
class RewardConfig:
    success_reward: float = 1.0
    failure_penalty: float = -1.0
    step_penalty: float = -0.1
    invalid_call_penalty: float = -0.5
    semantic_bonus: float = 0.2
    latency_threshold: float = 0.5
    wrong_tool_penalty: float = -0.3
    extra_step_penalty: float = -0.15


class Config:
    """Главный конфигурационный класс"""

    def __init__(self):
        print("\n" + "=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ КОНФИГУРАЦИИ")
        print("=" * 60)

        self.network = NetworkConfig()
        self.toolbench = ToolBenchConfig()
        self.rl = RLConfig()
        self.reward = RewardConfig()

        # ✅ Загружаем настройки модели из .env
        self.model_name = os.getenv('MODEL_NAME')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL')
        self.openai_api_token = os.getenv('OPENAI_API_TOKEN')
        self.system_prompt = os.getenv('SYSTEM_PROMPT')
        self.user_prompt = os.getenv('USER_PROMPT')
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '100'))
        self.min_request_timeout = float(os.getenv('MIN_REQUEST_TIMEOUT', '30.0'))

        # ✅ Валидация
        self._validate()

        # Загружаем данные из ToolBench
        print("\n📊 ЗАГРУЗКА ДАННЫХ ИЗ TOOLBENCH")
        self.loader = ToolBenchLoader(
            split=self.toolbench.split,
            sample_size=self.toolbench.sample_size
        )

        # Создаем селектор инструментов
        print("\n🎯 СОЗДАНИЕ СЕЛЕКТОРА ИНСТРУМЕНТОВ")
        self.tool_selector = ToolSelector(self.loader.tools)
        self.tool_selector.print_category_stats()

        # Выбираем инструменты для обучения
        print(f"\n🔧 ВЫБОРКА {self.toolbench.num_tools} ИНСТРУМЕНТОВ ДЛЯ ОБУЧЕНИЯ")
        selected_tools = []

        # 1. Берем инструменты из каждой категории
        tools_per_category = max(5, self.toolbench.num_tools // 10)
        for category, data in self.tool_selector.CATEGORIES.items():
            if data['tools']:
                category_tools = data['tools'][:tools_per_category]
                selected_tools.extend(category_tools)
                print(f"   {category}: взято {len(category_tools)} инструментов")

        # 2. Если не хватает, добавляем популярные инструменты
        if len(selected_tools) < self.toolbench.num_tools:
            remaining = self.toolbench.num_tools - len(selected_tools)
            sorted_tools = sorted(
                self.loader.tools,
                key=lambda x: len(x.get('description', '')),
                reverse=True
            )
            for tool in sorted_tools:
                if tool not in selected_tools:
                    selected_tools.append(tool)
                    remaining -= 1
                    if remaining == 0:
                        break
            print(f"   добавлено {remaining} популярных инструментов")

        self.tools = selected_tools[:self.toolbench.num_tools]
        print(f"\n✅ ИТОГО: {len(self.tools)} инструментов отобрано")

        # Распределение по категориям
        category_distribution = {}
        for tool in self.tools:
            cat = tool.get('category', 'Unknown')
            category_distribution[cat] = category_distribution.get(cat, 0) + 1

        print("\n📊 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
        for cat, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {cat}: {count} инструментов")

        # Получаем промпты для обучения
        print("\n📝 ПОДГОТОВКА ПРОМПТОВ ДЛЯ ОБУЧЕНИЯ")
        self.prompts = self.loader.get_training_prompts()

        valid_prompts = []
        tool_names = {t['name'] for t in self.tools}

        for prompt in self.prompts:
            relevant = [t for t in prompt.get('relevant_tools', []) if t['name'] in tool_names]
            if relevant:
                prompt['relevant_tools'] = relevant
                valid_prompts.append(prompt)

        self.prompts = valid_prompts
        print(f"   Всего промптов: {len(self.prompts)}")
        print(f"   Промптов с релевантными инструментами: {len(valid_prompts)}")

        print("\n" + "=" * 60)
        print("✅ КОНФИГУРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

    def _validate(self):
        """Проверка наличия обязательных параметров"""
        if not self.openai_base_url:
            raise ValueError("OPENAI_BASE_URL не установлен в .env файле")

        print(f"✅ Конфигурация загружена:")
        print(f"   Модель: {self.model_name}")
        print(f"   Base URL: {self.openai_base_url}")
        print(f"   Max concurrent: {self.max_concurrent_requests}")

    def get_tools_by_category(self, category: str) -> List[Dict]:
        return [t for t in self.tools if t.get('category', '').lower() == category.lower()]

    def get_tools_for_query(self, query: str, num_tools: int = 20) -> List[Dict]:
        return self.tool_selector.select_tools_for_query(query, num_tools)