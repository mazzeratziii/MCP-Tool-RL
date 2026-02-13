"""
MCP-Tool-RL с автоматической семантикой
"""

import asyncio
import logging
import argparse
from datetime import datetime

from core.tool_registry import registry
from core.semantic_searcher import searcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolRLSystem:
    """Система с автоматической семантикой"""

    def __init__(self):
        self.registry = registry
        self.searcher = searcher
        self.initialized = False

    def initialize(self):
        """Инициализация с авто-семантикой"""
        if self.initialized:
            return

        print("\n" + "="*70)
        print("MCP-TOOL-RL с АВТОМАТИЧЕСКОЙ СЕМАНТИКОЙ")
        print("="*70)

        # 1. Загружаем инструменты
        print("1. 📦 Загрузка инструментов...")
        tools = self.registry.get_sample_tools()
        print(f"   Загружено {len(tools)} инструментов")

        # 2. Показываем авто-сгенерированную семантику
        print("\n2. 🔑 Авто-генерируем ключевые слова и семантику...")
        self.registry.print_tool_keywords()

        # 3. Индексируем
        print("\n3. 📊 Индексация с улучшенной семантикой...")
        self.searcher.index_tools(tools)

        self.initialized = True
        print("\n✅ Система готова!")
        print("="*70)

    def process_query(self, query: str, verbose: bool = False):
        """Обрабатывает запрос с авто-расширением"""
        if not self.initialized:
            self.initialize()

        print(f"\n🔍 Запрос: '{query}'")

        # Поиск с авто-расширением
        results = self.searcher.search(query, top_k=3)

        if not results:
            print("❌ Не найдено подходящих инструментов")
            return

        # Показываем результаты
        print(f"\n📊 Результаты поиска:")
        for i, result in enumerate(results, 1):
            tool = result["tool"]
            print(f"\n   {i}. {tool.name} [{tool.category}]")
            print(f"      Сходство: {result['similarity']:.3f}")
            print(f"      Описание: {tool.description[:80]}...")

            if verbose and 'keywords' in result:
                print(f"      Ключевые слова: {', '.join(result['keywords'][:5])}")

    def debug_query(self, query: str):
        """Детальная отладка запроса"""
        print("\n" + "="*70)
        print(f"🔬 ОТЛАДКА ЗАПРОСА: '{query}'")
        print("="*70)

        # Показываем авто-расширение
        expanded = self.searcher._expand_query_automatically(query)
        print(f"\n📝 Авто-расширение запроса:")
        for i, eq in enumerate(expanded[:5], 1):
            print(f"   {i}. {eq}")
        print(f"   ... и ещё {len(expanded) - 5} вариантов")

        # Поиск
        results = self.searcher.search(query, top_k=5)

        print(f"\n🎯 Результаты поиска:")
        for i, result in enumerate(results, 1):
            tool = result["tool"]
            print(f"\n   {i}. {tool.name}")
            print(f"      Сходство: {result['similarity']:.3f}")
            print(f"      Категория: {tool.category}")
            print(f"      Ключевые слова инструмента: {', '.join(list(tool.keywords)[:10])}")

            # Анализируем совпадения
            query_words = set(query.lower().split())
            tool_words = set(' '.join(list(tool.keywords)[:20]).lower().split())
            matches = query_words.intersection(tool_words)

            if matches:
                print(f"      Совпадения: {', '.join(matches)}")

        print("\n" + "="*70)

def demo_semantic():
    """Демонстрация автоматической семантики"""
    system = MCPToolRLSystem()
    system.initialize()

    test_queries = [
        "Конвертируй 100 долларов в рубли",
        "Какая погода в Москве?",
        "Найди рейсы из Москвы в Лондон"
    ]

    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ АВТОМАТИЧЕСКОЙ СЕМАНТИКИ")
    print("="*70)

    for query in test_queries:
        system.process_query(query, verbose=True)

    print("\n" + "="*70)
    print("Для детальной отладки используйте:")
    print("  python main.py --debug \"ваш запрос\"")
    print("="*70)

def interactive_mode():
    """Интерактивный режим с авто-семантикой"""
    system = MCPToolRLSystem()
    system.initialize()

    print("\n" + "="*70)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ С АВТО-СЕМАНТИКОЙ")
    print("="*70)
    print("Команды:")
    print("  • Ваш запрос - обычный поиск")
    print("  • debug [запрос] - детальная отладка")
    print("  • semantics - показать семантику инструментов")
    print("  • exit - выход")
    print("="*70)

    while True:
        try:
            cmd = input("\n> ").strip()

            if cmd.lower() in ["exit", "quit", "выход"]:
                break

            if cmd.lower() == "semantics":
                registry.print_tool_semantics()
                continue

            if cmd.lower().startswith("debug "):
                query = cmd[6:].strip()
                if query:
                    system.debug_query(query)
                continue

            if cmd:
                system.process_query(cmd, verbose=True)

        except KeyboardInterrupt:
            print("\nЗавершение...")
            break

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='MCP-Tool-RL с авто-семантикой')
    parser.add_argument('--mode', choices=['demo', 'interactive', 'debug'],
                       default='demo', help='Режим работы')
    parser.add_argument('--query', type=str, help='Запрос для отладки')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')

    args = parser.parse_args()

    system = MCPToolRLSystem()

    if args.mode == 'debug' and args.query:
        system.initialize()
        system.debug_query(args.query)
    elif args.mode == 'demo':
        demo_semantic()
    elif args.mode == 'interactive':
        interactive_mode()
    else:
        print("Использование:")
        print("  py main.py --mode demo              # Демо-режим")
        print("  py main.py --mode interactive       # Интерактивный режим")
        print("  py main.py --mode debug --query 'текст'  # Отладка")

if __name__ == "__main__":
    main()