import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.toolbench_loader import ToolBenchLoader


def simple_test():
    """Упрощенный тест загрузки данных"""
    print("=" * 50)
    print("ТЕСТ ЗАГРУЗКИ TOOLBENCH")
    print("=" * 50)

    # Загружаем маленькую выборку
    loader = ToolBenchLoader(split="train", sample_size=10)

    print(f"\n Загружено {len(loader.data)} примеров")
    print(f" Найдено {len(loader.tools)} инструментов")

    # Показываем первые 3 примера
    print("\n Примеры запросов:")
    for i, item in enumerate(loader.data[:3]):
        print(f"\n  {i + 1}. {item['query'][:100]}...")
        print(f"     Домен: {item.get('domain', 'unknown')}")
        print(f"     Инструментов в запросе: {len(item['api_list'])}")

    # Показываем первые 3 инструмента
    print("\n🔧 Примеры инструментов:")
    for i, tool in enumerate(loader.tools[:3]):
        print(f"\n  {i + 1}. {tool['name']}")
        print(f"     Категория: {tool['category']}")
        print(f"     Описание: {tool['description'][:100]}...")

    print("\n✅ Тест завершен успешно!")


if __name__ == "__main__":
    simple_test()