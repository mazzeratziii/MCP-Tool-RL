
import torch
import sys
import os

# Добавляем путь к src в системный путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.environment.mcp_environment import MCPEnvironment


def explore_toolbench():
    """Исследование данных ToolBench"""
    config = Config()

    print("\n" + "=" * 50)
    print("ДАННЫЕ ИЗ TOOLBENCH")
    print("=" * 50)

    # Показываем несколько примеров запросов
    print("\n Примеры запросов пользователей:")
    for i, prompt in enumerate(config.prompts[:5]):
        print(f"{i + 1}. {prompt['query'][:100]}...")
        print(f"   Домен: {prompt['domain']}")
        print(f"   Релевантные инструменты: {[t['name'] for t in prompt['relevant_tools']]}")
        print()

    # Показываем инструменты
    print("\n🔧 Примеры инструментов:")
    categories = {}
    for tool in config.tools[:20]:
        cat = tool['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(tool['name'])

    for cat, tools in list(categories.items())[:5]:
        print(f"\n{cat}:")
        for tool in tools[:3]:
            print(f"  - {tool}")

    # Тестируем среду
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ СРЕДЫ")
    print("=" * 50)

    env = MCPEnvironment(config)

    # Берем первый промпт
    test_prompt = config.prompts[0]
    state = env.reset(test_prompt)

    print(f"\nЗапрос: {test_prompt['query']}")
    print(f"Домен: {test_prompt['domain']}")
    print(f"\nСостояние среды (первые 3 инструмента):")

    for tool in state['tools'][:3]:
        print(f"\n  {tool['name']}")
        print(f"    Доступен: {tool['available']}")
        print(f"    Задержка: {tool['latency']:.3f}s")
        print(f"    Семантическая близость: {tool['semantic_score']:.3f}")
        print(f"    Релевантен: {tool['is_relevant']}")

    # Пробуем выполнить действие
    if state['tools']:
        action = state['tools'][0]['name']  # берем первый инструмент
        print(f"\n👉 Пробуем вызвать: {action}")

        next_state, reward, done, info = env.step(action)
        print(f"   Награда: {reward:.2f}")
        print(f"   Успех: {info['success']}")
        print(f"   Задержка: {info['latency']:.3f}s")


if __name__ == "__main__":
    explore_toolbench()