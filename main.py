# main.py
import os
import sys
import platform
import ctypes
import argparse
import torch

# Решение для Windows: принудительно загружаем c10.dll до остальных импортов
if platform.system() == "Windows":
    try:
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.origin:
            torch_dir = os.path.dirname(torch_spec.origin)
            dll_path = os.path.join(torch_dir, "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
                print("✅ c10.dll успешно предварительно загружен")
    except Exception as e:
        print(f"⚠️ Предзагрузка c10.dll не удалась: {e}")

# Добавляем путь к src в системный путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.config import Config
from src.rl.train_grpo import NetMCPTrainer
from src.llm.llm_client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="NetMCP RL Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "evaluate", "interactive"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load (e.g., checkpoints/epoch_20)")

    args = parser.parse_args()

    # Загружаем конфигурацию (модель уже будет из .env)
    config = Config()
    config.rl.num_epochs = args.epochs

    # Создаем LLM клиент для использования в обучении
    llm_client = LLMClient(config)
    print(f"✅ LLM клиент создан для модели: {config.model_name}")

    # Создаем тренера с передачей LLM клиента
    trainer = NetMCPTrainer(config, llm_client)

    # Загружаем чекпоинт если указан
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if os.path.exists(checkpoint_path):
            try:
                trainer.load_checkpoint(checkpoint_path)
                print(f"✅ Загружен чекпоинт из {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки чекпоинта: {e}")
        else:
            print(f"⚠️ Чекпоинт {checkpoint_path} не найден")

    if args.mode == "train":
        trainer.train()
    elif args.mode == "evaluate":
        trainer.evaluate()
    elif args.mode == "interactive":
        run_interactive(trainer, llm_client)


def run_interactive(trainer, llm_client):
    """Интерактивный режим для тестирования с исправлением вызовов"""
    print("\n=== NetMCP Interactive Mode ===")
    print("Введите ваш запрос (или 'quit' для выхода):")

    # Получаем список всех доступных инструментов
    all_tools = trainer.config.tools
    print(f"\n📚 Загружено {len(all_tools)} инструментов")

    # Покажем первые несколько инструментов для примера
    print("\n📋 Примеры доступных инструментов:")
    for i, tool in enumerate(all_tools[:10]):
        print(f"   {i + 1}. {tool['name']}")

    while True:
        query = input("\n>>> ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        query_data = {
            'query': query,
            'domain': 'user_query',
            'relevant_tools': []
        }

        state = trainer.env.reset(query_data)
        print(f"\nОбработка запроса: {query}")

        # Получаем валидные инструменты
        valid_tools = [t['name'] for t in state['tools'] if t['available']]
        if not valid_tools:
            print("  ⚠️ Нет доступных инструментов для этого запроса")
            continue

        print(f"  📋 Доступно инструментов: {len(valid_tools)}")
        print(f"  📋 Первые 5 доступных инструментов:")
        for i, tool in enumerate(valid_tools[:5]):
            print(f"     {i + 1}. {tool}")

        for step in range(trainer.config.rl.max_steps):
            # Формируем промпт
            context = trainer._format_context(state)

            # Используем LLM клиент для генерации ответа
            response = llm_client.ask(context)

            tool_call = trainer._parse_tool_call(response)

            if tool_call:
                tool_name = tool_call['tool']
                print(f"  🤖 Модель вызвала: {tool_name}")

                # Исправляем tool_name если нужно
                if tool_name == 'tool_name' or tool_name not in valid_tools:
                    corrected = trainer._correct_tool_call(tool_name, valid_tools, query)
                    if corrected:
                        print(f"  🔧 Исправляем на: {corrected}")
                        tool_name = corrected
                    else:
                        print(f"  ❌ Не удалось исправить вызов")
                        break

                # Выполняем вызов
                next_state, reward, done, info = trainer.env.step(tool_name)
                print(f"  📊 Результат:")
                print(f"     - Инструмент: {tool_name}")
                print(f"     - Задержка: {info.get('latency', 0):.3f}s")
                print(f"     - Награда: {reward:.2f}")
                print(f"     - Успех: {'✅' if info.get('success') else '❌'}")

                if done:
                    break

                state = next_state
            else:
                print(f"  ⚠️ Модель не вызвала инструмент")
                break


if __name__ == "__main__":
    main()