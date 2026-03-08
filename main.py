import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import sys
import platform
import ctypes

# Решение для Windows: принудительно загружаем c10.dll до остальных импортов
if platform.system() == "Windows":
    try:
        # Сначала находим путь к torch, если он установлен
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.origin:
            torch_dir = os.path.dirname(torch_spec.origin)
            dll_path = os.path.join(torch_dir, "lib", "c10.dll")
            if os.path.exists(dll_path):
                # Загружаем DLL с указанием полного пути
                ctypes.CDLL(os.path.normpath(dll_path))
                print("c10.dll успешно предварительно загружен")
            else:
                print(f"c10.dll не найден по пути: {dll_path}")
    except Exception as e:
        print(f"Предзагрузка c10.dll не удалась: {e}")

# Теперь импортируем остальные модули
import argparse

# Добавляем путь к src в системный путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.config import Config
from src.rl.train_grpo import NetMCPTrainer


def main():
    parser = argparse.ArgumentParser(description="NetMCP RL Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "evaluate", "interactive"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    args = parser.parse_args()

    # Загружаем конфигурацию
    config = Config()
    config.rl.num_epochs = args.epochs
    config.model_name = args.model

    # Создаем тренера
    trainer = NetMCPTrainer(config)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "evaluate":
        trainer.evaluate()
    elif args.mode == "interactive":
        run_interactive(trainer)


def run_interactive(trainer):
    """Интерактивный режим для тестирования"""
    print("\n=== NetMCP Interactive Mode ===")
    print("Введите ваш запрос (или 'quit' для выхода):")

    while True:
        query = input("\n>>> ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        # Создаем структуру данных как в ToolBench
        query_data = {
            'query': query,
            'domain': 'user_query',
            'relevant_tools': []
        }

        state = trainer.env.reset(query_data)
        print(f"\nОбработка запроса: {query}")

        for step in range(trainer.config.rl.max_steps):
            context = trainer._format_context(state)
            inputs = trainer.tokenizer(context, return_tensors="pt")
            outputs = trainer.model.generate(**inputs, max_new_tokens=50)
            response = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)

            tool_call = trainer._parse_tool_call(response)
            if tool_call:
                next_state, reward, done, info = trainer.env.step(tool_call['tool'])
                print(f"  Шаг {step + 1}: Использую {tool_call['tool']}")
                print(f"    Задержка: {info.get('latency', 0):.3f}s")
                print(f"    Награда: {reward:.2f}")

                if done:
                    print(f"  Результат: {'Успех' if info.get('success') else 'Неудача'}")
                    break

                state = next_state
            else:
                print(f"  Шаг {step + 1}: Модель не вызвала инструмент")
                break


if __name__ == "__main__":
    main()