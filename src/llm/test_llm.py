# src/llm/test_llm.py
import sys
import os

# Добавляем путь к корневой папке проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm.llm_client import LLMClient
from src.config import Config


def test_llm():
    """Тестирование LLM клиента"""
    print("🔧 ТЕСТ LLM КЛИЕНТА")

    # Загружаем конфигурацию
    config = Config()
    print(f"\n📋 Конфигурация:")
    print(f"   Модель: {config.model_name}")
    print(f"   Base URL: {config.openai_base_url}")

    # Создаем клиент
    client = LLMClient(config)
    print(f"✅ LLM клиент создан")

    # Тестовые вопросы
    test_questions = [
        "What is 2+2?",
        "Explain what is artificial intelligence",
        "Who wrote Romeo and Juliet?"
    ]

    print(f"\n Тестирование вопросов...")

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Вопрос {i} ---")
        print(f"Q: {question}")

        try:
            response = client.ask(question)
            print(f"A: {response}")
        except Exception as e:
            print(f" Ошибка: {e}")

    print("\n Тест завершен!")


if __name__ == "__main__":
    test_llm()