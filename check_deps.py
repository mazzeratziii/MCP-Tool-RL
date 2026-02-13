"""
Проверка зависимостей и их установка
"""

import subprocess
import sys


def check_and_install(package, import_name=None):
    """Проверяет и устанавливает пакет"""
    if import_name is None:
        import_name = package

    try:
        __import__(import_name)
        print(f" {package} уже установлен")
        return True
    except ImportError:
        print(f"⏳ Установка {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f" {package} успешно установлен")
            return True
        except subprocess.CalledProcessError:
            print(f" Не удалось установить {package}")
            return False


def main():
    """Проверяет и устанавливает все зависимости"""
    print("=" * 60)
    print("Проверка зависимостей MCP-Tool-RL")
    print("=" * 60)

    dependencies = [
        ("datasets", "datasets"),
        ("sentence-transformers", "sentence_transformers"),
        ("numpy", "numpy"),
        ("aiohttp", "aiohttp"),
        ("scikit-learn", "sklearn"),
    ]

    all_ok = True
    for package, import_name in dependencies:
        if not check_and_install(package, import_name):
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print(" Все зависимости установлены!")
        print("Запустите: python main.py --mode demo")
    else:
        print(" Некоторые зависимости не установлены")
        print("Установите вручную: pip install -r requirements.txt")



if __name__ == "__main__":
    main()