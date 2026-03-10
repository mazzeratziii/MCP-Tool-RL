# test_training.py
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

print("=" * 60)
print("ТЕСТ ОБУЧЕНИЯ С ОТЛАДКОЙ")
print("=" * 60)

try:
    from src.config import Config
    from src.rl.train_grpo import NetMCPTrainer
    from src.prompts import get_dynamic_prompt, get_strict_prompt

    print("✅ Импорты успешны")
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    traceback.print_exc()
    sys.exit(1)

# Загружаем конфигурацию
print("\n📁 Загрузка конфигурации...")
try:
    config = Config()
    print(f"✅ Загружено {len(config.prompts)} промптов")
    print(f"✅ Загружено {len(config.tools)} инструментов")

    # Покажем первые 3 промпта
    print("\n📝 Примеры промптов:")
    for i, prompt in enumerate(config.prompts[:3]):
        print(f"  {i + 1}. {prompt['query'][:50]}...")
        print(f"     Домен: {prompt['domain']}")
        print(f"     Релевантные инструменты: {[t['name'] for t in prompt['relevant_tools']]}")
except Exception as e:
    print(f"❌ Ошибка загрузки конфигурации: {e}")
    traceback.print_exc()
    sys.exit(1)

# Создаем тренера
print("\n🤖 Создание тренера...")
try:
    trainer = NetMCPTrainer(config)
    print("✅ Тренер создан успешно")
except Exception as e:
    print(f"❌ Ошибка создания тренера: {e}")
    traceback.print_exc()
    sys.exit(1)

# Тестируем форматирование промпта
print("\n📝 Тест форматирования промпта...")
try:
    # Берем первый промпт и создаем состояние
    test_prompt = config.prompts[0]
    state = trainer.env.reset(test_prompt)

    # Форматируем контекст
    context = trainer._format_context(state)
    print(f"✅ Контекст сформатирован, длина: {len(context)} символов")
    print(f"\nПервые 200 символов контекста:\n{context[:200]}...")
except Exception as e:
    print(f"❌ Ошибка форматирования: {e}")
    traceback.print_exc()

# Тестируем парсинг вызова инструмента
print("\n🔧 Тест парсинга вызова инструмента...")
try:
    test_response = '<tool_call>Calculator.calculate</tool_call>'
    tool_call = trainer._parse_tool_call(test_response)
    print(f"✅ Парсинг успешен: {tool_call}")

    test_response_bad = 'Просто текст без вызова'
    tool_call = trainer._parse_tool_call(test_response_bad)
    print(f"✅ Некорректный ответ обработан: {tool_call}")
except Exception as e:
    print(f"❌ Ошибка парсинга: {e}")
    traceback.print_exc()

# Тестируем валидацию вызова
print("\n✅ Тест валидации вызова...")
try:
    valid = trainer._validate_tool_call({'tool': 'Calculator.calculate'}, state)
    print(f"  Валидный вызов: {valid}")

    valid = trainer._validate_tool_call({'tool': 'NonExistentTool'}, state)
    print(f"  Невалидный вызов: {valid}")
except Exception as e:
    print(f"❌ Ошибка валидации: {e}")
    traceback.print_exc()

# Тестируем исправление вызова
print("\n🔄 Тест исправления вызова...")
try:
    valid_tools = [t['name'] for t in state['tools']]
    corrected = trainer._correct_tool_call('wrong_tool', valid_tools, test_prompt['query'])
    print(f"  Исправленный вызов: {corrected}")
except Exception as e:
    print(f"❌ Ошибка исправления: {e}")
    traceback.print_exc()

# Тестируем сбор одной траектории
print("\n📊 Тест сбора одной траектории...")
try:
    # Собираем только первый промпт для теста
    trainer.config.rl.batch_size = 1
    trajectories = trainer._collect_trajectories()
    print(f"✅ Собрано {len(trajectories)} траекторий")

    if trajectories and trajectories[0]['steps']:
        traj = trajectories[0]
        print(f"  Промпт: {traj['prompt'][:50]}...")
        print(f"  Шагов: {len(traj['steps'])}")
        print(f"  Успех: {traj['success']}")

        for i, step in enumerate(traj['steps']):
            print(f"    Шаг {i + 1}: {step['action']} -> награда {step['reward']:.2f}")
    else:
        print("⚠️ Траектория пуста - модель не вызвала корректные инструменты")

        # Пробуем с жестким промптом
        print("\n🔄 Пробуем с жестким промптом...")
        strict_prompt = get_strict_prompt(test_prompt['query'], state['tools'])
        inputs = trainer.tokenizer(strict_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(trainer.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = trainer.model.generate(**inputs, max_new_tokens=30, temperature=0.1)

        response = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Ответ модели: {response}")

        tool_call = trainer._parse_tool_call(response)
        if tool_call:
            print(f"  Вызван инструмент: {tool_call}")
        else:
            print("  Модель не вызвала инструмент даже с жестким промптом")

except Exception as e:
    print(f"❌ Ошибка сбора траектории: {e}")
    traceback.print_exc()

print("\n✅ Тест завершен!")