import json
import os


def save_training_state(config, training_log, output_path="training_state"):
    """Сохраняет состояние обучения (не веса модели)"""
    os.makedirs(output_path, exist_ok=True)

    # Сохраняем конфигурацию
    with open(f"{output_path}/config.json", 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': config.model_name,
            'openai_base_url': config.openai_base_url,
            'rl_params': {
                'num_epochs': config.rl.num_epochs,
                'batch_size': config.rl.batch_size,
                'learning_rate': config.rl.learning_rate,
                'max_steps': config.rl.max_steps
            },
            'toolbench': {
                'sample_size': config.toolbench.sample_size,
                'num_tools': config.toolbench.num_tools
            }
        }, f, indent=2)

    # Сохраняем историю обучения
    with open(f"{output_path}/training_log.json", 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    # Сохраняем инструменты
    tools_data = []
    for tool in config.tools[:100]:
        tools_data.append({
            'name': tool['name'],
            'category': tool.get('category', 'Unknown'),
            'description': tool.get('description', '')[:200]
        })

    with open(f"{output_path}/tools.json", 'w', encoding='utf-8') as f:
        json.dump(tools_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Состояние обучения сохранено в {output_path}")


def load_training_state(output_path="training_state"):
    """Загружает сохранённое состояние"""
    with open(f"{output_path}/config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(f"{output_path}/training_log.json", 'r', encoding='utf-8') as f:
        training_log = json.load(f)

    print(f"✅ Загружено состояние из {output_path}")
    if training_log:
        print(f"   Последняя эпоха: {training_log[-1]['epoch']}")
        best_loss = min([m['proxy_loss'] for m in training_log]) if training_log else 0
        print(f"   Лучший прокси-loss: {best_loss:.4f}")

    return config, training_log