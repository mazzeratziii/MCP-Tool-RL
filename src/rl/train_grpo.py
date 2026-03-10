# src/rl/train_grpo.py
import os
import sys
import platform
import ctypes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.rl.reward_functions import GRPOToolReward

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
                print("✅ c10.dll успешно предварительно загружен в train_grpo")
    except Exception as e:
        print(f"⚠️ Предзагрузка c10.dll не удалась в train_grpo: {e}")


class NetMCPTrainer:
    """Тренер для обучения агента NetMCP с оптимизацией памяти"""

    def __init__(self, config: Config):
        self.config = config
        self.env = MCPEnvironment(config)
        self.reward_fn = GRPOToolReward(config)

        # Определяем устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        # Оптимизация памяти для GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB allocated")

        # Загружаем модель с оптимизациями
        print(f"Загрузка модели {config.model_name}...")
        try:
            # Используем 4-битную квантизацию для экономии памяти
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Модель успешно загружена с 4-битной квантизацией")

            # Используем PEFT/LoRA для эффективного обучения
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Подготавливаем модель для k-bit обучения
            self.model = prepare_model_for_kbit_training(self.model)

            # Конфигурация LoRA
            lora_config = LoraConfig(
                r=8,  # ранг
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()  # Покажет сколько параметров обучается

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise

        # Оптимизатор с меньшим размером batch
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.rl.learning_rate,
            weight_decay=0.01
        )

        # Gradient checkpointing для экономии памяти
        self.model.gradient_checkpointing_enable()

    def train(self):
        """Цикл обучения с подробной отладкой"""
        print("\n" + "=" * 50)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 50)
        print(f"Конфигурация:")
        print(f"  - Алгоритм: {self.config.rl.algorithm}")
        print(f"  - Эпох: {self.config.rl.num_epochs}")
        print(f"  - Batch size: {self.config.rl.batch_size}")
        print(f"  - Max steps: {self.config.rl.max_steps}")
        print(f"  - Всего промптов: {len(self.config.prompts)}")
        print(f"  - Trainable params: 2.2M / 1.5B (0.14%)")

        # История обучения
        loss_history = []
        reward_history = []

        for epoch in range(self.config.rl.num_epochs):
            print(f"\n{'=' * 40}")
            print(f"ЭПОХА {epoch + 1}/{self.config.rl.num_epochs}")
            print(f"{'=' * 40}")

            # Очищаем кэш GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU память до сбора: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            # Собираем траектории
            print("\n📊 Сбор траекторий...")
            trajectories = self._collect_trajectories()
            print(f"   Собрано {len(trajectories)} траекторий")

            # Проверяем содержимое траекторий
            valid_trajectories = 0
            for i, traj in enumerate(trajectories):
                if traj['steps']:
                    valid_trajectories += 1
                    print(f"   Траектория {i + 1}: {len(traj['steps'])} шагов, успех: {traj['success']}")

            print(f"   Валидных траекторий: {valid_trajectories}")

            if valid_trajectories == 0:
                print("❌ Нет валидных траекторий! Проверьте формат ответов модели.")
                continue

            # Обучаем на траекториях
            print("\n📈 Обучение на траекториях...")
            epoch_loss = 0
            epoch_reward = 0

            for traj_idx, traj in enumerate(trajectories):
                if not traj['steps']:
                    continue

                print(f"   Обучение на траектории {traj_idx + 1}...")
                loss = self._train_on_trajectory(traj)
                epoch_loss += loss

                # Считаем среднюю награду
                traj_reward = sum(step['reward'] for step in traj['steps'])
                epoch_reward += traj_reward

                print(f"     Потеря: {loss:.4f}, награда: {traj_reward:.2f}")

            if valid_trajectories > 0:
                avg_loss = epoch_loss / valid_trajectories
                avg_reward = epoch_reward / valid_trajectories
            else:
                avg_loss = 0
                avg_reward = 0

            loss_history.append(avg_loss)
            reward_history.append(avg_reward)

            print(f"\n📊 Итоги эпохи {epoch + 1}:")
            print(f"  Средняя потеря: {avg_loss:.4f}")
            print(f"  Средняя награда: {avg_reward:.2f}")
            print(f"  GPU память: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            # Оценка после каждой эпохи
            print("\n🔍 Оценка агента...")
            self.evaluate()

            # Сохраняем чекпоинт
            self._save_checkpoint(epoch)

        print("\n" + "=" * 50)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 50)
        print(f"Финальная потеря: {loss_history[-1]:.4f}")
        print(f"Финальная награда: {reward_history[-1]:.2f}")

        # Визуализация результатов
        self._plot_training_history(loss_history, reward_history)

    def _collect_trajectories(self):
        """Сбор траекторий с подробным логированием"""
        trajectories = []

        # Очищаем кэш перед сбором
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_prompts = self.config.prompts[:self.config.rl.batch_size]
        print(f"   Обработка {len(batch_prompts)} промптов...")

        for prompt_idx, prompt_data in enumerate(batch_prompts):
            print(f"   Промпт {prompt_idx + 1}: {prompt_data['query'][:50]}...")

            state = self.env.reset(prompt_data)
            trajectory = {
                'prompt': prompt_data['query'],
                'steps': [],
                'success': False
            }

            # Получаем список реальных инструментов для этого состояния
            valid_tool_names = [t['name'] for t in state['tools'] if t['available']]

            for step in range(self.config.rl.max_steps):
                # Используем обычный промпт
                context = self._format_context(state)
                inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Генерация
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        temperature=0.3,  # Уменьшаем температуру для более детерминированных ответов
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                tool_call = self._parse_tool_call(response)

                if tool_call:
                    tool_name = tool_call['tool']
                    print(f"     Шаг {step + 1}: вызван {tool_name}")

                    # Проверяем, существует ли инструмент
                    if tool_name in valid_tool_names:
                        next_state, reward, done, info = self.env.step(tool_name)

                        trajectory['steps'].append({
                            'state': state,
                            'action': tool_name,
                            'reward': reward,
                            'latency': info.get('latency', 0),
                            'success': info.get('success', False)
                        })

                        state = next_state
                        if done:
                            trajectory['success'] = info.get('success', False)
                            print(f"     Задача {'решена' if trajectory['success'] else 'провалена'}")
                            break
                    else:
                        print(f"     Некорректный вызов инструмента: {tool_name}")
                        print(f"     Допустимые инструменты: {valid_tool_names}")

                        # Пытаемся исправить невалидный вызов эвристически
                        corrected_tool = self._correct_tool_call(tool_name, valid_tool_names, prompt_data['query'])

                        if corrected_tool:
                            print(f"     Исправляем на: {corrected_tool}")
                            next_state, reward, done, info = self.env.step(corrected_tool)

                            trajectory['steps'].append({
                                'state': state,
                                'action': corrected_tool,
                                'reward': reward * 0.5,  # Уменьшенная награда за исправление
                                'latency': info.get('latency', 0),
                                'success': info.get('success', False)
                            })

                            state = next_state
                            if done:
                                trajectory['success'] = info.get('success', False)
                                print(f"     Задача {'решена' if trajectory['success'] else 'провалена'}")
                                break
                        else:
                            # Штрафуем за невалидный вызов
                            trajectory['steps'].append({
                                'state': state,
                                'action': tool_name,
                                'reward': self.config.reward.invalid_call_penalty,
                                'latency': 0,
                                'success': False
                            })

                            # Пробуем еще раз с жестким промптом
                            if step == 0:
                                print("     Пробуем с жестким промптом...")
                                from src.prompts import get_strict_prompt
                                strict_context = get_strict_prompt(prompt_data['query'], state['tools'])
                                inputs = self.tokenizer(strict_context, return_tensors="pt", truncation=True,
                                                        max_length=512)
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                                with torch.no_grad():
                                    outputs = self.model.generate(**inputs, max_new_tokens=30, temperature=0.1)

                                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                                tool_call = self._parse_tool_call(response)

                                if tool_call and tool_call['tool'] in valid_tool_names:
                                    print(f"     Повторная попытка: вызван {tool_call['tool']}")
                                    next_state, reward, done, info = self.env.step(tool_call['tool'])

                                    trajectory['steps'].append({
                                        'state': state,
                                        'action': tool_call['tool'],
                                        'reward': reward,
                                        'latency': info.get('latency', 0),
                                        'success': info.get('success', False)
                                    })

                                    state = next_state
                                    if done:
                                        trajectory['success'] = info.get('success', False)
                                    break
                            break
                else:
                    print(f"     Шаг {step + 1}: нет вызова инструмента")
                    break

            trajectories.append(trajectory)

        return trajectories

    def _correct_tool_call(self, wrong_tool, valid_tools, query):
        """Исправляет невалидный вызов инструмента"""

        # Специальная обработка для tool_name
        if wrong_tool == 'tool_name' or wrong_tool.startswith('tool_'):
            query_lower = query.lower()

            # Эвристики на основе ключевых слов
            if any(word in query_lower for word in ['+', '-', '*', '/', 'сколько', 'посчитай', 'calculate', 'math']):
                for tool in valid_tools:
                    if 'calc' in tool.lower() or 'math' in tool.lower():
                        return tool
            elif any(word in query_lower for word in ['погода', 'weather', 'температура', 'дождь']):
                for tool in valid_tools:
                    if 'weather' in tool.lower():
                        return tool
            elif any(word in query_lower for word in ['найди', 'поиск', 'search', 'информация']):
                for tool in valid_tools:
                    if 'search' in tool.lower() or 'web' in tool.lower():
                        return tool
            elif any(word in query_lower for word in ['база', 'данные', 'database', 'пользователь']):
                for tool in valid_tools:
                    if 'database' in tool.lower() or 'db' in tool.lower():
                        return tool

        # Если ничего не подошло, возвращаем первый доступный инструмент
        return valid_tools[0]

    def _train_on_trajectory(self, trajectory):
        """Обучение на одной траектории с оптимизацией памяти"""
        if not trajectory['steps']:
            return 0.0

        total_loss = 0

        for step in trajectory['steps']:
            # Очищаем градиенты
            self.optimizer.zero_grad()

            context = self._format_context(step['state'])
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Используем autocast для смешанной точности
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

            # Обратное распространение
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Очищаем ненужные тензоры
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += loss.item()

        return total_loss / len(trajectory['steps'])

    def _format_context(self, state):
        """Форматирование контекста с динамическим промптом"""
        from src.prompts import get_dynamic_prompt

        # Берем доступные инструменты из состояния
        available_tools = []
        for tool in state['tools']:
            if tool['available']:
                available_tools.append({
                    'name': tool['name'],
                    'description': tool['description'],
                    'category': tool['category']
                })

        # Если нет доступных инструментов, используем все
        if not available_tools:
            available_tools = state['tools']

        return get_dynamic_prompt(state['query'], available_tools)

    def _parse_tool_call(self, response):
        """Парсинг вызова инструмента из ответа модели"""
        import re
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, response)
        if match:
            return {'tool': match.group(1).strip()}
        return None

    def _validate_tool_call(self, tool_call, state):
        """Проверяет, существует ли вызываемый инструмент"""
        if not tool_call:
            return False

        tool_name = tool_call.get('tool')
        if not tool_name:
            return False

        # Проверяем, есть ли такой инструмент в состоянии
        valid_tools = [t['name'] for t in state['tools'] if t['available']]
        return tool_name in valid_tools

    def evaluate(self):
        """Оценка агента"""
        print("\n" + "=" * 50)
        print("ОЦЕНКА АГЕНТА")
        print("=" * 50)

        test_prompts = [
            "Найди информацию о искусственном интеллекте",
            "Сколько будет 15 * 8?",
            "Какая погода в Лондоне?",
            "Найди пользователя с id 123 в базе данных"
        ]

        for prompt in test_prompts:
            query_data = {'query': prompt, 'domain': 'test', 'relevant_tools': []}
            state = self.env.reset(query_data)
            print(f"\n📝 Запрос: {prompt}")

            valid_tool_names = [t['name'] for t in state['tools'] if t['available']]

            for step in range(self.config.rl.max_steps):
                context = self._format_context(state)
                inputs = self.tokenizer(context, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=30, temperature=0.1)

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                tool_call = self._parse_tool_call(response)

                if tool_call and tool_call['tool'] in valid_tool_names:
                    next_state, reward, done, info = self.env.step(tool_call['tool'])
                    print(f"  Шаг {step + 1}: {tool_call['tool']} (награда: {reward:.2f})")

                    if done:
                        print(f"  Результат: {'✅' if info.get('success') else '❌'}")
                        break
                    state = next_state
                else:
                    if tool_call:
                        print(f"  Шаг {step + 1}: некорректный вызов {tool_call['tool']}")
                    else:
                        print(f"  Шаг {step + 1}: нет вызова инструмента")
                    break

    def _save_checkpoint(self, epoch):
        """Сохранение чекпоинта модели"""
        checkpoint_dir = f"checkpoints/epoch_{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Сохраняем LoRA веса (они маленькие)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        print(f"  💾 Чекпоинт сохранен в {checkpoint_dir}")

    def _plot_training_history(self, loss_history, reward_history):
        """Визуализация истории обучения"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # График потерь
            ax1.plot(loss_history, 'b-', marker='o')
            ax1.set_xlabel('Эпоха')
            ax1.set_ylabel('Потеря')
            ax1.set_title('Динамика потерь')
            ax1.grid(True)

            # График наград
            ax2.plot(reward_history, 'g-', marker='o')
            ax2.set_xlabel('Эпоха')
            ax2.set_ylabel('Средняя награда')
            ax2.set_title('Динамика наград')
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig('training_history.png')
            print("  📊 Графики сохранены в training_history.png")
        except ImportError:
            print("  📊 Для визуализации установите matplotlib: pip install matplotlib")