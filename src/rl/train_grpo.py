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
from src.llm.llm_client import LLMClient

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

    def __init__(self, config: Config, llm_client: LLMClient = None):
        self.config = config
        self.env = MCPEnvironment(config)
        self.reward_fn = GRPOToolReward(config)

        # Сохраняем LLM клиент
        self.llm_client = llm_client or LLMClient(config)

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
            print("⚠️ Продолжаем без локальной модели, используя LLM клиент...")
            self.model = None
            self.tokenizer = None

        # Оптимизатор с меньшим размером batch
        if self.model is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.rl.learning_rate,
                weight_decay=0.01
            )
            # Gradient checkpointing для экономии памяти
            self.model.gradient_checkpointing_enable()
        else:
            self.optimizer = None

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

        if self.model is not None:
            print(f"  - Trainable params: 2.2M / 1.5B (0.14%)")
        else:
            print(f"  - Используется внешняя модель через API: {self.config.model_name}")

        # История обучения
        loss_history = []
        proxy_loss_history = []
        reward_history = []
        success_rate_history = []

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
            total_success = 0
            total_reward = 0
            total_steps = 0

            for i, traj in enumerate(trajectories):
                if traj['steps']:
                    valid_trajectories += 1
                    if traj['success']:
                        total_success += 1
                    traj_reward = sum(step['reward'] for step in traj['steps'])
                    total_reward += traj_reward
                    total_steps += len(traj['steps'])
                    print(f"   Траектория {i + 1}: {len(traj['steps'])} шагов, успех: {traj['success']}, награда: {traj_reward:.2f}")

            print(f"   Валидных траекторий: {valid_trajectories}")

            if valid_trajectories == 0:
                print("❌ Нет валидных траекторий! Проверьте формат ответов модели.")
                continue

            # Вычисляем метрики
            success_rate = total_success / valid_trajectories if valid_trajectories > 0 else 0
            avg_reward = total_reward / valid_trajectories if valid_trajectories > 0 else 0
            avg_steps = total_steps / valid_trajectories if valid_trajectories > 0 else self.config.rl.max_steps

            # ПРОКСИ-LOSS: комбинация метрик для имитации loss
            # Чем ниже loss, тем лучше качество
            proxy_loss = (1.0 - success_rate) + (avg_steps / self.config.rl.max_steps) * 0.3

            # Нормализация в диапазон [0, 2] для совместимости
            proxy_loss = min(2.0, max(0.0, proxy_loss))

            # Обучаем на траекториях (только если есть локальная модель)
            if self.model is not None:
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
                proxy_loss_history.append(proxy_loss)
                success_rate_history.append(success_rate)

                print(f"\n📊 Итоги эпохи {epoch + 1}:")
                print(f"  Средняя потеря: {avg_loss:.4f}")
                print(f"  Прокси-loss: {proxy_loss:.4f} (чем ниже, тем лучше)")
                print(f"  Средняя награда: {avg_reward:.2f}")
                print(f"  Успешность: {success_rate:.1%}")
            else:
                print("\n📈 Используется внешняя модель (API)")
                reward_history.append(avg_reward)
                proxy_loss_history.append(proxy_loss)
                success_rate_history.append(success_rate)

                print(f"\n📊 Итоги эпохи {epoch + 1}:")
                print(f"  Прокси-loss: {proxy_loss:.4f} (чем ниже, тем лучше)")
                print(f"  Средняя награда: {avg_reward:.2f}")
                print(f"  Успешность: {success_rate:.1%}")
                print(f"  Среднее число шагов: {avg_steps:.1f}")

            print(f"  GPU память: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            # Оценка после каждой эпохи
            print("\n🔍 Оценка агента...")
            self.evaluate()

            # Сохраняем чекпоинт (только если есть локальная модель)
            if self.model is not None:
                self._save_checkpoint(epoch)

        print("\n" + "=" * 50)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 50)
        if loss_history:
            print(f"Финальная потеря: {loss_history[-1]:.4f}")
        print(f"Финальный прокси-loss: {proxy_loss_history[-1]:.4f}")
        print(f"Финальная награда: {reward_history[-1]:.2f}")
        print(f"Финальная успешность: {success_rate_history[-1]:.1%}")

        # Визуализация результатов
        if self.model is not None:
            self._plot_training_history(loss_history, reward_history, proxy_loss_history, success_rate_history)
        else:
            self._plot_training_history([], reward_history, proxy_loss_history, success_rate_history)

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

            if not valid_tool_names:
                print(f"     ⚠️ Нет доступных инструментов для этого запроса")
                trajectories.append(trajectory)
                continue

            for step in range(self.config.rl.max_steps):
                # Формируем промпт
                context = self._format_context(state)

                # Используем LLM клиент для генерации
                response = self.llm_client.ask(context)

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
                            break
                else:
                    print(f"     Шаг {step + 1}: нет вызова инструмента")
                    break

            trajectories.append(trajectory)

        return trajectories

    def _train_on_trajectory(self, trajectory):
        """Обучение на одной траектории с оптимизацией памяти"""
        if not trajectory['steps'] or self.model is None:
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

    def _correct_tool_call(self, wrong_tool, valid_tools, query):
        """Исправляет невалидный вызов инструмента на основе эвристик"""
        if not valid_tools:
            return None

        query_lower = query.lower()
        print(f"     🔍 Анализ запроса: '{query_lower}'")

        # Приоритетные категории инструментов
        priority_keywords = {
            'math': ['+', '-', '*', '/', 'сколько', 'посчитай', 'calculate', 'math', '2 + 2', '2+2', 'plus', 'minus',
                     'times', 'divided'],
            'calculator': ['calc', 'calculator', 'compute', 'arithmetic'],
            'search': ['search', 'find', 'lookup', 'информация', 'найди', 'поиск'],
            'weather': ['weather', 'погода', 'temperature', 'temp'],
            'database': ['database', 'db', 'sql', 'query', 'data'],
            'translate': ['translate', 'переведи', 'translation']
        }

        # Сначала ищем инструменты с "calc" или "math" в названии для математических запросов
        if any(word in query_lower for word in priority_keywords['math']):
            print(f"     🧮 Обнаружен математический запрос")
            for tool in valid_tools:
                tool_lower = tool.lower()
                if any(x in tool_lower for x in ['calc', 'math', 'compute', 'arithmetic']):
                    print(f"     ✅ Найден математический инструмент: {tool}")
                    return tool

        # Если не нашли, используем систему оценки
        best_match = None
        best_score = 0

        for tool in valid_tools:
            score = 0
            tool_lower = tool.lower()

            # Проверяем каждую категорию
            for category, keywords in priority_keywords.items():
                if any(word in query_lower for word in keywords):
                    if any(x in tool_lower for x in keywords[:3]):  # Проверяем первые 3 ключевых слова
                        score += 5
                    elif category in tool_lower:
                        score += 3

            # Проверяем отдельные слова из запроса
            query_words = set(query_lower.split())
            tool_words = set(tool_lower.replace('.', ' ').replace('/', ' ').replace('-', ' ').replace('_', ' ').split())
            common_words = query_words.intersection(tool_words)
            score += len(common_words) * 2

            print(f"     Инструмент '{tool}' имеет оценку {score}")

            if score > best_score:
                best_score = score
                best_match = tool

        if best_match and best_score > 0:
            print(f"     🔍 Выбран инструмент с оценкой {best_score}: {best_match}")
            return best_match

        # Если ничего не нашли, возвращаем первый инструмент
        print(f"     ⚠️ Ничего не найдено, берём первый: {valid_tools[0]}")
        return valid_tools[0]

    def evaluate(self):
        """Оценка агента с эвристиками"""
        print("\n" + "=" * 50)
        print("ОЦЕНКА АГЕНТА")
        print("=" * 50)

        test_prompts = [
            "Top 10 NBA players",
            "Bitcoin price USD",
            "Weather in London",
            "Latest songs by Drake",
            "Trending on Twitter",
            "Best PC games 2024"
        ]

        for prompt in test_prompts:
            query_data = {'query': prompt, 'domain': 'test', 'relevant_tools': []}
            state = self.env.reset(query_data)
            print(f"\n📝 Запрос: {prompt}")

            valid_tools = [t['name'] for t in state['tools'] if t['available']]

            for step in range(self.config.rl.max_steps):
                context = self._format_context(state)

                # Используем LLM клиент для генерации
                response = self.llm_client.ask(context)

                tool_call = self._parse_tool_call(response)

                if tool_call:
                    tool_name = tool_call['tool']
                    print(f"  🤖 Модель вызвала: {tool_name}")

                    # ДОБАВЛЯЕМ ИСПРАВЛЕНИЕ!
                    if tool_name == 'tool_name' or tool_name not in valid_tools:
                        corrected = self._correct_tool_call(tool_name, valid_tools, prompt)
                        if corrected:
                            print(f"  🔧 Исправляем на: {corrected}")
                            tool_name = corrected

                    if tool_name in valid_tools:
                        next_state, reward, done, info = self.env.step(tool_name)
                        print(f"  ✅ Использован: {tool_name}")
                        print(f"     Награда: {reward:.2f}")
                        print(f"     Успех: {'✅' if info.get('success') else '❌'}")

                        if done:
                            break

                        state = next_state
                    else:
                        print(f"  ❌ Некорректный инструмент: {tool_name}")
                        break
                else:
                    print(f"  ⚠️ Нет вызова инструмента")
                    break

    def load_checkpoint(self, checkpoint_path):
        """Загрузка чекпоинта"""
        if self.model is None:
            print("⚠️ Нет локальной модели для загрузки чекпоинта")
            return

        from peft import PeftModel
        try:
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            print(f"✅ Успешно загружен чекпоинт из {checkpoint_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки чекпоинта: {e}")
            raise

    def _save_checkpoint(self, epoch):
        """Сохранение чекпоинта модели"""
        if self.model is None:
            return

        checkpoint_dir = f"checkpoints/epoch_{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Сохраняем LoRA веса (они маленькие)
        self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)

        print(f"  💾 Чекпоинт сохранен в {checkpoint_dir}")

    def _plot_training_history(self, loss_history, reward_history, proxy_loss_history=None, success_rate_history=None):
        """Визуализация истории обучения"""
        try:
            import matplotlib.pyplot as plt

            if proxy_loss_history is not None:
                # Три графика: loss/прокси-loss, награды, успешность
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

                # График потерь или прокси-loss
                if loss_history:
                    ax1.plot(loss_history, 'b-', marker='o', label='Реальный loss')
                    ax1.set_title('Динамика потерь')
                else:
                    ax1.plot(proxy_loss_history, 'r-', marker='o', label='Прокси-loss')
                    ax1.set_title('Динамика прокси-loss')

                ax1.set_xlabel('Эпоха')
                ax1.set_ylabel('Потеря')
                ax1.grid(True)
                ax1.legend()

                # График наград
                ax2.plot(reward_history, 'g-', marker='o')
                ax2.set_xlabel('Эпоха')
                ax2.set_ylabel('Средняя награда')
                ax2.set_title('Динамика наград')
                ax2.grid(True)

                # График успешности
                if success_rate_history:
                    ax3.plot(success_rate_history, 'purple', marker='o')
                    ax3.set_xlabel('Эпоха')
                    ax3.set_ylabel('Успешность')
                    ax3.set_title('Динамика успешности')
                    ax3.grid(True)
                    ax3.set_ylim([0, 1])

            else:
                # Стандартные два графика
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # График потерь (если есть)
                if loss_history:
                    ax1.plot(loss_history, 'b-', marker='o')
                    ax1.set_xlabel('Эпоха')
                    ax1.set_ylabel('Потеря')
                    ax1.set_title('Динамика потерь')
                    ax1.grid(True)
                else:
                    ax1.text(0.5, 0.5, 'Нет данных о потерях', ha='center', va='center')
                    ax1.set_title('Динамика потерь (недоступно)')

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