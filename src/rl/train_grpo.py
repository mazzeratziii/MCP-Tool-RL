# src/rl/train_grpo.py
import os
import sys
import platform
import ctypes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
                print(" c10.dll успешно предварительно загружен в train_grpo")
    except Exception as e:
        print(f" Предзагрузка c10.dll не удалась в train_grpo: {e}")


class NetMCPTrainer:
    """Тренер для обучения агента NetMCP"""

    def __init__(self, config: Config):
        self.config = config
        self.env = MCPEnvironment(config)
        self.reward_fn = GRPOToolReward(config)

        # Загружаем модель
        print(f"Загрузка модели {config.model_name}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(" Модель успешно загружена")
        except Exception as e:
            print(f" Ошибка загрузки модели: {e}")
            raise

        # Оптимизатор
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.rl.learning_rate)

    def train(self):
        """Настоящий цикл обучения"""
        print("\n" + "=" * 50)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 50)
        print(f"Конфигурация:")
        print(f"  - Алгоритм: {self.config.rl.algorithm}")
        print(f"  - Эпох: {self.config.rl.num_epochs}")
        print(f"  - Batch size: {self.config.rl.batch_size}")
        print(f"  - Max steps: {self.config.rl.max_steps}")

        for epoch in range(self.config.rl.num_epochs):
            print(f"\n--- Эпоха {epoch + 1}/{self.config.rl.num_epochs} ---")

            # Собираем траектории
            trajectories = self._collect_trajectories()

            # Обучаем на траекториях
            total_loss = 0
            for traj in trajectories:
                loss = self._train_on_trajectory(traj)
                total_loss += loss

            avg_loss = total_loss / len(trajectories) if trajectories else 0
            print(f"Средняя потеря: {avg_loss:.4f}")

            # Оценка после эпохи
            self.evaluate()

        print("\n Обучение завершено!")

    def _collect_trajectories(self):
        """Сбор траекторий взаимодействия"""
        trajectories = []

        for prompt_data in self.config.prompts[:self.config.rl.batch_size]:
            state = self.env.reset(prompt_data)
            trajectory = {
                'prompt': prompt_data['query'],
                'steps': [],
                'success': False
            }

            for step in range(self.config.rl.max_steps):
                # Генерируем действие
                context = self._format_context(state)
                inputs = self.tokenizer(context, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Парсим вызов инструмента
                tool_call = self._parse_tool_call(response)
                if not tool_call:
                    break

                # Выполняем действие
                next_state, reward, done, info = self.env.step(tool_call['tool'])

                # Сохраняем шаг
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

            trajectories.append(trajectory)

        return trajectories

    def _train_on_trajectory(self, trajectory):
        """Обучение на одной траектории"""
        if not trajectory['steps']:
            return 0.0

        total_loss = 0

        for step in trajectory['steps']:
            # Форматируем вход
            context = self._format_context(step['state'])
            inputs = self.tokenizer(context, return_tensors="pt")

            # Форвард проход
            self.optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # Бэквард проход
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(trajectory['steps'])

    def evaluate(self):
        """Оценка агента"""
        print("\n" + "=" * 50)
        print("ОЦЕНКА АГЕНТА")
        print("=" * 50)

        test_prompts = [
            "Найди информацию о искусственном интеллекте",
            "Сколько будет 15 * 8?",
            "Какая погода в Лондоне?"
        ]

        for prompt in test_prompts:
            query_data = {
                'query': prompt,
                'domain': 'test',
                'relevant_tools': []
            }
            state = self.env.reset(query_data)
            print(f"\n📝 Запрос: {prompt}")

            for step in range(self.config.rl.max_steps):
                context = self._format_context(state)
                inputs = self.tokenizer(context, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                tool_call = self._parse_tool_call(response)
                if tool_call:
                    next_state, reward, done, info = self.env.step(tool_call['tool'])
                    print(f"  Шаг {step + 1}: {tool_call['tool']} (награда: {reward:.2f})")

                    if done:
                        print(f"  Результат: {'Ок' if info.get('success') else 'Отмена'}")
                        break
                    state = next_state
                else:
                    print(f"  Шаг {step + 1}: нет вызова инструмента")
                    break

    def _format_context(self, state):
        """Форматирование контекста"""
        context = f"User query: {state['query']}\n\nAvailable tools:\n"
        for tool in state['tools'][:5]:
            status = "Ок" if tool['available'] else "Отмена"
            context += f"{status} {tool['name']} (latency: {tool['latency']:.2f}s)\n"
        context += "\nRespond with <tool_call>tool_name</tool_call> to use a tool."
        return context

    def _parse_tool_call(self, response):
        """Парсинг вызова инструмента"""
        import re
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, response)
        if match:
            return {'tool': match.group(1).strip()}
        return None