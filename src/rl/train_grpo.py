import os
import sys
import platform
import ctypes
import json
import csv
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.rl.reward_functions import GRPOToolReward
from src.llm.llm_client import LLMClient

if platform.system() == "Windows":
    try:
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.origin:
            torch_dir = os.path.dirname(torch_spec.origin)
            dll_path = os.path.join(torch_dir, "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception as e:
        print(f"c10.dll preload failed: {e}")


class NetMCPTrainer:
    def __init__(self, config: Config, llm_client: LLMClient = None):
        self.config = config

        # Ensure data is loaded before anything else
        if not config.tools:
            print("Loading data before initializing trainer...")
            config.load_data()

        self.reward_fn = GRPOToolReward(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading model {config.model_name}...")
        self.model = None
        self.tokenizer = None
        self.optimizer = None

        try:
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
            print("Model loaded with 4-bit quantization")

            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.rl.learning_rate,
                weight_decay=0.01
            )
            self.model.gradient_checkpointing_enable()

        except Exception as e:
            print(f"Model loading error: {e}")
            print("Continuing without local model")
            self.model = None
            self.tokenizer = None
            self.optimizer = None

        if self.model is not None:
            self.llm_client = llm_client or LLMClient(config, local_model=self.model, local_tokenizer=self.tokenizer)
        else:
            self.llm_client = llm_client or LLMClient(config)

        self.env = MCPEnvironment(config, self.llm_client)

        self.training_log = []
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def _save_training_metrics(self, epoch, proxy_loss, avg_reward, success_rate, avg_steps):
        epoch_data = {
            'epoch': epoch + 1,
            'proxy_loss': proxy_loss,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'timestamp': datetime.now().isoformat()
        }
        self.training_log.append(epoch_data)

        with open(f"{self.results_dir}/training_log.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)

        with open(f"{self.results_dir}/training_log.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'proxy_loss', 'avg_reward', 'success_rate', 'avg_steps', 'timestamp'])
            writer.writeheader()
            writer.writerows(self.training_log)

        if self.training_log:
            best_loss = min([m['proxy_loss'] for m in self.training_log])
            best_reward = max([m['avg_reward'] for m in self.training_log])
            best_epoch = min(self.training_log, key=lambda x: x['proxy_loss'])['epoch']

            best_metrics = {
                'best_proxy_loss': best_loss,
                'best_reward': best_reward,
                'best_epoch': best_epoch,
                'total_epochs': len(self.training_log),
                'final_epoch_data': self.training_log[-1]
            }

            with open(f"{self.results_dir}/best_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(best_metrics, f, indent=2, ensure_ascii=False)

    def _save_final_state(self):
        config_data = {
            'model_name': self.config.model_name,
            'openai_base_url': self.config.openai_base_url,
            'rl_params': {
                'num_epochs': self.config.rl.num_epochs,
                'batch_size': self.config.rl.batch_size,
                'learning_rate': self.config.rl.learning_rate,
                'max_steps': self.config.rl.max_steps,
                'temperature': self.config.rl.temperature
            },
            'toolbench': {
                'sample_size': self.config.toolbench.sample_size,
                'num_tools': self.config.toolbench.num_tools
            },
            'network': {
                'base_latency': self.config.network.base_latency,
                'failure_rate': self.config.network.failure_rate,
                'jitter': self.config.network.jitter
            }
        }

        with open(f"{self.results_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        tools_data = []
        for tool in self.config.tools[:100]:
            tools_data.append({
                'name': tool['name'],
                'category': tool.get('category', 'Unknown'),
                'description': tool.get('description', '')[:200]
            })

        with open(f"{self.results_dir}/tools_sample.json", 'w', encoding='utf-8') as f:
            json.dump(tools_data, f, indent=2, ensure_ascii=False)

        print(f"\nFinal state saved to {self.results_dir}/")

    def train(self):
        print("TRAINING STARTED")
        print(f"Configuration:")
        print(f"  Algorithm: {self.config.rl.algorithm}")
        print(f"  Epochs: {self.config.rl.num_epochs}")
        print(f"  Batch size: {self.config.rl.batch_size}")
        print(f"  Max steps: {self.config.rl.max_steps}")
        print(f"  Total prompts: {len(self.config.prompts)}")

        if self.model is not None:
            print(f"  Trainable params: 2.2M / 1.5B (0.14%)")
        else:
            print(f"  Using external API model: {self.config.model_name}")

        loss_history = []
        proxy_loss_history = []
        reward_history = []
        success_rate_history = []

        for epoch in range(self.config.rl.num_epochs):
            print(f"\n{'=' * 40}")
            print(f"EPOCH {epoch + 1}/{self.config.rl.num_epochs}")
            print(f"{'=' * 40}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("\nCollecting trajectories...")
            trajectories = self._collect_trajectories()
            print(f"  Collected {len(trajectories)} trajectories")

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

            print(f"  Valid trajectories: {valid_trajectories}")

            if valid_trajectories == 0:
                print("No valid trajectories")
                continue

            success_rate = total_success / valid_trajectories if valid_trajectories > 0 else 0
            avg_reward = total_reward / valid_trajectories if valid_trajectories > 0 else 0
            avg_steps = total_steps / valid_trajectories if valid_trajectories > 0 else self.config.rl.max_steps

            max_steps_value = self.config.rl.max_steps
            success_term = (1.0 - success_rate) * 0.7
            steps_term = (avg_steps / max_steps_value) * 0.1
            reward_term = max(0, 1 - avg_reward / 2) * 0.2

            proxy_loss = success_term + steps_term + reward_term
            proxy_loss = min(2.0, max(0.0, proxy_loss))

            if self.model is not None:
                print("\nTraining on trajectories...")
                epoch_loss = 0
                epoch_reward = 0

                for traj_idx, traj in enumerate(trajectories):
                    if not traj['steps']:
                        continue

                    loss = self._train_on_trajectory(traj)
                    epoch_loss += loss

                    traj_reward = sum(step['reward'] for step in traj['steps'])
                    epoch_reward += traj_reward

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

                print(f"\nEpoch {epoch + 1} results:")
                print(f"  Average loss: {avg_loss:.4f}")
                print(f"  Proxy loss: {proxy_loss:.4f}")
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Success rate: {success_rate:.1%}")
            else:
                reward_history.append(avg_reward)
                proxy_loss_history.append(proxy_loss)
                success_rate_history.append(success_rate)

                print(f"\nEpoch {epoch + 1} results:")
                print(f"  Proxy loss: {proxy_loss:.4f}")
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Success rate: {success_rate:.1%}")
                print(f"  Average steps: {avg_steps:.1f}")

            if torch.cuda.is_available():
                print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            self._save_training_metrics(epoch, proxy_loss, avg_reward, success_rate, avg_steps)

            print("\nEvaluating agent...")
            self.evaluate()

            if self.model is not None:
                self._save_checkpoint(epoch)

        print("TRAINING COMPLETED")
        if loss_history:
            print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Final proxy loss: {proxy_loss_history[-1]:.4f}")
        print(f"Final reward: {reward_history[-1]:.2f}")
        print(f"Final success rate: {success_rate_history[-1]:.1%}")

        self._save_final_state()

    def _collect_trajectories(self):
        trajectories = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_prompts = self.config.prompts[:self.config.rl.batch_size]

        for prompt_idx, prompt_data in enumerate(batch_prompts):
            state = self.env.reset(prompt_data)
            trajectory = {
                'prompt': prompt_data['query'],
                'steps': [],
                'success': False
            }

            valid_tool_names = [t['name'] for t in state['tools'] if t['available']]

            if not valid_tool_names:
                trajectories.append(trajectory)
                continue

            for step in range(self.config.rl.max_steps):
                context = self._format_context(state)
                response = self.llm_client.ask(context)
                tool_call = self._parse_tool_call(response)

                if tool_call:
                    tool_name = tool_call['tool']

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
                            break
                    else:
                        corrected_tool = self._correct_tool_call(tool_name, valid_tool_names, prompt_data['query'])

                        if corrected_tool:
                            next_state, reward, done, info = self.env.step(corrected_tool)

                            trajectory['steps'].append({
                                'state': state,
                                'action': corrected_tool,
                                'reward': reward * 0.5,
                                'latency': info.get('latency', 0),
                                'success': info.get('success', False)
                            })

                            state = next_state
                            if done:
                                trajectory['success'] = info.get('success', False)
                                break
                        else:
                            trajectory['steps'].append({
                                'state': state,
                                'action': tool_name,
                                'reward': self.config.reward.invalid_call_penalty,
                                'latency': 0,
                                'success': False
                            })
                            break
                else:
                    break

            trajectories.append(trajectory)

        return trajectories

    def _train_on_trajectory(self, trajectory):
        if not trajectory['steps'] or self.model is None:
            return 0.0

        total_loss = 0

        for step in trajectory['steps']:
            self.optimizer.zero_grad()

            context = self._format_context(step['state'])
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += loss.item()

        return total_loss / len(trajectory['steps'])

    def _format_context(self, state):
        from src.prompts import get_dynamic_prompt

        available_tools = []
        for tool in state['tools']:
            if tool['available']:
                available_tools.append({
                    'name': tool['name'],
                    'description': tool['description'],
                    'category': tool['category']
                })

        if not available_tools:
            available_tools = state['tools']

        return get_dynamic_prompt(state['query'], available_tools)

    def _parse_tool_call(self, response):
        import re
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, response)
        if match:
            return {'tool': match.group(1).strip()}
        return None

    def _correct_tool_call(self, wrong_tool, valid_tools, query):
        if not valid_tools:
            return None

        query_lower = query.lower()
        priority_keywords = {
            'math': ['+', '-', '*', '/', 'calculate', 'math'],
            'weather': ['weather', 'temperature', 'погода'],
            'search': ['search', 'find', 'lookup', 'найди', 'поиск'],
            'database': ['database', 'db', 'data', 'query']
        }

        best_match = None
        best_score = 0

        for tool in valid_tools:
            score = 0
            tool_lower = tool.lower()

            for category, keywords in priority_keywords.items():
                if any(word in query_lower for word in keywords):
                    if any(x in tool_lower for x in keywords[:3]):
                        score += 5
                    elif category in tool_lower:
                        score += 3

            query_words = set(query_lower.split())
            tool_words = set(tool_lower.replace('.', ' ').replace('/', ' ').replace('-', ' ').replace('_', ' ').split())
            common_words = query_words.intersection(tool_words)
            score += len(common_words) * 2

            if score > best_score:
                best_score = score
                best_match = tool

        if best_match and best_score > 0:
            return best_match

        return valid_tools[0]

    def evaluate(self):
        print("EVALUATING AGENT")

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
            print(f"\nQuery: {prompt}")

            valid_tools = [t['name'] for t in state['tools'] if t['available']]

            for step in range(self.config.rl.max_steps):
                context = self._format_context(state)
                response = self.llm_client.ask(context)
                tool_call = self._parse_tool_call(response)

                if tool_call:
                    tool_name = tool_call['tool']

                    if tool_name == 'tool_name' or tool_name not in valid_tools:
                        corrected = self._correct_tool_call(tool_name, valid_tools, prompt)
                        if corrected:
                            tool_name = corrected

                    if tool_name in valid_tools:
                        next_state, reward, done, info = self.env.step(tool_name)
                        print(f"  Used: {tool_name}")
                        print(f"    Reward: {reward:.2f}")
                        print(f"    Success: {'Yes' if info.get('success') else 'No'}")

                        if done:
                            break

                        state = next_state
                    else:
                        print(f"  Invalid tool: {tool_name}")
                        break
                else:
                    print(f"  No tool call")
                    break

    def load_checkpoint(self, checkpoint_path):
        if self.model is None:
            print("No local model for checkpoint loading")
            return

        from peft import PeftModel
        try:
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Checkpoint loading error: {e}")

    def _save_checkpoint(self, epoch):
        if self.model is None:
            return

        checkpoint_dir = f"checkpoints/epoch_{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)

        print(f"  Checkpoint saved to {checkpoint_dir}")