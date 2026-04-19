import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NetworkConfig:
    base_latency_range: Tuple[float, float] = (0.05, 1.0)
    jitter_range: Tuple[float, float] = (0.01, 0.2)
    failure_rate_range: Tuple[float, float] = (0.01, 0.2)
    congestion_factor_range: Tuple[float, float] = (0.5, 2.0)
    base_latency: float = 0.1
    jitter: float = 0.05
    failure_rate: float = 0.1
    congestion_factor: float = 1.0


@dataclass
class ToolBenchConfig:
    split: str = "train"
    sample_size: Optional[int] = 60000
    num_tools: int = 10000


@dataclass
class RLConfig:
    algorithm: str = "grpo"
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 100
    max_steps: int = 3
    kl_coef: float = 0.2
    temperature: float = 0.8
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    dropout: float = 0.1


@dataclass
class RewardConfig:
    success_reward: float = 3.0
    failure_penalty: float = -0.2
    step_penalty: float = -0.03
    invalid_call_penalty: float = -0.5
    semantic_bonus: float = 0.5
    latency_threshold: float = 1.0
    wrong_tool_penalty: float = -0.5
    extra_step_penalty: float = -0.1


class Config:
    def __init__(self):
        print("\n" + "=" * 60)
        print("INITIALIZING CONFIGURATION")
        print("=" * 60)

        self.network = NetworkConfig()
        self.toolbench = ToolBenchConfig()
        self.rl = RLConfig()
        self.reward = RewardConfig()

        self.model_name = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-1.5B-Instruct')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', '')
        self.openai_api_token = os.getenv('OPENAI_API_TOKEN', '')
        self.system_prompt = os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant.')
        self.user_prompt = os.getenv('USER_PROMPT', '')
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '100'))
        self.min_request_timeout = float(os.getenv('MIN_REQUEST_TIMEOUT', '60.0'))

        self.tools = []  # Will be populated after loading
        self.prompts = []  # Will be populated after loading
        self.train_prompts = []
        self.val_prompts = []
        self.loader = None
        self.tool_selector = None

        self._validate()

        print(f"Configuration loaded:")
        print(f"  Model: {self.model_name}")
        print(f"  Base URL: {self.openai_base_url or 'local model'}")
        print(f"  Learning rate: {self.rl.learning_rate}")
        print(f"  Batch size: {self.rl.batch_size}")

    def _validate(self):
        if not self.model_name:
            print("Warning: MODEL_NAME not set, using default")

    def load_data(self):
        """Load ToolBench data after configuration is ready"""
        from src.data.toolbench_loader import ToolBenchLoader
        from src.tools.tool_selector import ToolSelector

        print("\n" + "=" * 60)
        print("LOADING TOOLBENCH DATA")
        print("=" * 60)

        self.loader = ToolBenchLoader(
            split=self.toolbench.split,
            sample_size=self.toolbench.sample_size
        )

        print("\n" + "=" * 60)
        print("CREATING TOOL SELECTOR")
        print("=" * 60)

        self.tool_selector = ToolSelector(self.loader.tools)
        self.tool_selector.print_category_stats()

        print(f"\n" + "=" * 60)
        print(f"SELECTING {self.toolbench.num_tools} TOOLS FOR TRAINING")
        print("=" * 60)

        selected_tools = []
        tools_per_category = max(5, self.toolbench.num_tools // 10)

        for category, data in self.tool_selector.CATEGORIES.items():
            if data['tools']:
                category_tools = data['tools'][:tools_per_category]
                selected_tools.extend(category_tools)
                print(f"   {category}: selected {len(category_tools)} tools")

        if len(selected_tools) < self.toolbench.num_tools:
            remaining = self.toolbench.num_tools - len(selected_tools)
            sorted_tools = sorted(
                self.loader.tools,
                key=lambda x: len(x.get('description', '')),
                reverse=True
            )
            for tool in sorted_tools:
                if tool not in selected_tools:
                    selected_tools.append(tool)
                    remaining -= 1
                    if remaining == 0:
                        break
            print(f"   added {self.toolbench.num_tools - len(selected_tools) + remaining} popular tools")

        self.tools = selected_tools[:self.toolbench.num_tools]
        print(f"\nTotal: {len(self.tools)} tools selected")

        category_distribution = {}
        for tool in self.tools:
            cat = tool.get('category', 'Unknown')
            category_distribution[cat] = category_distribution.get(cat, 0) + 1

        print("\nCATEGORY DISTRIBUTION:")
        for cat, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {cat}: {count} tools")

        print("\n" + "=" * 60)
        print("PREPARING TRAINING PROMPTS")
        print("=" * 60)

        all_prompts = self.loader.get_training_prompts()

        valid_prompts = []
        tool_names = {t['name'] for t in self.tools}

        for prompt in all_prompts:
            relevant = [t for t in prompt.get('relevant_tools', []) if t['name'] in tool_names]
            if relevant:
                prompt['relevant_tools'] = relevant
                valid_prompts.append(prompt)

        split_idx = int(len(valid_prompts) * 0.8)
        self.train_prompts = valid_prompts[:split_idx]
        self.val_prompts = valid_prompts[split_idx:]
        self.prompts = self.train_prompts

        print(f"   Total prompts: {len(valid_prompts)}")
        print(f"   Train prompts: {len(self.train_prompts)}")
        print(f"   Val prompts: {len(self.val_prompts)}")

        print("\n" + "=" * 60)
        print("CONFIGURATION COMPLETE")
        print("=" * 60)

    def get_tools_by_category(self, category: str) -> List[Dict]:
        return [t for t in self.tools if t.get('category', '').lower() == category.lower()]