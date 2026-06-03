# Конфигурация проекта
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        """Load load dotenv."""
        return False

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
    sample_size: Optional[int] = 80000
    num_tools: int = 15000


@dataclass
class RLConfig:
    algorithm: str = "grpo"
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 100
    max_steps: int = 5
    kl_coef: float = 0.2
    temperature: float = 0.8
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    dropout: float = 0.1


@dataclass
class RewardConfig:
    success_reward: float = 5.0
    failure_penalty: float = 0.0
    step_penalty: float = -0.01
    invalid_call_penalty: float = -0.1
    semantic_bonus: float = 0.5
    latency_threshold: float = 1.0
    wrong_tool_penalty: float = -0.2
    extra_step_penalty: float = -0.05


class Config:
    def __init__(self):
        """Initialize the object."""
        print("\n" + "=" * 60)
        print("INITIALIZING CONFIGURATION")
        print("=" * 60)

        self.network = NetworkConfig()
        self.toolbench = ToolBenchConfig()
        self.rl = RLConfig()
        self.reward = RewardConfig()

        self.model_name = os.getenv('MODEL_NAME', '')
        self.openai_base_url = os.getenv('BASE_URL', '')
        self.openai_api_token = os.getenv('API_TOKEN', '')
        self.system_prompt = os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant.')
        self.user_prompt = os.getenv('USER_PROMPT', '')
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '100'))
        self.min_request_timeout = float(os.getenv('MIN_REQUEST_TIMEOUT', '60.0'))
        # Локальная модель может быть включена через USE_LOCAL_MODEL при необходимости

        self.tools = []
        self.prompts = []
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
        # При необходимости можно вывести флаг использования локальной модели

    def _validate(self):
        """Validate validate."""
        if not self.model_name:
            print("Warning: MODEL_NAME not set, using default")

    def load_data(self):
        """Load load data."""
        from src.data.toolbench_loader import ToolBenchLoader
        from src.tools.tool_selector import ToolSelector

        print("\n" + "-" * 30)
        print("LOADING TOOLBENCH DATA")
        print("-" * 30)

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

        # Добавляем fallback инструменты
        fallback_tools = [
            {
                "name": "Calculator.Evaluate",
                "category": "math",
                "description": "Evaluate mathematical expressions and arithmetic operations like 2+2, 10*5, sqrt(16), 100/4. Supports basic arithmetic (+, -, *, /), exponents, and common math functions.",
                "method": "GET",
                "required_parameters": [{"name": "expression", "type": "string", "description": "Mathematical expression to evaluate"}],
                "optional_parameters": []
            },
            {
                "name": "General.NoToolNeeded",
                "category": "general",
                "description": "Use when the query doesn't require any external tool call. For simple questions, greetings, or requests that can be answered directly without API calls.",
                "method": "GET",
                "required_parameters": [],
                "optional_parameters": []
            },
            {
                "name": "text.count",
                "category": "utility",
                "description": "Count characters, words and lines in a text. Use for word count, text length, line count and simple text statistics.",
                "method": "POST",
                "required_parameters": [{"name": "text", "type": "string", "description": "Input text"}],
                "optional_parameters": []
            },
            {
                "name": "text.extract_keywords",
                "category": "utility",
                "description": "Extract frequent keywords from text. Use for keyword extraction, text analysis and summarizing important terms.",
                "method": "POST",
                "required_parameters": [{"name": "text", "type": "string", "description": "Input text"}],
                "optional_parameters": [{"name": "limit", "type": "integer", "description": "Maximum number of keywords"}]
            },
            {
                "name": "text.slugify",
                "category": "utility",
                "description": "Create URL-friendly slug from text. Use for converting titles into lowercase hyphen-separated identifiers.",
                "method": "POST",
                "required_parameters": [{"name": "text", "type": "string", "description": "Input text"}],
                "optional_parameters": []
            },
            {
                "name": "regex.search",
                "category": "utility",
                "description": "Search text with a regular expression and return matches. Use for pattern matching in text.",
                "method": "POST",
                "required_parameters": [
                    {"name": "text", "type": "string", "description": "Input text"},
                    {"name": "pattern", "type": "string", "description": "Regular expression"}
                ],
                "optional_parameters": []
            },
            {
                "name": "datetime.now",
                "category": "datetime",
                "description": "Return current date and time. Use for current timestamp, current UTC time and date requests.",
                "method": "GET",
                "required_parameters": [],
                "optional_parameters": [{"name": "timezone", "type": "string", "description": "Timezone label"}]
            },
            {
                "name": "json.validate",
                "category": "utility",
                "description": "Validate JSON text and report whether it is valid. Use for checking JSON syntax.",
                "method": "POST",
                "required_parameters": [{"name": "text", "type": "string", "description": "JSON text"}],
                "optional_parameters": []
            },
            {
                "name": "json.format",
                "category": "utility",
                "description": "Pretty-print JSON text with indentation. Use for formatting or beautifying JSON.",
                "method": "POST",
                "required_parameters": [{"name": "text", "type": "string", "description": "JSON text"}],
                "optional_parameters": [{"name": "indent", "type": "integer", "description": "Indent size"}]
            },
            {
                "name": "hash.sha256",
                "category": "utility",
                "description": "Calculate SHA-256 hash for a text string. Use for hashing, checksums and fingerprints.",
                "method": "POST",
                "required_parameters": [{"name": "text", "type": "string", "description": "Input text"}],
                "optional_parameters": []
            },
            {
                "name": "random.uuid",
                "category": "utility",
                "description": "Generate a random UUID4 identifier. Use when a unique id is needed.",
                "method": "GET",
                "required_parameters": [],
                "optional_parameters": []
            },
            {
                "name": "random.password",
                "category": "utility",
                "description": "Generate a random password. Use when a random secure-looking password string is needed.",
                "method": "GET",
                "required_parameters": [],
                "optional_parameters": [{"name": "length", "type": "integer", "description": "Password length"}]
            }
        ]

        self.tools.extend(fallback_tools)
        print(f"\nTotal: {len(self.tools)} tools selected (including {len(fallback_tools)} fallback tools)")

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

    def get_tools_by_category(self, category: str) -> List[Dict]:
        """Return get tools by category."""
        return [t for t in self.tools if t.get('category', '').lower() == category.lower()]
