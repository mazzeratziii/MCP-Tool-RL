from datasets import load_dataset
import json
import random
import ast
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class ToolBenchLoader:
    def __init__(self, split="train", sample_size=None):
        print(f"\n{'=' * 60}")
        print(f"LOADING TOOLBENCH {split.upper()} SPLIT")
        print(f"{'=' * 60}")

        print("Loading dataset from HuggingFace...")
        self.dataset = load_dataset("Maurus/ToolBench", split=split)

        print(f"Dataset loaded successfully")
        print(f"  Size: {len(self.dataset)} examples")
        print(f"  Columns: {self.dataset.column_names}")

        if sample_size and sample_size < len(self.dataset):
            self.dataset = self.dataset.select(range(sample_size))
            print(f"  Using {sample_size} examples")

        print("\nProcessing data...")
        self.data = self._process_dataset()

        print("\nExtracting tools...")
        self.tools = self._extract_tools()

        print(f"\n{'=' * 60}")
        print(f"LOADING RESULTS:")
        print(f"  Examples: {len(self.data)}")
        print(f"  Tools: {len(self.tools)}")

        if self.tools:
            print(f"\nFirst 5 tools:")
            for i, tool in enumerate(self.tools[:5]):
                print(f"  {i + 1}. {tool['name']}")
                print(f"      Category: {tool.get('category', 'Unknown')}")
                print(f"      Description: {tool.get('description', '')[:100]}...")
        else:
            print("\nNo tools found in dataset")
            self._debug_dataset_structure()

    def _safe_parse_json(self, data: Any) -> Any:
        if isinstance(data, (dict, list)):
            return data

        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                try:
                    return ast.literal_eval(data)
                except:
                    try:
                        cleaned = data.replace('\\"', '"').replace("\\'", "'")
                        return json.loads(cleaned)
                    except:
                        pass
        return data

    def _process_dataset(self) -> List[Dict]:
        processed = []

        for idx, item in enumerate(tqdm(self.dataset, desc="Processing examples")):
            try:
                query = item.get('query', '')
                if not query:
                    continue

                api_list = item.get('api_list', [])
                api_list = self._safe_parse_json(api_list)

                if not isinstance(api_list, list):
                    api_list = []

                domain = item.get('domain', '')
                if isinstance(domain, bytes):
                    domain = domain.decode('utf-8')

                answer = item.get('answer', {})
                answer = self._safe_parse_json(answer)
                if not isinstance(answer, dict):
                    answer = {}

                embedding = item.get('embedding', [])
                embedding = self._safe_parse_json(embedding)
                if not isinstance(embedding, list):
                    embedding = []

                processed.append({
                    'query': query,
                    'api_list': api_list,
                    'domain': domain,
                    'answer': answer,
                    'embedding': embedding,
                    'query_id': str(item.get('query_id', idx))
                })

            except Exception as e:
                continue

        return processed

    def _extract_tools(self) -> List[Dict]:
        tools_dict = {}

        print("\nExtracting tools from dataset...")

        for idx, item in enumerate(tqdm(self.data, desc="Extracting tools")):
            api_list = item['api_list']

            if not api_list or not isinstance(api_list, list):
                continue

            for api in api_list:
                if not isinstance(api, dict):
                    continue

                tool_name = api.get('tool_name', '')
                api_name = api.get('api_name', '')

                if not tool_name or not api_name:
                    continue

                tool_key = f"{tool_name}.{api_name}"

                if tool_key not in tools_dict:
                    description = api.get('api_description', '')
                    if not description:
                        description = api.get('description', '')

                    category = api.get('category_name', 'Unknown')

                    required = api.get('required_parameters', [])
                    required = self._safe_parse_json(required)
                    if not isinstance(required, list):
                        required = []

                    optional = api.get('optional_parameters', [])
                    optional = self._safe_parse_json(optional)
                    if not isinstance(optional, list):
                        optional = []

                    examples = []
                    if description:
                        examples.append(description[:100])

                    tools_dict[tool_key] = {
                        'name': tool_key,
                        'tool_name': tool_name,
                        'api_name': api_name,
                        'description': description,
                        'category': category,
                        'required_parameters': required,
                        'optional_parameters': optional,
                        'method': api.get('method', 'GET'),
                        'examples': examples,
                        'base_latency': random.uniform(0.1, 0.5),
                        'failure_rate': random.uniform(0.01, 0.1)
                    }

        tools = list(tools_dict.values())
        print(f"\nFound {len(tools)} unique tools")
        return tools

    def _debug_dataset_structure(self):
        print("\nDEBUGGING DATASET STRUCTURE:")

        if len(self.dataset) > 0:
            first = self.dataset[0]
            print(f"\nFirst example keys: {list(first.keys())}")

            api_list = first.get('api_list', [])
            print(f"api_list type: {type(api_list)}")

            if isinstance(api_list, str):
                print(f"api_list (first 200 chars): {api_list[:200]}")

    def get_training_prompts(self) -> List[Dict]:
        prompts = []

        for item in self.data:
            relevant_tools = []
            target_tool = None

            answer = item.get('answer', {})
            if isinstance(answer, dict):
                tool_name = answer.get('tool_name')
                api_name = answer.get('api_name')
                if tool_name and api_name:
                    target_tool = f"{tool_name}.{api_name}"

            for api in item['api_list']:
                if isinstance(api, dict):
                    t_name = api.get('tool_name')
                    a_name = api.get('api_name')
                    if t_name and a_name:
                        tool_key = f"{t_name}.{a_name}"
                        for tool in self.tools:
                            if tool['name'] == tool_key:
                                relevant_tools.append(tool)
                                break

            prompts.append({
                'query': item['query'],
                'query_id': item['query_id'],
                'domain': item['domain'],
                'relevant_tools': relevant_tools,
                'target_tool': target_tool
            })

        return prompts

    def sample_tools(self, n: int = 10) -> List[Dict]:
        if not self.tools:
            print("No tools available for sampling")
            return []

        n = min(n, len(self.tools))
        sampled = random.sample(self.tools, n)
        print(f"Sampled {n} tools")
        return sampled