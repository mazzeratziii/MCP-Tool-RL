import time
import random
import json
from typing import Dict, Any, Tuple, Optional
from src.config import Config
from .network_emulator import NetworkEmulator
from .tool_registry import ToolRegistry


class MCPEnvironment:
    def __init__(self, config: Config, llm_client=None):
        self.config = config
        self.network = NetworkEmulator(config)

        # Ensure data is loaded before creating ToolRegistry
        if not config.tools:
            print("Loading data before creating environment...")
            config.load_data()

        self.tools = ToolRegistry(config)
        self.llm_client = llm_client

        self.current_query = None
        self.current_query_data = None
        self.relevant_tools = []
        self.step_count = 0
        self.used_tools = []

    def reset(self, query_data: Optional[Dict] = None):
        self.step_count = 0
        self.used_tools = []

        if query_data:
            self.current_query_data = query_data
            self.current_query = query_data['query']
            self.relevant_tools = query_data.get('relevant_tools', [])
        else:
            self.current_query_data = self._get_random_query()
            self.current_query = self.current_query_data['query']
            self.relevant_tools = self.current_query_data.get('relevant_tools', [])

        self.network.update_network_state()
        return self._get_current_state()

    def _get_random_query(self) -> Dict:
        random_tool = random.choice(self.config.tools)
        return {
            'query': f"How to use {random_tool['name']}?",
            'domain': random_tool.get('category', 'general'),
            'relevant_tools': [{'name': random_tool['name']}]
        }

    def _get_current_state(self) -> Dict[str, Any]:
        candidate_tools = self.tools.get_top_k_tools(self.current_query, k=10)

        tools_state = []
        for tool in candidate_tools:
            server_state = self.network.get_server_state(tool['name'])
            qos = self.network.get_qos_metrics(tool['name'])

            is_relevant = any(rt['name'] == tool['name'] for rt in self.relevant_tools)

            tool_state = {
                'name': tool['name'],
                'category': tool.get('category', 'general'),
                'description': tool.get('description', '')[:50] + "...",
                'available': server_state['available'],
                'latency': qos['avg_latency'],
                'stability': qos['stability'],
                'semantic_score': self.tools.semantic_similarity(self.current_query, tool['name']),
                'is_relevant': is_relevant,
                'used': tool['name'] in self.used_tools
            }
            tools_state.append(tool_state)

        return {
            'query': self.current_query,
            'query_domain': self.current_query_data.get('domain', 'unknown'),
            'step': self.step_count,
            'tools': tools_state,
            'total_tools': len(self.config.tools)
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self.step_count += 1
        self.used_tools.append(action)

        tool = self.tools.get_tool_by_name(action)
        if not tool:
            return (
                self._get_current_state(),
                self.config.reward.invalid_call_penalty,
                True,
                {'error': f'Invalid tool: {action}'}
            )

        server_state = self.network.get_server_state(tool['name'])
        if not server_state['available']:
            return (
                self._get_current_state(),
                -0.2,
                False,
                {'error': 'server unavailable', 'tool': action}
            )

        latency = self.network.get_current_latency(tool['name'], {'base_latency': tool.get('base_latency', 0.1)})
        time.sleep(latency * 0.01)

        success = random.random() > tool.get('failure_rate', 0.1)
        is_relevant = any(rt['name'] == action for rt in self.relevant_tools)

        reward = self._calculate_reward(tool, latency, success, is_relevant)

        response = self._generate_response(tool, success, is_relevant)

        done = (
                self.step_count >= self.config.rl.max_steps
                or (success and is_relevant)
                or len(self.used_tools) >= 3
        )

        info = {
            'latency': latency,
            'success': success,
            'is_relevant': is_relevant,
            'tool_used': action,
            'tool_category': tool.get('category', 'general'),
            'step': self.step_count,
            'response': response,
            'result': response
        }

        return self._get_current_state(), reward, done, info

    def _generate_response(self, tool: Dict, success: bool, is_relevant: bool) -> str:
        if not success:
            return f"Tool '{tool['name']}' could not process the request."

        if not is_relevant:
            return f"Tool '{tool['name']}' is not suitable for this request."

        tool_info = {
            "name": tool.get('name', 'unknown'),
            "category": tool.get('category', 'general'),
            "description": tool.get('description', 'No description available'),
            "parameters": tool.get('required_parameters', []),
            "examples": tool.get('examples', [])
        }

        if self.llm_client:
            prompt = f"""You are an assistant answering user questions based on tool data.

User query: {self.current_query}

Tool information from dataset:
{json.dumps(tool_info, ensure_ascii=False, indent=2)}

Rules:
1. Do not invent data not present in the dataset
2. Do not use external APIs
3. If the dataset lacks specific data, state that honestly
4. Use examples from the dataset to show how the tool works
5. Provide a helpful response in the language of the query

Formulate a response for the user."""
            try:
                response = self.llm_client.ask(prompt)
                return response
            except Exception:
                return self._fallback_response(tool_info)

        return self._fallback_response(tool_info)

    def _fallback_response(self, tool_info: Dict) -> str:
        response_parts = [
            f"Tool: {tool_info['name']}",
            f"Category: {tool_info['category']}",
            f"Description: {tool_info['description']}"
        ]

        if tool_info.get('examples'):
            response_parts.append(f"Examples: {tool_info['examples'][:2]}")

        if tool_info.get('parameters'):
            params = ", ".join(tool_info['parameters'][:5])
            response_parts.append(f"Parameters: {params}")

        return ". ".join(response_parts)

    def _calculate_reward(self, tool: Dict, latency: float, success: bool, is_relevant: bool) -> float:
        reward = 0.0

        if success and is_relevant:
            reward += self.config.reward.success_reward
            if self.step_count == 1:
                reward += 0.3
        elif success and not is_relevant:
            reward += 0.2
            reward += self.config.reward.wrong_tool_penalty
        elif not success and is_relevant:
            reward += self.config.reward.failure_penalty * 0.5
        else:
            reward += self.config.reward.failure_penalty

        if latency > self.config.reward.latency_threshold:
            reward -= 0.2

        reward += self.config.reward.step_penalty * self.step_count

        semantic_score = self.tools.semantic_similarity(self.current_query, tool['name'])
        if semantic_score > 0.7:
            reward += self.config.reward.semantic_bonus

        if self.used_tools.count(tool['name']) > 1:
            reward += self.config.reward.extra_step_penalty

        return reward