import os
import sys
import platform
import ctypes
import argparse
import torch

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.config import Config
from src.rl.train_grpo import NetMCPTrainer
from src.llm.llm_client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="NetMCP RL Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "evaluate", "interactive"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    config = Config()
    config.rl.num_epochs = args.epochs

    print(f"Configuration loaded for model: {config.model_name}")

    trainer = NetMCPTrainer(config)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if os.path.exists(checkpoint_path):
            try:
                trainer.load_checkpoint(checkpoint_path)
                print(f"Checkpoint loaded from {checkpoint_path}")
            except Exception as e:
                print(f"Checkpoint loading error: {e}")
        else:
            print(f"Checkpoint {checkpoint_path} not found")

    if args.mode == "train":
        trainer.train()
    elif args.mode == "evaluate":
        trainer.evaluate()
    elif args.mode == "interactive":
        run_interactive(trainer)


def run_interactive(trainer):
    print("\n" + "=" * 60)
    print("NetMCP Interactive Mode")
    print("=" * 60)
    print("Enter your query (or 'quit' to exit):")
    print()

    all_tools = trainer.config.tools
    print(f"Loaded {len(all_tools)} tools\n")

    while True:
        query = input(">>> ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        query_data = {
            'query': query,
            'domain': 'user_query',
            'relevant_tools': []
        }

        state = trainer.env.reset(query_data)
        print(f"\nProcessing query: {query}")
        print("-" * 50)

        valid_tools = [t['name'] for t in state['tools'] if t['available']]
        if not valid_tools:
            print("   No available tools for this query")
            print("-" * 50)
            continue

        response_text = None
        tool_used = None
        latency = 0

        for step in range(trainer.config.rl.max_steps):
            context = trainer._format_context(state)
            response = trainer.llm_client.ask(context)
            tool_call = trainer._parse_tool_call(response)

            if tool_call:
                tool_name = tool_call['tool']

                if tool_name == 'tool_name' or tool_name not in valid_tools:
                    corrected = trainer._correct_tool_call(tool_name, valid_tools, query)
                    if corrected:
                        tool_name = corrected
                    else:
                        continue

                next_state, reward, done, info = trainer.env.step(tool_name)
                latency = info.get('latency', 0)

                if info.get('success'):
                    response_text = info.get('response') or info.get('result')
                    tool_used = tool_name

                    if not response_text:
                        response_text = f"Request processed via tool '{tool_name}'"

                    print(f"\nRESPONSE:")
                    print(f"{response_text}")
                    print(f"\nDetails:")
                    print(f"  Tool: {tool_name}")
                    print(f"  Time: {latency:.3f} sec")
                    break
                else:
                    print(f"  Tool '{tool_name}' could not process the request")

                    if step < trainer.config.rl.max_steps - 1:
                        print(f"  Trying another tool...")
                        state = next_state
                    else:
                        print(f"\nFAILED TO PROCESS REQUEST")
                        print(f"  Please rephrase your query")
            else:
                if response and len(response) > 10 and '<tool_call>' not in response:
                    print(f"\nMODEL RESPONSE:")
                    print(f"{response}")
                else:
                    print(f"  Model could not respond to the query")
                break

        if response_text is None and step == trainer.config.rl.max_steps - 1:
            print(f"\nFAILED TO PROCESS REQUEST")
            print(f"  Please rephrase your query")

        print("-" * 50)


if __name__ == "__main__":
    main()