# Главный файл запуска
import os
import sys
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def main():
    """Run the module entry point."""
    parser = argparse.ArgumentParser(description="NetMCP-RL")
    parser.add_argument("--mode", default="train",
                        choices=["train", "evaluate", "interactive", "select", "benchmark", "analyze-log", "healthcheck"])
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--network-mode", default="deterministic",
                        choices=["deterministic", "controlled", "stochastic"])
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--profile",      default=None,
                        choices=["desktop", "laptop"],
                        help="Профиль оборудования. Если не указан, определяется автоматически.")
    parser.add_argument("--eval-episodes", type=int, default=200,
                        help="Number of episodes for evaluate mode")
    parser.add_argument("--use-hybrid", action="store_true",
                        help="Use hybrid mode: real MCP calls + emulation")
    parser.add_argument("--mcp-config", default="mcp_config.json",
                        help="Path to MCP configuration file")
    parser.add_argument("--query", default=None,
                        help="User query for lightweight select mode")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of candidate tools to rank in select mode")
    parser.add_argument("--semantic-threshold", type=float, default=0.55,
                        help="Minimum semantic score for select mode")
    parser.add_argument("--benchmark-policy", default="adaptive",
                        choices=["adaptive", "semantic", "sonar", "both", "all"],
                        help="Policy to evaluate in benchmark mode")
    parser.add_argument("--log-path", default="runs/selection_benchmark.jsonl",
                        help="JSONL output path for benchmark mode or input path for analyze-log")
    parser.add_argument("--example-limit", type=int, default=3,
                        help="Number of examples per error reason in analyze-log mode")
    parser.add_argument("--rerank-weight", type=float, default=0.35,
                        help="Weight of text reranker features in adaptive selection")
    parser.add_argument("--provider-group-weight", type=float, default=0.0,
                        help="Experimental provider-name group adjustment weight")
    parser.add_argument("--training-log-path", default="runs/training_metrics.csv",
                        help="CSV file for training metrics")
    parser.add_argument("--training-plot-path", default="runs/training_curve.png",
                        help="PNG file for training progress chart")
    parser.add_argument("--no-training-plot", action="store_true",
                        help="Disable training metric CSV/PNG generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed data/model loading logs")
    args = parser.parse_args()

    if args.mode == "analyze-log":
        run_analyze_log(args.log_path, example_limit=args.example_limit)
        return

    if args.mode == "healthcheck":
        run_project_healthcheck()
        return

    from src.config import Config
    from src.environment.network_emulator import NetworkMode

    with quiet_output(enabled=not args.verbose):
        config = Config()
        config.load_data()
    config.rl.batch_size  = max(1, args.batch_size)
    config.rl.num_epochs  = args.epochs
    config.training_log_path = args.training_log_path
    config.training_plot_path = args.training_plot_path
    config.training_plot_enabled = not args.no_training_plot

    network_mode_map = {
        "deterministic": NetworkMode.DETERMINISTIC,
        "controlled":    NetworkMode.CONTROLLED,
        "stochastic":    NetworkMode.STOCHASTIC,
    }
    network_mode = network_mode_map[args.network_mode]

    if args.mode == "select":
        run_select(
            config,
            query=args.query,
            network_mode=network_mode,
            top_k=args.top_k,
            semantic_threshold=args.semantic_threshold,
            rerank_weight=args.rerank_weight,
            provider_group_weight=args.provider_group_weight,
            verbose=args.verbose,
        )
        return

    if args.mode == "benchmark":
        run_benchmark(
            config,
            network_mode=network_mode,
            episodes=args.eval_episodes,
            policy=args.benchmark_policy,
            log_path=args.log_path,
            top_k=args.top_k,
            semantic_threshold=args.semantic_threshold,
            rerank_weight=args.rerank_weight,
            provider_group_weight=args.provider_group_weight,
            verbose=args.verbose,
        )
        return

    if not str(config.model_name or "").strip():
        if args.mode == "interactive":
            print("MODEL_NAME is not set. Starting lightweight adaptive interactive mode.")
            print("Set MODEL_NAME in .env to use the trained LLM/LoRA policy.")
            run_adaptive_interactive(
                config,
                network_mode=network_mode,
                top_k=args.top_k,
                semantic_threshold=args.semantic_threshold,
                rerank_weight=args.rerank_weight,
                provider_group_weight=args.provider_group_weight,
                verbose=args.verbose,
            )
            return
        print("MODEL_NAME is not set, so the LLM trainer cannot be started.")
        print("Add MODEL_NAME to .env, for example: MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct")
        print("For a no-LLM prototype use: python main.py --mode select --query \"...\"")
        return

    import torch
    from src.rl.train_grpo import NetMCPTrainer

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Автоматическое определение профиля
    if args.profile:
        config.profile = args.profile
    elif torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
        config.profile = "desktop" if vram_mb >= 7000 else "laptop"
        if args.verbose:
            print(f"Auto-detected profile: {config.profile} ({vram_mb} MB VRAM)")
    else:
        config.profile = "laptop"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── Выбор режима: гибридный или эмуляция ──────────────────────────
    if args.use_hybrid:
        from src.rl.train_grpo_hybrid import HybridMCPTrainer
        with quiet_output(enabled=not args.verbose):
            trainer = HybridMCPTrainer(config, mcp_config_path=args.mcp_config)
    else:
        with quiet_output(enabled=not args.verbose):
            trainer = NetMCPTrainer(config)

    print(f"Mode: {args.mode} | profile={config.profile} | tools={len(config.tools)} | network={network_mode.value}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            with quiet_output(enabled=not args.verbose):
                trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Checkpoint load error: {e}")

    if args.mode == "train":
        trainer.train()
    elif args.mode == "evaluate":
        trainer.evaluate(num_episodes=args.eval_episodes, network_mode=network_mode)
    elif args.mode == "interactive":
        run_interactive(trainer)


# Интерактивный режим

@contextmanager
def quiet_output(enabled: bool = True):
    """Handle quiet output."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def run_select(config, query: str, network_mode,
               top_k: int = 10, semantic_threshold: float = 0.55,
               rerank_weight: float = 0.35,
               provider_group_weight: float = 0.0,
               verbose: bool = False):
    """Run run select."""
    from src.environment.mcp_environment import MCPEnvironment
    from src.selection.adaptive_selector import AdaptiveToolSelector, candidates_from_environment_state
    from src.selection.query_normalization import expand_query_for_retrieval, is_too_ambiguous_query

    if not query:
        query = input("Query: ").strip()
    if not query:
        print("Empty query, nothing to select.")
        return

    retrieval_query = expand_query_for_retrieval(query)
    if is_too_ambiguous_query(query):
        print("Warning: the query is very short or ambiguous; selection may be noisy.")

    with quiet_output(enabled=not verbose):
        env = MCPEnvironment(config, network_mode=network_mode)
        state = env.reset({"query": retrieval_query, "domain": "user_query", "relevant_tools": []})
    state["query"] = query
    state["tools"] = state.get("tools", [])[:top_k]

    selector = AdaptiveToolSelector(
        semantic_threshold=semantic_threshold,
        rerank_weight=rerank_weight,
        provider_group_weight=provider_group_weight,
    )
    candidates = candidates_from_environment_state(state)
    result = selector.select(query, candidates)

    print("\n" + "=" * 60)
    print("Adaptive Tool Selection")
    print("=" * 60)
    print(f"Query:        {query}")
    print(f"Network mode: {network_mode.value}")
    print(f"Candidates:   {len(candidates)}")

    if result is None:
        print("No candidate passed the semantic threshold.")
        return

    print(f"\nSelected: {result.tool_name}")
    print(f"Score:    {result.score:.3f}")
    print(f"Reason:   {result.reason}")
    print("\nTop ranked candidates:")
    for row in result.candidates[:min(5, len(result.candidates))]:
        rank = int(row["rank"])
        tool = row["name"]
        print(
            f"  {rank}. {tool}  "
            f"score={row['score']:.3f} semantic={row['semantic']:.3f} "
            f"functional={row['functional']:.3f} "
            f"intent={row['query_intent']}->{row['tool_intent']} "
            f"qos={row['qos']:.3f} latency_penalty={row['latency_penalty']:.3f}"
        )


def run_benchmark(config, network_mode, episodes: int,
                  policy: str, log_path: str, top_k: int,
                  semantic_threshold: float, rerank_weight: float = 0.35,
                  provider_group_weight: float = 0.0,
                  verbose: bool = False):
    """Run run benchmark."""
    from src.selection.benchmark import SelectionBenchmark, print_summary

    benchmark = SelectionBenchmark(
        config,
        network_mode=network_mode,
        top_k=top_k,
        semantic_threshold=semantic_threshold,
        rerank_weight=rerank_weight,
        provider_group_weight=provider_group_weight,
    )

    if policy == "both":
        policies = ["semantic", "adaptive"]
    elif policy == "all":
        policies = ["semantic", "sonar", "adaptive"]
    else:
        policies = [policy]
    for item in policies:
        item_log_path = log_path
        if policy in {"both", "all"}:
            root, ext = os.path.splitext(log_path)
            item_log_path = f"{root}_{item}{ext or '.jsonl'}"

        with quiet_output(enabled=not verbose):
            summary = benchmark.run(
                policy=item,
                episodes=episodes,
                log_path=item_log_path,
            )
        print_summary(summary)


def run_analyze_log(log_path: str, example_limit: int = 3):
    """Run run analyze log."""
    from src.selection.log_analysis import SelectionLogAnalyzer, print_analysis

    analyzer = SelectionLogAnalyzer(example_limit=example_limit)
    summary = analyzer.analyze_file(log_path)
    print_analysis(summary)


def run_project_healthcheck():
    """Run run project healthcheck."""
    from src.selection.healthcheck import print_healthcheck, run_healthcheck

    print_healthcheck(run_healthcheck())


def run_adaptive_interactive(config, network_mode,
                             top_k: int = 10, semantic_threshold: float = 0.55,
                             rerank_weight: float = 0.35,
                             provider_group_weight: float = 0.0,
                             verbose: bool = False):
    """Run run adaptive interactive."""
    from src.environment.mcp_environment import MCPEnvironment
    from src.environment.network_emulator import NetworkMode
    from src.selection.adaptive_selector import AdaptiveToolSelector, candidates_from_environment_state
    from src.selection.query_normalization import expand_query_for_retrieval, is_too_ambiguous_query

    with quiet_output(enabled=not verbose):
        env = MCPEnvironment(config, network_mode=network_mode)
    selector = AdaptiveToolSelector(
        semantic_threshold=semantic_threshold,
        rerank_weight=rerank_weight,
        provider_group_weight=provider_group_weight,
    )
    nm_map = {
        "deterministic": NetworkMode.DETERMINISTIC,
        "controlled": NetworkMode.CONTROLLED,
        "stochastic": NetworkMode.STOCHASTIC,
    }

    print("\n" + "=" * 60)
    print("Adaptive Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /network deterministic | controlled | stochastic")
    print("  /network stats")
    print("  /help   /exit")
    print("=" * 60)
    print(f"Tools:   {len(config.tools)}")
    print(f"Network: {network_mode.value}\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "/exit"):
            break

        if query.startswith("/"):
            parts = query[1:].split()
            cmd = parts[0].lower() if parts else ""
            if cmd == "help":
                print("  /network [deterministic|controlled|stochastic|stats]")
                print("  /exit")
            elif cmd == "network" and len(parts) > 1:
                sub = parts[1].lower()
                if sub == "stats":
                    for key, value in env.get_network_stats().items():
                        print(f"  {key}: {value}")
                elif sub in nm_map:
                    env.set_network_mode(nm_map[sub])
                    print(f"Network -> {sub}")
                else:
                    print(f"Unknown network mode: {sub}")
            else:
                print(f"Unknown command: {cmd}")
            continue

        retrieval_query = expand_query_for_retrieval(query)
        if is_too_ambiguous_query(query):
            print("Warning: the query is very short or ambiguous; selection may be noisy.")

        state = env.reset({"query": retrieval_query, "domain": "user_query", "relevant_tools": []})
        state["query"] = query
        state["tools"] = state.get("tools", [])[:top_k]
        candidates = candidates_from_environment_state(state)
        result = selector.select(query, candidates)
        if result is None:
            print("No candidate passed the semantic threshold.")
            continue

        print(f"\nSelected: {result.tool_name}")
        print(f"Score:    {result.score:.3f}")
        print(f"Reason:   {result.reason}")
        print("Top candidates:")
        for row in result.candidates[:min(5, len(result.candidates))]:
            print(
                f"  {int(row['rank'])}. {row['name']}  "
                f"score={row['score']:.3f} semantic={row['semantic']:.3f} "
                f"functional={row['functional']:.3f} qos={row['qos']:.3f}"
            )

        _, _, _, info = env.step(result.tool_name)
        print(f"Success:  {info.get('success', False)}")
        print(f"Latency:  {info.get('latency', 0.0):.3f}s")
        print("-" * 50 + "\n")


def run_interactive(trainer: "NetMCPTrainer"):
    """Run run interactive."""
    from src.environment.network_emulator import NetworkMode
    from src.selection.query_normalization import expand_query_for_retrieval, is_too_ambiguous_query

    print("\n" + "=" * 60)
    print("NetMCP Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /network deterministic | controlled | stochastic")
    print("  /network stats")
    print("  /eval N   — quick eval on N val prompts")
    print("  /help   /exit")
    print("=" * 60)
    print(f"Profile: {trainer.p.name}")
    print(f"Tools:   {len(trainer.config.tools)}\n")

    nm_map = {
        "deterministic": NetworkMode.DETERMINISTIC,
        "controlled":    NetworkMode.CONTROLLED,
        "stochastic":    NetworkMode.STOCHASTIC,
    }

    trainer.model.eval()
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "/exit"):
            break

        # Команды
        if query.startswith("/"):
            parts = query[1:].split()
            cmd   = parts[0].lower() if parts else ""

            if cmd == "help":
                print("  /network [deterministic|controlled|stochastic|stats]")
                print("  /eval N   — evaluate on N val prompts")
                print("  /exit")

            elif cmd == "network" and len(parts) > 1:
                sub = parts[1].lower()
                if sub == "stats":
                    s = trainer.env.get_network_stats()
                    for k, v in s.items():
                        print(f"  {k}: {v}")
                elif sub in nm_map:
                    trainer.env.set_network_mode(nm_map[sub])
                    print(f"Network → {sub}")
                else:
                    print(f"Unknown: {sub}")

            elif cmd == "eval":
                n = int(parts[1]) if len(parts) > 1 else 50
                trainer.evaluate(num_episodes=n,
                                 network_mode=trainer.env.network.mode)

            else:
                print(f"Unknown command: {cmd}")
            continue

        # Обычный запрос
        rewritten_query = trainer.rewrite_query_for_retrieval(query)
        retrieval_query = _merge_retrieval_queries(
            query,
            expand_query_for_retrieval(query),
            rewritten_query,
        )
        if is_too_ambiguous_query(query):
            print("Warning: the query is very short or ambiguous; selection may be noisy.")
        if retrieval_query != query:
            print(f"Retrieval query: {retrieval_query}")

        query_data = {"query": retrieval_query, "domain": "user_query", "relevant_tools": []}
        state, tools, scores = trainer._get_tools_state(query_data)
        state["query"] = query
        print(f"\nTop tools for: '{query}'")
        print("-" * 50)

        import torch
        with torch.inference_mode():
            ids    = trainer._encode_context(state)
            embs   = trainer._build_tool_embs(tools)
            logits = trainer._forward_with_embs(ids, embs, scores)
            probs  = torch.softmax(logits, dim=-1)

        # Показываем top-5
        top5 = probs.topk(min(5, len(tools)))
        top_rows = []
        for rank, (prob, idx) in enumerate(
            zip(top5.values.tolist(), top5.indices.tolist()), 1
        ):
            tool = tools[idx]
            sem  = scores[idx]
            top_rows.append((rank, tool, prob, sem))
            print(f"  {rank}. {tool}")
            print(f"     model_prob={prob:.3f}  semantic={sem:.3f}")

        # Выполняем top-1
        best_idx = logits.argmax().item()
        best_tool = tools[best_idx]
        best_semantic = scores[best_idx]
        if _should_abstain_interactive(query, retrieval_query, top_rows):
            print("\nNo reliable tool selected.")
            print("Please clarify the task, for example: what data source or action do you need?")
            print("-" * 50 + "\n")
            continue

        execution_query_data = {
            "query": retrieval_query,
            "domain": "user_query",
            "relevant_tools": [{"name": best_tool}],
        }
        trainer.env.reset(execution_query_data)
        _, _, _, info = trainer.env.step(best_tool)
        print(f"\nExecuted: {best_tool}")
        print(f"Success:  {info.get('success', False)}")
        print(f"Latency:  {info.get('latency', 0):.3f}s")
        resp = info.get("response") or info.get("result", "")
        if resp:
            print(f"Response: {resp[:200]}")
        print("-" * 50 + "\n")


def _should_abstain_interactive(query: str, retrieval_query: str, top_rows) -> bool:
    """Handle should abstain interactive."""
    if not top_rows:
        return True

    _, best_tool, best_prob, best_semantic = top_rows[0]
    query_lower = query.lower()
    retrieval_lower = retrieval_query.lower()
    tool_lower = best_tool.lower()

    if best_semantic < 0.64:
        return True

    suspicious_tool_terms = (
        "login",
        "logout",
        "random word",
        "word of the day",
        "currency",
        "convert",
    )
    if any(term in tool_lower for term in suspicious_tool_terms):
        intent_terms = ("login", "logout", "currency", "convert", "translate")
        if not any(term in retrieval_lower for term in intent_terms):
            return True

    if len(query_lower.split()) <= 2 and best_semantic < 0.72 and best_prob > 0.85:
        return True

    if query != retrieval_query and best_semantic < 0.70:
        return True

    return False


def _merge_retrieval_queries(*queries: Optional[str]) -> str:
    """Merge merge retrieval queries."""
    seen = set()
    merged = []
    for query in queries:
        if not query:
            continue
        normalized = " ".join(str(query).split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            merged.append(normalized)
    return " ".join(merged)


if __name__ == "__main__":
    main()
