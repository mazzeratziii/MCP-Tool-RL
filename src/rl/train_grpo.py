# src/rl/train_grpo.py  v6 — multi-GPU profile
"""
Profiles
--------
  desktop   RTX 3070 8 GB  — full quality, fast
  laptop    4 GB VRAM       — memory-safe, slower

Usage:
  python main.py --mode train --epochs 30 --profile desktop
  python main.py --mode train --epochs 30 --profile laptop

Profile is passed via config.profile (set in main.py from --profile arg).
Falls back to 'laptop' if not set.
"""

import os
import gc
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.environment.mcp_environment import MCPEnvironment
from src.environment.network_emulator import NetworkMode
from src.rl.reward_functions import GRPOToolReward
from src.prompts import get_dynamic_prompt


# ── Hardware profiles ──────────────────────────────────────────────────────────

@dataclass
class HWProfile:
    name: str
    lora_r: int
    lora_alpha: int
    lora_targets: List[str]
    grpo_group_size: int
    train_set_size: int
    resample_every: int
    top_k_tools: int
    max_ctx_len: int
    entropy_coeff: float
    semantic_bias: float
    warmup_steps: int
    grad_clip: float
    use_double_quant: bool
    compute_dtype: torch.dtype
    sample_temperature: float = 1.0  # exploration temperature during collection

PROFILES: Dict[str, HWProfile] = {
    "desktop": HWProfile(
        name            = "desktop (RTX 3070 8 GB)",
        lora_r          = 16,
        lora_alpha      = 32,
        lora_targets    = ["q_proj", "v_proj", "k_proj", "o_proj"],
        grpo_group_size = 6,
        train_set_size  = 1500,  # was 800
        resample_every  = 3,     # was 5
        top_k_tools     = 20,
        max_ctx_len     = 512,
        entropy_coeff   = 0.02,   # was 0.005 — stronger regularisation
        semantic_bias   = 5.0,    # was 3.0 — increased for better relevance
        warmup_steps    = 100,
        grad_clip       = 0.5,
        use_double_quant= False,   # not needed on 8 GB
        compute_dtype   = torch.bfloat16,
    ),
    "laptop": HWProfile(
        name            = "laptop (4 GB VRAM)",
        lora_r          = 4,
        lora_alpha      = 8,
        lora_targets    = ["q_proj", "v_proj"],
        grpo_group_size = 3,     # 4→3: 25% fewer forwards
        train_set_size  = 600,   # 1000→600: faster epoch, diversity via resample
        resample_every  = 2,     # resample more often to compensate
        top_k_tools     = 15,    # 20→15: fewer tool embeds (~25% faster)
        max_ctx_len     = 256,   # 320→256: shorter context saves attention time
        entropy_coeff   = 0.02,   # was 0.005 — stronger regularisation
        semantic_bias   = 5.0,    # was 3.0 — increased for better relevance
        warmup_steps    = 80,
        grad_clip       = 0.5,
        use_double_quant= True,
        compute_dtype   = torch.float16,
        sample_temperature = 1.2,
    ),
}


class NetMCPTrainer:
    def __init__(self, config):
        self.config = config
        self.reward_fn = GRPOToolReward(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Select profile ────────────────────────────────────────────
        profile_name = getattr(config, "profile", "laptop")
        if profile_name not in PROFILES:
            print(f"Unknown profile '{profile_name}', falling back to 'laptop'")
            profile_name = "laptop"
        self.p = PROFILES[profile_name]
        print(f"\nDevice: {self.device}")
        print(f"Profile: {self.p.name}")
        if torch.cuda.is_available():
            total_mb = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
            print(f"VRAM: {total_mb} MB")

        # ── Model ─────────────────────────────────────────────────────
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.p.compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=self.p.use_double_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=True
        )
        self.model.gradient_checkpointing_enable()

        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=self.p.lora_r,
                lora_alpha=self.p.lora_alpha,
                target_modules=self.p.lora_targets,
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
        )
        self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.rl.learning_rate,
            weight_decay=config.rl.weight_decay,
            foreach=False,
        )
        self._update_step = 0

        self.env = MCPEnvironment(
            config, llm_client=None, network_mode=NetworkMode.DETERMINISTIC
        )
        self.llm_client = None

        # ── Fixed training pool ───────────────────────────────────────
        pool = [p for p in config.train_prompts if p.get("relevant_tools")]
        if not pool:
            pool = config.train_prompts
        self._train_pool = pool
        self.train_set = random.sample(pool, min(self.p.train_set_size, len(pool)))

        print(f"Train pool: {len(pool)} prompts  "
              f"train_set={len(self.train_set)}  "
              f"top_k={self.p.top_k_tools}  "
              f"group_size={self.p.grpo_group_size}")

        # Pre-cache all tool name embeddings (done once, saves ~80% of per-step tokenisation)
        print("Pre-caching tool embeddings...")
        self._tool_emb_cache: Dict[str, torch.Tensor] = {}
        embed_layer = self.model.get_input_embeddings()
        with torch.inference_mode():
            for tool in config.tools:
                name = tool["name"]
                ids = self.tokenizer(
                    name, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                self._tool_emb_cache[name] = embed_layer(ids).mean(dim=1).squeeze(0).detach().cpu()
        print(f"Cached {len(self._tool_emb_cache)} tool embeddings")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clear(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _current_lr(self) -> float:
        return self.config.rl.learning_rate * min(
            1.0, (self._update_step + 1) / self.p.warmup_steps
        )

    def _parse_tool_call(self, text: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.IGNORECASE | re.DOTALL)
        if m:
            name = m.group(1).strip()
            return name if name and name.lower() != "none" else None
        for token in text.split():
            token = token.strip(".,;\"'")
            if "." in token and len(token) > 3:
                return token
        return None

    # ------------------------------------------------------------------
    # Tool state helpers
    # ------------------------------------------------------------------

    def _get_tools_state(self, prompt: Dict):
        """Returns (state, tools_list, semantic_scores)."""
        state = self.env.reset(prompt)
        candidates = self.env.tools.get_top_k_tools(
            self.env.current_query, k=self.p.top_k_tools
        )

        # Минимальный порог релевантности для фильтрации
        MIN_SEMANTIC_THRESHOLD = 0.70

        tools_state = []
        for tool in candidates:
            sv = self.env.network.get_server_state(tool["name"])
            qos = self.env.network.get_qos_metrics(tool["name"])
            sem = self.env.tools.semantic_similarity(
                self.env.current_query, tool["name"]
            )

            # Пропускаем инструменты с низкой релевантностью
            if sem < MIN_SEMANTIC_THRESHOLD:
                continue

            is_rel = any(rt["name"] == tool["name"] for rt in self.env.relevant_tools)
            tools_state.append({
                "name": tool["name"],
                "category": tool.get("category", "general"),
                "description": tool.get("description", "")[:50] + "...",
                "available": sv["available"],
                "latency": qos["avg_latency"],
                "stability": qos["stability"],
                "semantic_score": sem,
                "is_relevant": is_rel,
                "used": False,
            })
        state["tools"] = tools_state
        tools = [t["name"] for t in tools_state]
        scores = [t["semantic_score"] for t in tools_state]
        return state, tools, scores

    # ------------------------------------------------------------------
    # Encode context and build tool embeddings — ONCE per group
    # ------------------------------------------------------------------

    def _encode_context(self, state: Dict) -> torch.Tensor:
        ctx = get_dynamic_prompt(state["query"], state["tools"])
        ids = self.tokenizer(
            ctx, return_tensors="pt",
            truncation=True, max_length=self.p.max_ctx_len, padding=False,
        ).input_ids.to(self.device)
        return ids

    @torch.inference_mode()
    def _build_tool_embs(self, tools: List[str]) -> torch.Tensor:
        """Return (n, d) matrix using pre-cached embeddings — no tokenisation per call."""
        embs = []
        embed_layer = self.model.get_input_embeddings()
        for t in tools:
            if t in self._tool_emb_cache:
                embs.append(self._tool_emb_cache[t].to(self.device))
            else:
                ids = self.tokenizer(
                    t, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                embs.append(embed_layer(ids).mean(dim=1).squeeze(0).detach())
        return torch.stack(embs, dim=0)   # (n, d)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_with_embs(self, input_ids: torch.Tensor,
                           tool_embs: torch.Tensor,
                           semantic_scores: Optional[List[float]] = None
                           ) -> torch.Tensor:
        out = self.model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :]          # (1, d)
        logits = torch.matmul(hidden, tool_embs.T).squeeze(0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        if semantic_scores is not None:
            bias = torch.tensor(semantic_scores, device=self.device, dtype=torch.float32)
            bias = (bias - bias.mean()) * self.p.semantic_bias
            logits = logits + bias
        return logits

    # ------------------------------------------------------------------
    # Build one GRPO group
    # ------------------------------------------------------------------

    def _make_group(self, prompt: Dict) -> Dict:
        state, tools, scores = self._get_tools_state(prompt)
        input_ids = self._encode_context(state)
        tool_embs = self._build_tool_embs(tools)   # inference_mode

        rollouts = []
        with torch.inference_mode():
            for _ in range(self.p.grpo_group_size):
                logits = self._forward_with_embs(input_ids, tool_embs, scores)
                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1e-8)
                probs = probs / (probs.sum() + 1e-8)
                # Apply temperature — higher = more exploration, fewer skipped groups
                if self.p.sample_temperature != 1.0:
                    logits_t = (logits / self.p.sample_temperature)
                    probs = torch.softmax(logits_t, dim=-1)
                    probs = torch.nan_to_num(probs, nan=1e-8)
                    probs = probs / (probs.sum() + 1e-8)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()

                tool_name = tools[idx.item()]
                self.env.reset(prompt)
                _, _, _, info = self.env.step(tool_name)

                # Вычисляем относительные метрики для reward
                avg_latency = sum(t['latency'] for t in state['tools']) / len(state['tools'])
                availability_ratio = sum(1 for t in state['tools'] if t['available']) / len(state['tools'])

                # Получаем метрики выбранного инструмента
                selected_tool = state['tools'][idx.item()]

                reward = self.reward_fn.compute_outcome_reward(
                    success=info.get("success", False),
                    steps=1,
                    is_relevant=info.get("is_relevant", False),
                    latency=info.get("latency", 0.0),
                    semantic_score=info.get("semantic_score", 0.0),
                    available=selected_tool.get("available", True),
                    stability=selected_tool.get("stability", 1.0),
                    avg_latency=avg_latency,
                    availability_ratio=availability_ratio,
                )
                rollouts.append({
                    "tool_idx": idx.item(),
                    "reward": reward,
                    "success": info.get("success", False),
                    "is_relevant": info.get("is_relevant", False),
                })

        # GRPO advantage
        rews = [r["reward"] for r in rollouts]
        mean_r = sum(rews) / len(rews)
        std_r = (sum((x - mean_r) ** 2 for x in rews) / len(rews)) ** 0.5
        for r in rollouts:
            adv = r["reward"] - mean_r
            r["advantage"] = adv / (std_r + 1e-8) if std_r > 1e-8 else 0.0

        return {
            "input_ids": input_ids,
            "tool_embs": tool_embs,
            "semantic_scores": scores,
            "rollouts": rollouts,
            "adv_std": std_r,
        }

    # ------------------------------------------------------------------
    # Gradient update for one group
    # ------------------------------------------------------------------

    def _train_group(self, group: Dict) -> Optional[float]:
        if group["adv_std"] < 1e-8:
            return None

        input_ids  = group["input_ids"]
        # clone: inference_mode tensors can't participate in backward
        tool_embs  = group["tool_embs"].clone()
        scores     = group["semantic_scores"]
        rollouts   = group["rollouts"]
        n          = len(rollouts)

        self.optimizer.zero_grad()
        total_loss = 0.0

        for r in rollouts:
            logits = self._forward_with_embs(input_ids, tool_embs, scores)
            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=1e-8)
            probs = probs / (probs.sum() + 1e-8)
            dist = torch.distributions.Categorical(probs)

            lp      = dist.log_prob(torch.tensor(r["tool_idx"], device=self.device))
            adv     = torch.tensor(r["advantage"], device=self.device, dtype=torch.float32)
            entropy = dist.entropy()

            loss = (-adv * lp - self.p.entropy_coeff * entropy) / n
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            total_loss += loss.item()

            del logits, probs, dist, lp, adv, entropy, loss

        if total_loss == 0.0:
            self.optimizer.zero_grad()
            return None

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.p.grad_clip)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._current_lr()
        self.optimizer.step()
        self._update_step += 1
        return total_loss

    # ------------------------------------------------------------------
    # train()
    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"GRPO TRAINING [{self.p.name}] — {self.config.rl.num_epochs} epochs")
        print(f"{'=' * 60}")

        best_relevance = 0.0

        for epoch in range(1, self.config.rl.num_epochs + 1):
            # НОВОЕ: Ротация сценариев каждые 5 эпох для разнообразия
            if epoch % 5 == 1:
                self.env.network.rotate_scenario()
                print(f"\n--- Epoch {epoch} [scenario: {self.env.network.current_scenario}] ---")

            # Resample train set for diversity
            if epoch % self.p.resample_every == 1:
                self.train_set = random.sample(
                    self._train_pool,
                    min(self.p.train_set_size, len(self._train_pool))
                )
                print(f"    [resample → {len(self.train_set)} queries]")
            else:
                print(f"\n--- Epoch {epoch}/{self.config.rl.num_epochs} ---")

            queries = self.train_set.copy()
            random.shuffle(queries)

            losses, adv_stds = [], []
            success_n = relevant_n = total_n = skipped = 0
            total_reward = 0.0

            for i, prompt in enumerate(queries):
                group = self._make_group(prompt)
                adv_stds.append(group["adv_std"])

                loss = self._train_group(group)
                if loss is None:
                    skipped += 1
                else:
                    losses.append(loss)

                for r in group["rollouts"]:
                    total_n += 1
                    total_reward += r["reward"]
                    success_n += int(r["success"])
                    relevant_n += int(r["is_relevant"])

                del group
                if i % 100 == 99:
                    self._clear()

            rel = relevant_n / max(total_n, 1)
            print(
                f"  loss={sum(losses)/max(len(losses),1):.4f}  "
                f"reward={total_reward/max(total_n,1):.3f}  "
                f"success={success_n/max(total_n,1):.2%}  "
                f"relevance={rel:.2%}  "
                f"rollouts={total_n}  skipped={skipped}  "
                f"updates={len(losses)}  "
                f"adv_std={sum(adv_stds)/max(len(adv_stds),1):.3f}  "
                f"lr={self._current_lr():.2e}"
            )

            if rel > best_relevance:
                best_relevance = rel
                self._save_checkpoint("best")
                print(f"  ★ New best relevance={best_relevance:.2%}")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

            self._clear()

        print("\nTraining complete.")

    # ------------------------------------------------------------------
    # evaluate()
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 200,
                 network_mode: NetworkMode = NetworkMode.CONTROLLED):
        print(f"\n{'=' * 60}")
        print(f"EVALUATION [{self.p.name}] — {num_episodes} ep, mode={network_mode.value}")
        print(f"{'=' * 60}")

        self.env.set_network_mode(network_mode)
        pool = [p for p in self.config.val_prompts if p.get("relevant_tools")]
        selected = random.sample(pool or self.config.val_prompts,
                                 min(num_episodes, len(pool or self.config.val_prompts)))

        success_t = relevant_t = total_t = 0
        top3_relevant_t = 0   # relevant tool appears in model's top-3

        # НОВОЕ: Метрики для оценки адаптации к сети
        total_latency = 0.0
        fast_tool_choices = 0  # выбран инструмент быстрее среднего
        available_tool_choices = 0  # выбран доступный инструмент

        self.model.eval()
        with torch.inference_mode():
            for prompt in selected:
                state, tools, scores = self._get_tools_state(prompt)
                ids    = self._encode_context(state)
                embs   = self._build_tool_embs(tools)
                logits = self._forward_with_embs(ids, embs, scores)

                # Top-1 greedy
                top1_idx  = logits.argmax().item()
                tool      = tools[top1_idx]
                self.env.reset(prompt)
                _, _, _, info = self.env.step(tool)
                total_t    += 1
                success_t  += int(info.get("success", False))
                relevant_t += int(info.get("is_relevant", False))

                # НОВОЕ: Сбор метрик адаптации к сети
                latency = info.get("latency", 0.0)
                total_latency += latency

                avg_latency = sum(t['latency'] for t in state['tools']) / len(state['tools'])
                if latency < avg_latency:
                    fast_tool_choices += 1

                selected_tool = state['tools'][top1_idx]
                if selected_tool.get('available', True):
                    available_tool_choices += 1

                # Top-3 accuracy: check if any of top-3 is relevant
                top3_indices = logits.topk(min(3, len(tools))).indices.tolist()
                relevant_names = {t["name"] for t in state["tools"]
                                  if t.get("is_relevant", False)}
                if any(tools[i] in relevant_names for i in top3_indices):
                    top3_relevant_t += 1

        self.model.train()
        self.env.set_network_mode(NetworkMode.DETERMINISTIC)
        n = max(total_t, 1)
        print(f"  Episodes:           {total_t}")
        print(f"  Success rate:       {success_t/n:.2%}")
        print(f"  Relevance@1:        {relevant_t/n:.2%}   (greedy top-1)")
        print(f"  Relevance@3:        {top3_relevant_t/n:.2%}  (relevant in top-3)")
        print(f"  Top-3 gap:          {(top3_relevant_t - relevant_t)/n:+.2%}")
        print(f"\n  Network Adaptation Metrics:")
        print(f"  Avg latency:        {total_latency/n:.3f}s")
        print(f"  Fast tool choices:  {fast_tool_choices/n:.2%}  (below avg latency)")
        print(f"  Available choices:  {available_tool_choices/n:.2%}  (chose available tools)")

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, label):
        d = f"checkpoints/{label}"
        os.makedirs(d, exist_ok=True)
        self.model.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        # Save profile so checkpoint can be resumed correctly
        import json
        meta = {
            "profile": [k for k, v in PROFILES.items() if v is self.p][0],
            "lora_r": self.p.lora_r,
            "lora_targets": self.p.lora_targets,
            "update_step": self._update_step,
        }
        with open(f"{d}/train_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Checkpoint: {d}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load LoRA adapter weights into the already-initialised model.
        Uses set_adapter / load_adapter instead of PeftModel.from_pretrained
        to avoid the 'multiple adapters' warning and missing-key errors when
        the checkpoint profile differs from the current one.
        """
        import os
        from safetensors.torch import load_file as st_load
        import torch

        print(f"Loading checkpoint from {checkpoint_path}...")

        # Try loading via PEFT's own adapter loader (no double-wrapping)
        try:
            self.model.load_adapter(checkpoint_path, adapter_name="default")
            print("Checkpoint loaded via load_adapter.")
            return
        except Exception as e:
            print(f"  load_adapter failed ({e}), trying manual weight load...")

        # Fallback: load safetensors / pytorch_model.bin directly
        st_path  = os.path.join(checkpoint_path, "adapter_model.safetensors")
        bin_path = os.path.join(checkpoint_path, "adapter_model.bin")

        if os.path.exists(st_path):
            state = st_load(st_path, device="cpu")
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu")
        else:
            print(f"  No adapter weights found in {checkpoint_path}, skipping.")
            return

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        loaded = len(state) - len(missing)
        print(f"  Loaded {loaded}/{len(state)} keys "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        print("Checkpoint loaded.")

    # Compatibility shim for interactive mode
    def _forward(self, context: str, tools: List[str], **kwargs) -> torch.Tensor:
        ids  = self.tokenizer(
            context, return_tensors="pt",
            truncation=True, max_length=self.p.max_ctx_len, padding=False,
        ).input_ids.to(self.device)
        embs = self._build_tool_embs(tools)
        with torch.inference_mode():
            return self._forward_with_embs(ids, embs, kwargs.get("semantic_scores"))