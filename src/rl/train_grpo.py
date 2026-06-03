# GRPO-обучение, версия 6, профиль для нескольких GPU
"""
Профили
--------
  desktop   RTX 3070 8 GB  — полное качество, быстро
  laptop    4 GB VRAM       — безопасно для памяти, медленнее

Использование:
  python main.py --mode train --epochs 30 --profile desktop
  python main.py --mode train --epochs 30 --profile laptop

Профиль передаётся через config.profile и задаётся в main.py аргументом --profile.
Если профиль не задан, используется 'laptop'.
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
from src.rl.training_plot import TrainingMetric, TrainingProgressLogger
from src.prompts import get_dynamic_prompt


# Профили оборудования

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
    sample_temperature: float = 1.0  # температура exploration во время сбора rollout

PROFILES: Dict[str, HWProfile] = {
    "desktop": HWProfile(
        name            = "desktop (RTX 3070 8 GB)",
        lora_r          = 16,
        lora_alpha      = 32,
        lora_targets    = ["q_proj", "v_proj", "k_proj", "o_proj"],
        grpo_group_size = 6,
        train_set_size  = 1500,  # было 800
        resample_every  = 3,     # было 5
        top_k_tools     = 20,
        max_ctx_len     = 512,
        entropy_coeff   = 0.02,   # было 0.005 — более сильная регуляризация
        semantic_bias   = 5.0,    # было 3.0 — увеличено для лучшей релевантности
        warmup_steps    = 100,
        grad_clip       = 0.5,
        use_double_quant= False,   # не требуется на 8 GB
        compute_dtype   = torch.bfloat16,
    ),
    "laptop": HWProfile(
        name            = "laptop (4 GB VRAM)",
        lora_r          = 4,
        lora_alpha      = 8,
        lora_targets    = ["q_proj", "v_proj"],
        grpo_group_size = 3,     # 4→3: на 25% меньше forward-проходов
        train_set_size  = 600,   # 1000→600: эпоха быстрее, разнообразие через resample
        resample_every  = 2,     # чаще пересэмплируем для компенсации
        top_k_tools     = 15,    # 20→15: меньше embeddings инструментов, примерно на 25% быстрее
        max_ctx_len     = 256,   # 320→256: более короткий контекст экономит время attention
        entropy_coeff   = 0.02,   # было 0.005 — более сильная регуляризация
        semantic_bias   = 5.0,    # было 3.0 — увеличено для лучшей релевантности
        warmup_steps    = 80,
        grad_clip       = 0.5,
        use_double_quant= True,
        compute_dtype   = torch.float16,
        sample_temperature = 1.2,
    ),
}


class NetMCPTrainer:
    def __init__(self, config):
        """Initialize the object."""
        self.config = config
        self.reward_fn = GRPOToolReward(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Выбор профиля
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

        # Модель
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
        self.progress_logger = TrainingProgressLogger(
            csv_path=getattr(config, "training_log_path", "runs/training_metrics.csv"),
            plot_path=getattr(config, "training_plot_path", "runs/training_curve.png"),
            enabled=getattr(config, "training_plot_enabled", True),
        )

        # Фиксированный пул обучения
        pool = [p for p in config.train_prompts if p.get("relevant_tools")]
        if not pool:
            pool = config.train_prompts
        self._train_pool = pool
        self.train_set = random.sample(pool, min(self.p.train_set_size, len(pool)))

        print(f"Train pool: {len(pool)} prompts  "
              f"train_set={len(self.train_set)}  "
              f"top_k={self.p.top_k_tools}  "
              f"group_size={self.p.grpo_group_size}")

        # Предварительно кешируем embeddings названий инструментов
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
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _clear(self):
        """Handle clear."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _current_lr(self) -> float:
        """Handle current lr."""
        return self.config.rl.learning_rate * min(
            1.0, (self._update_step + 1) / self.p.warmup_steps
        )

    def _parse_tool_call(self, text: str) -> Optional[str]:
        """Parse parse tool call."""
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

    def rewrite_query_for_retrieval(self, query: str) -> Optional[str]:
        """
        Просит загруженную base/LoRA-модель построить короткие английские ключевые слова для retrieval.

        Retrieval выполняется до ранжирования политикой. Если многоязычный или шумный
        пользовательский текст не находит подходящих кандидатов, обученная политика
        сможет лишь переупорядочить плохие варианты. Этот метод даёт retriever-у
        английское API-представление запроса, сохраняя исходный запрос для
        финального prompt ранжирования.
        """
        if not query or not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            return None

        prompt = (
            "Task: convert the request into 3-8 English search keywords for API/tool retrieval.\n"
            "Output only the keywords.\n"
            "Bad output: explanations, labels, field names, instructions.\n"
            "Request: weather in London\n"
            "weather current London\n"
            "Request: top 20 NFT collections\n"
            "top NFT collections ranking sales\n"
            f"Request: {query}\n"
        )

        was_training = self.model.training
        self.model.eval()
        try:
            ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=192,
                padding=False,
            ).input_ids.to(self.device)
            with torch.inference_mode():
                out = self.model.generate(
                    ids,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
            text = self._clean_retrieval_rewrite(text)
            return text or None
        except Exception as exc:
            print(f"Query rewrite skipped: {exc}")
            return None
        finally:
            if was_training:
                self.model.train()

    def _clean_retrieval_rewrite(self, text: str) -> str:
        """Clean clean retrieval rewrite."""
        text = (text or "").strip()
        text = text.splitlines()[0] if text else ""
        text = re.sub(r"^(keywords?|answer|query)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"[<>`\"']", " ", text)
        text = " ".join(text.split())
        bad_fragments = {
            "entities",
            "brands",
            "locations",
            "intent",
            "return keywords",
            "output only",
            "request",
            "explanation",
            "field names",
            "instructions",
        }
        lowered = text.lower()
        if any(fragment in lowered for fragment in bad_fragments):
            return ""
        if len(text) < 3 or len(text) > 180:
            return ""
        return text

    # ------------------------------------------------------------------
    # Работа с состоянием инструментов
    # ------------------------------------------------------------------

    def _get_tools_state(self, prompt: Dict):
        """Возвращает (state, tools_list, semantic_scores)."""
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
        if not tools_state:
            for tool in candidates[: max(3, min(self.p.top_k_tools, len(candidates)))]:
                sv = self.env.network.get_server_state(tool["name"])
                qos = self.env.network.get_qos_metrics(tool["name"])
                sem = self.env.tools.semantic_similarity(
                    self.env.current_query, tool["name"]
                )
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
    # Кодирование контекста и embeddings инструментов один раз на группу
    # ------------------------------------------------------------------

    def _encode_context(self, state: Dict) -> torch.Tensor:
        """Encode encode context."""
        ctx = get_dynamic_prompt(state["query"], state["tools"])
        ids = self.tokenizer(
            ctx, return_tensors="pt",
            truncation=True, max_length=self.p.max_ctx_len, padding=False,
        ).input_ids.to(self.device)
        return ids

    @torch.inference_mode()
    def _build_tool_embs(self, tools: List[str]) -> torch.Tensor:
        """Возвращает матрицу (n, d) из кешированных embeddings без токенизации на каждый вызов."""
        if not tools:
            raise ValueError("Нельзя построить embeddings для пустого списка инструментов")

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
    # Прямой проход
    # ------------------------------------------------------------------

    def _forward_with_embs(self, input_ids: torch.Tensor,
                           tool_embs: torch.Tensor,
                           semantic_scores: Optional[List[float]] = None
                           ) -> torch.Tensor:
        """Handle forward with embs."""
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
    # Создание одной GRPO-группы
    # ------------------------------------------------------------------

    def _make_group(self, prompt: Dict) -> Dict:
        """Handle make group."""
        state, tools, scores = self._get_tools_state(prompt)
        if not tools:
            return {
                "input_ids": None,
                "tool_embs": None,
                "semantic_scores": [],
                "rollouts": [],
                "adv_std": 0.0,
                "empty": True,
            }

        input_ids = self._encode_context(state)
        tool_embs = self._build_tool_embs(tools)   # режим inference_mode

        rollouts = []
        with torch.inference_mode():
            for _ in range(self.p.grpo_group_size):
                logits = self._forward_with_embs(input_ids, tool_embs, scores)
                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1e-8)
                probs = probs / (probs.sum() + 1e-8)
                # Применяем температуру: выше значение — больше exploration и меньше пропущенных групп
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

        # Преимущество для GRPO
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
    # Обновление градиента для одной группы
    # ------------------------------------------------------------------

    def _train_group(self, group: Dict) -> Optional[float]:
        """Train train group."""
        if group.get("empty"):
            return None

        if group["adv_std"] < 1e-8:
            return None

        input_ids  = group["input_ids"]
        # Клонируем: тензоры из inference_mode не участвуют в backward
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
    # Обучение
    # ------------------------------------------------------------------

    def train(self):
        """Train train."""
        print(f"\n{'=' * 60}")
        print(f"GRPO TRAINING [{self.p.name}] — {self.config.rl.num_epochs} epochs")
        print(f"{'=' * 60}")

        best_relevance = 0.0

        for epoch in range(1, self.config.rl.num_epochs + 1):
            # НОВОЕ: Ротация сценариев каждые 5 эпох для разнообразия
            if epoch % 5 == 1:
                self.env.network.rotate_scenario()
                print(f"\n--- Epoch {epoch} [scenario: {self.env.network.current_scenario}] ---")

            # Пересэмплируем train set для разнообразия
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
            avg_loss = sum(losses) / max(len(losses), 1)
            avg_reward = total_reward / max(total_n, 1)
            success_rate = success_n / max(total_n, 1)
            avg_adv_std = sum(adv_stds) / max(len(adv_stds), 1)
            current_lr = self._current_lr()
            print(
                f"  loss={avg_loss:.4f}  "
                f"reward={avg_reward:.3f}  "
                f"success={success_rate:.2%}  "
                f"relevance={rel:.2%}  "
                f"rollouts={total_n}  skipped={skipped}  "
                f"updates={len(losses)}  "
                f"adv_std={avg_adv_std:.3f}  "
                f"lr={current_lr:.2e}"
            )
            self.progress_logger.append(
                TrainingMetric(
                    epoch=epoch,
                    loss=avg_loss,
                    reward=avg_reward,
                    success_rate=success_rate,
                    relevance_rate=rel,
                    updates=len(losses),
                    skipped=skipped,
                    rollouts=total_n,
                    adv_std=avg_adv_std,
                    lr=current_lr,
                    scenario=str(getattr(self.env.network, "current_scenario", "")),
                )
            )

            if rel > best_relevance:
                best_relevance = rel
                self._save_checkpoint("best")
                print(f"  ★ New best relevance={best_relevance:.2%}")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

            self._clear()

        print("\nTraining complete.")
        if getattr(self.progress_logger, "enabled", False):
            print(f"Training metrics: {self.progress_logger.csv_path}")
            print(f"Training plot:    {self.progress_logger.plot_path}")

    # ------------------------------------------------------------------
    # Оценка
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 200,
                 network_mode: NetworkMode = NetworkMode.CONTROLLED):
        """Evaluate evaluate."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION [{self.p.name}] — {num_episodes} ep, mode={network_mode.value}")
        print(f"{'=' * 60}")

        self.env.set_network_mode(network_mode)
        pool = [p for p in self.config.val_prompts if p.get("relevant_tools")]
        selected = random.sample(pool or self.config.val_prompts,
                                 min(num_episodes, len(pool or self.config.val_prompts)))

        success_t = relevant_t = total_t = 0
        top3_relevant_t = 0   # релевантный инструмент есть в top-3 модели

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

                # Жадный Top-1
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

                # Точность Top-3: проверяем, есть ли релевантный инструмент в top-3
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
    # Сохранение и загрузка checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, label):
        """Save save checkpoint."""
        d = f"checkpoints/{label}"
        os.makedirs(d, exist_ok=True)
        self.model.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        # Сохраняем профиль, чтобы checkpoint можно было корректно продолжить
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
        Загружает веса LoRA-адаптера в уже инициализированную модель.
        Использует set_adapter / load_adapter вместо PeftModel.from_pretrained,
        чтобы избежать предупреждения multiple adapters и ошибок missing keys, когда
        профиль checkpoint отличается от текущего.
        """
        import os
        from safetensors.torch import load_file as st_load
        import torch

        print(f"Loading checkpoint from {checkpoint_path}...")

        # Пробуем загрузить через loader адаптеров PEFT без двойной обёртки
        try:
            self.model.load_adapter(checkpoint_path, adapter_name="default")
            print("Checkpoint loaded via load_adapter.")
            return
        except Exception as e:
            print(f"  load_adapter failed ({e}), trying manual weight load...")

        # Резервный путь: напрямую загружаем safetensors или pytorch_model.bin
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

    # Совместимость для интерактивного режима
    def _forward(self, context: str, tools: List[str], **kwargs) -> torch.Tensor:
        """Handle forward."""
        ids  = self.tokenizer(
            context, return_tensors="pt",
            truncation=True, max_length=self.p.max_ctx_len, padding=False,
        ).input_ids.to(self.device)
        embs = self._build_tool_embs(tools)
        with torch.inference_mode():
            return self._forward_with_embs(ids, embs, kwargs.get("semantic_scores"))
