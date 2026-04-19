import torch
from typing import List, Optional
from src.config import Config


class LLMClient:
    def __init__(self, config: Config, local_model=None, local_tokenizer=None):
        self.config = config
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer

        if self.local_model is not None:
            print(f"Using LOCAL model: {config.model_name}")
            self._use_local = True
        else:
            print(f"Using API model: {config.model_name}")
            self._use_local = False

    def ask(self, question: str) -> str:
        if not self._use_local or self.local_model is None:
            print("Warning: No local model available")
            return ""

        return self._ask_local(question)

    def ask_batch(self, questions: List[str]) -> List[str]:
        if not questions:
            return []
        if not self._use_local or self.local_model is None:
            print("Warning: No local model available")
            return [""] * len(questions)
        return [self._ask_local(q) for q in questions]

    def _ask_local(self, question: str) -> str:
        if self.local_model is None or self.local_tokenizer is None:
            return ""

        try:
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": question})

            text = self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.local_tokenizer(text, return_tensors="pt").to(self.local_model.device)

            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id,
                    repetition_penalty=1.05
                )

            response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response.startswith(text):
                response = response[len(text):].strip()

            return response

        except Exception as e:
            print(f"Local inference error: {e}")
            return ""