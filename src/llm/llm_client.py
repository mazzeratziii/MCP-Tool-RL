# llm_client.py
import asyncio
from typing import List, Optional
from src.llm.llm_wrapper import LLMWrapper
from src.config import Config  # ✅ правильный импорт


class LLMClient:
    """Клиент для работы с LLM через LLMWrapper"""

    def __init__(self, config: Optional[Config] = None):
        # Если конфиг не передан, создаём новый
        self.config = config or Config()

        # Валидация конфигурации (уже есть в __init__ Config)
        self.wrapper = LLMWrapper(
            model_name=self.config.model_name,
            openai_base_url=self.config.openai_base_url,
            openai_api_token=self.config.openai_api_token,
            sys_prompt=self.config.system_prompt,
            prompt=self.config.user_prompt,
            min_request_timeout=self.config.min_request_timeout,
            max_concurrent_requests=self.config.max_concurrent_requests
        )

    def ask(self, question: str) -> str:
        results = self.ask_batch([question])
        return results[0] if results else ""

    def ask_batch(self, questions: List[str]) -> List[str]:
        if not questions:
            return []
        print(f"🔄 Отправка {len(questions)} запросов к {self.config.model_name}...")
        results = self.wrapper.recognize_contents(questions)
        print(f"✅ Получено {len(results)} ответов")
        return results

    async def ask_async(self, question: str) -> str:
        results = await self.ask_batch_async([question])
        return results[0] if results else ""

    def ask_batch_async(self, questions: List[str]) -> List[str]:
        if not questions:
            return []
        print(f"🔄 Отправка {len(questions)} запросов к {self.config.model_name}...")
        results = self.wrapper.recognize_contents(questions)

        # Обработка пустых ответов
        processed_results = []
        for i, (q, r) in enumerate(zip(questions, results)):
            if not r or r.strip() == "" or r.strip() == "None":
                print(f"⚠️ Пустой ответ на вопрос {i + 1}, повторяем...")
                # Повторный запрос
                r = self.wrapper.recognize_contents([q])[0]
            processed_results.append(r)

        print(f"✅ Получено {len(processed_results)} ответов")
        return processed_results

    def create_child(self, system_prompt: Optional[str] = None, user_prompt: Optional[str] = None) -> 'LLMClient':
        child_wrapper = self.wrapper.create_child(
            prompt=user_prompt,
            sys_prompt=system_prompt
        )
        new_client = LLMClient.__new__(LLMClient)
        new_client.config = self.config
        new_client.wrapper = child_wrapper
        return new_client