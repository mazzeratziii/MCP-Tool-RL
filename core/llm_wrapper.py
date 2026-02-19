import asyncio
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from httpx import AsyncClient, Limits, Timeout
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError


@dataclass
class RecognizingContent:
    request: str
    text: Optional[str] = None


class LLMWrapper:
    DEFAULT_REQUEST_TIMEOUT = 600.0
    MAX_REQUEST_RETRIES = 10  # For APITimeoutError

    DEFAULT_CONCURRENCY = 10000
    MAX_ATTEMPTS_FOR_CONCURRENCY = 10  # For APIConnectionError

    TEMPERATURE = 0.1
    FREQUENCY_PENALTY = 0.02
    TOP_P = 0.95

    MAX_TOKENS = 1500
    SEED = 2025

    MIN_VISUAL_TOKENS = 8
    MAX_VISUAL_TOKENS = 1024

    def __init__(
            self,
            model_name: str,
            openai_base_url: str,
            openai_api_token: str,
            sys_prompt: str = "",
            prompt: str = "",
            min_request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
            max_concurrent_requests: int = DEFAULT_CONCURRENCY
    ):
        self._model_name = model_name
        self._base_url = openai_base_url
        self._prompt = prompt
        self._sys_prompt = sys_prompt
        self._api_token = openai_api_token
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._client: Optional[AsyncOpenAI] = None

        # For APITimeoutError
        self._min_request_timeout = max(min_request_timeout, 1)

        # For APIConnectionError
        self._max_concurrent_requests = max(max_concurrent_requests, 1)
        step = (self._max_concurrent_requests - 1) // self.MAX_ATTEMPTS_FOR_CONCURRENCY
        self._concurrency_step = max(step, 1)

    def create_child(self, prompt: Optional[str] = None, sys_prompt: Optional[str] = None) -> 'LLMWrapper':
        return LLMWrapper(model_name=self._model_name,
                          openai_api_token=self._api_token,
                          openai_base_url=self._base_url,
                          max_concurrent_requests=self._max_concurrent_requests,
                          min_request_timeout=self._min_request_timeout,
                          sys_prompt=sys_prompt or self._sys_prompt,
                          prompt=prompt or self._prompt)

    @asynccontextmanager
    async def _client_context(self, connections_num: int):
        max_connections = max(min(self._max_concurrent_requests, connections_num), 1)
        try:
            async with AsyncClient(timeout=Timeout(None, connect=5),
                                   limits=Limits(max_connections=max_connections,
                                                 max_keepalive_connections=max_connections)) as httpx_client:
                self._client = AsyncOpenAI(
                    api_key=self._api_token,
                    base_url=self._base_url,
                    http_client=httpx_client,
                    max_retries=0
                )
                self._semaphore = asyncio.Semaphore(max_connections)
                yield
        finally:
            if self._client is not None:
                await self._client.close()
                await asyncio.sleep(random.random())
            self._client = self._semaphore = None

    def recognize_contents(self, contents: list[str]) -> list[str]:
        recognizing_data = [RecognizingContent(content) for content in contents]

        def _run_async_in_thread():
            asyncio.run(self._recognize_content_async(recognizing_data))

        if not recognizing_data:
            return []

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_async_in_thread)
            future.result()

        return [img.text for img in recognizing_data]

    async def _recognize_content_async(self, recognizing_content: list[RecognizingContent]):
        """Request images recognition asynchronously."""
        content_for_recognition = [i for i in recognizing_content if i.text is None]
        connections_num = len(content_for_recognition)
        for attempt in range(1, self.MAX_ATTEMPTS_FOR_CONCURRENCY + 1):
            async with self._client_context(connections_num=connections_num):
                tasks = [
                    asyncio.create_task(self._recognize_content_with_retries(content, len(content_for_recognition)))
                    for content in content_for_recognition]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                api_connection_error = None
                for result in results:
                    if isinstance(result, Exception):
                        if isinstance(result, APIConnectionError):
                            api_connection_error = result
                        else:
                            raise result
                if api_connection_error is None:
                    return

                self._max_concurrent_requests -= self._concurrency_step
                content_for_recognition = [i for i in content_for_recognition if i.text is None]
                assert len(content_for_recognition) > 0
                recognized_images_num = len(tasks) - len(content_for_recognition)
                connections_num = max(min(recognized_images_num, len(content_for_recognition), connections_num), 1)
                warnings.warn(f'An APIConnectionError occurred, retrying for {len(content_for_recognition)} images '
                              f'with connections_num={connections_num} (attempt={attempt}).')

        warnings.warn(f'Maximum number of attempts to find a valid amount of concurrent requests '
                      f'exceeded. Reraise APIConnectionError.')
        raise api_connection_error

    async def _recognize_content_with_retries(self, content: RecognizingContent, total: int):
        """Process image request with retries on error"""
        async with self._semaphore:
            request_time = self._min_request_timeout
            initial = 0
            processed = 0
            for attempt in range(1, self.MAX_REQUEST_RETRIES + 1):
                try:
                    await self._recognize_element(content, request_time)
                    processed += 1
                    return
                except APITimeoutError:
                    if attempt == self.MAX_REQUEST_RETRIES:
                        warnings.warn('Maximum number of retries exceeded. Reraise APITimeoutError.')
                        raise
                    processed_requests = processed - initial
                    if processed_requests == 0:
                        request_time += self._min_request_timeout
                    else:
                        initial = processed
                        time_for_request = request_time // processed_requests + 1
                        new_requests_count = min(total - initial, self._max_concurrent_requests)
                        time_for_new_requests = time_for_request * new_requests_count
                        request_time = max(time_for_new_requests, self._min_request_timeout)
                    warnings.warn(f'An APITimeoutError occurred, retrying the request '
                                  f'with timeout={request_time} (attempt={attempt}).')
                    await asyncio.sleep(random.random())

    async def _recognize_element(self, content: RecognizingContent, timeout: float = DEFAULT_REQUEST_TIMEOUT):
        """Process single image request"""
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=self._build_messages(content),

            temperature=self.TEMPERATURE,
            frequency_penalty=self.FREQUENCY_PENALTY,
            top_p=self.TOP_P,

            timeout=timeout,
            max_tokens=self.MAX_TOKENS,
            seed=self.SEED,
        )
        content.text = response.choices[0].message.content

    def _build_messages(self, content: RecognizingContent) -> list[dict]:
        """Construct messages for openai api request."""
        return [
            {
                "role": "system",
                "content": self._sys_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'{self._prompt}\n\n{content.request}'},
                ]
            }
        ]
