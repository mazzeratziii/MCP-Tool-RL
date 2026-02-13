"""
Агент с архитектурой ReAct (Reasoning + Acting) для выполнения инструментов.
Интегрирует онлайн-загрузчик, семантический поиск и SONAR.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import random

from config.settings import config
from core.hf_loader import ToolSpec, loader
from selector.sonar import sonar
from utils.embeddings import embedder

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Результат выполнения инструмента"""
    success: bool
    data: Any
    tool_id: str
    tool_name: str
    execution_time_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReactAgent:
    """
    Агент с архитектурой ReAct для работы с инструментами ToolBench.
    Чередует рассуждение (reasoning) и действие (acting).
    """

    def __init__(self):
        self.config = config
        self.loader = loader
        self.sonar = sonar
        self.embedder = embedder

        # История выполнения
        self.execution_history: List[Dict[str, Any]] = []
        self.conversation_context: List[Dict[str, Any]] = []

        # Состояние агента
        self.max_history_size = 50
        self.thinking_depth = 3  # Максимальная глубина рассуждений

        logger.info("ReAct агент инициализирован")

    async def process_query(
            self,
            user_query: str,
            session_id: Optional[str] = None,
            max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Основной метод обработки запроса.
        Реализует цикл ReAct: Reasoning -> Acting -> Observing.

        Args:
            user_query: Запрос пользователя
            session_id: ID сессии (для контекста)
            max_steps: Максимум шагов выполнения

        Returns:
            Результат обработки
        """
        if not session_id:
            session_id = f"sess_{int(time.time())}_{random.randint(1000, 9999)}"

        logger.info(f"[{session_id}] Начало обработки: '{user_query}'")

        # Инициализация сессии
        session_start = time.time()
        session_result = {
            "session_id": session_id,
            "user_query": user_query,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "final_result": None,
            "success": False,
            "total_time_ms": 0
        }

        # Добавляем запрос в контекст
        self._add_to_context(session_id, "user", user_query)

        current_state = {
            "query": user_query,
            "remaining_steps": max_steps,
            "collected_data": {},
            "last_observation": None
        }

        step_count = 0

        try:
            # Основной цикл ReAct
            while current_state["remaining_steps"] > 0 and step_count < max_steps:
                step_count += 1
                step_id = f"{session_id}_step_{step_count}"

                logger.debug(f"[{session_id}] Шаг {step_count}: Reasoning")

                # Шаг 1: Reasoning - анализ текущего состояния
                reasoning_result = await self._reasoning_step(
                    current_state, step_id, session_id
                )

                if reasoning_result.get("should_stop", False):
                    logger.info(f"[{session_id}] Reasoning решил остановиться")
                    break

                # Шаг 2: Acting - выбор и выполнение инструмента
                action_result = await self._acting_step(
                    reasoning_result, step_id, session_id
                )

                # Сохраняем шаг
                step_info = {
                    "step": step_count,
                    "reasoning": reasoning_result,
                    "action": action_result,
                    "timestamp": datetime.now().isoformat()
                }
                session_result["steps"].append(step_info)

                # Шаг 3: Observing - обновление состояния
                current_state = self._update_state(
                    current_state, reasoning_result, action_result
                )

                current_state["remaining_steps"] -= 1

                # Проверяем, достигли ли мы цели
                if self._is_goal_achieved(current_state, user_query):
                    logger.info(f"[{session_id}] Цель достигнута на шаге {step_count}")
                    break

            # Формируем финальный результат
            final_result = self._generate_final_response(
                current_state, user_query, session_id
            )

            session_result.update({
                "final_result": final_result,
                "success": True,
                "total_time_ms": (time.time() - session_start) * 1000,
                "steps_count": step_count,
                "end_time": datetime.now().isoformat()
            })

            # Добавляем результат в контекст
            self._add_to_context(
                session_id,
                "assistant",
                final_result.get("response", "No response")
            )

        except Exception as e:
            logger.error(f"[{session_id}] Ошибка обработки: {e}")
            session_result.update({
                "success": False,
                "error": str(e),
                "total_time_ms": (time.time() - session_start) * 1000
            })

        # Сохраняем сессию в историю
        self.execution_history.append(session_result)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]

        logger.info(
            f"[{session_id}] Обработка завершена за {session_result['total_time_ms']:.1f}ms, "
            f"успех: {session_result['success']}, шагов: {len(session_result['steps'])}"
        )

        return session_result

    async def _reasoning_step(
            self,
            state: Dict[str, Any],
            step_id: str,
            session_id: str
    ) -> Dict[str, Any]:
        """
        Шаг рассуждения (Reasoning).
        Анализирует текущее состояние и решает, что делать дальше.

        Args:
            state: Текущее состояние
            step_id: ID шага
            session_id: ID сессии

        Returns:
            Решение о следующем действии
        """
        query = state["query"]
        last_observation = state.get("last_observation")
        collected_data = state.get("collected_data", {})

        # Анализируем, нужны ли дополнительные инструменты
        reasoning_result = {
            "step_id": step_id,
            "analysis": {},
            "decision": "continue",
            "should_stop": False,
            "target_tool_category": None,
            "extracted_parameters": {}
        }

        # Простой анализатор (в production заменить на LLM)
        query_lower = query.lower()

        # Определяем категорию инструмента по ключевым словам
        category_keywords = {
            "weather": ["погода", "температура", "ветер", "дождь"],
            "finance": ["курс", "валюта", "доллар", "евро", "рубль", "конверт"],
            "transportation": ["рейс", "поезд", "самолет", "билет", "расписание"],
            "shopping": ["цена", "купить", "товар", "магазин", "скидка"],
            "entertainment": ["фильм", "музыка", "игра", "сериал", "кино"]
        }

        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                reasoning_result["target_tool_category"] = category
                break

        # Извлекаем простые параметры
        reasoning_result["extracted_parameters"] = self._extract_simple_parameters(query)

        # Проверяем, достаточно ли данных для ответа
        if last_observation and last_observation.get("success", False):
            # Если последний вызов был успешным, можно остановиться
            reasoning_result["decision"] = "respond"
            reasoning_result["should_stop"] = True

        logger.debug(f"[{session_id}] Reasoning: {reasoning_result}")
        return reasoning_result

    async def _acting_step(
            self,
            reasoning: Dict[str, Any],
            step_id: str,
            session_id: str
    ) -> Dict[str, Any]:
        """
        Шаг действия (Acting).
        Выбирает и выполняет инструмент.

        Args:
            reasoning: Результат reasoning шага
            step_id: ID шага
            session_id: ID сессии

        Returns:
            Результат выполнения
        """
        query = reasoning.get("analysis", {}).get("original_query", "")
        category = reasoning.get("target_tool_category")
        extracted_params = reasoning.get("extracted_parameters", {})

        # Контекст для селектора
        context = {}
        if category:
            context["category"] = category

        # Выбираем инструмент с помощью SONAR
        start_time = time.time()
        selected_tool, selection_info = await self.sonar.select_tool(
            query=query,
            context=context if context else None
        )

        selection_time = (time.time() - start_time) * 1000

        if not selected_tool:
            return {
                "success": False,
                "error": "No tool selected",
                "selection_info": selection_info,
                "execution_time_ms": selection_time
            }

        # Выполняем инструмент
        execution_start = time.time()
        execution_result = await self._execute_tool(
            selected_tool,
            extracted_params,
            session_id
        )
        execution_time = (time.time() - execution_start) * 1000

        # Обновляем сетевые метрики
        self.sonar.update_network_metrics(
            selected_tool.id,
            execution_time,
            execution_result.success
        )

        action_result = {
            "success": execution_result.success,
            "tool": {
                "id": selected_tool.id,
                "name": selected_tool.name,
                "category": selected_tool.category
            },
            "execution_result": execution_result.data,
            "selection_info": selection_info,
            "selection_time_ms": selection_time,
            "execution_time_ms": execution_time,
            "total_time_ms": selection_time + execution_time,
            "error": execution_result.error,
            "step_id": step_id
        }

        logger.debug(f"[{session_id}] Acting: выполнен {selected_tool.name}, успех: {execution_result.success}")
        return action_result

    async def _execute_tool(
            self,
            tool: ToolSpec,
            parameters: Dict[str, Any],
            session_id: str
    ) -> ExecutionResult:
        """
        Выполняет инструмент ToolBench.
        В режиме симуляции генерирует реалистичные ответы.

        Args:
            tool: Инструмент для выполнения
            parameters: Параметры вызова
            session_id: ID сессии

        Returns:
            Результат выполнения
        """
        start_time = time.time()

        try:
            # Проверяем обязательные параметры
            missing_params = []
            for param_name in tool.required_params:
                if param_name not in parameters:
                    missing_params.append(param_name)

            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")

            if tool.is_simulated or self.config.network.enable_simulation:
                # Режим симуляции
                simulated_data = self._simulate_tool_response(tool, parameters)
                success = True
                error = None
            else:
                # Реальный вызов API (заглушка для будущей реализации)
                # В production здесь будет HTTP-запрос
                simulated_data = self._simulate_tool_response(tool, parameters)
                success = True
                error = None

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=success,
                data=simulated_data,
                tool_id=tool.id,
                tool_name=tool.name,
                execution_time_ms=execution_time,
                error=error,
                metadata={
                    "simulated": tool.is_simulated,
                    "parameters_used": parameters
                }
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"[{session_id}] Ошибка выполнения {tool.name}: {e}")

            return ExecutionResult(
                success=False,
                data=None,
                tool_id=tool.id,
                tool_name=tool.name,
                execution_time_ms=execution_time,
                error=str(e)
            )

    def _simulate_tool_response(
            self,
            tool: ToolSpec,
            parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Генерирует реалистичный симулированный ответ инструмента.

        Args:
            tool: Инструмент
            parameters: Параметры вызова

        Returns:
            Симулированный ответ
        """
        category = tool.category.lower()

        # Генерация ответов по категориям
        if "weather" in category:
            city = parameters.get("city", "Москва")
            return {
                "city": city,
                "temperature": random.uniform(-10, 30),
                "humidity": random.uniform(30, 90),
                "conditions": random.choice(["ясно", "облачно", "дождь", "снег"]),
                "wind_speed": random.uniform(0, 15),
                "timestamp": datetime.now().isoformat(),
                "source": "simulated_weather_api"
            }

        elif "finance" in category:
            amount = parameters.get("amount", 100)
            from_curr = parameters.get("from_currency", "USD")
            to_curr = parameters.get("to_currency", "EUR")

            # Примерные курсы
            rates = {
                "USD": 1.0,
                "EUR": 0.92,
                "RUB": 90.5,
                "GBP": 0.79
            }

            from_rate = rates.get(from_curr.upper(), 1.0)
            to_rate = rates.get(to_curr.upper(), 1.0)

            converted = amount * (to_rate / from_rate)

            return {
                "amount": amount,
                "from_currency": from_curr,
                "to_currency": to_curr,
                "converted_amount": round(converted, 2),
                "exchange_rate": round(to_rate / from_rate, 4),
                "timestamp": datetime.now().isoformat()
            }

        elif "transportation" in category:
            origin = parameters.get("origin", "Москва")
            destination = parameters.get("destination", "Санкт-Петербург")

            return {
                "origin": origin,
                "destination": destination,
                "departure_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                "arrival_time": (datetime.now() + timedelta(hours=4)).isoformat(),
                "price": random.uniform(1000, 5000),
                "carrier": random.choice(["Аэрофлот", "S7", "Уральские авиалинии"]),
                "available_seats": random.randint(10, 200)
            }

        # Общий случай
        return {
            "status": "success",
            "result": f"Выполнен {tool.name} с параметрами: {parameters}",
            "simulated": True,
            "timestamp": datetime.now().isoformat(),
            "tool_id": tool.id
        }

    def _extract_simple_parameters(self, query: str) -> Dict[str, Any]:
        """
        Извлекает параметры из запроса (простая реализация).
        В production заменить на LLM-based extractor.

        Args:
            query: Пользовательский запрос

        Returns:
            Извлечённые параметры
        """
        params = {}
        query_lower = query.lower()

        # Простые эвристики для демонстрации
        import re

        # Города
        city_keywords = ["в москве", "в лондоне", "в париже", "в берлине"]
        for phrase in city_keywords:
            if phrase in query_lower:
                params["city"] = phrase.split()[-1]
                break

        # Числа
        numbers = re.findall(r'\d+', query)
        if numbers:
            params["amount"] = float(numbers[0])

        # Валюты
        currency_map = {
            "доллар": "USD",
            "евро": "EUR",
            "рубл": "RUB",
            "фунт": "GBP"
        }

        for word, code in currency_map.items():
            if word in query_lower:
                if "from_currency" not in params:
                    params["from_currency"] = code
                else:
                    params["to_currency"] = code

        return params

    def _update_state(
            self,
            current_state: Dict[str, Any],
            reasoning: Dict[str, Any],
            action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Обновляет состояние после выполнения действия"""
        updated = current_state.copy()

        # Сохраняем результат последнего действия
        if action.get("success"):
            updated["last_observation"] = {
                "success": True,
                "data": action.get("execution_result"),
                "tool": action.get("tool", {})
            }

            # Добавляем данные в собранные
            collected = updated.get("collected_data", {})
            tool_name = action.get("tool", {}).get("name", "unknown")
            collected[tool_name] = action.get("execution_result")
            updated["collected_data"] = collected

        return updated

    def _is_goal_achieved(
            self,
            state: Dict[str, Any],
            original_query: str
    ) -> bool:
        """Проверяет, достигнута ли цель запроса"""
        # Простая проверка: если есть успешное наблюдение, цель достигнута
        last_obs = state.get("last_observation")
        if last_obs and last_obs.get("success"):
            return True

        # Можно добавить более сложную логику
        return False

    def _generate_final_response(
            self,
            state: Dict[str, Any],
            original_query: str,
            session_id: str
    ) -> Dict[str, Any]:
        """Генерирует финальный ответ пользователю"""
        collected_data = state.get("collected_data", {})
        last_obs = state.get("last_observation")

        if not collected_data and not last_obs:
            return {
                "response": "Не удалось обработать ваш запрос. Попробуйте переформулировать.",
                "status": "error",
                "data_available": False
            }

        # Формируем ответ на основе собранных данных
        response_parts = []

        if last_obs and last_obs.get("success"):
            data = last_obs.get("data", {})
            tool_name = last_obs.get("tool", {}).get("name", "инструмент")

            # Генерация ответа по категории инструмента
            if isinstance(data, dict):
                if "temperature" in data:  # Погода
                    response = (
                        f"Погода в {data.get('city', 'городе')}: "
                        f"{data.get('temperature', 0):.1f}°C, "
                        f"{data.get('conditions', 'ясно')}. "
                        f"Ветер: {data.get('wind_speed', 0):.1f} м/с."
                    )
                elif "converted_amount" in data:  # Конвертация валют
                    response = (
                        f"{data.get('amount', 0)} {data.get('from_currency', '')} = "
                        f"{data.get('converted_amount', 0):.2f} {data.get('to_currency', '')}. "
                        f"Курс: {data.get('exchange_rate', 0):.4f}"
                    )
                else:
                    response = f"Получены данные от {tool_name}: {data}"
            else:
                response = f"Результат от {tool_name}: {data}"

            response_parts.append(response)

        # Добавляем данные из других шагов
        for tool_name, data in collected_data.items():
            if tool_name != last_obs.get("tool", {}).get("name", ""):
                response_parts.append(f"Дополнительно: {tool_name} вернул {data}")

        final_response = " ".join(response_parts) if response_parts else "Данные обработаны"

        return {
            "response": final_response,
            "status": "success",
            "data_available": True,
            "collected_data_summary": list(collected_data.keys()),
            "tools_used": len(collected_data)
        }

    def _add_to_context(self, session_id: str, role: str, content: str):
        """Добавляет сообщение в контекст сессии"""
        self.conversation_context.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Ограничиваем размер контекста
        if len(self.conversation_context) > self.max_history_size:
            self.conversation_context = self.conversation_context[-self.max_history_size:]

    def get_session_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Получает контекст конкретной сессии"""
        return [
            msg for msg in self.conversation_context
            if msg["session_id"] == session_id
        ]

    def get_agent_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы агента"""
        return {
            "total_sessions": len(self.execution_history),
            "successful_sessions": sum(1 for s in self.execution_history if s.get("success")),
            "avg_session_time_ms": (
                np.mean([s.get("total_time_ms", 0) for s in self.execution_history])
                if self.execution_history else 0
            ),
            "avg_steps_per_session": (
                np.mean([len(s.get("steps", [])) for s in self.execution_history])
                if self.execution_history else 0
            ),
            "most_used_tools": self._get_most_used_tools()
        }

    def _get_most_used_tools(self) -> List[Dict[str, Any]]:
        """Возвращает наиболее используемые инструменты"""
        tool_usage = {}

        for session in self.execution_history:
            for step in session.get("steps", []):
                tool = step.get("action", {}).get("tool", {})
                tool_id = tool.get("id")
                if tool_id:
                    tool_usage[tool_id] = tool_usage.get(tool_id, 0) + 1

        # Сортируем по использованию
        sorted_usage = sorted(
            tool_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return [
            {"tool_id": tool_id, "usage_count": count}
            for tool_id, count in sorted_usage
        ]


# Глобальный экземпляр агента
agent = ReactAgent()