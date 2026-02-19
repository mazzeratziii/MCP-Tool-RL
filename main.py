"""
Главный файл: интеграция всех компонентов с загрузкой инструментов ToolBench.
Добавлена проверка релевантности и обработка неподходящих запросов.
Поддержка больших списков инструментов (до 1000+).
"""

import asyncio
import argparse
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.registry import registry
from core.embedder import searcher
from mcp.server import MCPServer
from mcp.client import MCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPToolRLSystem:
    """Главный класс системы, объединяющий загрузку, поиск и MCP."""

    def __init__(self, relevance_threshold: float = 0.3):
        """
        Args:
            relevance_threshold: Минимальное сходство для использования инструмента
                                 (ниже этого порога считаем, что нет подходящего инструмента)
        """
        self.relevance_threshold = relevance_threshold
        self.registry = registry
        self.searcher = searcher
        self.mcp_server = MCPServer()
        self.mcp_client: Optional[MCPClient] = None
        self.initialized = False
        self.history = []
        self.server_task = None

    async def initialize(self, tool_limit: int = 50, categories: Optional[list] = None):
        """Инициализация системы: загрузка инструментов, индексация, регистрация в MCP."""
        if self.initialized:
            return

        print("\n" + "=" * 70)
        print("🚀 MCP-TOOL-RL с ИНСТРУМЕНТАМИ TOOLBENCH")
        print("=" * 70)

        # 1. Загрузка инструментов из ToolBench
        print(f"📦 Загрузка инструментов из ToolBench (лимит: {tool_limit})...")
        tools = self.registry.load_from_toolbench(limit=tool_limit, categories=categories)
        print(f"✅ Загружено {len(tools)} инструментов")

        # Показываем первые несколько инструментов
        print("\n📋 Первые 5 инструментов:")
        for i, tool in enumerate(tools[:5]):
            print(f"  {i+1}. {tool.name} (id: {tool.id}, категория: {tool.category})")

        if len(tools) > 5:
            print(f"  ... и ещё {len(tools) - 5} инструментов")

        # 2. Индексация для семантического поиска
        print("\n📊 Индексация для семантического поиска...")
        self.searcher.index_tools(tools)

        # 3. Регистрация инструментов в MCP с уникальными именами
        print("\n📡 Регистрация инструментов в MCP...")
        registered_count = 0
        for tool in tools:
            properties = {}
            required = []

            for param in tool.parameters:
                if isinstance(param, dict):
                    param_name = param.get('name', '')
                    param_type = param.get('type', 'string')
                    param_desc = param.get('description', '')[:100]  # Обрезаем описания параметров
                else:
                    param_name = str(param)
                    param_type = 'string'
                    param_desc = ''

                if param_name:
                    properties[param_name] = {"type": param_type, "description": param_desc}
                    if param_name in tool.required_params:
                        required.append(param_name)

            input_schema = {"type": "object", "properties": properties, "required": required}

            # Обрезаем длинные описания для MCP
            tool_description = tool.description[:200] + "..." if len(tool.description) > 200 else tool.description

            self.mcp_server.register_tool(
                name=tool.id,
                description=tool_description,
                input_schema=input_schema,
                handler=lambda args, t=tool: self._handle_tool_call(t, args)
            )
            registered_count += 1

        print(f"✅ Зарегистрировано {registered_count} инструментов")

        # 4. Запуск MCP сервера в фоне
        print(f"\n🔄 Запуск MCP сервера на localhost:8765...")
        self.server_task = asyncio.create_task(self.mcp_server.start())

        self.initialized = True
        print("\n✅ Система готова!")
        print(f"⚡ Порог релевантности: {self.relevance_threshold}")
        print("=" * 70)

    async def _handle_tool_call(self, tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Обработчик вызова инструмента (заглушка, можно заменить реальным API)."""
        logger.info(f"🔧 Вызов инструмента: {tool.name} (id: {tool.id}) с аргументами {arguments}")
        return {
            "tool": tool.name,
            "tool_id": tool.id,
            "arguments": arguments,
            "result": f"Выполнен {tool.name}",
            "simulated": True
        }

    async def connect_client(self):
        """Подключает внутреннего MCP-клиента к серверу."""
        self.mcp_client = MCPClient()
        try:
            await self.mcp_client.connect()
            logger.info("✅ MCP клиент подключен успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения MCP клиента: {e}")
            # Продолжаем работу без клиента? Или прерываем?
            # В данном случае прерываем, так как без клиента нельзя вызывать инструменты
            raise

    def _categorize_query(self, query: str) -> Dict[str, Any]:
        """
        Анализирует запрос и определяет его категорию и параметры.
        Возвращает словарь с информацией о запросе.
        """
        q = query.lower()
        result = {
            "original": query,
            "category": None,
            "keywords": [],
            "has_numbers": bool(re.search(r'\d+', query)),
            "has_cities": False
        }

        # Определяем возможные категории по ключевым словам
        category_keywords = {
            "weather": ["погод", "температур", "дожд", "снег", "ветер", "град", "осадк", "прогноз"],
            "finance": ["доллар", "евро", "рубл", "валют", "конверт", "деньг", "курс", "цен", "стоимост"],
            "transportation": ["рейс", "билет", "самолет", "авиа", "полет", "перелет", "поезд", "транспорт"],
            "data": ["данн", "информац", "справк", "найди", "поиск", "покажи"],
            "general": ["что", "как", "почему", "когда", "где", "кто"]
        }

        # Собираем ключевые слова и определяем категорию
        for cat, keywords in category_keywords.items():
            for kw in keywords:
                if kw in q:
                    result["keywords"].append(kw)
                    if result["category"] is None:
                        result["category"] = cat

        # Проверяем наличие городов
        cities = ["москва", "питер", "санкт-петербург", "лондон", "париж", "берлин", "нью-йорк", "уфа", "казань"]
        for city in cities:
            if city in q:
                result["has_cities"] = True
                result["keywords"].append(city)
                break

        return result

    async def process_query(self, query: str):
        """Обработка запроса с проверкой релевантности."""
        if not self.initialized:
            await self.initialize()

        start = datetime.now()
        print(f"\n🔍 Запрос: '{query}'")

        # Анализируем запрос
        query_info = self._categorize_query(query)
        print(f"📊 Анализ запроса: категория={query_info['category']}, ключевые слова={query_info['keywords']}")

        # Семантический поиск (берём топ-3 инструмента для анализа)
        results = self.searcher.search(query, top_k=3)

        if not results:
            print("❌ Не найдено инструментов в базе")
            return

        # Проверяем, достаточно ли высокое сходство у лучшего инструмента
        best_result = results[0]
        best_tool = best_result["tool"]
        best_sim = best_result["similarity"]

        # Если сходство ниже порога, считаем что нет подходящего инструмента
        if best_sim < self.relevance_threshold:
            print(f"\n⚠️  Сходство с лучшим инструментом ({best_sim:.3f}) ниже порога ({self.relevance_threshold})")
            print(f"   Лучший инструмент: {best_tool.name} (категория: {best_tool.category})")
            print(f"   Описание: {best_tool.description[:100]}...")

            # Анализируем, почему не подошло
            if query_info['category'] and query_info['category'] != best_tool.category:
                print(f"\n💡 Запрос относится к категории '{query_info['category']}', но лучший инструмент из '{best_tool.category}'")
                print(f"   Возможно, в системе нет инструментов для категории '{query_info['category']}'")
            elif query_info['has_cities'] and "city" not in str(best_tool.parameters).lower():
                print(f"\n💡 Запрос содержит город, но выбранный инструмент не принимает параметр 'city'")
            elif query_info['has_numbers'] and not any(p.get('type') == 'number' for p in best_tool.parameters):
                print(f"\n💡 Запрос содержит числа, но выбранный инструмент не принимает числовые параметры")

            print("\n❌ Не могу ответить на этот вопрос - нет подходящего инструмента")
            print("   Попробуйте переформулировать запрос или задать вопрос по другим темам.")

            # Сохраняем в историю как неудачный запрос
            self.history.append({
                "query": query,
                "tool": best_tool.name,
                "tool_id": best_tool.id,
                "similarity": best_sim,
                "success": False,
                "reason": "below_threshold",
                "time_ms": (datetime.now() - start).total_seconds() * 1000
            })
            return

        # Если сходство достаточно высокое, используем инструмент
        print(f"🎯 Выбран инструмент: {best_tool.name} (id: {best_tool.id}, сходство: {best_sim:.3f})")

        params = self._extract_parameters(query, best_tool)
        print(f"📝 Параметры: {params}")

        # Подключаем клиент, если ещё не подключён
        if not self.mcp_client:
            await self.connect_client()

        # Вызываем инструмент
        result = await self.mcp_client.call_tool(best_tool.id, params)

        print("\n📊 Результат вызова:")
        print("-" * 40)
        if result.get("status") == "success":
            print(f"✅ Успешно")
            print(f"   Результат: {result.get('result', {}).get('result', 'Нет данных')}")
            print(f"   Время выполнения: {result.get('execution_time_ms', 0):.1f}ms")
        else:
            print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        print("-" * 40)
        print(f"⏱️  Всего: {(datetime.now() - start).total_seconds() * 1000:.1f}ms")

        # Сохраняем в историю
        self.history.append({
            "query": query,
            "tool": best_tool.name,
            "tool_id": best_tool.id,
            "similarity": best_sim,
            "success": result.get("status") == "success",
            "time_ms": (datetime.now() - start).total_seconds() * 1000
        })

    def _extract_parameters(self, query: str, tool) -> Dict[str, Any]:
        """Извлечение параметров из запроса."""
        params = {}
        q = query.lower()

        # Определяем тип инструмента по категории или имени
        if "weather" in tool.category or "погод" in tool.name.lower():
            cities = ["москва", "питер", "санкт-петербург", "уфа", "казань", "новосибирск", "екатеринбург"]
            for city in cities:
                if city in q:
                    params["city"] = city.capitalize()
                    break

        elif "finance" in tool.category or "валют" in tool.name.lower() or "доллар" in q or "евро" in q:
            nums = re.findall(r'\d+', query)
            if nums:
                params["amount"] = float(nums[0])

            if "доллар" in q:
                params["from_currency"] = "USD"
            elif "евро" in q:
                params["from_currency"] = "EUR"
            elif "рубл" in q:
                params["from_currency"] = "RUB"

            if "рубл" in q and params.get("from_currency") != "RUB":
                params["to_currency"] = "RUB"
            elif "доллар" in q and params.get("from_currency") != "USD":
                params["to_currency"] = "USD"
            elif "евро" in q and params.get("from_currency") != "EUR":
                params["to_currency"] = "EUR"

        elif "transportation" in tool.category or "рейс" in tool.name.lower() or "билет" in q:
            cities = ["москва", "питер", "лондон", "париж", "берлин", "нью-йорк"]
            found = [c for c in cities if c in q]
            if len(found) >= 2:
                params["origin"] = found[0].capitalize()
                params["destination"] = found[1].capitalize()
            elif len(found) == 1:
                params["destination"] = found[0].capitalize()

        elif "data" in tool.category or "user" in tool.name.lower():
            # Для data-инструментов пытаемся извлечь ID, имя и т.д.
            nums = re.findall(r'\d+', query)
            if nums:
                params["id"] = int(nums[0])

            words = q.split()
            for word in words:
                if len(word) > 3 and word not in ["какой", "какая", "какое", "сколько", "найди", "покажи"]:
                    params["name"] = word.capitalize()
                    break

        return params

    async def interactive_mode(self, tool_limit: int = 50):
        """Интерактивный режим с вводом запросов пользователем."""
        await self.initialize(tool_limit=tool_limit)
        await self.connect_client()

        print("\n" + "=" * 70)
        print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ (MCP)")
        print("=" * 70)
        print("Команды:")
        print("  • запрос - поиск и вызов инструмента")
        print("  • tools - список инструментов (первые 20)")
        print("  • stats - статистика")
        print("  • threshold [значение] - изменить порог релевантности")
        print("  • search [текст] - поиск инструментов по тексту")
        print("  • exit - выход")
        print("=" * 70)

        while True:
            try:
                cmd = input("\n> ").strip()

                if cmd.lower() in ('exit', 'quit', 'выход'):
                    break

                if cmd == 'tools':
                    tools = self.mcp_client.tools_cache
                    print(f"\n📦 Инструменты MCP ({len(tools)}):")
                    for i, t in enumerate(tools[:20], 1):
                        name = t['name']
                        desc = t.get('description', '')[:50] + '...' if len(t.get('description', '')) > 50 else t.get('description', '')
                        print(f"  {i:2d}. {name} - {desc}")
                    if len(tools) > 20:
                        print(f"\n  ... и ещё {len(tools) - 20} инструментов")
                    continue

                if cmd == 'stats':
                    print("\n📊 СТАТИСТИКА")
                    print(f"Запросов обработано: {len(self.history)}")
                    if self.history:
                        succ = sum(1 for h in self.history if h.get('success', False))
                        failed = len(self.history) - succ
                        avg_time = sum(h['time_ms'] for h in self.history) / len(self.history)
                        print(f"Успешных: {succ}")
                        print(f"Неудачных: {failed}")
                        print(f"Среднее время: {avg_time:.1f}ms")
                        print(f"Текущий порог: {self.relevance_threshold}")

                    mcp_stats = self.mcp_server.get_stats()
                    print(f"\nMCP сервер:")
                    print(f"  Инструментов: {mcp_stats['tools']}")
                    print(f"  Вызовов всего: {mcp_stats['total_calls']}")
                    print(f"  Успешных вызовов: {mcp_stats['successful_calls']}")
                    print(f"  Успешность: {mcp_stats['success_rate'] * 100:.1f}%")
                    continue

                if cmd.startswith('threshold '):
                    try:
                        new_threshold = float(cmd.split()[1])
                        if 0 <= new_threshold <= 1:
                            self.relevance_threshold = new_threshold
                            print(f"✅ Порог релевантности изменён на {new_threshold}")
                        else:
                            print("❌ Порог должен быть между 0 и 1")
                    except ValueError:
                        print("❌ Использование: threshold [значение от 0 до 1]")
                    continue

                if cmd.startswith('search '):
                    search_text = cmd[7:].strip()
                    if search_text:
                        results = self.searcher.search(search_text, top_k=5)
                        print(f"\n🔍 Результаты поиска для '{search_text}':")
                        for i, r in enumerate(results, 1):
                            tool = r["tool"]
                            sim = r["similarity"]
                            print(f"  {i}. {tool.name} [{tool.category}] - сходство: {sim:.3f}")
                    else:
                        print("❌ Укажите текст для поиска")
                    continue

                if cmd:
                    await self.process_query(cmd)

            except KeyboardInterrupt:
                print("\n\nЗавершение работы...")
                break
            except Exception as e:
                logger.error(f"Ошибка в интерактивном режиме: {e}")
                print(f"❌ Произошла ошибка: {e}")

        # Отключаем клиент при выходе
        if self.mcp_client:
            await self.mcp_client.disconnect()

    async def demo_mode(self, tool_limit: int = 50, categories: Optional[list] = None):
        """Демонстрационный режим с фиксированными тестовыми запросами."""
        await self.initialize(tool_limit=tool_limit, categories=categories)
        await self.connect_client()

        test_queries = [
            "Конвертируй 100 долларов в рубли",
            "Какая погода в Москве?",
            "Найди рейсы из Москвы в Лондон",
            "Сколько будет 2+2?",  # Неподходящий запрос
            "Какой сегодня день?",  # Неподходящий запрос
            "Покажи котиков"        # Неподходящий запрос
        ]

        print("\n" + "=" * 70)
        print("🎯 ДЕМОНСТРАЦИЯ MCP С TOOLBENCH")
        print("=" * 70)

        for i, q in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] ", end="")
            await self.process_query(q)
            print("\n" + "-" * 50)
            await asyncio.sleep(0.5)  # Небольшая пауза между запросами

        # Итоговая статистика
        print("\n" + "=" * 70)
        print("📊 ИТОГОВАЯ СТАТИСТИКА ДЕМО")
        print("=" * 70)
        succ = sum(1 for h in self.history if h.get('success', False))
        total = len(self.history)
        print(f"Всего запросов: {total}")
        print(f"Успешных: {succ}")
        print(f"Неудачных: {total - succ}")
        if total > 0:
            print(f"Процент успеха: {succ/total*100:.1f}%")

        # Отключаем клиент
        if self.mcp_client:
            await self.mcp_client.disconnect()


async def main():
    """Точка входа с разбором аргументов командной строки."""
    parser = argparse.ArgumentParser(description="MCP-Tool-RL с интеграцией ToolBench")
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo',
                        help='Режим работы: demo (тестовые запросы) или interactive (интерактивный)')
    parser.add_argument('--limit', type=int, default=50,
                        help='Количество инструментов для загрузки (по умолчанию 50, макс 1000)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Порог релевантности (от 0 до 1)')
    parser.add_argument('--categories', nargs='+',
                        help='Фильтр по категориям (например, weather finance)')

    args = parser.parse_args()

    # Ограничиваем лимит для предотвращения перегрузки
    if args.limit > 1000:
        print("⚠️ Лимит ограничен 1000 инструментов для стабильности")
        args.limit = 1000

    system = MCPToolRLSystem(relevance_threshold=args.threshold)

    try:
        if args.mode == 'demo':
            await system.demo_mode(tool_limit=args.limit, categories=args.categories)
        else:  # interactive
            await system.interactive_mode(tool_limit=args.limit)
    except KeyboardInterrupt:
        print("\n\n👋 Завершение работы по запросу пользователя")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Произошла критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Гарантированное отключение клиента
        if hasattr(system, 'mcp_client') and system.mcp_client:
            try:
                await system.mcp_client.disconnect()
            except:
                pass
        print("\n👋 Программа завершена")


if __name__ == "__main__":
    asyncio.run(main())