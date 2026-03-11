# src/tools/tool_selector.py
import re
from collections import Counter


class ToolSelector:
    """Интеллектуальный выбор инструментов на основе запроса"""

    # Категории инструментов и их ключевые слова
    CATEGORIES = {
        'sports': {
            'keywords': ['nba', 'nfl', 'mlb', 'soccer', 'football', 'basketball', 'baseball',
                         'player', 'team', 'score', 'game', 'match', 'tournament', 'championship',
                         'league', 'sports', 'athlete', 'stats', 'statistics', 'nhl', 'fifa',
                         'olympics', 'tennis', 'golf', 'cricket', 'rugby', 'boxing'],
            'tools': []
        },
        'crypto': {
            'keywords': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
                         'blockchain', 'nft', 'token', 'wallet', 'mining', 'coin', 'altcoin',
                         'defi', 'web3', 'metamask', 'binance', 'coinbase', 'solana', 'cardano'],
            'tools': []
        },
        'finance': {
            'keywords': ['stock', 'market', 'price', 'trading', 'forex', 'currency', 'dollar',
                         'euro', 'usd', 'eur', 'exchange', 'rate', 'financial', 'investment',
                         'bank', 'interest', 'loan', 'mortgage', 'credit', 'debt', 'asset'],
            'tools': []
        },
        'music': {
            'keywords': ['song', 'music', 'artist', 'album', 'spotify', 'shazam', 'lyrics',
                         'playlist', 'track', 'singer', 'band', 'concert', 'festival', 'genre',
                         'rock', 'pop', 'jazz', 'classical', 'hip-hop', 'rap'],
            'tools': []
        },
        'food': {
            'keywords': ['restaurant', 'food', 'recipe', 'cafe', 'menu', 'dinner', 'lunch',
                         'breakfast', 'meal', 'cooking', 'kitchen', 'cuisine', 'chef', 'bakery',
                         'pizza', 'burger', 'pasta', 'sushi', 'vegan', 'vegetarian'],
            'tools': []
        },
        'technology': {
            'keywords': ['tech', 'software', 'hardware', 'gpu', 'cpu', 'computer', 'laptop',
                         'phone', 'smartphone', 'app', 'application', 'digital', 'coding',
                         'programming', 'python', 'javascript', 'api', 'database', 'server'],
            'tools': []
        },
        'weather': {
            'keywords': ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sun', 'cloud',
                         'climate', 'humidity', 'wind', 'storm', 'hurricane', 'tornado',
                         'meteorology', 'precipitation', 'atmosphere'],
            'tools': []
        },
        'travel': {
            'keywords': ['hotel', 'flight', 'travel', 'trip', 'vacation', 'tour', 'destination',
                         'city', 'country', 'airport', 'booking', 'reservation', 'holiday',
                         'tourism', 'sightseeing', 'landmark', 'attraction'],
            'tools': []
        },
        'education': {
            'keywords': ['learn', 'course', 'class', 'tutorial', 'education', 'school',
                         'university', 'college', 'student', 'teacher', 'professor', 'academic',
                         'study', 'knowledge', 'training', 'workshop', 'seminar'],
            'tools': []
        },
        'health': {
            'keywords': ['health', 'medical', 'doctor', 'hospital', 'clinic', 'patient',
                         'treatment', 'medicine', 'drug', 'fitness', 'workout', 'exercise',
                         'diet', 'nutrition', 'wellness', 'therapy', 'covid', 'vaccine'],
            'tools': []
        },
        'gaming': {
            'keywords': ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam',
                         'pc game', 'console', 'multiplayer', 'rpg', 'fps', 'rts', 'mmo',
                         'minecraft', 'fortnite', 'cod', 'league of legends', 'dota'],
            'tools': []
        },
        'social': {
            'keywords': ['social', 'facebook', 'twitter', 'instagram', 'tiktok', 'linkedin',
                         'youtube', 'reddit', 'discord', 'telegram', 'whatsapp', 'messenger',
                         'post', 'tweet', 'share', 'like', 'comment', 'follower'],
            'tools': []
        },
        'news': {
            'keywords': ['news', 'headline', 'article', 'journal', 'media', 'press',
                         'newspaper', 'magazine', 'blog', 'update', 'breaking', 'latest'],
            'tools': []
        }
    }

    def __init__(self, all_tools):
        """Инициализация с полным списком инструментов"""
        print(f"   Загружено {len(all_tools)} инструментов для категоризации")
        self.all_tools = all_tools
        self._categorize_tools()

    def _categorize_tools(self):
        """Распределение инструментов по категориям на основе их названий и описаний"""
        categorized = 0

        for tool in self.all_tools:
            tool_text = f"{tool['name']} {tool.get('description', '')}".lower()
            categorized_flag = False

            for category, data in self.CATEGORIES.items():
                for keyword in data['keywords']:
                    if keyword in tool_text:
                        data['tools'].append(tool)
                        categorized_flag = True
                        break
                if categorized_flag:
                    break

            if categorized_flag:
                categorized += 1

        print(f"   Категоризировано {categorized} инструментов")

    def select_tools_for_query(self, query, num_tools=20):
        """Выбор инструментов, наиболее подходящих для запроса"""
        query_lower = query.lower()

        # Определяем категории запроса
        query_categories = []
        category_scores = {}

        for category, data in self.CATEGORIES.items():
            score = 0
            for keyword in data['keywords']:
                if keyword in query_lower:
                    score += 1
            if score > 0:
                query_categories.append(category)
                category_scores[category] = score

        # Сортируем категории по релевантности
        query_categories.sort(key=lambda x: category_scores[x], reverse=True)

        if query_categories:
            print(f"   Определены категории: {', '.join(query_categories[:3])}")

        selected_tools = []
        used_tools = set()

        # Сначала берем инструменты из определенных категорий
        tools_per_category = max(3, num_tools // (len(query_categories) or 1))

        for category in query_categories:
            tools = self.CATEGORIES[category]['tools']
            # Сортируем инструменты по длине описания (более информативные)
            sorted_tools = sorted(tools, key=lambda x: len(x.get('description', '')), reverse=True)

            count = 0
            for tool in sorted_tools:
                if tool['name'] not in used_tools:
                    selected_tools.append(tool)
                    used_tools.add(tool['name'])
                    count += 1
                    if count >= tools_per_category:
                        break

        # Если не хватает, добавляем популярные инструменты из других категорий
        if len(selected_tools) < num_tools:
            remaining = num_tools - len(selected_tools)
            # Берем инструменты с самыми длинными описаниями
            all_sorted = sorted(self.all_tools, key=lambda x: len(x.get('description', '')), reverse=True)

            for tool in all_sorted:
                if tool['name'] not in used_tools:
                    selected_tools.append(tool)
                    used_tools.add(tool['name'])
                    remaining -= 1
                    if remaining == 0:
                        break

        return selected_tools[:num_tools]

    def print_category_stats(self):
        """Вывод статистики по категориям"""
        print("\n📊 СТАТИСТИКА ИНСТРУМЕНТОВ ПО КАТЕГОРИЯМ:")

        # Сортируем категории по количеству инструментов
        sorted_categories = sorted(
            self.CATEGORIES.items(),
            key=lambda x: len(x[1]['tools']),
            reverse=True
        )

        for category, data in sorted_categories:
            count = len(data['tools'])
            if count > 0:
                percentage = (count / len(self.all_tools)) * 100
                bar = "█" * int(percentage / 2)
                print(f"   {category:15} {count:5} ({percentage:5.1f}%) {bar}")

        total_categorized = sum(len(data['tools']) for data in self.CATEGORIES.values())
        print(f"\n   Всего категоризировано: {total_categorized}/{len(self.all_tools)}")
        print(f"   Не категоризировано: {len(self.all_tools) - total_categorized}")