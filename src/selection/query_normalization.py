from __future__ import annotations

import re


RU_KEYWORD_EXPANSIONS = {
    "погода": "weather current weather temperature forecast",
    "температура": "weather temperature current weather",
    "прогноз": "weather forecast",
    "ветер": "weather wind",
    "влажность": "weather humidity",
    "давление": "weather pressure",
    "переведи": "translate translation language",
    "перевод": "translate translation language",
    "найди": "search find lookup",
    "поиск": "search find lookup",
    "рассчитай": "calculate calculator arithmetic",
    "посчитай": "calculate calculator arithmetic",
    "топ": "top ranking trending popular",
    "лучшие": "top best ranking",
    "нфт": "nft non fungible token top nft sales collections ranking",
    "статья": "article articles news search",
    "статьи": "article articles news search",
    "новости": "news search latest articles",
    "ии": "ai artificial intelligence",
    "сок": "juice drink beverage product search grocery",
    "напиток": "drink beverage product search grocery",
    "товар": "product item search details",
    "продукт": "product item search details",
}

RU_CITY_TRANSLITERATIONS = {
    "москва": "moscow",
    "уфа": "ufa",
    "казань": "kazan",
    "самара": "samara",
    "спб": "saint petersburg",
    "питер": "saint petersburg",
    "санкт": "saint petersburg",
    "екатеринбург": "yekaterinburg",
    "новосибирск": "novosibirsk",
    "сочи": "sochi",
    "добрый": "dobry dobrui",
}


def expand_query_for_retrieval(query: str) -> str:
    """Add English retrieval hints for short Russian user queries."""
    if not query:
        return query

    q = _normalize_common_typos(query.lower())
    q_compact = _collapse_repeated_letters(q)
    additions = []

    for ru_term, expansion in RU_KEYWORD_EXPANSIONS.items():
        if ru_term in q or ru_term in q_compact:
            additions.append(expansion)

    words = set(re.findall(r"[а-яё]+", q))
    words.update(re.findall(r"[а-яё]+", q_compact))
    for ru_city, latin_city in RU_CITY_TRANSLITERATIONS.items():
        if ru_city in words:
            additions.append(latin_city)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def is_too_ambiguous_query(query: str) -> bool:
    """?????????, ??????? ?? ???????? ??? ????????????? ??????."""
    tokens = re.findall(r"[\wа-яё]+", query.lower(), flags=re.IGNORECASE)
    if not tokens:
        return True
    meaningful = [token for token in tokens if len(token) > 2]
    return len(meaningful) == 1 and len(meaningful[0]) <= 4


def _collapse_repeated_letters(value: str) -> str:
    """?????????? ????????????? ????? ? ??????."""
    return re.sub(r"([a-zа-яё])\1+", r"\1", value, flags=re.IGNORECASE)


def _normalize_common_typos(value: str) -> str:
    """?????????? ?????? ???????? ????? retrieval."""
    replacements = {
        "нтф": "нфт",
        "тнф": "нфт",
        "тны": "нфт",
        "тнв": "нфт",
    }
    for typo, fixed in replacements.items():
        value = value.replace(typo, fixed)
    return value
