# src/prompts.py

def get_dynamic_prompt(query, tools):
    """Создает промпт с актуальным списком инструментов и примерами"""

    # Формируем список доступных инструментов
    tools_text = ""
    tool_names = []
    for i, tool in enumerate(tools[:5], 1):
        tool_names.append(tool['name'])
        tools_text += f"{i}. {tool['name']}\n"
        tools_text += f"   Описание: {tool['description']}\n"
        tools_text += f"   Категория: {tool['category']}\n\n"

    # Создаем примеры на основе реальных инструментов
    examples = []
    for tool in tools[:3]:  # Берем первые 3 инструмента для примеров
        if 'search' in tool['name'].lower():
            examples.append(('найди информацию про искусственный интеллект', tool['name']))
        elif 'weather' in tool['name'].lower():
            examples.append(('погода в москве', tool['name']))
        elif 'calc' in tool['name'].lower():
            examples.append(('2 + 2', tool['name']))
        elif 'database' in tool['name'].lower() or 'db' in tool['name'].lower():
            examples.append(('найди пользователя с id 123', tool['name']))

    examples_text = ""
    for user_q, tool_name in examples:
        examples_text += f"User: {user_q}\nAssistant: <tool_call>{tool_name}</tool_call>\n\n"

    prompt = f"""You are an AI assistant that helps users by calling the appropriate tools.
You MUST respond ONLY with <tool_call>tool_name</tool_call> format, nothing else.

Available tools:

{tools_text}

Examples of correct responses:

{examples_text}

IMPORTANT RULES:
1. Respond ONLY with <tool_call>tool_name</tool_call>
2. tool_name must be EXACTLY one of these tools: {', '.join(tool_names)}
3. Do NOT add any other text, explanations, or punctuation
4. NEVER respond with "tool_name" - use the actual tool names from the list
5. If no tool is appropriate, respond with <tool_call>none</tool_call>

User: {query}
Assistant: """
    return prompt


def get_strict_prompt(query, tools):
    """Жесткий промпт для случаев, когда модель не следует инструкциям"""
    tool_names = [t['name'] for t in tools[:5]]
    return f"""CRITICAL: You MUST choose ONE tool from this EXACT list: {', '.join(tool_names)}

User: {query}
Assistant: <tool_call>"""


SYSTEM_PROMPT = """You are an AI assistant that helps users by calling the appropriate tools.
Always respond with <tool_call>tool_name</tool_call> format."""


def get_evaluation_prompt(query, tools):
    """Промпт для оценки без обучения"""
    tools_list = "\n".join([f"- {t['name']}: {t['description']}" for t in tools[:5]])
    return f"""Available tools:
{tools_list}

User: {query}
Assistant: """