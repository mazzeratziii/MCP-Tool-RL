# MCP-Tool-RL
Система для обучения LLM выбору инструментов с учётом состояния сети.

# Создание активации виртуального окружения
 **Windows** \
`py -3.10 -m venv venv` \
`.\venv\Scripts\Activate.ps1`

 **Linux/Mac**\
`python3.10 -m venv venv` \
`source venv/bin/activate`

# Установка зависимостей 
 **Обновление pip** \
`python -m pip install --upgrade pip`

**Установка пакетов** \
`pip install -r requirements.txt`

# Обучение модели
**Базовое обучение (20 эпох)**\
`python main.py --mode train --epochs 20`

**Увеличить до 40 эпох для лучшего результата**\
`python main.py --mode train --epochs 40`

**С загрузкой конкретного чекпоинта**\
`python main.py --mode train --epochs 20 --checkpoint checkpoints/epoch_20`

# Интерактивный режим(тестирование)
**С лучшей моделью**\
`python main.py --mode interactive --checkpoint checkpoints/epoch_40`

**Без указания чекпоинта (базовая модель)** \
`python main.py --mode interactive`

# Оценка модели
`python main.py --mode evaluate --checkpoint checkpoints/epoch_40`