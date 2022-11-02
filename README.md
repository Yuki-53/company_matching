# company_matching

Решение позволяет сопоставлять название компании с базой данных, например, для поиска информации о предыдущих взаимодействиях.

После ввода названия компании из базы данных выдается список топ k наиболее похожих компаний.


Для настройки среды: `pip install -r ./requirements.txt`

Структура репозитория:

`./reference/description.md` - подробное описание

`./experiments/` - директория с ipython ноутбуками

`./main.py` - основной скрипт для демонстрации работы

`./utils.py` - вспомогательные функции

`./requirements.txt` - файл с зависимостями

`./catboost_model.bin` - веса модели

Ссылка на данные и модель https://drive.google.com/drive/folders/1ZyMQxfISZcS6uanY5zbKYdPgMF21Frco?usp=share_link
