"""Main menu (FR-16, stage 1) and small keyboard helpers."""

from __future__ import annotations

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


class Menu:
    DAY = "📅 Мой день"
    ADD = "➕ Добавить"
    TASKS = "📋 Задачи"
    GOALS = "🎯 Проекты/цели"
    HABITS = "🔁 Привычки"
    WORK = "▶️ Начать работу"
    REPORT = "📊 Отчёт"
    SETTINGS = "⚙️ Настройки"
    MEMORY = "🧠 Память"

    ALL: tuple[str, ...] = (DAY, ADD, TASKS, GOALS, HABITS, WORK, REPORT, SETTINGS, MEMORY)


def main_menu() -> ReplyKeyboardMarkup:
    rows = [
        [
            KeyboardButton(text=Menu.DAY),
            KeyboardButton(text=Menu.ADD),
            KeyboardButton(text=Menu.TASKS),
        ],
        [
            KeyboardButton(text=Menu.WORK),
            KeyboardButton(text=Menu.GOALS),
            KeyboardButton(text=Menu.HABITS),
        ],
        [
            KeyboardButton(text=Menu.REPORT),
            KeyboardButton(text=Menu.MEMORY),
            KeyboardButton(text=Menu.SETTINGS),
        ],
    ]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True, is_persistent=True)
