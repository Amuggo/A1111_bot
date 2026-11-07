"""
Тонкий слой над aiogram 3 — всё остальное в проекте общается с Telegram через него.

Зачем он нужен. Бот состоит из ~100 клавиатур и ~200 обращений к Telegram. aiogram строит клавиатуры из
готовых списков рядов (``InlineKeyboardMarkup(inline_keyboard=[[...]])``), а в коде проекта они собираются
по кнопке (``markup.add(...)``, ``markup.row(...)``). Здесь эти построители повторены поверх типов aiogram,
поэтому структура клавиатур и их callback_data остались прежними, а для Telegram по-прежнему отправляются
настоящие ``aiogram.types.InlineKeyboardMarkup`` / ``InlineKeyboardButton``.

Методы ``BotAPI`` — это обычные вызовы ``aiogram.Bot`` (send_message, edit_message_text, send_photo, ...),
только с привычным порядком аргументов; ответом на сообщение служит ``ReplyParameters``.
"""
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Union

from aiogram import Bot
from aiogram.types import LabeledPrice
from aiogram.types import (
    BotCommand as _BotCommand,
    BufferedInputFile,
    InlineKeyboardButton as _InlineKeyboardButton,
    InlineKeyboardMarkup as _InlineKeyboardMarkup,
    KeyboardButton as _KeyboardButton,
    Message,
    ReplyKeyboardMarkup as _ReplyKeyboardMarkup,
    ReplyParameters,
)


# ===== Клавиатуры =====
class InlineKeyboardMarkup:
    """Построитель инлайн-клавиатуры: add() раскладывает кнопки по row_width в ряд, row() — один ряд."""

    def __init__(self, row_width: int = 3):
        self.row_width = max(1, row_width)
        self._rows: List[List[_InlineKeyboardButton]] = []

    def add(self, *buttons: _InlineKeyboardButton) -> "InlineKeyboardMarkup":
        for i in range(0, len(buttons), self.row_width):
            self._rows.append(list(buttons[i:i + self.row_width]))
        return self

    def row(self, *buttons: _InlineKeyboardButton) -> "InlineKeyboardMarkup":
        if buttons:
            self._rows.append(list(buttons))
        return self

    def build(self) -> _InlineKeyboardMarkup:
        return _InlineKeyboardMarkup(inline_keyboard=[list(r) for r in self._rows])

    @property
    def keyboard(self) -> List[List[_InlineKeyboardButton]]:
        return self._rows


def InlineKeyboardButton(text: str, callback_data: Optional[str] = None, url: Optional[str] = None) -> _InlineKeyboardButton:
    return _InlineKeyboardButton(text=text, callback_data=callback_data, url=url)


class ReplyKeyboardMarkup:
    """Построитель обычной (нижней) клавиатуры."""

    def __init__(self, resize_keyboard: bool = False, row_width: int = 3, one_time_keyboard: bool = False):
        self.resize_keyboard = resize_keyboard
        self.one_time_keyboard = one_time_keyboard
        self.row_width = max(1, row_width)
        self._rows: List[List[_KeyboardButton]] = []

    def add(self, *buttons: _KeyboardButton) -> "ReplyKeyboardMarkup":
        for i in range(0, len(buttons), self.row_width):
            self._rows.append(list(buttons[i:i + self.row_width]))
        return self

    def row(self, *buttons: _KeyboardButton) -> "ReplyKeyboardMarkup":
        if buttons:
            self._rows.append(list(buttons))
        return self

    def build(self) -> _ReplyKeyboardMarkup:
        return _ReplyKeyboardMarkup(
            keyboard=[list(r) for r in self._rows],
            resize_keyboard=self.resize_keyboard,
            one_time_keyboard=self.one_time_keyboard,
        )


def KeyboardButton(text: str) -> _KeyboardButton:
    return _KeyboardButton(text=text)


def BotCommand(command: str, description: str) -> _BotCommand:
    return _BotCommand(command=command, description=description)


# Пространство имён, совместимое с прежним `from telebot import types`
types = SimpleNamespace(
    InlineKeyboardMarkup=InlineKeyboardMarkup,
    InlineKeyboardButton=InlineKeyboardButton,
    ReplyKeyboardMarkup=ReplyKeyboardMarkup,
    KeyboardButton=KeyboardButton,
    BotCommand=BotCommand,
)


def _markup(reply_markup: Any) -> Any:
    """Построитель → готовая aiogram-клавиатура; готовые объекты aiogram проходят как есть."""
    return reply_markup.build() if hasattr(reply_markup, "build") else reply_markup


def _reply_parameters(reply_to_message_id: Optional[int]) -> Optional[ReplyParameters]:
    if reply_to_message_id is None:
        return None
    return ReplyParameters(message_id=reply_to_message_id, allow_sending_without_reply=True)


def _input_file(data: Union[bytes, bytearray, Any], default_name: str) -> BufferedInputFile:
    """bytes / BytesIO / BufferedInputFile → BufferedInputFile (имя файла берётся из атрибута .name)."""
    if isinstance(data, BufferedInputFile):
        return data
    if isinstance(data, (bytes, bytearray)):
        return BufferedInputFile(bytes(data), filename=default_name)
    name = getattr(data, "name", None) or default_name
    payload = data.getvalue() if hasattr(data, "getvalue") else (data.seek(0) or data.read())
    return BufferedInputFile(payload, filename=str(name))


class BotAPI:
    """Обёртка над aiogram.Bot с привычными сигнатурами методов (все методы — корутины)."""

    def __init__(self, bot: Bot):
        self.bot = bot

    async def send_message(self, chat_id: int, text: str, parse_mode: Optional[str] = None,
                           reply_markup: Any = None, reply_to_message_id: Optional[int] = None) -> Message:
        return await self.bot.send_message(
            chat_id=chat_id, text=text, parse_mode=parse_mode,
            reply_markup=_markup(reply_markup), reply_parameters=_reply_parameters(reply_to_message_id),
        )

    async def reply_to(self, message: Message, text: str, parse_mode: Optional[str] = None,
                       reply_markup: Any = None) -> Message:
        return await self.send_message(message.chat.id, text, parse_mode=parse_mode, reply_markup=reply_markup,
                                       reply_to_message_id=message.message_id)

    async def edit_message_text(self, text: str, chat_id: int, message_id: int,
                                parse_mode: Optional[str] = None, reply_markup: Any = None) -> Any:
        return await self.bot.edit_message_text(
            text=text, chat_id=chat_id, message_id=message_id, parse_mode=parse_mode,
            reply_markup=_markup(reply_markup),
        )

    async def edit_message_reply_markup(self, chat_id: int, message_id: int, reply_markup: Any = None) -> Any:
        return await self.bot.edit_message_reply_markup(
            chat_id=chat_id, message_id=message_id, reply_markup=_markup(reply_markup),
        )

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        return await self.bot.delete_message(chat_id=chat_id, message_id=message_id)

    async def answer_callback_query(self, callback_query_id: str, text: Optional[str] = None,
                                    show_alert: bool = False) -> bool:
        return await self.bot.answer_callback_query(
            callback_query_id=callback_query_id, text=text, show_alert=show_alert or None,
        )

    async def send_photo(self, chat_id: int, photo: Any, caption: Optional[str] = None,
                         parse_mode: Optional[str] = None, reply_markup: Any = None,
                         reply_to_message_id: Optional[int] = None) -> Message:
        return await self.bot.send_photo(
            chat_id=chat_id, photo=_input_file(photo, "image.png"), caption=caption, parse_mode=parse_mode,
            reply_markup=_markup(reply_markup), reply_parameters=_reply_parameters(reply_to_message_id),
        )

    async def send_document(self, chat_id: int, document: Any, caption: Optional[str] = None,
                            parse_mode: Optional[str] = None, reply_markup: Any = None,
                            reply_to_message_id: Optional[int] = None) -> Message:
        return await self.bot.send_document(
            chat_id=chat_id, document=_input_file(document, "file.png"), caption=caption, parse_mode=parse_mode,
            reply_markup=_markup(reply_markup), reply_parameters=_reply_parameters(reply_to_message_id),
        )

    async def get_file(self, file_id: str) -> Any:
        return await self.bot.get_file(file_id)

    async def download_file(self, file_path: str) -> bytes:
        """Скачивает файл Telegram и возвращает его байты (aiogram отдаёт BytesIO)."""
        buffer = await self.bot.download_file(file_path)
        return buffer.getvalue() if buffer is not None else b""

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        return await self.bot.send_chat_action(chat_id=chat_id, action=action)

    async def get_me(self) -> Any:
        return await self.bot.get_me()

    async def send_invoice(self, chat_id: int, title: str, description: str, payload: str, stars: int,
                           label: str) -> Message:
        """Счёт в Telegram Stars (валюта XTR; платёжный провайдер не нужен)."""
        return await self.bot.send_invoice(
            chat_id=chat_id, title=title, description=description, payload=payload, currency="XTR",
            prices=[LabeledPrice(label=label, amount=int(stars))],
        )

    async def refund_star_payment(self, user_id: int, charge_id: str) -> Any:
        return await self.bot.refund_star_payment(user_id=user_id, telegram_payment_charge_id=charge_id)

    async def set_my_commands(self, commands: Iterable[_BotCommand]) -> bool:
        return await self.bot.set_my_commands(list(commands))
