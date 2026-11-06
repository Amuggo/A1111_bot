import bootstrap_env  # noqa: F401,E402  — ДОЛЖЕН идти первым: переключает скрипт на библиотеки из папки env/
import asyncio
import base64
import hashlib
import html
import io
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress
from typing import Optional

import pandas as pd
from PIL import Image

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from tg import BotAPI, types

try:
    import segmentation
except Exception as _seg_error:  # ImportError, а также AttributeError у новых версий mediapipe (нет mp.solutions)
    segmentation = None
    print(f"⚠️ Модуль segmentation недоступен ({_seg_error}). Редактирование тела/фона/позы будет недоступно.")

sys.path.insert(0, os.path.dirname(__file__))

from Aapi import ( 
    preprocess_prompt,
    generate_image, generate_txt2img_advanced, check_api_available,
    upscale_image, generate_img2img, swap_face_with_reactor,
    generate_inpaint
)
# Импортируем ControlNet-функции из нового модуля
from controlnet_manager import (
    get_controlnet_sources,
    get_controlnet_preview_by_source,
    generate_txt2img_controlnet,
    generate_controlnet
)
from settings import (
    get_user_settings, save_user_settings, get_user_negative_prompt,
    DEFAULT_SETTINGS, DEFAULT_NEGATIVE_PROMPT, get_diversity_mode,
    get_prompt_processing_enabled, set_prompt_processing_enabled,
    set_diversity_mode, set_user_negative_prompt,
    get_user_lora_enabled, set_user_lora_enabled, get_available_loras,
    set_available_loras, fetch_loras_from_api,
    get_user_lora_weights, cycle_user_lora_weight,
    get_available_models, refresh_checkpoints_on_server, get_user_model,
    set_user_model, get_default_model, get_default_model_name,
    update_user_settings, get_available_embeddings, refresh_embeddings_on_server,
    get_user_negative_embeddings, toggle_user_negative_embedding, set_user_negative_embeddings,
    get_available_samplers, get_user_sampler, set_user_sampler, get_active_lora_tags,
    hires_plan, HIRES_MAX_SIDE, MAX_PRESETS, list_presets, save_preset, apply_preset, delete_preset
)
from face_manager import (
    save_user_face, get_user_faces, get_active_face, set_active_face,
    delete_face, get_face_preview, get_face_stats,
    check_reactor_available as check_reactor,
    get_face_enabled, set_face_enabled, get_active_face_name,
    FACE_MODES, FACE_MODE_LABELS, get_face_mode, set_face_mode, build_face_model,
    get_face_context, apply_face_swap, face_features_available,
    get_face_weights, cycle_face_weight, rename_face
)
from face_check import check_face_photo

from task_queue import QueueFull, Task, TaskQueue  # noqa: E402
from http_client import close_session  # noqa: E402
import app_config as config  # noqa: E402  (загружает .env и переменные окружения)
import prompt_logger  # noqa: E402  (логирование промптов пользователей)

log = logging.getLogger("bot")

# Токен берётся из переменной окружения TELEGRAM_BOT_TOKEN (см. .env.example).
# НЕ храните токен в коде — при попадании в git его придётся отзывать через @BotFather!
BOT_TOKEN = config.BOT_TOKEN or os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    # Уточняем причину: возможно, .env существует, но переменная в нём не заполнена
    from app_config import _find_dotenv_files
    env_files = _find_dotenv_files()
    if env_files:
        hint = (
            f"   Найден файл {env_files[0]}, но TELEGRAM_BOT_TOKEN в нём не задан.\n"
            "   Проверьте строку вида:  TELEGRAM_BOT_TOKEN=123456:ABC...\n"
            "   (без пробелов вокруг '=', без кавычек-двойников целиком вместо значения)"
        )
    else:
        hint = (
            "   Файл .env не найден ни рядом с кодом (code/.env), ни в корне проекта,\n"
            "   ни в текущей директории. Скопируйте .env.example в .env и заполните токен,\n"
            "   либо задайте переменную:  export TELEGRAM_BOT_TOKEN=\"123456:ABC...\""
        )
    print(
        "❌ Переменная окружения TELEGRAM_BOT_TOKEN не задана.\n"
        f"{hint}\n"
        "   Получить токен: https://t.me/BotFather",
        file=sys.stderr,
    )
    sys.exit(1)

# aiogram: бот, роутер и диспетчер. Всё работает в ОДНОМ потоке на цикле событий asyncio — пул потоков
# (BOT_WORKER_THREADS) больше не нужен; параллельность даёт неблокирующий ввод-вывод и очередь задач (task_queue).
_aiogram_bot = Bot(token=BOT_TOKEN)
bot = BotAPI(_aiogram_bot)
bot_api = bot  # то же самое: алиас для хуков запуска
router = Router(name="main")
dp = Dispatcher()

# Очередь задач создаётся при старте (on_startup), когда уже запущен цикл событий
task_queue: Optional[TaskQueue] = None


class RememberUserMiddleware(BaseMiddleware):
    """Запоминает @username авторов сообщений и нажатий кнопок ДО вызова хендлеров — для лога «пользователь : промпт»."""

    async def __call__(self, handler, event, data):
        prompt_logger.remember_user(getattr(event, "from_user", None))
        return await handler(event, data)


dp.message.outer_middleware(RememberUserMiddleware())
dp.callback_query.outer_middleware(RememberUserMiddleware())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SafeStore:
    """
    Словарь с ограничением размера: при переполнении вытесняются самые старые записи
    (иначе last_images росли бы без ограничений и держали в памяти десятки мегабайт на пользователя).
    Бот работает в одном потоке на цикле событий, поэтому отдельные операции со словарём атомарны.
    """

    def __init__(self, maxsize=500):
        self._data = {}
        self._maxsize = maxsize

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data.pop(key, None)  # обновлённая запись становится самой свежей
        self._data[key] = value
        while len(self._data) > self._maxsize:
            self._data.pop(next(iter(self._data)))

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def pop(self, key, *default):
        return self._data.pop(key, *default)

    def setdefault(self, key, default=None):
        if key not in self._data:
            self[key] = default
        return self._data[key]


# Глобальные хранилища данных (потокобезопасные, с ограничением размера)
last_images = SafeStore(maxsize=30)       # BytesIO последних картинок (тяжёлые — держим немного)
last_message_ids = SafeStore()
last_seeds = SafeStore()
last_configs = SafeStore()
temp_data = SafeStore()
pagination_data = SafeStore()
preview_messages = SafeStore()  # user_id -> message_id сообщения с превью

ITEMS_PER_PAGE = 20

# Тексты кнопок главного меню (не должны обрабатываться как промпты)
BUTTON_TEXTS = [
    "🚀 Сгенерировать",
    "⚙️ Настройки",
    "🔍 Апскейл",
    "🎭 Управление лицом",
    "🎮 ControlNet",
]

# ===== ПЕРЕВОД КАТЕГОРИЙ (оставлены здесь, т.к. только для отображения) =====
CATEGORY_TRANSLATION = {
    'NSFW_Squatting': 'Присед',
    'NSFW_standing': 'Стоя',
    'NSFW_suspended': 'Подвешенная',
    'NSFW_split_leg': 'Шпагат',
    'NSFW_sitting': 'Сидя',
    'NSFW_lying': 'Лежа',
    'NSFW_Kneeling': 'На коленях',
    'NSFW_all_fours': 'На четвереньках',
    'metalstocks': 'Металлические стойки',
    'lying': 'Лежа',
    'unknown': 'Неизвестно',
}
# ===== ОПИСАНИЯ ПОЗ ДЛЯ ПРОМПТА (тоже для отображения) =====
POSE_DESCRIPTIONS = {
    'NSFW_Squatting': 'squatting, deep squat, knees bent',
    'NSFW_standing': 'standing upright, full body, straight posture',
    'NSFW_suspended': 'suspended in air, hanging, floating',
    'NSFW_split_leg': 'split legs, wide leg stance, stretching, legs apart',
    'NSFW_sitting': 'sitting, seated, resting',
    'NSFW_lying': 'lying down, horizontal, laying',
    'NSFW_Kneeling': 'kneeling, on knees, kneeling down',
    'NSFW_all_fours': 'on all fours, crawling position',
    'metalstocks': 'in metal stocks, restrained, bound',
    'lying': 'lying down',
    'unknown': 'human pose',
}

REACTOR_AVAILABLE = False  # определяется при старте бота (on_startup)

# =========== ОЧЕРЕДЬ ЗАДАЧ ===========

def _progress_text(task, progress, eta, state=None):
    percent = int(progress * 100)
    bar_length = 20
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    steps = ""
    if state and state.get("sampling_steps"):
        steps = f" — шаг {state.get('sampling_step', 0)}/{state['sampling_steps']}"
    eta_text = f"⏱️ Осталось: {eta:.1f} сек" if eta > 0 else "⏳ Финальная обработка..."
    return f"🎨 **Генерация** (задача №{task.id})\n\n`[{bar}]` {percent}%{steps}\n\n{eta_text}"

def make_callbacks(task):
    """Коллбэки прогресса/статуса для задачи: обновляют её статус-сообщение (с кнопкой «Отменить»)."""
    async def on_progress(progress, eta, state=None):
        await task.edit_status(_progress_text(task, progress, eta, state), parse_mode="Markdown")

    async def on_status(text):
        await task.edit_status(f"🎨 **Генерация** (задача №{task.id})\n\n{text}", parse_mode="Markdown", force=True)

    return on_progress, on_status

async def enqueue(chat_id, user_id, kind, job, reply_to_message_id=None):
    """Ставит задачу в очередь и сразу отвечает «⏳ Задача №N в очереди». Возвращает Task или None (очередь полна)."""
    if task_queue is None:
        await bot.send_message(chat_id, "⚠️ Бот ещё запускается, повторите через несколько секунд.")
        return None
    try:
        return await task_queue.submit(chat_id, user_id, kind, job, reply_to_message_id=reply_to_message_id)
    except QueueFull as error:
        await bot.send_message(chat_id, f"⚠️ {error}", reply_to_message_id=reply_to_message_id)
        return None




# =========== «СЛЕДУЮЩИЙ ШАГ» (замена register_next_step_handler) ===========
# Хендлеры-диалоги («введите промпт», «введите имя лица» ...) ждут ОДНО следующее сообщение чата. В aiogram для этого
# обычно используют FSM; здесь сохранена прежняя простая семантика: следующее сообщение из чата уходит в
# зарегистрированный обработчик (раньше, чем в любые другие), а команда /cancel по-прежнему работает.
pending_input = SafeStore(maxsize=5000)  # chat_id -> (async handler, args)

def register_next_step(chat_id, handler, *args):
    pending_input[chat_id] = (handler, args)

async def _has_pending(message: Message) -> bool:
    return message.chat.id in pending_input and not (message.text or "").startswith("/cancel")

@router.message(_has_pending)
async def _dispatch_pending_input(message: Message):
    entry = pending_input.pop(message.chat.id, None)
    if entry is None:
        return
    handler, args = entry
    await handler(message, *args)


# =========== ОБЩИЕ ХЕЛПЕРЫ ===========

def _h(value):
    """Экранирует текст для parse_mode="HTML" (промпты пользователей могут содержать <, >, &, `, _, *)."""
    return html.escape(str(value), quote=False)

def _cut(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "…"

def _face_caption_line(ctx, swapped) -> str:
    """Строка подписи: как именно было применено лицо (FaceID в генерации и/или ReActor-замена)."""
    if not ctx:
        return ""
    parts = []
    if ctx.use_faceid:
        fw, lw = ctx.weights or (None, None)
        parts.append("FaceID (лицо задано при генерации" + (f", сила {fw}, LoRA {lw}" if fw is not None else "") + ")")
    if swapped:
        parts.append("ReActor-замена по модели" if ctx.model_file else "ReActor-замена")
    return f"🎭 <b>Лицо:</b> {_h(', '.join(parts))}\n" if parts else ""

def _text(message):
    """Текст сообщения или пустая строка (фото, стикер и т.п. раньше роняли обработчики на None.strip())."""
    return (getattr(message, "text", None) or "").strip()

async def _safe_answer(call, text=None, show_alert=False):
    """answer_callback_query, который не падает: на один callback можно ответить только один раз и недолго."""
    try:
        await bot.answer_callback_query(call.id, text, show_alert=show_alert)
    except Exception:
        pass

async def _report_error(chat_id, status_message_id, error):
    """Показывает ошибку на месте статус-сообщения (или новым сообщением, если его нельзя изменить)."""
    text = _cut(f"❌ Ошибка: {error}", 3900)
    try:
        await bot.edit_message_text(text, chat_id, status_message_id)
    except Exception:
        try:
            await bot.send_message(chat_id, text)
        except Exception as send_error:
            print(f"⚠️ Не удалось сообщить об ошибке пользователю: {send_error}")

def _is_cancel_text(text):
    """Кнопка меню или команда вместо ожидаемого текста — отмена текущего шага."""
    return (not text) or text in BUTTON_TEXTS or text.startswith('/')

# --- Отправка изображений: фото или файл ---------------------------------------------------------
# Telegram принимает как «фото» только файлы до 10 МБ, с суммой сторон не более 10000 px и
# соотношением сторон не более 20:1; при этом фото ещё и пережимается. Всё, что не проходит,
# отправляется документом (файлом) без потери качества — раньше такая отправка падала с ошибкой,
# и готовая (часто апскейленная) картинка пропадала.
TG_PHOTO_MAX_BYTES = 10 * 1024 * 1024 - 64 * 1024
TG_PHOTO_MAX_SUM_SIDES = 10000
TG_PHOTO_MAX_RATIO = 20
TG_DOCUMENT_MAX_BYTES = 49 * 1024 * 1024
TG_CAPTION_LIMIT = 1024
TG_MESSAGE_LIMIT = 4096
# Размер вплоть до которого фото отправляется «как фото» можно занизить, чтобы большие картинки
# уходили без пережатия: переменная окружения SEND_AS_FILE_ABOVE_MB (по умолчанию — лимит Telegram).
try:
    _SEND_AS_FILE_ABOVE = min(TG_PHOTO_MAX_BYTES, int(float(os.environ.get("SEND_AS_FILE_ABOVE_MB", "0")) * 1024 * 1024)) or TG_PHOTO_MAX_BYTES
except ValueError:
    _SEND_AS_FILE_ABOVE = TG_PHOTO_MAX_BYTES

def _image_bytes(image):
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "getvalue"):
        return image.getvalue()
    image.seek(0)
    return image.read()

def _image_info(data):
    """(ширина, высота, формат) либо (0, 0, None), если PIL не смог прочитать."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height, (img.format or "").upper()
    except Exception:
        return 0, 0, None

def _photo_limit_reason(data, width, height):
    """Почему изображение нельзя отправить фото (None — можно)."""
    if len(data) > _SEND_AS_FILE_ABOVE:
        return f"вес {len(data) / 1024 / 1024:.1f} МБ"
    if width and height:
        if width + height > TG_PHOTO_MAX_SUM_SIDES:
            return f"разрешение {width}×{height}"
        if max(width, height) / max(1, min(width, height)) > TG_PHOTO_MAX_RATIO:
            return f"пропорции {width}×{height}"
    return None

def _shrink_for_document(data):
    """Если файл больше лимита Telegram на документы (50 МБ) — пересохраняет в JPEG, при необходимости уменьшая."""
    if len(data) <= TG_DOCUMENT_MAX_BYTES:
        return data, None
    try:
        with Image.open(io.BytesIO(data)) as img:
            rgb = img.convert("RGB")
        for scale in (1.0, 0.75, 0.5, 0.35):
            candidate = rgb if scale == 1.0 else rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.LANCZOS)
            for quality in (95, 88, 80):
                buf = io.BytesIO()
                candidate.save(buf, format="JPEG", quality=quality)
                if buf.tell() <= TG_DOCUMENT_MAX_BYTES:
                    return buf.getvalue(), "jpg"
    except Exception as e:
        print(f"⚠️ Не удалось уменьшить файл: {e}")
    return data, None

def _is_parse_error(error):
    return "can't parse entities" in str(error).lower()

def _is_photo_rejected(error):
    text = str(error).lower()
    return any(marker in text for marker in (
        "photo_invalid_dimensions", "photo_save_file_invalid", "image_process_failed",
        "wrong file identifier", "file is too big", "wrong type of the web page content",
        "failed to get http url content", "request entity too large", "photo_ext_invalid",
    ))

async def send_image_smart(chat_id, image, caption=None, parse_mode=None, reply_to_message_id=None,
                     reply_markup=None, filename=None, force_document=False):
    """
    Отправляет картинку как фото, а если она не подходит под лимиты Telegram по весу/разрешению/пропорциям
    (или фото отклонено сервером) — как файл. Возвращает объект отправленного сообщения.

    * подпись длиннее 1024 символов уходит отдельным сообщением (иначе Telegram отклонял всю отправку);
    * если в подписи оказалась некорректная разметка — повторяет отправку обычным текстом.
    """
    data = _image_bytes(image)
    if not data:
        raise Exception("Пустое изображение — нечего отправлять")
    width, height, fmt = _image_info(data)
    reason = "по запросу" if force_document else _photo_limit_reason(data, width, height)
    as_document = bool(reason)
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif"}.get(fmt, "png")

    def make_caption(is_document):
        text = caption or ""
        if is_document and reason and reason != "по запросу":
            text += f"\n📎 Отправлено файлом без сжатия ({reason})"
        return text

    async def send(is_document):
        payload = data
        out_ext = ext
        if is_document:
            payload, new_ext = await asyncio.to_thread(_shrink_for_document, payload)  # декодирование/кодирование — в потоке
            out_ext = new_ext or ext
        text = make_caption(is_document)
        separate_text = None
        if len(text) > TG_CAPTION_LIMIT:
            separate_text, text = text, None
        buf = io.BytesIO(payload)
        buf.name = filename or (f"image_{width}x{height}.{out_ext}" if is_document and width else f"image.{out_ext}")
        kwargs = dict(reply_to_message_id=reply_to_message_id, reply_markup=reply_markup)
        if text:
            kwargs.update(caption=text, parse_mode=parse_mode)
        try:
            if is_document:
                sent = await bot.send_document(chat_id, buf, **kwargs)
            else:
                sent = await bot.send_photo(chat_id, buf, **kwargs)
        except Exception as e:
            if text and parse_mode and _is_parse_error(e):
                kwargs.pop("parse_mode", None)
                kwargs["caption"] = re.sub(r"<[^>]+>", "", html.unescape(text)) if parse_mode == "HTML" else text
                buf.seek(0)
                sent = await bot.send_document(chat_id, buf, **kwargs) if is_document else await bot.send_photo(chat_id, buf, **kwargs)
            else:
                raise
        if separate_text:
            body = _cut(separate_text, TG_MESSAGE_LIMIT)
            try:
                await bot.send_message(chat_id, body, parse_mode=parse_mode, reply_to_message_id=sent.message_id)
            except Exception as e:
                if _is_parse_error(e):
                    plain = re.sub(r"<[^>]+>", "", html.unescape(body)) if parse_mode == "HTML" else body
                    await bot.send_message(chat_id, plain, reply_to_message_id=sent.message_id)
                else:
                    print(f"⚠️ Не удалось отправить подпись отдельным сообщением: {e}")
        return sent

    if as_document:
        print(f"📎 Отправляю файлом ({reason})")
        return await send(True)
    try:
        return await send(False)
    except Exception as e:
        if _is_photo_rejected(e):
            print(f"📎 Telegram отклонил фото ({e}); отправляю файлом")
            reason = "Telegram не принял как фото"
            return await send(True)
        raise

# =========== КЛАВИАТУРЫ ===========

def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_generate = types.KeyboardButton("🚀 Сгенерировать")
    btn_settings = types.KeyboardButton("⚙️ Настройки")
    btn_upscale = types.KeyboardButton("🔍 Апскейл")
    btn_face = types.KeyboardButton("🎭 Управление лицом")
    btn_controlnet = types.KeyboardButton("🎮 ControlNet")
    markup.row(btn_generate, btn_settings)
    markup.row(btn_upscale, btn_face)
    markup.row(btn_controlnet)
    return markup

async def get_settings_inline_keyboard(settings, user_id):
    markup = types.InlineKeyboardMarkup(row_width=2)
    btn_res = types.InlineKeyboardButton(f"📐 Размер: {settings['width']}x{settings['height']}", callback_data="set_res")
    btn_steps = types.InlineKeyboardButton(f"⏱️ Шаги: {settings['steps']}", callback_data="set_steps")
    btn_cfg = types.InlineKeyboardButton(f"🎚️ CFG: {settings['cfg_scale']}", callback_data="set_cfg")
    emb_count = len(await get_user_negative_embeddings(user_id))
    btn_neg = types.InlineKeyboardButton(f"🚫 Негатив" + (f" (🧬{emb_count})" if emb_count else ""), callback_data="set_neg")
    btn_sampler = types.InlineKeyboardButton(f"🎯 Сэмплер: {settings.get('sampler_name', 'DPM++ 2M')}", callback_data="set_sampler")  # <--- НОВАЯ КНОПКА
    diversity_mode = await get_diversity_mode(user_id)
    diversity_text = {
        "auto": "🔄 Разнообразие: авто",
        "high": "🎲 Разнообразие: макс",
        "low": "🎯 Разнообразие: мин",
        "off": "⚪ Разнообразие: выкл"
    }.get(diversity_mode, "🔄 Разнообразие: авто")
    btn_diversity = types.InlineKeyboardButton(diversity_text, callback_data="set_diversity")
    prompt_processing = await get_prompt_processing_enabled(user_id)
    prompt_text = "🤖 Обработка: ✅ Вкл" if prompt_processing else "🤖 Обработка: ❌ Выкл"
    btn_prompt = types.InlineKeyboardButton(prompt_text, callback_data="toggle_prompt_processing")
    btn_lora = types.InlineKeyboardButton("🎛️ LoRA", callback_data="manage_lora")
    model_label = await _model_display_name(user_id)
    if len(model_label) > 30:
        model_label = model_label[:29] + "…"
    btn_model = types.InlineKeyboardButton(f"🧩 Модель: {model_label}", callback_data="set_model")
    markup.add(btn_model)
    markup.add(btn_res)
    markup.add(btn_steps, btn_cfg)
    markup.add(btn_sampler, btn_neg)  # <--- Добавляем кнопку сэмплера
    markup.add(btn_diversity, btn_prompt)
    plan = hires_plan(settings)
    hires_label = f"🔬 Hires: ✅ {plan['scale']}x" if plan else ("🔬 Hires: ⚠️ нет места" if settings.get("hires_enabled") else "🔬 Hires: ❌")
    markup.add(types.InlineKeyboardButton(hires_label, callback_data="set_hires"),
               types.InlineKeyboardButton("📦 Пресеты", callback_data="set_presets"))
    markup.add(btn_lora)
    return markup

def get_resolution_keyboard(current_width, current_height):
    markup = types.InlineKeyboardMarkup(row_width=3)
    resolutions = [
        ("512x512", 512, 512), ("512x768", 512, 768), ("768x512", 768, 512),
        ("768x768", 768, 768), ("512x1024", 512, 1024), ("1024x512", 1024, 512),
        ("768x1024", 768, 1024), ("1024x768", 1024, 768), ("1024x1024", 1024, 1024),
        ("1024x1280", 1024, 1280), ("1280x1024", 1280, 1024)
    ]
    for label, w, h in resolutions:
        if w == current_width and h == current_height:
            label = f"✅ {label}"
        markup.add(types.InlineKeyboardButton(label, callback_data=f"res_{w}_{h}"))
    return markup

def get_cfg_keyboard(current_cfg):
    markup = types.InlineKeyboardMarkup(row_width=4)
    for cfg in [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
        label = f"✅ {cfg}" if cfg == current_cfg else str(cfg)
        markup.add(types.InlineKeyboardButton(label, callback_data=f"cfg_{cfg}"))
    return markup

def get_steps_keyboard(current_steps):
    markup = types.InlineKeyboardMarkup(row_width=5)
    for steps in [10, 15, 20, 25, 28, 30, 35, 40, 45, 50]:
        label = f"✅ {steps}" if steps == current_steps else str(steps)
        markup.add(types.InlineKeyboardButton(label, callback_data=f"steps_{steps}"))
    return markup

def get_upscale_keyboard():
    markup = types.InlineKeyboardMarkup(row_width=3)
    markup.add(
        types.InlineKeyboardButton("2x", callback_data="upscale_2"),
        types.InlineKeyboardButton("3x", callback_data="upscale_3"),
        types.InlineKeyboardButton("4x", callback_data="upscale_4")
    )
    return markup

def get_image_actions_keyboard(seed=None):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("🔍 Увеличить", callback_data="action_upscale"),
        types.InlineKeyboardButton("🎨 Редактировать", callback_data="action_img2img"),
        types.InlineKeyboardButton("🔄 Повторить", callback_data="action_repeat")
    )
    return markup

def get_controlnet_actions_keyboard(seed=None):
    """Клавиатура для действий после генерации ControlNet (с кнопкой 'тот же скелет')"""
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("🔍 Увеличить", callback_data="action_upscale"),
        types.InlineKeyboardButton("🎨 Редактировать", callback_data="action_img2img"),
        types.InlineKeyboardButton("🔄 Повторить", callback_data="action_repeat"),
        types.InlineKeyboardButton("🦴 Тот же скелет", callback_data="action_same_skeleton")
    )
    return markup

# =========== CONTROLNET МЕНЮ С КАТЕГОРИЯМИ И ПАГИНАЦИЕЙ ===========

@router.message(F.text == "🎮 ControlNet")
async def controlnet_menu(message: Message) -> None:
    user_id = str(message.chat.id)
    sources = await get_controlnet_sources()
    if not sources:
        await bot.send_message(message.chat.id, "❌ В папке `controlnet_images` нет изображений или читаемых .parquet.\nДобавьте туда скелеты (PNG, JPG, Parquet) и попробуйте снова.", parse_mode="Markdown")
        return

    # Собираем уникальные категории
    categories = set()
    for src in sources:
        cat = src.get('category', 'unknown')
        categories.add(cat)
    categories = sorted(categories)

    # Сохраняем все источники в pagination_data
    pagination_data[user_id] = {
        'sources': sources,
        'category': None,  # None означает "все категории"
        'page': 0,
        'total_pages': (len(sources) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE,
        'list_message_id': None  # будет хранить ID сообщения со списком
    }

    # Создаём кнопки категорий
    markup = types.InlineKeyboardMarkup(row_width=2)
    for cat in categories:
        # Переводим название категории
        cat_ru = CATEGORY_TRANSLATION.get(cat, cat)
        markup.add(types.InlineKeyboardButton(f"📂 {cat_ru}", callback_data=f"ctrl_cat_{cat}"))

    # Добавляем кнопку "Все категории"
    markup.add(types.InlineKeyboardButton("📋 Все позы", callback_data="ctrl_cat_all"))

    await bot.send_message(
        message.chat.id,
        "🏷️ **Выберите категорию позы:**",
        parse_mode="Markdown",
        reply_markup=markup
    )

@router.callback_query(F.data.startswith("ctrl_cat_"))
async def handle_controlnet_category(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    cat = call.data.replace("ctrl_cat_", "")
    if cat == "all":
        category = None
        cat_display = "Все позы"
    else:
        category = cat
        cat_display = CATEGORY_TRANSLATION.get(cat, cat)

    if user_id not in pagination_data:
        # если данных нет – перезапускаем меню
        await controlnet_menu(call.message)
        await bot.answer_callback_query(call.id)
        return

    sources = pagination_data[user_id]['sources']
    if category is not None:
        filtered = [s for s in sources if s.get('category') == category]
    else:
        filtered = sources

    if not filtered:
        await bot.answer_callback_query(call.id, "❌ В этой категории нет поз")
        return

    # Обновляем данные
    pagination_data[user_id]['category'] = category
    pagination_data[user_id]['page'] = 0
    pagination_data[user_id]['total_pages'] = (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    pagination_data[user_id]['filtered_sources'] = filtered
    # Удаляем сохранённый ID сообщения списка, если был
    pagination_data[user_id]['list_message_id'] = None

    # Удаляем возможное старое превью
    if user_id in preview_messages:
        try:
            await bot.delete_message(call.message.chat.id, preview_messages[user_id])
        except Exception:
            pass
        del preview_messages[user_id]

    # Удаляем сообщение с категориями (текущее)
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass

    await show_controlnet_page(call.message, user_id, 0)
    await bot.answer_callback_query(call.id)

async def show_controlnet_page(message, user_id, page):
    data = pagination_data.get(user_id)
    if not data:
        await bot.send_message(message.chat.id, "❌ Сессия истекла. Нажмите кнопку ControlNet заново.")
        return

    category = data.get('category')
    filtered = data.get('filtered_sources')
    if filtered is None:
        # если нет фильтра – используем все источники
        filtered = data['sources']
        data['filtered_sources'] = filtered

    total_pages = (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    if page < 0 or page >= total_pages:
        page = 0
    data['page'] = page
    data['total_pages'] = total_pages

    start = page * ITEMS_PER_PAGE
    end = min(start + ITEMS_PER_PAGE, len(filtered))
    page_sources = filtered[start:end]

    markup = types.InlineKeyboardMarkup(row_width=1)
    for i, src in enumerate(page_sources):
        idx = start + i
        markup.add(types.InlineKeyboardButton(src['label'], callback_data=f"ctrl_preview_{idx}"))

    # Навигация
    nav_buttons = []
    if category is not None:
        nav_buttons.append(types.InlineKeyboardButton("◀️ Назад к категориям", callback_data="ctrl_back_categories"))
    if page > 0:
        nav_buttons.append(types.InlineKeyboardButton("◀️ Назад", callback_data=f"ctrl_page_{page-1}"))
    nav_buttons.append(types.InlineKeyboardButton(f"📄 {page+1}/{total_pages}", callback_data="ctrl_page_info"))
    if page < total_pages - 1:
        nav_buttons.append(types.InlineKeyboardButton("Вперед ▶️", callback_data=f"ctrl_page_{page+1}"))
    markup.row(*nav_buttons)

    markup.add(types.InlineKeyboardButton("🔄 Обновить список", callback_data="ctrl_refresh"))

    # Заголовок с категорией
    if category is not None:
        cat_ru = CATEGORY_TRANSLATION.get(category, category)
        header = f"🏷️ **Категория:** {cat_ru}\n"
    else:
        header = "📋 **Все позы**\n"
    
    # Отправляем новое сообщение со списком
    sent_msg = await bot.send_message(
        message.chat.id,
        f"{header}📄 Страница {page+1} из {total_pages}\n\nВыберите позу для предпросмотра:",
        parse_mode="Markdown",
        reply_markup=markup
    )
    # Сохраняем ID сообщения списка
    data['list_message_id'] = sent_msg.message_id

@router.callback_query(F.data == "ctrl_back_categories")
async def handle_controlnet_back_categories(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    # Удаляем текущее сообщение со списком поз
    data = pagination_data.get(user_id, {})
    list_msg_id = data.get('list_message_id')
    if list_msg_id:
        try:
            await bot.delete_message(call.message.chat.id, list_msg_id)
        except Exception:
            pass
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    # Сбрасываем фильтр
    if user_id in pagination_data:
        pagination_data[user_id]['category'] = None
        pagination_data[user_id]['filtered_sources'] = None
        pagination_data[user_id]['list_message_id'] = None
    # Показываем категории
    await controlnet_menu(call.message)
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("ctrl_page_"))
async def handle_controlnet_page(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    if call.data == "ctrl_page_info":
        await bot.answer_callback_query(call.id, "Информация о странице")
        return
    try:
        page = int(call.data.split("_")[2])
    except Exception:
        await bot.answer_callback_query(call.id, "Ошибка")
        return
    if user_id not in pagination_data:
        await bot.answer_callback_query(call.id, "Сессия истекла")
        return
    # Удаляем предыдущее сообщение списка
    data = pagination_data[user_id]
    old_list_id = data.get('list_message_id')
    if old_list_id:
        try:
            await bot.delete_message(call.message.chat.id, old_list_id)
        except Exception:
            pass
        data['list_message_id'] = None
    # Удаляем текущее сообщение (это callback от кнопки навигации, оно тоже может быть удалено)
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    await show_controlnet_page(call.message, user_id, page)
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data == "ctrl_refresh")
async def handle_controlnet_refresh(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    sources = await get_controlnet_sources()
    if not sources:
        await bot.edit_message_text("❌ Нет доступных поз. Добавьте файлы в папку controlnet_images", call.message.chat.id, call.message.message_id)
        await bot.answer_callback_query(call.id)
        return

    # Обновляем данные, сохраняя текущую категорию
    category = pagination_data.get(user_id, {}).get('category')
    pagination_data[user_id] = {
        'sources': sources,
        'category': category,
        'page': 0,
        'total_pages': 0,
        'filtered_sources': None,
        'list_message_id': None
    }
    # Пересчитываем filtered_sources
    if category is not None:
        filtered = [s for s in sources if s.get('category') == category]
        pagination_data[user_id]['filtered_sources'] = filtered
    else:
        filtered = sources

    if not filtered:
        await bot.edit_message_text("❌ В этой категории нет поз. Вернитесь к выбору категории.", call.message.chat.id, call.message.message_id)
        await bot.answer_callback_query(call.id)
        return

    pagination_data[user_id]['total_pages'] = (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

    # Удаляем старое сообщение списка (текущее)
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    if user_id in preview_messages:
        try:
            await bot.delete_message(call.message.chat.id, preview_messages[user_id])
        except Exception:
            pass
        del preview_messages[user_id]

    await show_controlnet_page(call.message, user_id, 0)
    await bot.answer_callback_query(call.id, "🔄 Список обновлен!")

@router.callback_query(F.data.startswith("ctrl_preview_"))
async def handle_controlnet_preview(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    try:
        idx = int(call.data.split("_")[2])
    except Exception:
        await bot.answer_callback_query(call.id, "Ошибка")
        return
    data = pagination_data.get(user_id)
    if not data:
        await bot.answer_callback_query(call.id, "Сессия истекла")
        return
    filtered = data.get('filtered_sources')
    if filtered is None:
        filtered = data['sources']
    if idx < 0 or idx >= len(filtered):
        await bot.answer_callback_query(call.id, "Источник не найден")
        return
    source = filtered[idx]
    preview = await get_controlnet_preview_by_source(source)
    if not preview:
        await bot.answer_callback_query(call.id, "❌ Не удалось загрузить изображение")
        return
    
    # Удаляем сообщение со списком
    list_msg_id = data.get('list_message_id')
    if list_msg_id:
        try:
            await bot.delete_message(call.message.chat.id, list_msg_id)
        except Exception:
            pass
        data['list_message_id'] = None
    # Удаляем текущее сообщение (кнопка предпросмотра) – оно уже не нужно
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass

    # Удаляем старое превью
    if user_id in preview_messages:
        try:
            await bot.delete_message(call.message.chat.id, preview_messages[user_id])
        except Exception:
            pass
        del preview_messages[user_id]
    
    # Кнопки под превью
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("✅ Использовать", callback_data=f"ctrl_use_{idx}"),
        types.InlineKeyboardButton("❌ Назад к списку", callback_data="ctrl_back_to_list")
    )
    markup.add(types.InlineKeyboardButton("🔄 Случайная поза", callback_data="ctrl_random"))
    sent = await bot.send_photo(
        call.message.chat.id,
        preview,
        caption=f"🦴 **Поза:** {source.get('full_label', source['label'])}\n\nВыберите действие:",
        parse_mode="Markdown",
        reply_markup=markup
    )
    preview_messages[user_id] = sent.message_id
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data == "ctrl_back_to_list")
async def handle_controlnet_back_to_list(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    # Удаляем сообщение с превью
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    if user_id in preview_messages:
        del preview_messages[user_id]
    if user_id in pagination_data:
        # Удаляем сохранённый list_message_id, если есть (на случай, если он остался)
        pagination_data[user_id]['list_message_id'] = None
        page = pagination_data[user_id].get('page', 0)
        await show_controlnet_page(call.message, user_id, page)
    else:
        await controlnet_menu(call.message)
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("ctrl_use_"))
async def handle_controlnet_use(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    try:
        idx = int(call.data.split("_")[2])
    except Exception:
        await bot.answer_callback_query(call.id, "Ошибка")
        return
    data = pagination_data.get(user_id)
    if not data:
        await bot.answer_callback_query(call.id, "Сессия истекла")
        return
    filtered = data.get('filtered_sources')
    if filtered is None:
        filtered = data['sources']
    if idx < 0 or idx >= len(filtered):
        await bot.answer_callback_query(call.id, "Источник не найден")
        return
    source = filtered[idx]
    # Сохраняем выбранный источник
    temp_data.setdefault(user_id, {})['controlnet_source'] = {
        'type': source['type'],
        'file': source.get('file'),
        'index': source.get('index'),
        'path': source.get('path'),
        'img_col': source.get('img_col'),
        'full_label': source.get('full_label', source['label'])
    }
    # Удаляем сообщение с превью
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    if user_id in preview_messages:
        del preview_messages[user_id]
    await bot.send_message(
        call.message.chat.id,
        f"✍️ **Введите промпт** для генерации с позой:\n`{source.get('full_label', source['label'])}`\n\nНапример: *девушка в платье, городской фон*",
        parse_mode="Markdown"
    )
    register_next_step(call.message.chat.id, process_controlnet_prompt, user_id, source)
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data == "ctrl_random")
async def handle_controlnet_random(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    data = pagination_data.get(user_id)
    if not data:
        await bot.answer_callback_query(call.id, "Сессия истекла")
        return
    filtered = data.get('filtered_sources')
    if filtered is None:
        filtered = data['sources']
    if not filtered:
        await bot.answer_callback_query(call.id, "Нет доступных поз")
        return
    import random
    idx = random.randint(0, len(filtered) - 1)
    # Удаляем текущее сообщение с превью
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    if user_id in preview_messages:
        del preview_messages[user_id]
    # Имитируем вызов preview с новым idx
    fake_call = type('FakeCall', (), {'data': f"ctrl_preview_{idx}", 'message': call.message, 'id': call.id})()
    await handle_controlnet_preview(fake_call)

async def process_controlnet_prompt(message, user_id, source=None):
    """
    Обрабатывает промпт для ControlNet.
    source может быть передан при первом выборе, либо взят из temp_data при повторной генерации.
    """
    prompt = _text(message)
    if _is_cancel_text(prompt):
        await bot.send_message(message.chat.id, "❌ Нужен текстовый промпт (не кнопка, команда или файл). Отменено.")
        return
    
    # Если source не передан, берём из temp_data (для повторной генерации)
    if source is None:
        data = temp_data.get(user_id, {}).get('controlnet_source')
        if not data:
            await bot.send_message(message.chat.id, "❌ Скелет не выбран. Начните заново через кнопку ControlNet.")
            return
        # Восстанавливаем источник из сохранённых данных
        sources = await get_controlnet_sources()
        target = None
        for s in sources:
            if s['type'] == data['type']:
                if data['type'] == 'parquet' and s['file'] == data['file'] and s['index'] == data['index']:
                    target = s
                    break
                elif data['type'] == 'image' and s.get('path') == data.get('path'):
                    target = s
                    break
        if target is None:
            await bot.send_message(message.chat.id, "❌ Источник не найден. Возможно, файл был удалён.")
            return
        source = target
        # также обновим данные в temp_data, чтобы сохранить актуальный full_label
        data['full_label'] = source.get('full_label', source['label'])
    else:
        # Если source передан, сохраняем его в temp_data (если ещё не сохранён)
        temp_data.setdefault(user_id, {})['controlnet_source'] = {
            'type': source['type'],
            'file': source.get('file'),
            'index': source.get('index'),
            'path': source.get('path'),
            'img_col': source.get('img_col'),
            'full_label': source.get('full_label', source['label'])
        }

    control_img = await get_controlnet_preview_by_source(source)
    if not control_img:
        await bot.send_message(message.chat.id, "❌ Не удалось загрузить скелет.")
        return

    # --- АВТОПОДСТРОЙКА РАЗРЕШЕНИЯ ПОД СКЕЛЕТ ---
    settings = await get_user_settings(user_id)
    resolution = source.get('resolution')
    if resolution and resolution != 'unknown':
        try:
            if 'x' in resolution:
                parts = resolution.split('x')
            else:
                # Формат "ширинавысота" (например, 512768)
                if len(resolution) == 6:
                    parts = [resolution[:3], resolution[3:]]
                else:
                    parts = [resolution[:4], resolution[4:]]  # для 1024x...
            w = int(parts[0])
            h = int(parts[1])
            settings['width'] = w
            settings['height'] = h
            print(f"📐 Использую разрешение скелета: {w}x{h}")
        except Exception as e:
            print(f"⚠️ Не удалось распарсить разрешение {resolution}: {e}")

    # --- Логирование промпта пользователя (ControlNet) ---

    gen_id = await prompt_logger.log_request(
        user_id, "controlnet", prompt,
        settings=settings,
        negative_prompt=await get_user_negative_prompt(user_id),
        controlnet_source=source.get('full_label', source.get('label')),
    )
    async def job(task: Task):
        start_time = time.time()
        on_progress, on_status = make_callbacks(task)
        try:
            # Сохраняем оригинальный промпт для отображения
            original_prompt = prompt
            # Обработка промпта через Aapi.preprocess_prompt (она использует настройки пользователя)
            processed_prompt = await preprocess_prompt(user_id, prompt, on_status)
            # Добавляем LoRA через Aapi (там уже есть get_active_lora_tags)
            lora_tags = await get_active_lora_tags(user_id)
            if lora_tags:
                processed_prompt = f"{processed_prompt}, {lora_tags}"
                await on_status(f"🎛️ Добавлены LoRA: {lora_tags}")

            face_ctx = await get_face_context(user_id, need_controlnet_slot=True)
            if face_ctx and face_ctx.note:
                await on_status(f"⚠️ {face_ctx.note}")
            generated_image = await generate_txt2img_controlnet(
                processed_prompt,  # используем обработанный промпт
                control_image=control_img,
                module="none",
                model="control_v11p_sd15_openpose",
                settings=settings,
                progress_callback=on_progress,
                status_callback=on_status,
                user_id=user_id,
                face_image=face_ctx.image if face_ctx and face_ctx.use_faceid else None,
                face_weights=face_ctx.weights if face_ctx else None
            )
            final_image = generated_image
            swapped = False
            if face_ctx and face_ctx.use_swap:
                await on_status("🎭 Заменяю лицо...")
                final_image, swapped = await apply_face_swap(face_ctx, generated_image)
            face_replaced = bool(face_ctx and (face_ctx.use_faceid or swapped))
            try:
                await bot.delete_message(message.chat.id, task.status_message_id)
            except Exception:
                pass
        
            # Формируем капшн с полной информацией
            caption = (
                f"✅ <b>ControlNet генерация</b>\n"
                f"🦴 Скелет: <code>{_h(source.get('full_label', source['label']))}</code>\n"
                f"✨ <b>Исходный промпт:</b>\n<code>{_h(_cut(original_prompt, 600))}</code>\n\n"
                f"🤖 <b>Обработанный промпт:</b>\n<code>{_h(_cut(processed_prompt, 1000))}</code>\n\n"
                f"📐 <b>Размер:</b> {settings.get('width', 512)}x{settings.get('height', 768)}\n"
                f"⏱️ <b>Шаги:</b> {settings.get('steps', 20)}\n"
                f"🎚️ <b>CFG:</b> {settings.get('cfg_scale', 7.0)}\n"
                f"🎲 <b>Seed:</b> <code>{getattr(generated_image, 'seed', 'N/A')}</code>\n"
            )
            caption += _face_caption_line(face_ctx, swapped)
            sent_msg = await send_image_smart(message.chat.id, final_image, caption=caption, parse_mode="HTML")
            last_images[user_id] = final_image
            last_message_ids[user_id] = sent_msg.message_id
            # Сохраняем конфигурацию для повтора
            config = {
                "type": "controlnet",
                "prompt": processed_prompt,
                "original_prompt": original_prompt,
                "settings": dict(settings),
                "seed": getattr(generated_image, "seed", 0),
                "face_enabled": face_replaced,
                "controlnet_source": {
                    'type': source['type'],
                    'file': source.get('file'),
                    'index': source.get('index'),
                    'path': source.get('path'),
                    'img_col': source.get('img_col'),
                    'full_label': source.get('full_label', source['label'])
                }
            }
            last_configs[user_id] = config
            # --- Логирование результата ControlNet-генерации ---
            await prompt_logger.log_result(
                gen_id, user_id, "controlnet", original_prompt,
                processed_prompt=processed_prompt,
                seed=getattr(generated_image, "seed", None),
                settings=settings, duration_sec=time.time() - start_time,
                face_replaced=face_replaced,
                controlnet_source=source.get('full_label', source.get('label')),
            )
            # Отправляем клавиатуру с кнопкой "тот же скелет"
            await bot.send_message(message.chat.id, "🔧 **Действия с изображением:**", reply_markup=get_controlnet_actions_keyboard())
        except asyncio.CancelledError:
            await prompt_logger.log_error(gen_id, user_id, "controlnet", prompt, "Отменено пользователем", duration_sec=time.time() - start_time)
            raise
        except Exception as e:
            await prompt_logger.log_error(gen_id, user_id, "controlnet", prompt, e, duration_sec=time.time() - start_time)
            await _report_error(message.chat.id, task.status_message_id, e)
        finally:
            # Не удаляем controlnet_source, чтобы можно было повторно использовать (если не было ошибки)
            # Но если мы хотим, чтобы при повторном выборе скелета он перезаписывался, то оставляем.
            pass
    await enqueue(message.chat.id, user_id, "controlnet", job)
# =========== КОМАНДЫ ДЛЯ РАБОТЫ С СЭМПЛЕРАМИ ===========

@router.message(Command('samplers'))
async def show_samplers(message: Message) -> None:
    """Показывает список доступных сэмплеров"""
    try:
        samplers = await get_available_samplers()
        if not samplers:
            await bot.reply_to(message, "❌ Не удалось получить список сэмплеров")
            return

        user_id = str(message.chat.id)
        current = await get_user_sampler(user_id)

        # Раньше здесь была опечатка (current_master вместо current_marker) — команда всегда падала с NameError.
        header = f"⭐ <b>Текущий сэмплер:</b> <code>{_h(current)}</code>\n\n<b>Доступные сэмплеры:</b>\n"
        lines = [f"• <code>{_h(s)}</code>" for s in samplers]

        # Режем по границам строк, чтобы не разрывать HTML-теги посреди сообщения
        chunks, chunk = [], header
        for line in lines:
            if len(chunk) + len(line) + 1 > 3900:
                chunks.append(chunk)
                chunk = ""
            chunk += line + "\n"
        chunks.append(chunk)
        for part in chunks:
            await bot.send_message(message.chat.id, part, parse_mode="HTML")
    except Exception as e:
        await bot.reply_to(message, f"❌ Ошибка: {e}")

@router.message(Command('setsampler'))
async def set_sampler_command(message: Message) -> None:
    """Устанавливает сэмплер для пользователя"""
    user_id = str(message.chat.id)

    # Разбираем команду
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        # Показываем текущий сэмплер (подсказки раньше ссылались на несуществующие команды «!samplers»)
        current = await get_user_sampler(user_id)
        await bot.reply_to(
            message,
            f"📌 <b>Текущий сэмплер:</b> <code>{_h(current)}</code>\n\n"
            f"📋 Используйте <code>/setsampler &lt;название&gt;</code> для смены\n"
            f"🔍 Просмотр всех: /samplers\n\n"
            f"Пример: <code>/setsampler DPM++ 2M Karras</code>",
            parse_mode="HTML"
        )
        return

    sampler_name = parts[1].strip()

    try:
        # Проверяем, существует ли такой сэмплер
        samplers = await get_available_samplers()
        if not samplers:
            await bot.reply_to(message, "❌ Не удалось получить список сэмплеров")
            return

        # Ищем точное совпадение (без учёта регистра), затем частичное
        exact = [s for s in samplers if s.lower() == sampler_name.lower()]
        matching = exact or [s for s in samplers if sampler_name.lower() in s.lower()]

        if len(matching) == 1:
            await set_user_sampler(user_id, matching[0])
            await bot.reply_to(message, f"✅ <b>Сэмплер изменён:</b> <code>{_h(matching[0])}</code>", parse_mode="HTML")
        elif len(matching) > 1:
            suggestions = "\n".join([f"• <code>{_h(s)}</code>" for s in matching[:10]])
            await bot.reply_to(
                message,
                f"❓ <b>Найдено несколько сэмплеров:</b>\n{suggestions}\n\n"
                f"Уточните название. Все доступные: /samplers",
                parse_mode="HTML"
            )
        else:
            await bot.reply_to(
                message,
                f"❌ <b>Сэмплер не найден:</b> <code>{_h(sampler_name)}</code>\n\n"
                f"Все доступные: /samplers",
                parse_mode="HTML"
            )

    except Exception as e:
        await bot.reply_to(message, f"❌ Ошибка: {e}")

# =========== ОБРАБОТЧИК "ТОТ ЖЕ СКЕЛЕТ" ===========

@router.callback_query(F.data == "action_same_skeleton")
async def handle_same_skeleton(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    # Проверяем наличие сохранённого скелета в last_configs
    config = last_configs.get(user_id)
    if not config or config.get('type') != 'controlnet':
        await bot.answer_callback_query(call.id, "❌ Нет сохранённого скелета для повторной генерации.", show_alert=True)
        return
    source_data = config.get('controlnet_source')
    if not source_data:
        await bot.answer_callback_query(call.id, "❌ Данные скелета отсутствуют.", show_alert=True)
        return
    # Восстанавливаем источник из данных
    sources = await get_controlnet_sources()
    target = None
    for s in sources:
        if s['type'] == source_data['type']:
            if source_data['type'] == 'parquet' and s['file'] == source_data['file'] and s['index'] == source_data['index']:
                target = s
                break
            elif source_data['type'] == 'image' and s.get('path') == source_data.get('path'):
                target = s
                break
    if target is None:
        await bot.answer_callback_query(call.id, "❌ Скелет не найден (возможно, файл удалён).", show_alert=True)
        return
    # Сохраняем скелет в temp_data для process_controlnet_prompt
    temp_data.setdefault(user_id, {})['controlnet_source'] = {
        'type': target['type'],
        'file': target.get('file'),
        'index': target.get('index'),
        'path': target.get('path'),
        'img_col': target.get('img_col'),
        'full_label': target.get('full_label', target['label'])
    }
    await bot.answer_callback_query(call.id, "✍️ Введите новый промпт для генерации с тем же скелетом.")
    # Запрашиваем новый промпт
    await bot.send_message(
        call.message.chat.id,
        f"✍️ **Введите новый промпт** для генерации с той же позой:\n`{target.get('full_label', target['label'])}`\n\n(можно изменить настройки перед отправкой)",
        parse_mode="Markdown"
    )
    # Регистрируем следующий шаг, передавая None как source (будет взят из temp_data)
    register_next_step(call.message.chat.id, process_controlnet_prompt, user_id, None)

# =========== ОБРАБОТЧИКИ ОСТАЛЬНЫХ КНОПОК ===========

@router.message(Command("start", "help"))
async def send_welcome(message: Message) -> None:
    await get_user_settings(str(message.chat.id))
    await bot.send_message(
        message.chat.id,
        "🤖 **Stable Diffusion Бот**\n\n📝 **Быстрая генерация:** просто отправь любой текст\n\n🚀 **Сгенерировать** - расширенный режим\n⚙️ **Настройки** - изменить параметры\n🔍 **Апскейл** - увеличить картинку (или просто пришли мне любое фото)\n🧩 **Модель** - выбор в настройках или командой /model\n🎭 **Управление лицом** - добавление нескольких лиц для замены",
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )

@router.message(F.text == "🚀 Сгенерировать")
async def extended_generation_start(message: Message) -> None:
    msg = await bot.send_message(message.chat.id, "✍️ Отправь промпт для генерации:")
    register_next_step(msg.chat.id, extended_generation_prompt)

@router.message(F.text == "⚙️ Настройки")
async def show_settings(message: Message) -> None:
    user_id = str(message.chat.id)
    settings = await get_user_settings(user_id)
    current_neg = (await get_user_negative_prompt(user_id)).replace("`", "'")
    neg_preview = current_neg if len(current_neg) <= 80 else current_neg[:80] + "…"
    emb_names = await get_user_negative_embeddings(user_id)
    emb_line = f"\n🧬 **Эмбеддинги:** `{_cut(', '.join(emb_names), 150).replace(chr(96), chr(39))}`" if emb_names else ""
    prompt_processing = await get_prompt_processing_enabled(user_id)
    model_display = await _model_display_name(user_id)
    plan = hires_plan(settings)
    hires_line = f"🔬 Hires: ✅ {plan['scale']}x → {plan['width']}×{plan['height']}\n" if plan else "🔬 Hires: ❌ выкл\n"
    text = (
        "🎛️ **Текущие настройки:**\n\n"
        f"📐 Размер: {settings['width']}x{settings['height']}\n"
        f"⏱️ Шаги: {settings['steps']}\n"
        f"🎚️ CFG Scale: {settings['cfg_scale']}\n"
        f"🎯 Сэмплер: {settings.get('sampler_name', 'DPM++ 2M')}\n"  # <--- Добавлена строка
        f"🧩 Модель: `{model_display.replace('`', '')}`\n"
        f"{hires_line}"
        f"🤖 **Обработка промпта:** {'✅ Включена' if prompt_processing else '❌ Выключена'}\n\n"
        f"🚫 **Негативный промпт:**\n`{neg_preview}`{emb_line}"
    )
    await bot.send_message(message.chat.id, text, parse_mode="Markdown", reply_markup=await get_settings_inline_keyboard(settings, user_id))

@router.message(F.text == "🔍 Апскейл")
async def ask_for_upscale(message: Message) -> None:
    user_id = str(message.chat.id)
    if user_id not in last_images:
        await bot.reply_to(message, "❌ Сначала сгенерируй картинку или просто пришли мне фото для апскейла!")
        return
    await bot.send_message(message.chat.id, "Выбери коэффициент увеличения:", reply_markup=get_upscale_keyboard())

@router.message(F.text == "🎭 Управление лицом")
async def face_management(message: Message) -> None:
    if not await face_features_available():
        await bot.reply_to(message, "❌ Функция лиц недоступна: в WebUI не найден ни ReActor, ни FaceID (ControlNet). Подробности — в README, раздел «Лица».")
        return
    user_id = str(message.chat.id)
    stats = await get_face_stats(user_id)
    face_enabled = await get_face_enabled(user_id)
    mode = await get_face_mode(user_id)
    faces = await get_user_faces(user_id)
    models_count = sum(1 for f in faces if f.get("model"))
    fw, lw = await get_face_weights(user_id)
    try:
        from Aapi import get_faceid_setup
        setup = await get_faceid_setup()
    except Exception as e:
        setup = {"ok": False, "reason": str(e)}
    markup = types.InlineKeyboardMarkup(row_width=1)
    status_text = "✅ Включено" if face_enabled else "❌ Выключено"
    buttons = [
        types.InlineKeyboardButton(f"🎭 Лицо в генерации: {status_text}", callback_data="face_toggle_main"),
        types.InlineKeyboardButton(f"🧠 Режим: {FACE_MODE_LABELS[mode]}", callback_data="face_mode_cycle"),
        types.InlineKeyboardButton(f"🎚️ Сила лица (FaceID): {fw:g}", callback_data="face_weight_faceid"),
        types.InlineKeyboardButton(f"🧴 Вес LoRA лица: {'выкл.' if lw <= 0 else format(lw, 'g')}", callback_data="face_weight_lora"),
        types.InlineKeyboardButton("📸 Добавить новое лицо", callback_data="face_add"),
        types.InlineKeyboardButton("🧬 Модель лица из нескольких фото", callback_data="face_model_new"),
    ]
    if stats["total_faces"] > 0:
        buttons.append(types.InlineKeyboardButton("📋 Список лиц", callback_data="face_list"))
    markup.add(*buttons)
    active_face = stats.get("active_face") or "не выбрано"
    faceid_line = "✅ готов" if setup.get("ok") else f"⚠️ не готов: {_h(setup.get('reason') or '—')}"
    info_text = (
        "🎭 <b>Управление лицами</b>\n\n"
        f"📸 <b>Всего лиц:</b> {stats['total_faces']} (с моделью ReActor: {models_count})\n"
        f"⭐ <b>Активное лицо:</b> {_h(active_face)}\n"
        f"🧠 <b>FaceID:</b> {faceid_line}\n"
        f"🎚️ <b>Сила лица:</b> {fw:g} · <b>LoRA:</b> {'выкл.' if lw <= 0 else format(lw, 'g')}\n\n"
        "💡 <b>Режимы:</b>\n"
        "🔁 <b>Замена после генерации</b> — ReActor подменяет лицо на готовой картинке (по модели лица, если она создана).\n"
        "🧠 <b>FaceID в генерации</b> — лицо задаётся прямо при генерации (IP-Adapter), без замены.\n"
        "🧠+🔁 <b>Оба</b> — FaceID, а сверху ReActor для максимального сходства.\n\n"
        "🎚️ Сила лица — насколько жёстко FaceID держит сходство (выше — похожее, но «деревянное»; ниже — живее). "
        "LoRA лица усиливает сходство, обычно 0.5–0.7.\n\n"
        "✨ Для лучшего результата используйте чёткие фото анфас; модель из 3–10 фото точнее, чем из одного."
    )
    await bot.send_message(message.chat.id, info_text, parse_mode="HTML", reply_markup=markup)

# =========== СТАТИСТИКА ЛОГА ПРОМПТОВ ===========
# (Обработчик должен быть зарегистрирован ДО универсального текстового обработчика ниже —
#  telebot вызывает первый подходящий, иначе команда /prompts никогда не срабатывала.)

@router.message(Command('prompts'))
async def show_prompt_log(message: Message) -> None:
    """Показывает последние записями из лога промптов (для администратора)."""
    admin_id = os.environ.get("ADMIN_USER_ID")
    if admin_id and str(message.chat.id) != admin_id:
        await bot.reply_to(message, "⛔ Доступ только для администратора (задайте ADMIN_USER_ID в .env).")
        return
    records = await prompt_logger.tail(limit=15)
    if not records:
        await bot.reply_to(message, "📭 Лог промптов пуст.")
        return
    lines = []
    for rec in records:
        event = rec.get("event", "?")
        user = rec.get("username") or rec.get("user_id", "?")
        ptype = rec.get("type", "?")
        prompt_text = (rec.get("original_prompt") or "")[:60]
        ts = (rec.get("ts") or "")[11:19]
        extra = f" seed={rec['seed']}" if rec.get("seed") is not None else ""
        extra += f" ⏱{rec['duration_sec']}s" if rec.get("duration_sec") is not None else ""
        lines.append(f"`{ts}` [{event}/{ptype}] user `{user}`: {prompt_text}{extra}")
    text = "📜 **Последние записи лога промптов**\n\n" + "\n".join(lines)
    try:
        await bot.send_message(message.chat.id, text, parse_mode="Markdown")
    except Exception:
        await bot.send_message(message.chat.id, text.replace("`", ""))

# =========== ВЫБОР МОДЕЛИ (CHECKPOINT) ===========

MODELS_PER_PAGE = 8

def _model_id(model_name):
    return hashlib.md5(model_name.encode("utf-8")).hexdigest()[:10]

async def _model_display_name(user_id):
    """Имя модели пользователя для настроек: выбранная, либо «по умолчанию»."""
    chosen = await get_user_model(user_id)
    if chosen:
        return chosen
    default_name = await get_default_model_name()
    return f"{default_name} (по умолч.)" if default_name else "по умолчанию"

async def _build_models_menu(user_id, page):
    models = await get_available_models()
    chosen = await get_user_model(user_id)
    default_name = await get_default_model_name()
    markup = types.InlineKeyboardMarkup(row_width=1)
    if not models:
        text = ("⚠️ <b>Не удалось получить список моделей.</b>\n\n"
                "Проверьте, что WebUI запущен с флагом <code>--api</code>, и нажмите «Обновить».")
        markup.add(types.InlineKeyboardButton("🔄 Обновить список", callback_data="model_refresh"))
        markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
        return text, markup

    pages = (len(models) + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE
    page = max(0, min(page, pages - 1))
    current_label = chosen or (f"{default_name} (по умолчанию)" if default_name else "по умолчанию")
    text = (f"🧩 <b>Выбор модели</b>\n\n"
            f"Текущая: <code>{html.escape(current_label)}</code>\n"
            f"Всего моделей: {len(models)}\n\n"
            f"ℹ️ После смены модели первая генерация займёт больше времени — модель загружается в память.")
    for model in models[page * MODELS_PER_PAGE:(page + 1) * MODELS_PER_PAGE]:
        is_current = (model["name"] == chosen) or (chosen is None and default_name == model["name"])
        label = ("✅ " if is_current else "") + model["name"]
        if len(label) > 60:
            label = label[:59] + "…"
        markup.add(types.InlineKeyboardButton(label, callback_data=f"model_set_{_model_id(model['name'])}"))
    if pages > 1:
        markup.row(
            types.InlineKeyboardButton("◀️", callback_data=f"model_page_{(page - 1) % pages}"),
            types.InlineKeyboardButton(f"{page + 1}/{pages}", callback_data="model_noop"),
            types.InlineKeyboardButton("▶️", callback_data=f"model_page_{(page + 1) % pages}"),
        )
    markup.row(
        types.InlineKeyboardButton("🔄 Обновить список", callback_data="model_refresh"),
        types.InlineKeyboardButton("↩️ По умолчанию", callback_data="model_default"),
    )
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
    return text, markup

async def _show_models_menu(chat_id, user_id, page=0, message_id=None):
    text, markup = await _build_models_menu(user_id, page)
    if message_id is not None:
        try:
            await bot.edit_message_text(text, chat_id, message_id, parse_mode="HTML", reply_markup=markup)
            return
        except Exception as e:
            if "message is not modified" in str(e):
                return
    await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)

@router.message(Command('model', 'models'))
async def models_command(message: Message) -> None:
    await _show_models_menu(message.chat.id, str(message.chat.id), 0)

@router.callback_query(F.data == "set_model")
async def handle_model_button(call: CallbackQuery) -> None:
    await _show_models_menu(call.message.chat.id, str(call.message.chat.id), 0, call.message.message_id)
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("model_page_"))
async def handle_model_page(call: CallbackQuery) -> None:
    try:
        page = int(call.data[len("model_page_"):])
    except ValueError:
        page = 0
    await _show_models_menu(call.message.chat.id, str(call.message.chat.id), page, call.message.message_id)
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data == "model_noop")
async def handle_model_noop(call: CallbackQuery) -> None:
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data == "model_refresh")
async def handle_model_refresh(call: CallbackQuery) -> None:
    await bot.answer_callback_query(call.id, "Обновляю список...")
    await refresh_checkpoints_on_server()           # WebUI пересканирует папку с моделями
    models = await get_available_models(force_refresh=True)
    await _show_models_menu(call.message.chat.id, str(call.message.chat.id), 0, call.message.message_id)

@router.callback_query(F.data.startswith("model_set_"))
async def handle_model_choice(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    model_id = call.data[len("model_set_"):]
    available_models = await get_available_models()
    model = next((m for m in available_models if _model_id(m["name"]) == model_id), None)
    if not model:
        await bot.answer_callback_query(call.id, "Модель не найдена — обновите список", show_alert=True)
        return
    await set_user_model(user_id, model["name"])
    await bot.answer_callback_query(call.id, f"✅ Модель: {model['name']}"[:200])
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    await show_settings(call.message)

@router.callback_query(F.data == "model_default")
async def handle_model_default(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    await set_user_model(user_id, None)
    await bot.answer_callback_query(call.id, "Используется модель по умолчанию")
    try:
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    await show_settings(call.message)

# =========== АПСКЕЙЛ ФОТО, ПРИСЛАННЫХ ПОЛЬЗОВАТЕЛЕМ ===========

MAX_UPSCALE_SIDE = 8192              # максимальная сторона результата, px (защита от нехватки памяти)
MAX_RESULT_FILE_BYTES = 45 * 1024 * 1024   # лимит Telegram для отправки ботом — 50 МБ
PENDING_UPSCALE_LIMIT = 30
pending_upscale = SafeStore(maxsize=PENDING_UPSCALE_LIMIT)  # (chat_id, id сообщения с кнопками) -> байты исходного изображения

def _remember_pending_upscale(key, data):
    pending_upscale[key] = data   # SafeStore потокобезопасен и сам вытесняет самые старые записи

async def _offer_photo_upscale(message, file_id):
    """Скачивает присланное изображение и предлагает выбрать коэффициент увеличения."""
    try:
        file_info = await bot.get_file(file_id)
        data = await bot.download_file(file_info.file_path)
    except Exception as e:
        reason = "файл слишком большой (Telegram даёт ботам скачивать до 20 МБ)" if "too big" in str(e).lower() else str(e)
        await bot.reply_to(message, f"❌ Не удалось скачать изображение: {reason}")
        return
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        await bot.reply_to(message, "❌ Не удалось прочитать изображение. Пришлите файл JPG, PNG или WEBP.")
        return
    factors = [f for f in (2, 3, 4) if max(width, height) * f <= MAX_UPSCALE_SIDE]
    if not factors:
        await bot.reply_to(message, f"❌ Изображение {width}×{height} слишком большое для апскейла "
                              f"(результат не должен превышать {MAX_UPSCALE_SIDE}px по стороне).")
        return
    markup = types.InlineKeyboardMarkup(row_width=3)
    markup.add(*[types.InlineKeyboardButton(f"{f}x → {width * f}×{height * f}", callback_data=f"pup_{f}") for f in factors])
    markup.add(types.InlineKeyboardButton("❌ Отмена", callback_data="pup_cancel"))
    sent = await bot.reply_to(message, f"🔍 Получено изображение {width}×{height}.\nВо сколько раз увеличить?", reply_markup=markup)
    _remember_pending_upscale((message.chat.id, sent.message_id), data)

@router.message(F.photo)
async def handle_photo_for_upscale(message: Message) -> None:
    await _offer_photo_upscale(message, message.photo[-1].file_id)

@router.message(F.document.mime_type.startswith("image/"))
async def handle_image_document_for_upscale(message: Message) -> None:
    # Изображение, отправленное «как файл», приходит без сжатия Telegram — лучшее качество для апскейла
    await _offer_photo_upscale(message, message.document.file_id)

@router.callback_query(F.data.startswith("pup_"))
async def handle_photo_upscale_callback(call: CallbackQuery) -> None:
    chat_id = call.message.chat.id
    user_id = str(chat_id)
    key = (chat_id, call.message.message_id)
    action = call.data[len("pup_"):]
    if action == "cancel":
        pending_upscale.pop(key, None)
        await bot.answer_callback_query(call.id, "Отменено")
        try:
            await bot.delete_message(chat_id, call.message.message_id)
        except Exception:
            pass
        return
    data = pending_upscale.pop(key, None)
    if data is None:
        await bot.answer_callback_query(call.id, "Изображение устарело — пришлите его снова", show_alert=True)
        return
    try:
        factor = int(action)
    except ValueError:
        await bot.answer_callback_query(call.id)
        return
    await bot.answer_callback_query(call.id, f"Увеличиваю в {factor}x...")
    try:  # сообщение с кнопками выбора коэффициента заменяет статус-сообщение задачи
        await bot.delete_message(chat_id, call.message.message_id)
    except Exception:
        pass
    gen_id = await prompt_logger.log_request(user_id, "upscale", original_prompt="", upscale_factor=factor, source="user_photo")

    def _prepare_result(raw_input: bytes, upscaled_bytes: bytes):
        """CPU-работа с PIL (размеры, перекодирование тяжёлого PNG в JPEG) — выполняется в потоке."""
        with Image.open(io.BytesIO(raw_input)) as src:
            src_size = src.size
        raw, ext = upscaled_bytes, "png"
        with Image.open(io.BytesIO(raw)) as res:
            res_size = res.size
            if len(raw) > MAX_RESULT_FILE_BYTES:      # слишком тяжёлый PNG — пересохраняем в JPEG
                buf = io.BytesIO()
                res.convert("RGB").save(buf, format="JPEG", quality=95)
                raw, ext = buf.getvalue(), "jpg"
        return src_size, res_size, raw, ext

    async def job(task: Task):
        start_time = time.time()
        try:
            await task.edit_status(f"🔍 Задача №{task.id}: апскейл {factor}x…", force=True)
            upscaled = await upscale_image(io.BytesIO(data), upscale_factor=float(factor))
            src_size, res_size, raw, ext = await asyncio.to_thread(_prepare_result, data, upscaled.getvalue())
            out = io.BytesIO(raw)
            out.name = f"upscaled_{factor}x.{ext}"
            caption = f"✅ {src_size[0]}×{src_size[1]} → {res_size[0]}×{res_size[1]} ({factor}x)"
            # Отправляем файлом: как «фото» Telegram пересжал бы результат и часть выигрыша от апскейла пропала бы
            await bot.send_document(chat_id, out, caption=caption)
            await prompt_logger.log_result(gen_id, user_id, "upscale", "", duration_sec=time.time() - start_time, upscale_factor=factor)
            await task.delete_status()
        except asyncio.CancelledError:
            await prompt_logger.log_error(gen_id, user_id, "upscale", "", "Отменено пользователем", duration_sec=time.time() - start_time)
            raise
        except Exception as e:
            await prompt_logger.log_error(gen_id, user_id, "upscale", "", e, duration_sec=time.time() - start_time)
            await _report_error(chat_id, task.status_message_id, e)

    await enqueue(chat_id, user_id, "upscale", job)

# =========== ОТМЕНА ЗАДАЧ ===========

@router.message(Command("cancel"))
async def cancel_command(message: Message):
    """/cancel — отменяет все задачи пользователя (в очереди и выполняющиеся) и ожидание ввода текста."""
    pending_input.pop(message.chat.id, None)
    cancelled = await task_queue.cancel_user(message.chat.id) if task_queue is not None else 0
    await bot.reply_to(message, f"❌ Отменено задач: {cancelled}" if cancelled else "ℹ️ Нет активных задач для отмены.")

@router.callback_query(F.data.startswith("task_cancel_"))
async def handle_task_cancel(call: CallbackQuery):
    """Кнопка «❌ Отменить» под статус-сообщением задачи."""
    try:
        task_id = int(call.data[len("task_cancel_"):])
    except ValueError:
        await _safe_answer(call)
        return
    result = await task_queue.cancel(task_id, call.message.chat.id) if task_queue is not None else "unknown"
    answers = {
        "queued": "Задача удалена из очереди",
        "running": "Останавливаю генерацию…",
        "forbidden": "Это чужая задача",
        "unknown": "Задача уже завершена",
    }
    await _safe_answer(call, answers.get(result, "Готово"))

# =========== HIRES. FIX И ПРЕСЕТЫ ===========

HIRES_SCALES = (1.25, 1.5, 2.0)
HIRES_DENOISES = (0.4, 0.5, 0.6)
HIRES_UPSCALERS = (("Latent", "Latent (быстрее)"), ("ESRGAN_4x", "ESRGAN 4x (резче, дольше)"))

async def _show_hires_menu(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    settings = await get_user_settings(user_id)
    enabled = bool(settings.get("hires_enabled"))
    scale, denoise = float(settings.get("hires_scale", 1.5)), float(settings.get("hires_denoise", 0.5))
    upscaler = settings.get("hires_upscaler", "Latent")
    plan = hires_plan(settings)
    if not enabled:
        info = "Выключен: картинки генерируются за один проход."
    elif plan:
        shrink = f" (ограничено {HIRES_MAX_SIDE}px)" if plan["scale"] < scale - 0.01 else ""
        info = f"Итог: <b>{plan['width']}×{plan['height']}</b>, масштаб {plan['scale']}x{shrink}; второй проход — {plan['steps']} шагов"
    else:
        info = f"⚠️ Исходный размер уже близок к лимиту {HIRES_MAX_SIDE}px — hires будет пропущен. Уменьшите размер в настройках."
    text = (f"🔬 <b>Hires. fix</b> — {'✅ включён' if enabled else '❌ выключен'}\n\n{info}\n\n"
            "Генерирует картинку, увеличивает её и дорисовывает детали: чётче лицо и текстуры.\n"
            "⏱ Время генерации примерно ×2. Для 8 ГБ видеопамяти итоговая сторона ограничена, а масштаб 2x лучше брать "
            "с небольшим исходным размером.")
    markup = types.InlineKeyboardMarkup(row_width=3)
    markup.add(types.InlineKeyboardButton("❌ Выключить" if enabled else "✅ Включить", callback_data="hires_toggle"))
    markup.row(*[types.InlineKeyboardButton(("✅ " if abs(v - scale) < 0.01 else "") + f"{v}x", callback_data=f"hires_scale_{v}") for v in HIRES_SCALES])
    markup.row(*[types.InlineKeyboardButton(("✅ " if abs(v - denoise) < 0.01 else "") + f"Денойз {v}", callback_data=f"hires_den_{v}") for v in HIRES_DENOISES])
    label = dict(HIRES_UPSCALERS).get(upscaler, upscaler)
    markup.add(types.InlineKeyboardButton(f"🔁 Апскейлер: {label}", callback_data="hires_up"))
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
    await _edit_or_send(call.message.chat.id, call.message.message_id, text, markup)

@router.callback_query(F.data == "set_hires")
async def handle_hires_menu(call: CallbackQuery) -> None:
    await _safe_answer(call)
    await _show_hires_menu(call)

@router.callback_query(F.data.startswith("hires_"))
async def handle_hires_change(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    data = call.data
    settings = await get_user_settings(user_id)
    try:
        if data == "hires_toggle":
            await update_user_settings(user_id, {"hires_enabled": not settings.get("hires_enabled")})
        elif data.startswith("hires_scale_") and float(data[12:]) in HIRES_SCALES:
            await update_user_settings(user_id, {"hires_scale": float(data[12:])})
        elif data.startswith("hires_den_") and float(data[10:]) in HIRES_DENOISES:
            await update_user_settings(user_id, {"hires_denoise": float(data[10:])})
        elif data == "hires_up":
            names = [n for n, _ in HIRES_UPSCALERS]
            current = settings.get("hires_upscaler", "Latent")
            await update_user_settings(user_id, {"hires_upscaler": names[(names.index(current) + 1) % len(names)] if current in names else names[0]})
    except ValueError:
        pass
    await _safe_answer(call)
    await _show_hires_menu(call)

def _preset_id(name: str) -> str:
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:8]

def _preset_summary(preset: dict) -> str:
    model = preset.get("sd_model") or "по умолч."
    hires = f", hires {preset.get('hires_scale')}x" if preset.get("hires_enabled") else ""
    return f"{_cut(model, 22)} · {preset.get('width')}×{preset.get('height')} · {preset.get('steps')} шагов{hires}"

async def _show_presets_menu(chat_id: int, user_id: str, message_id: Optional[int] = None) -> None:
    presets = await list_presets(user_id)
    lines = [f"• <b>{_h(name)}</b> — {_h(_preset_summary(p))}" for name, p in presets.items()]
    text = ("📦 <b>Пресеты</b>\n\nСохраняют модель, LoRA, негатив и эмбеддинги, размер, шаги, CFG, сэмплер и Hires — "
            "переключение одним нажатием.\n\n" + ("\n".join(lines) if lines else "Пока нет ни одного. Настройте бота и нажмите «Сохранить текущие».")
            + f"\n\nСохранено: {len(presets)}/{MAX_PRESETS}")
    markup = types.InlineKeyboardMarkup(row_width=2)
    for name in presets:
        markup.row(types.InlineKeyboardButton(f"▶️ {_cut(name, 24)}", callback_data=f"preset_use_{_preset_id(name)}"),
                   types.InlineKeyboardButton("🗑", callback_data=f"preset_del_{_preset_id(name)}"))
    markup.add(types.InlineKeyboardButton("➕ Сохранить текущие настройки", callback_data="preset_save"))
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
    await _edit_or_send(chat_id, message_id, text, markup)

async def _preset_by_id(user_id: str, pid: str) -> Optional[str]:
    return next((n for n in await list_presets(user_id) if _preset_id(n) == pid), None)

@router.callback_query(F.data == "set_presets")
async def handle_presets_menu(call: CallbackQuery) -> None:
    await _safe_answer(call)
    await _show_presets_menu(call.message.chat.id, str(call.message.chat.id), call.message.message_id)

@router.callback_query(F.data == "preset_save")
async def handle_preset_save(call: CallbackQuery) -> None:
    await _safe_answer(call)
    msg = await bot.send_message(call.message.chat.id, f"✏️ Введите название пресета (до 24 символов).\n"
                                 "Если такое название уже есть — оно будет перезаписано. /cancel — отмена.")
    register_next_step(msg.chat.id, save_preset_name, call.message.message_id)

async def save_preset_name(message: Message, menu_message_id: int) -> None:
    user_id, name = str(message.chat.id), " ".join(_text(message).split())
    if _is_cancel_text(name) or len(name) > 24:
        await bot.send_message(message.chat.id, "❌ Отменено: нужно короткое текстовое название (до 24 символов).")
        return
    result = await save_preset(user_id, name)
    if result == "limit":
        await bot.send_message(message.chat.id, f"⚠️ Достигнут лимит {MAX_PRESETS} пресетов — удалите ненужный (🗑).")
    else:
        await bot.send_message(message.chat.id, f"✅ Пресет «{name}» {'обновлён' if result == 'updated' else 'сохранён'}")
    await _show_presets_menu(message.chat.id, user_id, None)
    with suppress(Exception):
        await bot.delete_message(message.chat.id, menu_message_id)

@router.callback_query(F.data.startswith("preset_use_"))
async def handle_preset_use(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    name = await _preset_by_id(user_id, call.data[len("preset_use_"):])
    if name is None or not await apply_preset(user_id, name):
        await _safe_answer(call, "Пресет не найден", show_alert=True)
        return
    await _safe_answer(call, f"✅ Применён: {name}"[:200])
    with suppress(Exception):
        await bot.delete_message(call.message.chat.id, call.message.message_id)
    await show_settings(call.message)

@router.callback_query(F.data.startswith("preset_del_"))
async def handle_preset_delete(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    name = await _preset_by_id(user_id, call.data[len("preset_del_"):])
    if name is not None:
        await delete_preset(user_id, name)
    await _safe_answer(call, f"🗑 Удалён: {name}" if name else "Уже удалён")
    await _show_presets_menu(call.message.chat.id, user_id, call.message.message_id)

# =========== ОБРАБОТЧИК ТЕКСТА (ПРОМПТЫ) ===========

@router.message(F.text)
async def handle_quick_generation(message: Message) -> None:
    # Игнорируем команды, ответы, кнопки и слишком короткие сообщения
    if not message.text or message.text.startswith('/'):
        return
    if message.reply_to_message:
        return
    if message.text in BUTTON_TEXTS:
        return
    if any(message.text.startswith(emoji) for emoji in ['🚀', '⚙️', '🔍', '🎭', '🎮']):
        return
    if len(message.text.strip()) < 2:
        return
    user_id = str(message.chat.id)
    prompt = message.text.strip()
    settings = await get_user_settings(user_id)
    # --- Логирование промпта пользователя (до начала генерации) ---
    gen_id = await prompt_logger.log_request(
        user_id, "txt2img", prompt,
        settings=settings,
        negative_prompt=await get_user_negative_prompt(user_id),
    )
    async def job(task: Task):
        start_time = time.time()
        on_progress, on_status = make_callbacks(task)
        try:
            face_ctx = await get_face_context(user_id)
            if face_ctx and face_ctx.note:
                await on_status(f"⚠️ {face_ctx.note}")
            image_file, used_seed, original_prompt, processed_prompt = await generate_image(
                prompt, settings, user_id, on_progress, on_status, fixed_seed=None,
                face_image=face_ctx.image if face_ctx and face_ctx.use_faceid else None,
                face_weights=face_ctx.weights if face_ctx else None
            )
            final_image_file = image_file
            swapped = False
            if face_ctx and face_ctx.use_swap:
                await on_status("🎭 Заменяю лицо...")
                final_image_file, swapped = await apply_face_swap(face_ctx, image_file)
            face_replacement_success = bool(face_ctx and (face_ctx.use_faceid or swapped))
            try:
                await bot.delete_message(message.chat.id, task.status_message_id)
            except Exception:
                pass
            caption = (
                f"✨ <b>Исходный промпт:</b>\n<code>{_h(_cut(original_prompt, 700))}</code>\n\n"
                f"🤖 <b>Обработанный промпт:</b>\n<code>{_h(_cut(processed_prompt, 1200))}</code>\n\n"
                f"📐 <b>Размер:</b> {settings.get('width', 512)}x{settings.get('height', 768)}\n"
                f"⏱️ <b>Шаги:</b> {settings.get('steps', 20)}\n"
                f"🎚️ <b>CFG:</b> {settings.get('cfg_scale', 7.0)}\n"
                f"🎯 <b>Сэмплер:</b> <code>{_h(settings.get('sampler_name'))}</code>\n"
                f"🎲 <b>Seed:</b> <code>{used_seed}</code>\n"
                + (f"🔬 <b>Hires:</b> {hires_plan(settings)['scale']}x → {hires_plan(settings)['width']}×{hires_plan(settings)['height']}\n"
                   if hires_plan(settings) else "")
            )
            caption += _face_caption_line(face_ctx, swapped)
            # Фото либо файл (если не проходит по весу/разрешению), подпись длиннее 1024 — отдельным сообщением
            sent_msg = await send_image_smart(message.chat.id, final_image_file, caption=caption, parse_mode="HTML",
                                        reply_to_message_id=message.message_id)
            # В конфиг пишем ПОЛНУЮ копию настроек (с моделью, сэмплером и т.д.) и итоговые промпты, ушедшие в WebUI:
            # по ним «Повторить» воспроизводит ту же картинку (раньше терялись сэмплер и модель, а промпт обрабатывался заново)
            config = {
                "type": "txt2img",
                "prompt": processed_prompt,
                "original_prompt": original_prompt,
                "final_prompt": getattr(image_file, "final_prompt", None),
                "final_negative": getattr(image_file, "final_negative", None),
                "settings": dict(settings),
                "seed": used_seed,
                "face_enabled": face_replacement_success
            }
            last_configs[user_id] = config
            # --- Логирование результата генерации ---
            await prompt_logger.log_result(
                gen_id, user_id, "txt2img", original_prompt,
                processed_prompt=processed_prompt, seed=used_seed,
                settings=settings, duration_sec=time.time() - start_time,
                face_replaced=face_replacement_success,
            )
            last_images[user_id] = final_image_file
            last_message_ids[user_id] = sent_msg.message_id
            last_seeds[user_id] = used_seed
            await bot.send_message(message.chat.id, "🔧 **Действия с изображением:**", reply_markup=get_image_actions_keyboard(), reply_to_message_id=sent_msg.message_id)
        except asyncio.CancelledError:
            await prompt_logger.log_error(gen_id, user_id, "txt2img", prompt, "Отменено пользователем", duration_sec=time.time() - start_time)
            raise
        except Exception as e:
            await prompt_logger.log_error(gen_id, user_id, "txt2img", prompt, e, duration_sec=time.time() - start_time)
            await _report_error(message.chat.id, task.status_message_id, e)
    await enqueue(message.chat.id, user_id, "txt2img", job, reply_to_message_id=message.message_id)

# =========== ОСТАЛЬНЫЕ ОБРАБОТЧИКИ (ЛИЦА, НАСТРОЙКИ, АПСКЕЙЛ и т.д.) ===========

@router.callback_query(F.data == "face_toggle_main")
async def handle_face_toggle_main(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    current = await get_face_enabled(user_id)
    new_state = not current
    await set_face_enabled(user_id, new_state)
    status = "включена ✅" if new_state else "выключена ❌"
    await bot.answer_callback_query(call.id, f"Лицо в генерации {status}")
    await face_management(call.message)

@router.callback_query(F.data.in_({"face_weight_faceid", "face_weight_lora"}))
async def handle_face_weight(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    kind = "faceid" if call.data == "face_weight_faceid" else "lora"
    value = await cycle_face_weight(user_id, kind)
    label = "Сила лица" if kind == "faceid" else "Вес LoRA лица"
    await _safe_answer(call, f"{label}: {'выкл.' if kind == 'lora' and value <= 0 else format(value, 'g')}")
    await face_management(call.message)

@router.callback_query(F.data == "face_mode_cycle")
async def handle_face_mode_cycle(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    current = await get_face_mode(user_id)
    new_mode = FACE_MODES[(FACE_MODES.index(current) + 1) % len(FACE_MODES)]
    await set_face_mode(user_id, new_mode)
    await _safe_answer(call, f"Режим: {FACE_MODE_LABELS[new_mode]}")
    await face_management(call.message)

@router.callback_query(F.data == "face_add")
async def handle_face_add(call: CallbackQuery) -> None:
    msg = await bot.send_message(call.message.chat.id, "📸 **Добавление нового лица**\n\nОтправьте **четкое фото лица** анфас.\n\n**Рекомендации:**\n• Хорошее освещение, без теней\n• Лицо занимает 70-80% кадра\n• Нейтральное выражение\n• Минимум аксессуаров\n\nПосле отправки фото, вы сможете задать имя для этого лица.", parse_mode="Markdown")
    register_next_step(msg.chat.id, save_new_face, call.message.chat.id)

async def save_new_face(message, original_chat_id):
    user_id = str(message.chat.id)
    # Принимаем и обычное фото, и изображение, отправленное «как файл» (без сжатия — для лица это лучше)
    document = getattr(message, "document", None)
    if message.photo:
        file_id = message.photo[-1].file_id
    elif document and (getattr(document, "mime_type", None) or "").startswith("image/"):
        file_id = document.file_id
    else:
        await bot.send_message(message.chat.id, "❌ Пожалуйста, отправьте фото.")
        return
    try:
        file_info = await bot.get_file(file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        # Лица хранятся как JPEG: приводим формат к единому и заодно проверяем, что это действительно картинка
        def to_jpeg(raw: bytes) -> bytes:
            with Image.open(io.BytesIO(raw)) as face_img:
                converted = io.BytesIO()
                face_img.convert("RGB").save(converted, format="JPEG", quality=95)
                return converted.getvalue()

        try:
            jpeg_bytes = await asyncio.to_thread(to_jpeg, downloaded_file)
        except Exception:
            await bot.send_message(message.chat.id, "❌ Не удалось прочитать изображение. Отправьте фото в формате JPG/PNG.")
            return
        ok, reason = await check_face_photo(jpeg_bytes)
        if not ok:
            await bot.send_message(message.chat.id, f"❌ {reason}\nОтправьте другое фото.")
            register_next_step(message.chat.id, save_new_face, original_chat_id)
            return
        temp_face = io.BytesIO(jpeg_bytes)
        temp_data.setdefault(user_id, {})['temp_face'] = temp_face
        msg = await bot.send_message(message.chat.id, "✏️ **Введите имя для этого лица**\n\nНапример: 'я', 'друг', 'коллега' и т.д.\n\nИли отправьте 'default' для стандартного имени.", parse_mode="Markdown")
        register_next_step(msg.chat.id, set_face_name, original_chat_id)
    except Exception as e:
        await bot.send_message(message.chat.id, f"❌ Ошибка: {e}")

async def set_face_name(message, original_chat_id):
    user_id = str(message.chat.id)
    face_name = (message.text or "").strip().lower()
    if face_name == "default":
        existing_faces = await get_user_faces(user_id)
        face_name = f"face_{len(existing_faces) + 1}"
    if not _valid_face_name(face_name):
        msg = await bot.send_message(message.chat.id, "❌ Некорректное имя (до 32 символов, без / \\ : * ? \" < > |). Введите другое имя:")
        register_next_step(msg.chat.id, set_face_name, original_chat_id)
        return
    faces = await get_user_faces(user_id)
    if any(f["name"] == face_name for f in faces):
        msg = await bot.send_message(message.chat.id, f"❌ Лицо с именем '{face_name}' уже существует. Введите другое имя:")
        register_next_step(msg.chat.id, set_face_name, original_chat_id)
        return
    temp_face = temp_data.get(user_id, {}).get('temp_face')
    if temp_face is not None:
        success = await save_user_face(user_id, temp_face, face_name)
        if success:
            await bot.send_message(message.chat.id, f"✅ <b>Лицо '{_h(face_name)}' успешно сохранено!</b>\n\nТеперь вы можете выбрать его как активное в меню управления лицами.", parse_mode="HTML")
            preview = await get_face_preview(user_id, face_name)
            if preview:
                await bot.send_photo(message.chat.id, preview, caption=f"📸 {face_name}")
            await _build_model_and_report(message.chat.id, user_id, face_name)
        else:
            await bot.send_message(message.chat.id, "❌ Ошибка при сохранении лица.")
        temp_data.get(user_id, {}).pop('temp_face', None)
    else:
        await bot.send_message(message.chat.id, "❌ Ошибка: фото не найдено.")

# --- Вспомогательные функции для меню лиц ---------------------------------
# Причина бага с удалением: после выбора лица бот отправляет ФОТО-сообщение
# (превью), а кнопки «Удалить» / «Назад» пытались сделать edit_message_text
# именно для него. Telegram не позволяет редактировать текст у фото-сообщения
# (400: there is no text in the message to edit), поэтому ничего не происходило.
# Также имя лица в callback_data могло превысить лимит Telegram в 64 байта
# (кириллица = 2 байта на символ), поэтому теперь в кнопках передаётся короткий id.

def _face_id(face_name):
    return hashlib.md5(face_name.encode("utf-8")).hexdigest()[:10]

async def _find_face_by_id(user_id, face_id):
    faces = await get_user_faces(user_id)
    return next((f for f in faces if _face_id(f["name"]) == face_id), None)

async def _replace_message(call, text, markup=None):
    """Заменяет сообщение с кнопками на текстовое (работает и для фото-сообщений)."""
    chat_id = call.message.chat.id
    message_id = call.message.message_id
    if getattr(call.message, "content_type", "text") == "text":
        try:
            await bot.edit_message_text(text, chat_id, message_id, parse_mode="HTML", reply_markup=markup)
            return
        except Exception as e:
            if "message is not modified" in str(e):
                return
    try:
        await bot.delete_message(chat_id, message_id)
    except Exception:
        pass
    await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)

def _valid_face_name(name):
    return bool(name) and len(name) <= 32 and not any(c in name for c in '/\\:*?"<>|') and name not in (".", "..")

# ---------------------------------------------------------------------------

@router.callback_query(F.data == "face_list")
async def handle_face_list(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    faces = await get_user_faces(user_id)
    markup = types.InlineKeyboardMarkup(row_width=1)
    if not faces:
        markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="face_back"))
        await _replace_message(call, "📭 У вас нет сохранённых лиц.", markup)
        await _safe_answer(call)
        return
    for face in faces:
        status = ("⭐ " if face["is_active"] else "📸 ") + ("🧬 " if face.get("model") else "")
        markup.add(types.InlineKeyboardButton(f"{status}{face['name']}", callback_data=f"face_select_{_face_id(face['name'])}"))
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="face_back"))
    await _replace_message(call, "📋 <b>Список ваших лиц</b>\n\nВыберите лицо для управления:", markup)
    await _safe_answer(call)

@router.callback_query(F.data.startswith("face_select_"))
async def handle_face_select(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    face = await _find_face_by_id(user_id, call.data[len("face_select_"):])
    if not face:
        await _safe_answer(call, "Лицо не найдено")
        return
    face_name = face["name"]
    fid = _face_id(face_name)
    markup = types.InlineKeyboardMarkup(row_width=2)
    if not face["is_active"]:
        markup.add(types.InlineKeyboardButton("⭐ Сделать активным", callback_data=f"face_activate_{fid}"))
    markup.add(types.InlineKeyboardButton(
        "🧬 Пересоздать модель ReActor" if face.get("model") else "🧬 Создать модель ReActor",
        callback_data=f"face_mkmodel_{fid}"))
    markup.add(
        types.InlineKeyboardButton("✏️ Переименовать", callback_data=f"face_rename_{fid}"),
        types.InlineKeyboardButton("🗑️ Удалить", callback_data=f"face_delete_{fid}"),
        types.InlineKeyboardButton("◀️ Назад к списку", callback_data="face_list")
    )
    preview = await get_face_preview(user_id, face_name)
    status_text = "⭐ Активно" if face["is_active"] else "📸 Не активно"
    model_text = "🧬 Модель ReActor создана" if face.get("model") else "▫️ Модели ReActor нет"
    caption = f"<b>{html.escape(face_name)}</b>\n{status_text}\n{model_text}"
    if preview:
        await bot.send_photo(call.message.chat.id, preview, caption=caption, parse_mode="HTML", reply_markup=markup)
        try:
            await bot.delete_message(call.message.chat.id, call.message.message_id)
        except Exception:
            pass
    else:
        await _replace_message(call, caption, markup)
    await _safe_answer(call)

@router.callback_query(F.data.startswith("face_activate_"))
async def handle_face_activate(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    face = await _find_face_by_id(user_id, call.data[len("face_activate_"):])
    if face and await set_active_face(user_id, face["name"]):
        await _safe_answer(call, f"Активное лицо: {face['name']}")
    else:
        await _safe_answer(call, "Ошибка при активации", show_alert=True)
        return
    await handle_face_list(call)

@router.callback_query(F.data.startswith("face_rename_"))
async def handle_face_rename(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    face = await _find_face_by_id(user_id, call.data[len("face_rename_"):])
    if not face:
        await _safe_answer(call, "Лицо не найдено")
        return
    old_name = face["name"]
    msg = await bot.send_message(call.message.chat.id, f"✏️ <b>Введите новое имя для '{html.escape(old_name)}':</b>", parse_mode="HTML")
    register_next_step(msg.chat.id, process_face_rename, user_id, old_name, call.message.chat.id)
    await _safe_answer(call)

async def process_face_rename(message, user_id, old_name, chat_id):
    new_name = (message.text or "").strip().lower()
    if not _valid_face_name(new_name):
        await bot.send_message(chat_id, "❌ Некорректное имя (до 32 символов, без / \\ : * ? \" < > |).")
        return
    if new_name == old_name:
        await bot.send_message(chat_id, "❌ Имя не изменено.")
        return
    faces = await get_user_faces(user_id)
    if any(f["name"] == new_name for f in faces):
        await bot.send_message(chat_id, f"❌ Лицо с именем '{new_name}' уже существует.")
        return
    success = await rename_face(user_id, old_name, new_name)
    if success:
        await bot.send_message(chat_id, f"✅ Лицо переименовано: '{old_name}' → '{new_name}'")
    else:
        await bot.send_message(chat_id, "❌ Ошибка при переименовании.")

@router.callback_query(F.data.startswith("face_delete_"))
async def handle_face_delete(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    face = await _find_face_by_id(user_id, call.data[len("face_delete_"):])
    if not face:
        await _safe_answer(call, "Лицо не найдено")
        return
    fid = _face_id(face["name"])
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("✅ Да, удалить", callback_data=f"face_confirm_delete_{fid}"),
        types.InlineKeyboardButton("❌ Нет, отмена", callback_data="face_list")
    )
    await _replace_message(call, f"⚠️ <b>Удалить лицо '{html.escape(face['name'])}'?</b>\n\nЭто действие нельзя отменить.", markup)
    await _safe_answer(call)

@router.callback_query(F.data.startswith("face_confirm_delete_"))
async def handle_face_confirm_delete(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    face = await _find_face_by_id(user_id, call.data[len("face_confirm_delete_"):])
    if not face:
        await _safe_answer(call, "Лицо не найдено или уже удалено")
        await handle_face_list(call)
        return
    face_name = face["name"]
    if await delete_face(user_id, face_name):
        await _safe_answer(call, f"Лицо '{face_name}' удалено")
        await handle_face_list(call)
    else:
        await _safe_answer(call, "Ошибка при удалении", show_alert=True)

@router.callback_query(F.data == "face_back")
async def handle_face_back(call: CallbackQuery) -> None:
    await face_management(call.message)

# --- Модели лиц ReActor ----------------------------------------------------
# Модель лица — это .safetensors с усреднённым эмбеддингом лица (models/reactor/faces на стороне WebUI).
# ReActor применяет её быстрее и стабильнее, чем разбор исходного фото при каждой замене.

MODEL_MAX_PHOTOS = 10
_model_sessions: dict = {}   # user_id -> {"name": str, "photos": [jpeg bytes]}

async def _build_model_and_report(chat_id, user_id, face_name, extra_images=None, quiet_unavailable=True):
    """Строит модель ReActor и сообщает результат. Ошибки не роняют сценарий: лицо остаётся доступным как фото."""
    if not await check_reactor():
        if not quiet_unavailable:
            await bot.send_message(chat_id, "⚠️ ReActor недоступен — модель лица создать нельзя (лицо сохранено как фото).")
        return False
    status = await bot.send_message(chat_id, "🧬 Создаю модель лица ReActor...")
    try:
        model_file = await build_face_model(user_id, face_name, extra_images)
        text = f"🧬 <b>Модель лица создана:</b> <code>{_h(model_file)}</code>\nТеперь замена будет идти по модели."
        ok = True
    except Exception as e:
        text = f"⚠️ Не удалось создать модель ReActor: {_h(_cut(e, 500))}\nЛицо сохранено, замена пойдёт по фото."
        ok = False
    try:
        await bot.edit_message_text(text, chat_id, status.message_id, parse_mode="HTML")
    except Exception:
        await bot.send_message(chat_id, text, parse_mode="HTML")
    return ok

async def _download_face_jpeg(message) -> Optional[bytes]:
    """Скачивает фото (или картинку-файл) из сообщения и приводит к JPEG. None — если это не изображение."""
    document = getattr(message, "document", None)
    if message.photo:
        file_id = message.photo[-1].file_id
    elif document and (getattr(document, "mime_type", None) or "").startswith("image/"):
        file_id = document.file_id
    else:
        return None
    file_info = await bot.get_file(file_id)
    raw = await bot.download_file(file_info.file_path)

    def to_jpeg(data: bytes) -> bytes:
        with Image.open(io.BytesIO(data)) as img:
            out = io.BytesIO()
            img.convert("RGB").save(out, format="JPEG", quality=95)
            return out.getvalue()

    try:
        return await asyncio.to_thread(to_jpeg, raw if isinstance(raw, bytes) else raw.read())
    except Exception:
        return None

@router.callback_query(F.data.startswith("face_mkmodel_"))
async def handle_face_mkmodel(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    face = await _find_face_by_id(user_id, call.data[len("face_mkmodel_"):])
    if not face:
        await _safe_answer(call, "Лицо не найдено")
        return
    await _safe_answer(call, "Создаю модель...")
    await _build_model_and_report(call.message.chat.id, user_id, face["name"], quiet_unavailable=False)

@router.callback_query(F.data == "face_model_new")
async def handle_face_model_new(call: CallbackQuery) -> None:
    await _safe_answer(call)
    msg = await bot.send_message(
        call.message.chat.id,
        "🧬 <b>Модель лица из нескольких фото</b>\n\n"
        "Введите имя для нового лица (до 32 символов). Затем пришлите 3–10 фото одного человека "
        "(можно альбомом, лучше «как файл»), а в конце отправьте слово <b>готово</b>.\n\n"
        "Чем разнообразнее ракурсы и освещение при чётком лице, тем точнее модель.",
        parse_mode="HTML")
    register_next_step(msg.chat.id, model_name_step, call.message.chat.id)

async def model_name_step(message, original_chat_id):
    user_id = str(message.chat.id)
    name = (message.text or "").strip().lower()
    faces = await get_user_faces(user_id)
    if name == "default":
        name = f"face_{len(faces) + 1}"
    if not _valid_face_name(name):
        msg = await bot.send_message(message.chat.id, "❌ Некорректное имя (до 32 символов, без / \\ : * ? \" < > |). Введите другое имя:")
        register_next_step(msg.chat.id, model_name_step, original_chat_id)
        return
    if any(f["name"] == name for f in faces):
        msg = await bot.send_message(message.chat.id, f"❌ Лицо '{name}' уже существует. Введите другое имя:")
        register_next_step(msg.chat.id, model_name_step, original_chat_id)
        return
    _model_sessions[user_id] = {"name": name, "photos": []}
    msg = await bot.send_message(message.chat.id, f"📸 Присылайте фото для «{_h(name)}» (до {MODEL_MAX_PHOTOS}). Когда закончите — напишите <b>готово</b>.", parse_mode="HTML")
    register_next_step(msg.chat.id, model_collect_step, original_chat_id)

async def model_collect_step(message, original_chat_id):
    user_id = str(message.chat.id)
    session = _model_sessions.get(user_id)
    if session is None:
        await bot.send_message(message.chat.id, "❌ Сессия создания модели не найдена. Начните заново из меню лиц.")
        return
    text = (message.text or "").strip().lower()
    if text in ("готово", "done", "/done"):
        _model_sessions.pop(user_id, None)
        photos = session["photos"]
        if not photos:
            await bot.send_message(message.chat.id, "❌ Фото не получено, модель не создана.")
            return
        name = session["name"]
        if not await save_user_face(user_id, io.BytesIO(photos[0]), name):
            await bot.send_message(message.chat.id, "❌ Ошибка при сохранении лица.")
            return
        await bot.send_message(message.chat.id, f"✅ Лицо <b>{_h(name)}</b> сохранено ({len(photos)} фото).", parse_mode="HTML")
        await _build_model_and_report(message.chat.id, user_id, name, extra_images=photos, quiet_unavailable=False)
        return
    # Перерегистрируемся СРАЗУ: фото альбома приходят пачкой и не должны проскакивать в другие обработчики
    register_next_step(message.chat.id, model_collect_step, original_chat_id)
    if len(session["photos"]) >= MODEL_MAX_PHOTOS:
        await bot.send_message(message.chat.id, f"Достигнут лимит {MODEL_MAX_PHOTOS} фото — напишите <b>готово</b>.", parse_mode="HTML")
        return
    try:
        jpeg = await _download_face_jpeg(message)
    except Exception as e:
        await bot.send_message(message.chat.id, f"❌ Ошибка загрузки: {_h(_cut(e, 200))}", parse_mode="HTML")
        return
    if jpeg is None:
        await bot.send_message(message.chat.id, "❌ Нужно фото (или картинка файлом). Либо напишите <b>готово</b>.", parse_mode="HTML")
        return
    ok, reason = await check_face_photo(jpeg)
    if not ok:
        await bot.send_message(message.chat.id, f"❌ Фото пропущено: {_h(reason)}", parse_mode="HTML")
        return
    session["photos"].append(jpeg)
    await bot.send_message(message.chat.id, f"✅ Принято фото {len(session['photos'])}/{MODEL_MAX_PHOTOS}. Ещё или <b>готово</b>.", parse_mode="HTML")

# =========== НАСТРОЙКИ (CALLBACK) ===========

@router.callback_query(F.data == "toggle_prompt_processing")
async def handle_toggle_prompt_processing(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    current = await get_prompt_processing_enabled(user_id)
    new_state = not current
    await set_prompt_processing_enabled(user_id, new_state)
    status = "включена ✅" if new_state else "выключена ❌"
    await bot.answer_callback_query(call.id, f"Обработка промпта {status}")
    await show_settings(call.message)

@router.callback_query(F.data == "set_diversity")
async def handle_diversity(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    settings = await get_user_settings(user_id)
    modes = ["auto", "high", "low", "off"]
    current = settings.get("diversity_mode", "auto")
    if current not in modes:
        current = "auto"
    next_index = (modes.index(current) + 1) % len(modes)
    new_mode = modes[next_index]
    await set_diversity_mode(user_id, new_mode)
    settings = await get_user_settings(user_id)
    try:
        await bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=await get_settings_inline_keyboard(settings, user_id))
    except Exception as e:
        if "message is not modified" not in str(e):
            raise
    await _safe_answer(call, f"Разнообразие: {new_mode}")

@router.callback_query(F.data == "set_res")
async def handle_resolution_button(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    settings = await get_user_settings(user_id)
    await bot.edit_message_text("📐 **Выбери разрешение:**\n\nДоступны размеры от 512x512 до 1024x1280", call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_resolution_keyboard(settings['width'], settings['height']))
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("res_"))
async def handle_resolution_choice(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    _, w, h = call.data.split("_")
    w, h = int(w), int(h)
    await update_user_settings(user_id, {"width": w, "height": h})
    await bot.answer_callback_query(call.id, f"Размер установлен: {w}x{h}")
    await show_settings(call.message)

@router.callback_query(F.data == "set_cfg")
async def handle_cfg_button(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    settings = await get_user_settings(user_id)
    await bot.edit_message_text("🎚️ **Выбери CFG Scale:**\n\n• 5-6: больше креатива\n• 7-8: золотая середина\n• 9-10: строгое следование промпту", call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_cfg_keyboard(settings['cfg_scale']))
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("cfg_"))
async def handle_cfg_choice(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    cfg = float(call.data.split("_")[1])
    await update_user_settings(user_id, {"cfg_scale": cfg})
    await bot.answer_callback_query(call.id, f"CFG установлен: {cfg}")
    await show_settings(call.message)

@router.callback_query(F.data == "set_steps")
async def handle_steps_button(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    settings = await get_user_settings(user_id)
    await bot.edit_message_text("⏱️ **Выбери количество шагов:**\n\n• 15-20: быстро\n• 25-35: оптимально\n• 40-50: максимально", call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_steps_keyboard(settings['steps']))
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("steps_"))
async def handle_steps_choice(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    steps = int(call.data.split("_")[1])
    await update_user_settings(user_id, {"steps": steps})
    await bot.answer_callback_query(call.id, f"Шагов: {steps}")
    await show_settings(call.message)

# =========== НЕГАТИВНЫЙ ПРОМПТ И НЕГАТИВНЫЕ ЭМБЕДДИНГИ ===========

EMBEDDINGS_PER_PAGE = 10

def _emb_id(name):
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:10]

async def _edit_or_send(chat_id, message_id, text, markup):
    """Редактирует сообщение меню (если можно), иначе отправляет новое."""
    if message_id is not None:
        try:
            await bot.edit_message_text(text, chat_id, message_id, parse_mode="HTML", reply_markup=markup)
            return
        except Exception as e:
            if "message is not modified" in str(e):
                return
    await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)

async def _show_negative_menu(chat_id, user_id, message_id=None):
    negative = await get_user_negative_prompt(user_id)
    is_default = negative == DEFAULT_NEGATIVE_PROMPT
    embeddings = await get_user_negative_embeddings(user_id)
    text = (
        "🚫 <b>Негативный промпт</b>\n\n"
        f"<b>Текст{' (стандартный)' if is_default else ''}:</b>\n<code>{_h(_cut(negative, 400))}</code>\n\n"
        f"🧬 <b>Эмбеддинги:</b> {_h(_cut(', '.join(embeddings), 400)) if embeddings else 'не выбраны'}\n\n"
        "Эмбеддинги берутся из WebUI и добавляются в негативный промпт при каждой генерации."
    )
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("✏️ Изменить текст", callback_data="neg_text"),
        types.InlineKeyboardButton(f"🧬 Эмбеддинги ({len(embeddings)})", callback_data="neg_emb"),
    )
    if not is_default:
        markup.add(types.InlineKeyboardButton("↩️ Сбросить текст на стандартный", callback_data="neg_reset"))
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
    await _edit_or_send(chat_id, message_id, text, markup)

@router.callback_query(F.data == "set_neg")
async def handle_neg_button(call: CallbackQuery) -> None:
    await _safe_answer(call)
    await _show_negative_menu(call.message.chat.id, str(call.message.chat.id), call.message.message_id)

@router.callback_query(F.data == "neg_text")
async def handle_neg_text_button(call: CallbackQuery) -> None:
    await _safe_answer(call)
    msg = await bot.send_message(call.message.chat.id, "✏️ Отправь свой негативный промпт\n\nИли отправь <code>сбросить</code> для стандартного", parse_mode="HTML")
    register_next_step(msg.chat.id, save_custom_negative, call.message.chat.id, call.message.message_id)

async def save_custom_negative(message, chat_id, original_msg_id):
    user_id = str(message.chat.id)
    new_neg = _text(message)
    if not new_neg or new_neg.startswith('/') or new_neg in BUTTON_TEXTS:
        # Фото, стикер, команда или кнопка вместо текста раньше сохранялись/роняли обработчик
        await bot.send_message(chat_id, "❌ Отменено: нужен текст негативного промпта.")
        return
    if new_neg.lower() == "сбросить":
        await set_user_negative_prompt(user_id, None)
        await bot.send_message(chat_id, "✅ Сброшено до стандартного!")
    else:
        await set_user_negative_prompt(user_id, new_neg)
        await bot.send_message(chat_id, f"✅ Сохранено:\n<code>{_h(_cut(new_neg, 100))}</code>", parse_mode="HTML")
    try:
        await bot.delete_message(chat_id, original_msg_id)
    except Exception:
        pass
    await show_settings(message)

@router.callback_query(F.data == "neg_reset")
async def handle_neg_reset(call: CallbackQuery) -> None:
    await set_user_negative_prompt(str(call.message.chat.id), None)
    await _safe_answer(call, "Сброшено до стандартного")
    await _show_negative_menu(call.message.chat.id, str(call.message.chat.id), call.message.message_id)

async def _build_embeddings_menu(user_id, page):
    names = await get_available_embeddings()
    markup = types.InlineKeyboardMarkup(row_width=1)
    if names is None or not names:
        if names is None:
            text = ("⚠️ <b>Не удалось получить список эмбеддингов.</b>\n\n"
                    "Проверьте, что WebUI запущен с флагом <code>--api</code>, и нажмите «Обновить».")
        else:
            text = ("📭 <b>В WebUI нет загруженных эмбеддингов.</b>\n\n"
                    "Положите файлы в папку <code>embeddings</code> и нажмите «Обновить».")
        markup.add(types.InlineKeyboardButton("🔄 Обновить список", callback_data="negemb_refresh"))
        markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="set_neg"))
        return text, markup

    selected = set(await get_user_negative_embeddings(user_id))
    pages = (len(names) + EMBEDDINGS_PER_PAGE - 1) // EMBEDDINGS_PER_PAGE
    page = max(0, min(page, pages - 1))
    text = (
        "🧬 <b>Негативные эмбеддинги</b>\n\n"
        f"Выбрано: {len(selected)} из {len(names)}\n"
        f"{_h(_cut(', '.join(sorted(selected, key=str.lower)), 500)) if selected else 'Ничего не выбрано'}\n\n"
        "Нажимайте на эмбеддинги, чтобы включать и выключать их — можно выбрать несколько."
    )
    for name in names[page * EMBEDDINGS_PER_PAGE:(page + 1) * EMBEDDINGS_PER_PAGE]:
        label = ("✅ " if name in selected else "⬜ ") + _cut(name, 48)
        markup.add(types.InlineKeyboardButton(label, callback_data=f"negemb_t_{_emb_id(name)}_{page}"))
    if pages > 1:
        markup.row(
            types.InlineKeyboardButton("◀️", callback_data=f"negemb_p_{(page - 1) % pages}"),
            types.InlineKeyboardButton(f"{page + 1}/{pages}", callback_data="negemb_noop"),
            types.InlineKeyboardButton("▶️", callback_data=f"negemb_p_{(page + 1) % pages}"),
        )
    markup.row(
        types.InlineKeyboardButton("🔄 Обновить", callback_data="negemb_refresh"),
        types.InlineKeyboardButton("🧹 Снять все", callback_data="negemb_clear"),
    )
    markup.add(types.InlineKeyboardButton("↩️ Рекомендованные по умолчанию", callback_data="negemb_default"))
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="set_neg"))
    return text, markup

async def _show_embeddings_menu(call, page):
    text, markup = await _build_embeddings_menu(str(call.message.chat.id), page)
    await _edit_or_send(call.message.chat.id, call.message.message_id, text, markup)

@router.callback_query(F.data == "neg_emb")
async def handle_neg_embeddings_button(call: CallbackQuery) -> None:
    await _safe_answer(call)
    await _show_embeddings_menu(call, 0)

@router.callback_query(F.data.startswith("negemb_p_"))
async def handle_negemb_page(call: CallbackQuery) -> None:
    try:
        page = int(call.data[len("negemb_p_"):])
    except ValueError:
        page = 0
    await _safe_answer(call)
    await _show_embeddings_menu(call, page)

@router.callback_query(F.data == "negemb_noop")
async def handle_negemb_noop(call: CallbackQuery) -> None:
    await _safe_answer(call)

@router.callback_query(F.data.startswith("negemb_t_"))
async def handle_negemb_toggle(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    try:
        emb_id, page_text = call.data[len("negemb_t_"):].rsplit("_", 1)
        page = int(page_text)
    except ValueError:
        await _safe_answer(call)
        return
    available_embeddings = await get_available_embeddings() or []
    name = next((n for n in available_embeddings if _emb_id(n) == emb_id), None)
    if name is None:
        await _safe_answer(call, "Эмбеддинг не найден — обновите список", show_alert=True)
        return
    state = await toggle_user_negative_embedding(user_id, name)
    await _safe_answer(call, f"{'✅ Добавлен' if state else '➖ Убран'}: {name}"[:200])
    await _show_embeddings_menu(call, page)

@router.callback_query(F.data == "negemb_clear")
async def handle_negemb_clear(call: CallbackQuery) -> None:
    await set_user_negative_embeddings(str(call.message.chat.id), [])
    await _safe_answer(call, "Выбор снят")
    await _show_embeddings_menu(call, 0)

@router.callback_query(F.data == "negemb_default")
async def handle_negemb_default(call: CallbackQuery) -> None:
    await set_user_negative_embeddings(str(call.message.chat.id), None)
    await _safe_answer(call, "Возвращены рекомендованные")
    await _show_embeddings_menu(call, 0)

@router.callback_query(F.data == "negemb_refresh")
async def handle_negemb_refresh(call: CallbackQuery) -> None:
    await _safe_answer(call, "Обновляю список...")
    await refresh_embeddings_on_server()
    await get_available_embeddings(force_refresh=True)
    await _show_embeddings_menu(call, 0)

# =========== АПСКЕЙЛ, ПОВТОР, РЕДАКТИРОВАНИЕ ===========

@router.callback_query(F.data.startswith("upscale_"))
async def handle_upscale_callback(call: CallbackQuery) -> None:
    chat_id = call.message.chat.id
    user_id = str(chat_id)
    source_image = last_images.get(user_id)
    if source_image is None:
        await bot.answer_callback_query(call.id, "Нет изображения!")
        return
    factor = int(call.data.split("_")[1])
    await bot.answer_callback_query(call.id, f"Увеличиваю в {factor}x...")
    try:  # меню выбора коэффициента больше не нужно: прогресс будет в статус-сообщении задачи
        await bot.delete_message(chat_id, call.message.message_id)
    except Exception:
        pass
    # --- Логирование апскейла (без промпта, но с историей действий пользователя) ---
    gen_id = await prompt_logger.log_request(user_id, "upscale", original_prompt="", upscale_factor=factor)

    async def job(task: Task):
        start_time = time.time()
        try:
            await task.edit_status(f"🔍 Задача №{task.id}: апскейл {factor}x…", force=True)
            upscaled = await upscale_image(source_image, upscale_factor=float(factor))
            await prompt_logger.log_result(gen_id, user_id, "upscale", "", duration_sec=time.time() - start_time, upscale_factor=factor)
            # После апскейла картинка часто весит >10 МБ или шире 10000 px по сумме сторон — тогда уйдёт файлом
            await send_image_smart(chat_id, upscaled, caption=f"✅ {factor}x")
            last_images[user_id] = upscaled
            await task.delete_status()
        except asyncio.CancelledError:
            await prompt_logger.log_error(gen_id, user_id, "upscale", "", "Отменено пользователем", duration_sec=time.time() - start_time)
            raise
        except Exception as e:
            await prompt_logger.log_error(gen_id, user_id, "upscale", "", e, duration_sec=time.time() - start_time)
            await _report_error(chat_id, task.status_message_id, e)

    await enqueue(chat_id, user_id, "upscale", job)

@router.callback_query(F.data == "action_upscale")
async def handle_action_upscale(call: CallbackQuery) -> None:
    await bot.answer_callback_query(call.id)
    await bot.send_message(call.message.chat.id, "Выбери коэффициент:", reply_markup=get_upscale_keyboard())

@router.callback_query(F.data == "action_repeat")
async def handle_repeat_generation(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    if user_id not in last_configs:
        await bot.answer_callback_query(call.id, "Нет сохраненной конфигурации для повтора", show_alert=True)
        return
    config = last_configs[user_id]
    # Если это ControlNet, предлагаем повторить с тем же скелетом?
    # Но кнопка "Повторить" остаётся для обычных генераций, для ControlNet используем отдельную кнопку "тот же скелет"
    # Поэтому здесь обрабатываем только обычные (txt2img)
    if config.get('type') == 'controlnet':
        await bot.answer_callback_query(call.id, "Для ControlNet используйте кнопку 'Тот же скелет'", show_alert=True)
        return

    cfg = config.get('settings', {})
    confirm_text = (
        "🔄 <b>Повтор генерации</b>\n\n"
        f"📝 <b>Промпт:</b>\n<code>{_h(_cut(config.get('original_prompt', ''), 100))}</code>\n\n"
        f"📐 Размер: {cfg.get('width', '?')}x{cfg.get('height', '?')}\n"
        f"⏱️ Шаги: {cfg.get('steps', '?')}\n"
        f"🎚️ CFG: {cfg.get('cfg_scale', '?')}\n"
        f"🎲 Seed: <code>{config.get('seed')}</code>\n"
    )
    if config.get('face_enabled'):
        confirm_text += "🎭 Замена лица: включена\n"
    confirm_text += "\n✅ Повторить с этими настройками?"
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("✅ Да, повторить", callback_data="repeat_confirm"),
        types.InlineKeyboardButton("❌ Нет, отмена", callback_data="repeat_cancel")
    )
    await bot.send_message(call.message.chat.id, confirm_text, parse_mode="HTML", reply_markup=markup)
    await _safe_answer(call)

@router.callback_query(F.data == "repeat_confirm")
async def repeat_confirm(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    if user_id not in last_configs:
        await bot.answer_callback_query(call.id, "Нет сохраненной конфигурации", show_alert=True)
        return
    config = last_configs[user_id]
    # Если это ControlNet - перенаправляем на "тот же скелет"
    if config.get('type') == 'controlnet':
        await bot.answer_callback_query(call.id, "Для ControlNet используйте кнопку 'Тот же скелет'", show_alert=True)
        return
    # Отвечаем на callback СРАЗУ: раньше ответ шёл после многоминутной генерации — кнопка «висела» с индикатором,
    # а запоздалый ответ падал с «query is too old».
    await _safe_answer(call, "Повторяю генерацию...")
    chat_id = call.message.chat.id
    try:
        await bot.edit_message_text("🔄 Повторяю генерацию...", chat_id, call.message.message_id)
    except Exception:
        pass
    async def job(task: Task):
        start_time = time.time()
        on_progress, on_status = make_callbacks(task)
        try:
            # Точный повтор: промпты берутся ровно такими, какие ушли в WebUI в прошлый раз
            # (без повторного перевода через Ollama, без дублирования LoRA и новых «фраз разнообразия»)
            face_ctx = await get_face_context(user_id) if config.get('face_enabled', False) else None
            if face_ctx and face_ctx.note:
                await on_status(f"⚠️ {face_ctx.note}")
            image_file, used_seed, original_prompt, processed_prompt = await generate_image(
                config['prompt'], dict(config['settings']), user_id, on_progress, on_status, fixed_seed=config['seed'],
                exact_prompt=config.get('final_prompt'), exact_negative=config.get('final_negative'),
                face_image=face_ctx.image if face_ctx and face_ctx.use_faceid else None,
                face_weights=face_ctx.weights if face_ctx else None
            )
            final_image_file = image_file
            swapped = False
            if face_ctx and face_ctx.use_swap:
                await on_status("🎭 Заменяю лицо...")
                final_image_file, swapped = await apply_face_swap(face_ctx, image_file)
            face_replacement_success = bool(face_ctx and (face_ctx.use_faceid or swapped))
            try:
                await bot.delete_message(chat_id, task.status_message_id)
            except Exception:
                pass
            cfg = config['settings']
            caption = (
                f"✨ <b>Исходный промпт:</b>\n<code>{_h(_cut(config['original_prompt'], 700))}</code>\n\n"
                f"🤖 <b>Обработанный промпт:</b>\n<code>{_h(_cut(processed_prompt, 1200))}</code>\n\n"
                f"📐 <b>Размер:</b> {cfg.get('width')}x{cfg.get('height')}\n"
                f"⏱️ <b>Шаги:</b> {cfg.get('steps')}\n"
                f"🎚️ <b>CFG:</b> {cfg.get('cfg_scale')}\n"
                f"🎲 <b>Seed:</b> <code>{used_seed}</code> (повтор)\n"
            )
            caption += _face_caption_line(face_ctx, swapped)
            sent_msg = await send_image_smart(chat_id, final_image_file, caption=caption, parse_mode="HTML")
            last_images[user_id] = final_image_file
            last_message_ids[user_id] = sent_msg.message_id
            last_seeds[user_id] = used_seed
            await bot.send_message(chat_id, "🔧 **Действия с изображением:**", reply_markup=get_image_actions_keyboard(), reply_to_message_id=sent_msg.message_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await _report_error(chat_id, task.status_message_id, e)
    await enqueue(chat_id, user_id, "repeat", job)

@router.callback_query(F.data == "repeat_cancel")
async def repeat_cancel(call: CallbackQuery) -> None:
    try:
        await bot.edit_message_text("❌ Повтор отменен", call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    await _safe_answer(call)

# =========== РАСШИРЕННАЯ ГЕНЕРАЦИЯ ===========

async def extended_generation_prompt(message: Message) -> None:
    text = _text(message)
    if _is_cancel_text(text):
        await bot.send_message(message.chat.id, "❌ Отменено.")
        return
    user_id = str(message.chat.id)
    prompt = text
    temp_data.setdefault(user_id, {})['prompt'] = prompt
    current_neg = await get_user_negative_prompt(user_id)
    is_custom = current_neg != DEFAULT_NEGATIVE_PROMPT
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("✅ Стандартный негатив", callback_data="ext_neg_default"),
        types.InlineKeyboardButton(f"✏️ Свой негатив{' ✓' if is_custom else ''}", callback_data="ext_neg_custom"),
        types.InlineKeyboardButton("⚙️ Изменить в настройках", callback_data="ext_neg_settings")
    )
    await bot.send_message(message.chat.id, f"📝 <b>Промпт:</b> <i>{_h(_cut(prompt, 100))}</i>\n\nВыбери негативный промпт:", parse_mode="HTML", reply_markup=markup)

@router.callback_query(F.data.startswith("ext_neg_"))
async def handle_extended_neg_choice(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    prompt = temp_data.get(user_id, {}).get('prompt')
    if not prompt:
        await bot.answer_callback_query(call.id, "Сессия истекла")
        await bot.send_message(call.message.chat.id, "Нажми кнопку генерации заново.")
        return
    if call.data == "ext_neg_default":
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        neg_source = "стандартный"
    elif call.data == "ext_neg_custom":
        negative_prompt = await get_user_negative_prompt(user_id)
        if negative_prompt == DEFAULT_NEGATIVE_PROMPT:
            await bot.answer_callback_query(call.id, "Сначала задай свой негатив в настройках!")
            return
        neg_source = "свой"
    elif call.data == "ext_neg_settings":
        await bot.answer_callback_query(call.id)
        await show_settings(call.message)
        return
    else:
        return
    # Сессия использована: убираем промпт сразу, чтобы двойное нажатие кнопки не запустило генерацию дважды
    session = temp_data.get(user_id)
    if session is not None:
        session.pop('prompt', None)
    await bot.answer_callback_query(call.id, f"Генерация с {neg_source} негативом...")
    # --- Логирование промпта пользователя (расширенная генерация) ---
    settings = await get_user_settings(user_id)
    gen_id = await prompt_logger.log_request(
        user_id, "txt2img_advanced", prompt,
        settings=await get_user_settings(user_id), negative_prompt=negative_prompt,
    )
    async def job(task: Task):
        start_time = time.time()
        on_progress, on_status = make_callbacks(task)
        try:
            images, used_seed = await generate_txt2img_advanced(prompt, negative_prompt, settings, on_progress, user_id=user_id)
            if not images:
                raise Exception("WebUI не вернул изображений")
            try:
                await bot.delete_message(call.message.chat.id, task.status_message_id)
            except Exception:
                pass
            for image_file in images:
                caption = (
                    f"✨ <b>Промпт:</b>\n<code>{_h(_cut(prompt, 900))}</code>\n\n"
                    f"📐 <b>Размер:</b> {settings.get('width', 512)}x{settings.get('height', 768)}\n"
                    f"⏱️ <b>Шаги:</b> {settings.get('steps', 20)}\n"
                    f"🎚️ <b>CFG:</b> {settings.get('cfg_scale', 7.0)}\n"
                    f"🚫 <b>Негатив:</b> {neg_source}\n"
                    f"🎲 <b>Seed:</b> <code>{used_seed}</code>\n"
                )
                sent_msg = await send_image_smart(call.message.chat.id, image_file, caption=caption, parse_mode="HTML")
                last_images[user_id] = image_file
                last_message_ids[user_id] = sent_msg.message_id
                last_seeds[user_id] = used_seed
                config = {
                    "type": "txt2img",
                    "prompt": prompt,
                    "original_prompt": prompt,
                    "final_prompt": getattr(image_file, "final_prompt", None),
                    "final_negative": getattr(image_file, "final_negative", None),
                    "settings": dict(settings),
                    "seed": used_seed,
                    "face_enabled": False
                }
                last_configs[user_id] = config
            await prompt_logger.log_result(
                gen_id, user_id, "txt2img_advanced", prompt,
                processed_prompt=prompt, seed=used_seed, settings=settings,
                duration_sec=time.time() - start_time, negative_prompt=negative_prompt,
            )
            await bot.send_message(call.message.chat.id, "🔧 **Действия с изображением:**", reply_markup=get_image_actions_keyboard())
        except asyncio.CancelledError:
            await prompt_logger.log_error(gen_id, user_id, "txt2img_advanced", prompt, "Отменено пользователем", duration_sec=time.time() - start_time)
            raise
        except Exception as e:
            await prompt_logger.log_error(gen_id, user_id, "txt2img_advanced", prompt, e, duration_sec=time.time() - start_time)
            await _report_error(call.message.chat.id, task.status_message_id, e)
        temp_data.pop(user_id, None)  # раньше del по несуществующему ключу падал с KeyError при гонке
    await enqueue(call.message.chat.id, user_id, "txt2img_advanced", job)


# =========== РЕДАКТИРОВАНИЕ (IMG2IMG) ===========

async def apply_face_replacement_if_enabled(user_id, image):
    # Правка готового изображения: FaceID тут не при чём, лицо ставится ReActor-заменой (по модели, если она есть)
    ctx = await get_face_context(user_id, allow_faceid=False)
    if not ctx or not ctx.use_swap:
        return image
    print(f"🎭 Применяю замену лица для пользователя {user_id}")
    result, ok = await apply_face_swap(ctx, image)
    print("✅ Замена лица выполнена успешно" if ok else "❌ Замена лица не удалась")
    return result

@router.callback_query(F.data == "action_img2img")
async def show_edit_options(call: CallbackQuery) -> None:
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("🎨 Всё изображение", callback_data="edit_all"),
        types.InlineKeyboardButton("🧍 Тело", callback_data="edit_body"),
        types.InlineKeyboardButton("🌄 Фон", callback_data="edit_background"),
        types.InlineKeyboardButton("👗 Одежда", callback_data="edit_clothes"),
        types.InlineKeyboardButton("🚶 Поза (ControlNet)", callback_data="edit_pose")
    )
    await bot.edit_message_text("Выберите, что хотите изменить:", call.message.chat.id, call.message.message_id, reply_markup=markup)
    await bot.answer_callback_query(call.id)
@router.callback_query(F.data == "set_sampler")
async def handle_sampler_button(call: CallbackQuery) -> None:
    """Показывает список сэмплеров для выбора"""
    user_id = str(call.message.chat.id)
    from settings import get_available_samplers, get_user_sampler
    samplers = await get_available_samplers()
    current = await get_user_sampler(user_id)
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    for sampler in samplers[:12]:  # Показываем первые 12 для компактности
        label = f"✅ {sampler}" if sampler == current else sampler
        markup.add(types.InlineKeyboardButton(label, callback_data=f"sampler_set_{sampler}"))
    
    # Если сэмплеров больше 12, добавляем кнопку "Ещё"
    if len(samplers) > 12:
        markup.add(types.InlineKeyboardButton("📋 Показать все", callback_data="sampler_show_all"))
    
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
    
    await bot.edit_message_text(
        f"🎯 **Выберите сэмплер**\n\n"
        f"Текущий: `{current}`\n\n"
        f"Доступно: {len(samplers)}",
        call.message.chat.id,
        call.message.message_id,
        parse_mode="Markdown",
        reply_markup=markup
    )
    await bot.answer_callback_query(call.id)

@router.callback_query(F.data.startswith("sampler_set_"))
async def handle_sampler_choice(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    sampler_name = call.data[len("sampler_set_"):]
    await set_user_sampler(user_id, sampler_name)
    
    await bot.answer_callback_query(call.id, f"✅ Сэмплер: {sampler_name}")
    # Возвращаемся к настройкам
    await show_settings(call.message)

@router.callback_query(F.data == "sampler_show_all")
async def handle_sampler_show_all(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    from settings import get_available_samplers, get_user_sampler
    samplers = await get_available_samplers()
    current = await get_user_sampler(user_id)
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    for sampler in samplers:
        label = f"✅ {sampler}" if sampler == current else sampler
        markup.add(types.InlineKeyboardButton(label, callback_data=f"sampler_set_{sampler}"))
    
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="set_sampler"))
    
    await bot.edit_message_text(
        f"📋 **Все сэмплеры**\nТекущий: `{current}`",
        call.message.chat.id,
        call.message.message_id,
        parse_mode="Markdown",
        reply_markup=markup
    )
    await bot.answer_callback_query(call.id)
    
@router.callback_query(F.data.startswith("edit_"))
async def handle_edit_type(call: CallbackQuery) -> None:
    edit_type = call.data.split("_")[1]
    user_id = str(call.message.chat.id)
    if last_images.get(user_id) is None:
        await bot.answer_callback_query(call.id, "Нет изображения для редактирования", show_alert=True)
        return
    # Саму картинку в temp_data больше не кладём (она нигде не читалась, но удерживала память): берётся last_images
    temp_data.setdefault(user_id, {})['edit_type'] = edit_type
    await bot.edit_message_text(f"✍️ Введите описание изменений для **{edit_type}**:\n\nНапример: 'добавить мускулы' или 'заменить фон на городской пейзаж'", call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    register_next_step(call.message.chat.id, process_edit_prompt, edit_type, user_id, call.message.message_id)
    await bot.answer_callback_query(call.id)

async def process_edit_prompt(message, edit_type, user_id, original_msg_id):
    prompt = _text(message)
    if _is_cancel_text(prompt):
        await bot.send_message(message.chat.id, "❌ Нужен текст промпта (не кнопка, команда или файл). Редактирование отменено.")
        return
    original_image = last_images.get(user_id)
    if not original_image:
        await bot.send_message(message.chat.id, "❌ Изображение не найдено. Отмена.")
        return
    settings = await get_user_settings(user_id)
    # --- Логирование промпта пользователя (редактирование img2img/inpaint) ---
    gen_type = {"all": "img2img", "pose": "controlnet_edit"}.get(edit_type, "inpaint")
    gen_id = await prompt_logger.log_request(
        user_id, gen_type, prompt, settings=settings, edit_type=edit_type,
        negative_prompt=await get_user_negative_prompt(user_id),
    )
    async def job(task: Task):
        start_time = time.time()
        on_progress, on_status = make_callbacks(task)
        try:
            if edit_type == "all":
                new_image = await edit_all_image(prompt, original_image, settings, on_progress, on_status, user_id=user_id)
            elif edit_type == "body":
                new_image = await edit_body_part(prompt, original_image, settings, "body", on_progress, on_status, user_id=user_id)
            elif edit_type == "background":
                new_image = await edit_body_part(prompt, original_image, settings, "background", on_progress, on_status, user_id=user_id)
            elif edit_type == "clothes":
                new_image = await edit_body_part(prompt, original_image, settings, "clothes", on_progress, on_status, user_id=user_id)
            elif edit_type == "pose":
                new_image = await edit_pose(prompt, original_image, settings, on_progress, on_status, user_id=user_id)
            else:
                raise ValueError(f"Неизвестный тип редактирования: {edit_type}")
            final_image = await apply_face_replacement_if_enabled(user_id, new_image)
            try:
                await bot.delete_message(message.chat.id, task.status_message_id)
            except Exception:
                pass
            sent_msg = await send_image_smart(message.chat.id, final_image,
                                        caption=f"✅ Редактирование <b>{_h(edit_type)}</b>\nПромпт: <code>{_h(_cut(prompt, 800))}</code>", parse_mode="HTML")
            last_images[user_id] = final_image
            last_message_ids[user_id] = sent_msg.message_id
            last_configs[user_id] = {
                "type": "txt2img",
                "prompt": prompt,
                "original_prompt": prompt,
                "settings": dict(settings),
                "seed": 0,
                "face_enabled": await get_face_enabled(user_id)
            }
            await prompt_logger.log_result(
                gen_id, user_id, gen_type, prompt, processed_prompt=prompt,
                settings=settings, duration_sec=time.time() - start_time, edit_type=edit_type,
            )
            await bot.send_message(message.chat.id, "🔧 Действия с изображением:", reply_markup=get_image_actions_keyboard())
        except asyncio.CancelledError:
            await prompt_logger.log_error(gen_id, user_id, gen_type, prompt, "Отменено пользователем", duration_sec=time.time() - start_time)
            raise
        except Exception as e:
            await prompt_logger.log_error(gen_id, user_id, gen_type, prompt, e, duration_sec=time.time() - start_time)
            await _report_error(message.chat.id, task.status_message_id, e)
        temp_data.get(user_id, {}).pop('edit_type', None)
    await enqueue(message.chat.id, user_id, "edit", job)

async def edit_all_image(prompt, image, settings, progress_callback=None, status_callback=None, user_id="default"):
    if status_callback:
        await status_callback("🔄 Генерация через img2img...")
    images = await generate_img2img(prompt, image, settings, progress_callback, user_id=user_id)
    return images[0] if images else image

async def edit_body_part(prompt, image, settings, part, progress_callback=None, status_callback=None, user_id="default"):
    from segmentation import get_mask_for_part
    if status_callback:
        await status_callback(f"🔄 Выделяю {part}...")
    mask = await asyncio.to_thread(get_mask_for_part, image, part)  # OpenCV/MediaPipe — CPU-работа, не в цикле событий
    if mask is None:
        raise Exception(f"Не удалось выделить {part}. Попробуйте другой тип.")
    if status_callback:
        await status_callback(f"🔄 Перерисовываю {part}...")
    new_image = await generate_inpaint(prompt, image, mask, settings, progress_callback, user_id=user_id)
    return new_image

async def edit_pose(prompt, image, settings, progress_callback=None, status_callback=None, user_id="default"):
    try:
        from segmentation import get_pose_control_image
    except ImportError:
        raise Exception("Для позы требуется установить mediapipe и opencv.")
    if status_callback:
        await status_callback("🔄 Детектирую позу...")
    control_image = await asyncio.to_thread(get_pose_control_image, image)
    if control_image is None:
        raise Exception("Не удалось детектировать позу на изображении.")
    if status_callback:
        await status_callback("🔄 Генерация с ControlNet...")
    # Используем функцию generate_controlnet из controlnet_manager
    new_image = await generate_controlnet(prompt, image, control_image, "openpose", "control_v11p_sd15_openpose", settings, progress_callback, user_id=user_id)
    return new_image
# =========== ОБРАБОТЧИКИ LORA ===========

def _lora_id(lora_name):
    return hashlib.md5(lora_name.encode("utf-8")).hexdigest()[:10]

async def handle_manage_lora(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    available = await get_available_loras()
    lora_enabled = await get_user_lora_enabled(user_id)
    markup = types.InlineKeyboardMarkup(row_width=2)
    lora_weights = await get_user_lora_weights(user_id)
    for lora_name, info in available.items():
        status = "✅" if lora_enabled.get(lora_name, True) else "❌"
        btn_text = _cut(f"{status} {info['name']}", 40)
        # callback_data у Telegram ограничена 64 байтами — длинные/русские имена LoRA её превышали и ломали меню
        markup.add(
            types.InlineKeyboardButton(btn_text, callback_data=f"lora_toggle_{_lora_id(lora_name)}"),
            types.InlineKeyboardButton(f"⚖️ {lora_weights.get(lora_name, 1.0):g}", callback_data=f"lora_weight_{_lora_id(lora_name)}"),
        )
    # Кнопка обновления списка
    markup.add(types.InlineKeyboardButton("🔄 Обновить список LoRA", callback_data="refresh_lora_list"))
    markup.add(types.InlineKeyboardButton("◀️ Назад в настройки", callback_data="back_to_settings"))

    text = f"🎛️ **Управление LoRA**\n\nНажмите на название — включить/отключить, на ⚖️ — сменить вес (0.4 → 0.6 → 0.8 → 1.0 → 1.2).\nВключённые LoRA автоматически добавляются в промпт при генерации.\n\nДоступно: {len(available)} LoRA"

    try:
        await bot.edit_message_text(
            text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode="Markdown",
            reply_markup=markup
        )
    except Exception as e:
        if "message is not modified" not in str(e):
            # В случае другой ошибки – удаляем и отправляем новое сообщение
            try:
                await bot.delete_message(call.message.chat.id, call.message.message_id)
            except Exception:
                pass
            await bot.send_message(
                call.message.chat.id,
                text,
                parse_mode="Markdown",
                reply_markup=markup
            )
    await _safe_answer(call)

@router.callback_query(F.data == "manage_lora")
async def handle_manage_lora_button(call: CallbackQuery) -> None:
    await handle_manage_lora(call)

@router.callback_query(F.data.startswith("lora_toggle_"))
async def handle_lora_toggle(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    lora_id = call.data[len("lora_toggle_"):]
    available = await get_available_loras()
    lora_name = next((name for name in available if _lora_id(name) == lora_id), None)
    if lora_name is None:
        await _safe_answer(call, "❌ LoRA не найдена — обновите список", show_alert=True)
        return
    current = (await get_user_lora_enabled(user_id)).get(lora_name, True)
    new_state = not current
    await set_user_lora_enabled(user_id, lora_name, new_state)
    status = "включена ✅" if new_state else "выключена ❌"
    await _safe_answer(call, f"LoRA {available[lora_name]['name']} {status}"[:200])
    # Перерисовываем меню (редактируем то же сообщение)
    await handle_manage_lora(call)

@router.callback_query(F.data.startswith("lora_weight_"))
async def handle_lora_weight(call: CallbackQuery) -> None:
    user_id = str(call.message.chat.id)
    lora_id = call.data[len("lora_weight_"):]
    available = await get_available_loras()
    lora_name = next((name for name in available if _lora_id(name) == lora_id), None)
    if lora_name is None:
        await _safe_answer(call, "❌ LoRA не найдена — обновите список", show_alert=True)
        return
    weight = await cycle_user_lora_weight(user_id, lora_name)
    await _safe_answer(call, f"Вес {available[lora_name]['name']}: {weight:g}"[:200])
    await handle_manage_lora(call)

@router.callback_query(F.data == "refresh_lora_list")
async def handle_refresh_lora(call: CallbackQuery) -> None:
    new_loras = await get_available_loras(force_refresh=True)
    if new_loras:
        await _safe_answer(call, "✅ Список обновлён!")
        await handle_manage_lora(call)
    else:
        await _safe_answer(call, "❌ Не удалось обновить список", show_alert=True)

@router.callback_query(F.data == "back_to_settings")
async def handle_back_to_settings(call: CallbackQuery) -> None:
    await show_settings(call.message)
    await bot.answer_callback_query(call.id)

# =========== ЗАПУСК БОТА ===========

# =========== ЗАПУСК БОТА ===========

async def on_startup(bot: Bot):
    """Хук aiogram: прогрев кэшей и запуск воркеров очереди (цикл событий уже работает)."""
    global REACTOR_AVAILABLE, task_queue
    print("🤖 Бот запущен!")

    # Reactor (замена лиц)
    try:
        REACTOR_AVAILABLE = await check_reactor()
    except Exception:
        REACTOR_AVAILABLE = False
    print("✅ Reactor API доступен" if REACTOR_AVAILABLE else "⚠️ Reactor API недоступен, замена лиц будет отключена")

    # Предупреждение о недоступном Stable Diffusion API (чтобы не «падать» молча при генерации)
    try:
        if await check_api_available():
            print(f"✅ Stable Diffusion API доступен: {config.SD_BASE_URL}")
        else:
            print(f"⚠️ Stable Diffusion API НЕдоступен по адресу {config.SD_BASE_URL}. Проверьте, запущен ли WebUI с --api")
    except Exception as e:
        print(f"⚠️ Не удалось проверить доступность SD API: {e}")

    # Справочники WebUI грузим параллельно (кэш с TTL и asyncio.Lock — дубликатов запросов не будет)
    async def warm(name, coro):
        try:
            return await coro
        except Exception as e:
            print(f"⚠️ {name}: {e}")
            return None

    loras, embeddings, models, sources = await asyncio.gather(
        warm("LoRA", get_available_loras(force_refresh=True)),
        warm("эмбеддинги", get_available_embeddings(force_refresh=True)),
        warm("модели", get_available_models(force_refresh=True)),
        # индекс поз (os.walk + parquet) строится один раз в потоке
        warm("ControlNet", get_controlnet_sources(force_refresh=True)),
    )
    print(f"✅ Загружено {len(loras)} LoRA с сервера" if loras else "⚠️ Не удалось загрузить LoRA с сервера, используем дефолтные")
    print(f"🧬 Эмбеддингов в WebUI: {len(embeddings) if embeddings is not None else 'не удалось получить'}")
    default_model = await warm("модель по умолчанию", get_default_model())  # фиксируем модель, загруженную в WebUI сейчас
    print(f"🧩 Моделей в WebUI: {len(models or [])}; по умолчанию: {default_model or 'не определена'}")
    print(f"📊 Всего поз в базе: {len(sources or [])}")

    # Список команд в меню Telegram
    try:
        await bot_api.set_my_commands([
            types.BotCommand("start", "Начать / главное меню"),
            types.BotCommand("help", "Справка"),
            types.BotCommand("samplers", "Список сэмплеров"),
            types.BotCommand("setsampler", "Выбрать сэмплер"),
            types.BotCommand("model", "Выбрать модель"),
            types.BotCommand("cancel", "Отменить мои задачи"),
            types.BotCommand("prompts", "Лог промптов (админ)"),
        ])
    except Exception as e:
        print(f"⚠️ Не удалось установить список команд: {e}")

    task_queue = TaskQueue(bot_api, workers=config.QUEUE_WORKERS, maxsize=config.QUEUE_MAXSIZE,
                           max_per_user=config.QUEUE_MAX_PER_USER)
    await task_queue.start()
    print(f"📝 Лог промптов: {config.PROMPT_LOG_FILE} ({'вкл' if config.PROMPT_LOG_ENABLED else 'выкл'})")
    print(f"⚙️ Режим: {config.BOT_MODE}; воркеров очереди: {config.QUEUE_WORKERS}; запросов к WebUI одновременно: {config.SD_MAX_PARALLEL}")


async def on_shutdown(bot: Bot):
    """Хук aiogram: корректная остановка — прерываем задачи (и генерацию в WebUI), закрываем HTTP-сеанс."""
    global task_queue
    print("🛑 Останавливаюсь...")
    if task_queue is not None:
        await task_queue.shutdown()
        task_queue = None
    await close_session()


dp.include_router(router)
dp.startup.register(on_startup)
dp.shutdown.register(on_shutdown)


async def run_webhook():
    """Режим BOT_MODE=webhook: aiohttp-сервер + SimpleRequestHandler aiogram (нужны WEBHOOK_URL и желательно WEBHOOK_SECRET)."""
    from aiohttp import web
    from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

    if not config.WEBHOOK_URL:
        raise SystemExit("❌ BOT_MODE=webhook требует WEBHOOK_URL (например https://example.com) в .env")
    if not config.WEBHOOK_SECRET:
        print("⚠️ WEBHOOK_SECRET не задан — запросы к вебхуку не будут проверяться. Рекомендуется задать секрет.")
    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=_aiogram_bot, secret_token=config.WEBHOOK_SECRET).register(app, path=config.WEBHOOK_PATH)
    setup_application(app, dp, bot=_aiogram_bot)  # связывает хуки startup/shutdown диспетчера с жизненным циклом сервера
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, host=config.WEBAPP_HOST, port=config.WEBAPP_PORT).start()
        await _aiogram_bot.set_webhook(
            url=config.WEBHOOK_URL + config.WEBHOOK_PATH,
            secret_token=config.WEBHOOK_SECRET,
            allowed_updates=dp.resolve_used_update_types(),
        )
        print(f"🌐 Webhook: {config.WEBHOOK_URL}{config.WEBHOOK_PATH} (слушаю {config.WEBAPP_HOST}:{config.WEBAPP_PORT})")
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        import signal
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):  # Windows
                pass
        await stop.wait()
    finally:
        await runner.cleanup()  # вызывает on_shutdown диспетчера


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if config.BOT_MODE == "webhook":
            await run_webhook()
        else:
            # long polling: SIGINT/SIGTERM обрабатываются aiogram — вызываются хуки on_shutdown
            await dp.start_polling(_aiogram_bot, polling_timeout=config.POLLING_TIMEOUT,
                                   allowed_updates=dp.resolve_used_update_types())
    finally:
        await _aiogram_bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
