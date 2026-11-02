import os
import sys
import io
import time
import base64
import re
import telebot
import pandas as pd
from telebot import types 
from face_manager import rename_face
import json
import html
import hashlib
import threading
from PIL import Image

try:
    import segmentation
except Exception as _seg_error:  # ImportError, а также AttributeError у новых версий mediapipe (нет mp.solutions)
    segmentation = None
    print(f"⚠️ Модуль segmentation недоступен ({_seg_error}). Редактирование тела/фона/позы будет недоступно.")

sys.path.insert(0, os.path.dirname(__file__))

from Aapi import (
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
    set_available_loras, get_user_lora_enabled, set_user_lora_enabled,
    fetch_loras_from_api,
    get_available_models, refresh_checkpoints_on_server, get_user_model,
    set_user_model, get_default_model, get_default_model_name,
    update_user_settings, get_available_embeddings, refresh_embeddings_on_server,
    get_user_negative_embeddings, toggle_user_negative_embedding, set_user_negative_embeddings,
    get_available_samplers, get_user_sampler, set_user_sampler
)
from face_manager import (
    save_user_face, get_user_faces, get_active_face, set_active_face,
    delete_face, get_face_preview, get_face_stats,
    check_reactor_available as check_reactor,
    get_face_enabled, set_face_enabled, get_active_face_name
)

import app_config as config  # noqa: E402  (загружает .env и переменные окружения)
import prompt_logger  # noqa: E402  (логирование промптов пользователей)

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

# Число рабочих потоков telebot. Раньше было 2 (по умолчанию): две долгие генерации одновременно блокировали
# весь бот, включая кнопки меню. Можно изменить переменной окружения BOT_WORKER_THREADS.
try:
    _WORKER_THREADS = max(2, int(os.environ.get("BOT_WORKER_THREADS", "8")))
except ValueError:
    _WORKER_THREADS = 8
bot = telebot.TeleBot(BOT_TOKEN, threaded=True, num_threads=_WORKER_THREADS)

# Запоминаем @username авторов сообщений и нажатий кнопок ДО вызова хендлеров,
# чтобы лог промптов («пользователь : промпт») знал имя пользователя.
_orig_process_new_messages = bot.process_new_messages
_orig_process_new_callback_query = bot.process_new_callback_query

def _process_new_messages_with_names(messages):
    for m in messages:
        prompt_logger.remember_user(getattr(m, "from_user", None))
    return _orig_process_new_messages(messages)

def _process_new_callback_query_with_names(callbacks):
    for c in callbacks:
        prompt_logger.remember_user(getattr(c, "from_user", None))
    return _orig_process_new_callback_query(callbacks)

bot.process_new_messages = _process_new_messages_with_names
bot.process_new_callback_query = _process_new_callback_query_with_names

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SafeStore:
    """
    Потокобезопасный словарь с ограничением размера.
    telebot обрабатывает апдейты в нескольких потоках, поэтому обычные dict'ы со сложными операциями
    («проверить — создать — записать») давали гонки и KeyError. Здесь каждая операция атомарна, а при
    переполнении вытесняются самые старые записи (раньше last_images росли без ограничений и держали в
    памяти десятки мегабайт на пользователя).
    """

    def __init__(self, maxsize=500):
        self._data = {}
        self._lock = threading.RLock()
        self._maxsize = maxsize

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data.pop(key, None)  # обновлённая запись становится самой свежей
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.pop(next(iter(self._data)))

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def __len__(self):
        with self._lock:
            return len(self._data)

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def pop(self, key, *default):
        with self._lock:
            return self._data.pop(key, *default)

    def setdefault(self, key, default=None):
        with self._lock:
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

REACTOR_AVAILABLE = check_reactor()
if REACTOR_AVAILABLE:
    print("✅ Reactor API доступен")
else:
    print("⚠️ Reactor API недоступен, замена лиц будет отключена")

# =========== ФУНКЦИИ ОБНОВЛЕНИЯ ПРОГРЕССА ===========

def update_progress_message(message_id, chat_id, progress, eta):
    percent = int(progress * 100)
    bar_length = 20
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    eta_text = f"⏱️ Осталось: {eta:.1f} сек" if eta > 0 else "⏳ Финальная обработка..."
    text = f"🎨 **Генерация**\n\n`[{bar}]` {percent}%\n\n{eta_text}"
    try:
        bot.edit_message_text(text, chat_id, message_id, parse_mode="Markdown")
    except:
        pass

def update_status_message(message_id, chat_id, status_text):
    try:
        bot.edit_message_text(f"🎨 **Генерация**\n\n{status_text}", chat_id, message_id, parse_mode="Markdown")
    except:
        pass

# =========== ОБЩИЕ ХЕЛПЕРЫ ===========

def _h(value):
    """Экранирует текст для parse_mode="HTML" (промпты пользователей могут содержать <, >, &, `, _, *)."""
    return html.escape(str(value), quote=False)

def _cut(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "…"

def _text(message):
    """Текст сообщения или пустая строка (фото, стикер и т.п. раньше роняли обработчики на None.strip())."""
    return (getattr(message, "text", None) or "").strip()

def _safe_answer(call, text=None, show_alert=False):
    """answer_callback_query, который не падает: на один callback можно ответить только один раз и недолго."""
    try:
        bot.answer_callback_query(call.id, text, show_alert=show_alert)
    except Exception:
        pass

def _report_error(chat_id, status_message_id, error):
    """Показывает ошибку на месте статус-сообщения (или новым сообщением, если его нельзя изменить)."""
    text = _cut(f"❌ Ошибка: {error}", 3900)
    try:
        bot.edit_message_text(text, chat_id, status_message_id)
    except Exception:
        try:
            bot.send_message(chat_id, text)
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

def send_image_smart(chat_id, image, caption=None, parse_mode=None, reply_to_message_id=None,
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

    def send(is_document):
        payload = data
        out_ext = ext
        if is_document:
            payload, new_ext = _shrink_for_document(payload)
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
                sent = bot.send_document(chat_id, buf, **kwargs)
            else:
                sent = bot.send_photo(chat_id, buf, **kwargs)
        except Exception as e:
            if text and parse_mode and _is_parse_error(e):
                kwargs.pop("parse_mode", None)
                kwargs["caption"] = re.sub(r"<[^>]+>", "", html.unescape(text)) if parse_mode == "HTML" else text
                buf.seek(0)
                sent = bot.send_document(chat_id, buf, **kwargs) if is_document else bot.send_photo(chat_id, buf, **kwargs)
            else:
                raise
        if separate_text:
            body = _cut(separate_text, TG_MESSAGE_LIMIT)
            try:
                bot.send_message(chat_id, body, parse_mode=parse_mode, reply_to_message_id=sent.message_id)
            except Exception as e:
                if _is_parse_error(e):
                    plain = re.sub(r"<[^>]+>", "", html.unescape(body)) if parse_mode == "HTML" else body
                    bot.send_message(chat_id, plain, reply_to_message_id=sent.message_id)
                else:
                    print(f"⚠️ Не удалось отправить подпись отдельным сообщением: {e}")
        return sent

    if as_document:
        print(f"📎 Отправляю файлом ({reason})")
        return send(True)
    try:
        return send(False)
    except Exception as e:
        if _is_photo_rejected(e):
            print(f"📎 Telegram отклонил фото ({e}); отправляю файлом")
            reason = "Telegram не принял как фото"
            return send(True)
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

def get_settings_inline_keyboard(settings, user_id):
    markup = types.InlineKeyboardMarkup(row_width=2)
    btn_res = types.InlineKeyboardButton(f"📐 Размер: {settings['width']}x{settings['height']}", callback_data="set_res")
    btn_steps = types.InlineKeyboardButton(f"⏱️ Шаги: {settings['steps']}", callback_data="set_steps")
    btn_cfg = types.InlineKeyboardButton(f"🎚️ CFG: {settings['cfg_scale']}", callback_data="set_cfg")
    emb_count = len(get_user_negative_embeddings(user_id))
    btn_neg = types.InlineKeyboardButton(f"🚫 Негатив" + (f" (🧬{emb_count})" if emb_count else ""), callback_data="set_neg")
    btn_sampler = types.InlineKeyboardButton(f"🎯 Сэмплер: {settings.get('sampler_name', 'DPM++ 2M')}", callback_data="set_sampler")  # <--- НОВАЯ КНОПКА
    diversity_mode = get_diversity_mode(user_id)
    diversity_text = {
        "auto": "🔄 Разнообразие: авто",
        "high": "🎲 Разнообразие: макс",
        "low": "🎯 Разнообразие: мин",
        "off": "⚪ Разнообразие: выкл"
    }.get(diversity_mode, "🔄 Разнообразие: авто")
    btn_diversity = types.InlineKeyboardButton(diversity_text, callback_data="set_diversity")
    prompt_processing = get_prompt_processing_enabled(user_id)
    prompt_text = "🤖 Обработка: ✅ Вкл" if prompt_processing else "🤖 Обработка: ❌ Выкл"
    btn_prompt = types.InlineKeyboardButton(prompt_text, callback_data="toggle_prompt_processing")
    btn_lora = types.InlineKeyboardButton("🎛️ LoRA", callback_data="manage_lora")
    model_label = _model_display_name(user_id)
    if len(model_label) > 30:
        model_label = model_label[:29] + "…"
    btn_model = types.InlineKeyboardButton(f"🧩 Модель: {model_label}", callback_data="set_model")
    markup.add(btn_model)
    markup.add(btn_res)
    markup.add(btn_steps, btn_cfg)
    markup.add(btn_sampler, btn_neg)  # <--- Добавляем кнопку сэмплера
    markup.add(btn_diversity, btn_prompt)
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

@bot.message_handler(func=lambda message: message.text == "🎮 ControlNet")
def controlnet_menu(message):
    user_id = str(message.chat.id)
    sources = get_controlnet_sources()
    if not sources:
        bot.send_message(message.chat.id, "❌ В папке `controlnet_images` нет изображений или читаемых .parquet.\nДобавьте туда скелеты (PNG, JPG, Parquet) и попробуйте снова.", parse_mode="Markdown")
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

    bot.send_message(
        message.chat.id,
        "🏷️ **Выберите категорию позы:**",
        parse_mode="Markdown",
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("ctrl_cat_"))
def handle_controlnet_category(call):
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
        controlnet_menu(call.message)
        bot.answer_callback_query(call.id)
        return

    sources = pagination_data[user_id]['sources']
    if category is not None:
        filtered = [s for s in sources if s.get('category') == category]
    else:
        filtered = sources

    if not filtered:
        bot.answer_callback_query(call.id, "❌ В этой категории нет поз")
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
            bot.delete_message(call.message.chat.id, preview_messages[user_id])
        except:
            pass
        del preview_messages[user_id]

    # Удаляем сообщение с категориями (текущее)
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass

    show_controlnet_page(call.message, user_id, 0)
    bot.answer_callback_query(call.id)

def show_controlnet_page(message, user_id, page):
    data = pagination_data.get(user_id)
    if not data:
        bot.send_message(message.chat.id, "❌ Сессия истекла. Нажмите кнопку ControlNet заново.")
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
    sent_msg = bot.send_message(
        message.chat.id,
        f"{header}📄 Страница {page+1} из {total_pages}\n\nВыберите позу для предпросмотра:",
        parse_mode="Markdown",
        reply_markup=markup
    )
    # Сохраняем ID сообщения списка
    data['list_message_id'] = sent_msg.message_id

@bot.callback_query_handler(func=lambda call: call.data == "ctrl_back_categories")
def handle_controlnet_back_categories(call):
    user_id = str(call.message.chat.id)
    # Удаляем текущее сообщение со списком поз
    data = pagination_data.get(user_id, {})
    list_msg_id = data.get('list_message_id')
    if list_msg_id:
        try:
            bot.delete_message(call.message.chat.id, list_msg_id)
        except:
            pass
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    # Сбрасываем фильтр
    if user_id in pagination_data:
        pagination_data[user_id]['category'] = None
        pagination_data[user_id]['filtered_sources'] = None
        pagination_data[user_id]['list_message_id'] = None
    # Показываем категории
    controlnet_menu(call.message)
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("ctrl_page_"))
def handle_controlnet_page(call):
    user_id = str(call.message.chat.id)
    if call.data == "ctrl_page_info":
        bot.answer_callback_query(call.id, "Информация о странице")
        return
    try:
        page = int(call.data.split("_")[2])
    except:
        bot.answer_callback_query(call.id, "Ошибка")
        return
    if user_id not in pagination_data:
        bot.answer_callback_query(call.id, "Сессия истекла")
        return
    # Удаляем предыдущее сообщение списка
    data = pagination_data[user_id]
    old_list_id = data.get('list_message_id')
    if old_list_id:
        try:
            bot.delete_message(call.message.chat.id, old_list_id)
        except:
            pass
        data['list_message_id'] = None
    # Удаляем текущее сообщение (это callback от кнопки навигации, оно тоже может быть удалено)
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    show_controlnet_page(call.message, user_id, page)
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == "ctrl_refresh")
def handle_controlnet_refresh(call):
    user_id = str(call.message.chat.id)
    sources = get_controlnet_sources()
    if not sources:
        bot.edit_message_text("❌ Нет доступных поз. Добавьте файлы в папку controlnet_images", call.message.chat.id, call.message.message_id)
        bot.answer_callback_query(call.id)
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
        bot.edit_message_text("❌ В этой категории нет поз. Вернитесь к выбору категории.", call.message.chat.id, call.message.message_id)
        bot.answer_callback_query(call.id)
        return

    pagination_data[user_id]['total_pages'] = (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

    # Удаляем старое сообщение списка (текущее)
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    if user_id in preview_messages:
        try:
            bot.delete_message(call.message.chat.id, preview_messages[user_id])
        except:
            pass
        del preview_messages[user_id]

    show_controlnet_page(call.message, user_id, 0)
    bot.answer_callback_query(call.id, "🔄 Список обновлен!")

@bot.callback_query_handler(func=lambda call: call.data.startswith("ctrl_preview_"))
def handle_controlnet_preview(call):
    user_id = str(call.message.chat.id)
    try:
        idx = int(call.data.split("_")[2])
    except:
        bot.answer_callback_query(call.id, "Ошибка")
        return
    data = pagination_data.get(user_id)
    if not data:
        bot.answer_callback_query(call.id, "Сессия истекла")
        return
    filtered = data.get('filtered_sources')
    if filtered is None:
        filtered = data['sources']
    if idx < 0 or idx >= len(filtered):
        bot.answer_callback_query(call.id, "Источник не найден")
        return
    source = filtered[idx]
    preview = get_controlnet_preview_by_source(source)
    if not preview:
        bot.answer_callback_query(call.id, "❌ Не удалось загрузить изображение")
        return
    
    # Удаляем сообщение со списком
    list_msg_id = data.get('list_message_id')
    if list_msg_id:
        try:
            bot.delete_message(call.message.chat.id, list_msg_id)
        except:
            pass
        data['list_message_id'] = None
    # Удаляем текущее сообщение (кнопка предпросмотра) – оно уже не нужно
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass

    # Удаляем старое превью
    if user_id in preview_messages:
        try:
            bot.delete_message(call.message.chat.id, preview_messages[user_id])
        except:
            pass
        del preview_messages[user_id]
    
    # Кнопки под превью
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("✅ Использовать", callback_data=f"ctrl_use_{idx}"),
        types.InlineKeyboardButton("❌ Назад к списку", callback_data="ctrl_back_to_list")
    )
    markup.add(types.InlineKeyboardButton("🔄 Случайная поза", callback_data="ctrl_random"))
    sent = bot.send_photo(
        call.message.chat.id,
        preview,
        caption=f"🦴 **Поза:** {source.get('full_label', source['label'])}\n\nВыберите действие:",
        parse_mode="Markdown",
        reply_markup=markup
    )
    preview_messages[user_id] = sent.message_id
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == "ctrl_back_to_list")
def handle_controlnet_back_to_list(call):
    user_id = str(call.message.chat.id)
    # Удаляем сообщение с превью
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    if user_id in preview_messages:
        del preview_messages[user_id]
    if user_id in pagination_data:
        # Удаляем сохранённый list_message_id, если есть (на случай, если он остался)
        pagination_data[user_id]['list_message_id'] = None
        page = pagination_data[user_id].get('page', 0)
        show_controlnet_page(call.message, user_id, page)
    else:
        controlnet_menu(call.message)
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("ctrl_use_"))
def handle_controlnet_use(call):
    user_id = str(call.message.chat.id)
    try:
        idx = int(call.data.split("_")[2])
    except:
        bot.answer_callback_query(call.id, "Ошибка")
        return
    data = pagination_data.get(user_id)
    if not data:
        bot.answer_callback_query(call.id, "Сессия истекла")
        return
    filtered = data.get('filtered_sources')
    if filtered is None:
        filtered = data['sources']
    if idx < 0 or idx >= len(filtered):
        bot.answer_callback_query(call.id, "Источник не найден")
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
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    if user_id in preview_messages:
        del preview_messages[user_id]
    bot.send_message(
        call.message.chat.id,
        f"✍️ **Введите промпт** для генерации с позой:\n`{source.get('full_label', source['label'])}`\n\nНапример: *девушка в платье, городской фон*",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(call.message, process_controlnet_prompt, user_id, source)
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == "ctrl_random")
def handle_controlnet_random(call):
    user_id = str(call.message.chat.id)
    data = pagination_data.get(user_id)
    if not data:
        bot.answer_callback_query(call.id, "Сессия истекла")
        return
    filtered = data.get('filtered_sources')
    if filtered is None:
        filtered = data['sources']
    if not filtered:
        bot.answer_callback_query(call.id, "Нет доступных поз")
        return
    import random
    idx = random.randint(0, len(filtered) - 1)
    # Удаляем текущее сообщение с превью
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    if user_id in preview_messages:
        del preview_messages[user_id]
    # Имитируем вызов preview с новым idx
    fake_call = type('FakeCall', (), {'data': f"ctrl_preview_{idx}", 'message': call.message, 'id': call.id})()
    handle_controlnet_preview(fake_call)

def process_controlnet_prompt(message, user_id, source=None):
    """
    Обрабатывает промпт для ControlNet.
    source может быть передан при первом выборе, либо взят из temp_data при повторной генерации.
    """
    prompt = _text(message)
    if _is_cancel_text(prompt):
        bot.send_message(message.chat.id, "❌ Нужен текстовый промпт (не кнопка, команда или файл). Отменено.")
        return
    
    # Если source не передан, берём из temp_data (для повторной генерации)
    if source is None:
        data = temp_data.get(user_id, {}).get('controlnet_source')
        if not data:
            bot.send_message(message.chat.id, "❌ Скелет не выбран. Начните заново через кнопку ControlNet.")
            return
        # Восстанавливаем источник из сохранённых данных
        sources = get_controlnet_sources()
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
            bot.send_message(message.chat.id, "❌ Источник не найден. Возможно, файл был удалён.")
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

    control_img = get_controlnet_preview_by_source(source)
    if not control_img:
        bot.send_message(message.chat.id, "❌ Не удалось загрузить скелет.")
        return

    # --- АВТОПОДСТРОЙКА РАЗРЕШЕНИЯ ПОД СКЕЛЕТ ---
    settings = get_user_settings(user_id)
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

    status_msg = bot.send_message(message.chat.id, "🎨 **Генерация с ControlNet**\n\n🔄 Подготовка...", parse_mode="Markdown")
    # --- Логирование промпта пользователя (ControlNet) ---
    gen_id = prompt_logger.log_request(
        user_id, "controlnet", prompt,
        settings=settings,
        negative_prompt=get_user_negative_prompt(user_id),
        controlnet_source=source.get('full_label', source.get('label')),
    )
    start_time = time.time()
    def on_progress(progress, eta):
        update_progress_message(status_msg.message_id, message.chat.id, progress, eta)
    def on_status(text):
        update_status_message(status_msg.message_id, message.chat.id, text)

    try:
        # Сохраняем оригинальный промпт для отображения
        original_prompt = prompt
        # Обработка промпта через Aapi.preprocess_prompt (она использует настройки пользователя)
        from Aapi import preprocess_prompt
        processed_prompt = preprocess_prompt(user_id, prompt, on_status)
        # Добавляем LoRA через Aapi (там уже есть get_active_lora_tags)
        from settings import get_active_lora_tags
        lora_tags = get_active_lora_tags(user_id)
        if lora_tags:
            processed_prompt = f"{processed_prompt}, {lora_tags}"
            on_status(f"🎛️ Добавлены LoRA: {lora_tags}")

        generated_image = generate_txt2img_controlnet(
            processed_prompt,  # используем обработанный промпт
            control_image=control_img,
            module="none",
            model="control_v11p_sd15_openpose",
            settings=settings,
            progress_callback=on_progress,
            status_callback=on_status,
            user_id=user_id
        )
        on_status("🎭 Заменяю лицо...")
        final_image = apply_face_replacement_if_enabled(user_id, generated_image)
        face_replaced = False
        if REACTOR_AVAILABLE and get_face_enabled(user_id) and get_active_face(user_id):
            face_replaced = True
        try:
            bot.delete_message(message.chat.id, status_msg.message_id)
        except:
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
        if face_replaced:
            caption += "🎭 <b>Лицо заменено</b>\n"
        sent_msg = send_image_smart(message.chat.id, final_image, caption=caption, parse_mode="HTML")
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
        prompt_logger.log_result(
            gen_id, user_id, "controlnet", original_prompt,
            processed_prompt=processed_prompt,
            seed=getattr(generated_image, "seed", None),
            settings=settings, duration_sec=time.time() - start_time,
            face_replaced=face_replaced,
            controlnet_source=source.get('full_label', source.get('label')),
        )
        # Отправляем клавиатуру с кнопкой "тот же скелет"
        bot.send_message(message.chat.id, "🔧 **Действия с изображением:**", reply_markup=get_controlnet_actions_keyboard())
    except Exception as e:
        prompt_logger.log_error(gen_id, user_id, "controlnet", prompt, e, duration_sec=time.time() - start_time)
        _report_error(message.chat.id, status_msg.message_id, e)
    finally:
        # Не удаляем controlnet_source, чтобы можно было повторно использовать (если не было ошибки)
        # Но если мы хотим, чтобы при повторном выборе скелета он перезаписывался, то оставляем.
        pass
# =========== КОМАНДЫ ДЛЯ РАБОТЫ С СЭМПЛЕРАМИ ===========

@bot.message_handler(commands=['samplers'])
def show_samplers(message):
    """Показывает список доступных сэмплеров"""
    try:
        samplers = get_available_samplers()
        if not samplers:
            bot.reply_to(message, "❌ Не удалось получить список сэмплеров")
            return

        user_id = str(message.chat.id)
        current = get_user_sampler(user_id)

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
            bot.send_message(message.chat.id, part, parse_mode="HTML")
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")

@bot.message_handler(commands=['setsampler'])
def set_sampler_command(message):
    """Устанавливает сэмплер для пользователя"""
    user_id = str(message.chat.id)

    # Разбираем команду
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        # Показываем текущий сэмплер (подсказки раньше ссылались на несуществующие команды «!samplers»)
        current = get_user_sampler(user_id)
        bot.reply_to(
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
        samplers = get_available_samplers()
        if not samplers:
            bot.reply_to(message, "❌ Не удалось получить список сэмплеров")
            return

        # Ищем точное совпадение (без учёта регистра), затем частичное
        exact = [s for s in samplers if s.lower() == sampler_name.lower()]
        matching = exact or [s for s in samplers if sampler_name.lower() in s.lower()]

        if len(matching) == 1:
            set_user_sampler(user_id, matching[0])
            bot.reply_to(message, f"✅ <b>Сэмплер изменён:</b> <code>{_h(matching[0])}</code>", parse_mode="HTML")
        elif len(matching) > 1:
            suggestions = "\n".join([f"• <code>{_h(s)}</code>" for s in matching[:10]])
            bot.reply_to(
                message,
                f"❓ <b>Найдено несколько сэмплеров:</b>\n{suggestions}\n\n"
                f"Уточните название. Все доступные: /samplers",
                parse_mode="HTML"
            )
        else:
            bot.reply_to(
                message,
                f"❌ <b>Сэмплер не найден:</b> <code>{_h(sampler_name)}</code>\n\n"
                f"Все доступные: /samplers",
                parse_mode="HTML"
            )

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")

# =========== ОБРАБОТЧИК "ТОТ ЖЕ СКЕЛЕТ" ===========

@bot.callback_query_handler(func=lambda call: call.data == "action_same_skeleton")
def handle_same_skeleton(call):
    user_id = str(call.message.chat.id)
    # Проверяем наличие сохранённого скелета в last_configs
    config = last_configs.get(user_id)
    if not config or config.get('type') != 'controlnet':
        bot.answer_callback_query(call.id, "❌ Нет сохранённого скелета для повторной генерации.", show_alert=True)
        return
    source_data = config.get('controlnet_source')
    if not source_data:
        bot.answer_callback_query(call.id, "❌ Данные скелета отсутствуют.", show_alert=True)
        return
    # Восстанавливаем источник из данных
    sources = get_controlnet_sources()
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
        bot.answer_callback_query(call.id, "❌ Скелет не найден (возможно, файл удалён).", show_alert=True)
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
    bot.answer_callback_query(call.id, "✍️ Введите новый промпт для генерации с тем же скелетом.")
    # Запрашиваем новый промпт
    bot.send_message(
        call.message.chat.id,
        f"✍️ **Введите новый промпт** для генерации с той же позой:\n`{target.get('full_label', target['label'])}`\n\n(можно изменить настройки перед отправкой)",
        parse_mode="Markdown"
    )
    # Регистрируем следующий шаг, передавая None как source (будет взят из temp_data)
    bot.register_next_step_handler(call.message, process_controlnet_prompt, user_id, None)

# =========== ОБРАБОТЧИКИ ОСТАЛЬНЫХ КНОПОК ===========

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    get_user_settings(str(message.chat.id))
    bot.send_message(
        message.chat.id,
        "🤖 **Stable Diffusion Бот**\n\n📝 **Быстрая генерация:** просто отправь любой текст\n\n🚀 **Сгенерировать** - расширенный режим\n⚙️ **Настройки** - изменить параметры\n🔍 **Апскейл** - увеличить картинку (или просто пришли мне любое фото)\n🧩 **Модель** - выбор в настройках или командой /model\n🎭 **Управление лицом** - добавление нескольких лиц для замены",
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )

@bot.message_handler(func=lambda message: message.text == "🚀 Сгенерировать")
def extended_generation_start(message):
    msg = bot.send_message(message.chat.id, "✍️ Отправь промпт для генерации:")
    bot.register_next_step_handler(msg, extended_generation_prompt)

@bot.message_handler(func=lambda message: message.text == "⚙️ Настройки")
def show_settings(message):
    user_id = str(message.chat.id)
    settings = get_user_settings(user_id)
    current_neg = get_user_negative_prompt(user_id).replace("`", "'")
    neg_preview = current_neg if len(current_neg) <= 80 else current_neg[:80] + "…"
    emb_names = get_user_negative_embeddings(user_id)
    emb_line = f"\n🧬 **Эмбеддинги:** `{_cut(', '.join(emb_names), 150).replace(chr(96), chr(39))}`" if emb_names else ""
    prompt_processing = get_prompt_processing_enabled(user_id)
    text = (
        "🎛️ **Текущие настройки:**\n\n"
        f"📐 Размер: {settings['width']}x{settings['height']}\n"
        f"⏱️ Шаги: {settings['steps']}\n"
        f"🎚️ CFG Scale: {settings['cfg_scale']}\n"
        f"🎯 Сэмплер: {settings.get('sampler_name', 'DPM++ 2M')}\n"  # <--- Добавлена строка
        f"🧩 Модель: `{_model_display_name(user_id).replace('`', '')}`\n"
        f"🤖 **Обработка промпта:** {'✅ Включена' if prompt_processing else '❌ Выключена'}\n\n"
        f"🚫 **Негативный промпт:**\n`{neg_preview}`{emb_line}"
    )
    bot.send_message(message.chat.id, text, parse_mode="Markdown", reply_markup=get_settings_inline_keyboard(settings, user_id))

@bot.message_handler(func=lambda message: message.text == "🔍 Апскейл")
def ask_for_upscale(message):
    user_id = str(message.chat.id)
    if user_id not in last_images:
        bot.reply_to(message, "❌ Сначала сгенерируй картинку или просто пришли мне фото для апскейла!")
        return
    bot.send_message(message.chat.id, "Выбери коэффициент увеличения:", reply_markup=get_upscale_keyboard())

@bot.message_handler(func=lambda message: message.text == "🎭 Управление лицом")
def face_management(message):
    if not REACTOR_AVAILABLE:
        bot.reply_to(message, "❌ Функция замены лиц временно недоступна. Попробуйте позже.")
        return
    user_id = str(message.chat.id)
    stats = get_face_stats(user_id)
    face_enabled = get_face_enabled(user_id)
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = []
    status_text = "✅ Включена" if face_enabled else "❌ Выключена"
    buttons.append(types.InlineKeyboardButton(f"🎭 Замена лиц: {status_text}", callback_data="face_toggle_main"))
    buttons.append(types.InlineKeyboardButton("📸 Добавить новое лицо", callback_data="face_add"))
    if stats["total_faces"] > 0:
        buttons.append(types.InlineKeyboardButton("📋 Список лиц", callback_data="face_list"))
    markup.add(*buttons)
    active_face = stats.get("active_face", "не выбрано")
    info_text = (
        "🎭 **Управление лицами**\n\n"
        f"📸 **Всего лиц:** {stats['total_faces']}\n"
        f"⭐ **Активное лицо:** {active_face}\n\n"
        "💡 **Как работает:**\n"
        "1. Добавьте одно или несколько лиц\n"
        "2. Выберите активное лицо\n"
        "3. Включите замену лиц\n"
        "4. Генерируйте изображения - лица будут заменены!\n\n"
        "✨ **Совет:** Для лучшего качества используйте фото с хорошим освещением"
    )
    bot.send_message(message.chat.id, info_text, parse_mode="Markdown", reply_markup=markup)

# =========== СТАТИСТИКА ЛОГА ПРОМПТОВ ===========
# (Обработчик должен быть зарегистрирован ДО универсального текстового обработчика ниже —
#  telebot вызывает первый подходящий, иначе команда /prompts никогда не срабатывала.)

@bot.message_handler(commands=['prompts'])
def show_prompt_log(message):
    """Показывает последние записями из лога промптов (для администратора)."""
    admin_id = os.environ.get("ADMIN_USER_ID")
    if admin_id and str(message.chat.id) != admin_id:
        bot.reply_to(message, "⛔ Доступ только для администратора (задайте ADMIN_USER_ID в .env).")
        return
    records = prompt_logger.tail(limit=15)
    if not records:
        bot.reply_to(message, "📭 Лог промптов пуст.")
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
        bot.send_message(message.chat.id, text, parse_mode="Markdown")
    except Exception:
        bot.send_message(message.chat.id, text.replace("`", ""))

# =========== ВЫБОР МОДЕЛИ (CHECKPOINT) ===========

MODELS_PER_PAGE = 8

def _model_id(model_name):
    return hashlib.md5(model_name.encode("utf-8")).hexdigest()[:10]

def _model_display_name(user_id):
    """Имя модели пользователя для настроек: выбранная, либо «по умолчанию»."""
    chosen = get_user_model(user_id)
    if chosen:
        return chosen
    default_name = get_default_model_name()
    return f"{default_name} (по умолч.)" if default_name else "по умолчанию"

def _build_models_menu(user_id, page):
    models = get_available_models()
    chosen = get_user_model(user_id)
    default_name = get_default_model_name()
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

def _show_models_menu(chat_id, user_id, page=0, message_id=None):
    text, markup = _build_models_menu(user_id, page)
    if message_id is not None:
        try:
            bot.edit_message_text(text, chat_id, message_id, parse_mode="HTML", reply_markup=markup)
            return
        except Exception as e:
            if "message is not modified" in str(e):
                return
    bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)

@bot.message_handler(commands=['model', 'models'])
def models_command(message):
    _show_models_menu(message.chat.id, str(message.chat.id), 0)

@bot.callback_query_handler(func=lambda call: call.data == "set_model")
def handle_model_button(call):
    _show_models_menu(call.message.chat.id, str(call.message.chat.id), 0, call.message.message_id)
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("model_page_"))
def handle_model_page(call):
    try:
        page = int(call.data[len("model_page_"):])
    except ValueError:
        page = 0
    _show_models_menu(call.message.chat.id, str(call.message.chat.id), page, call.message.message_id)
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == "model_noop")
def handle_model_noop(call):
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == "model_refresh")
def handle_model_refresh(call):
    bot.answer_callback_query(call.id, "Обновляю список...")
    refresh_checkpoints_on_server()           # WebUI пересканирует папку с моделями
    models = get_available_models(force_refresh=True)
    _show_models_menu(call.message.chat.id, str(call.message.chat.id), 0, call.message.message_id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("model_set_"))
def handle_model_choice(call):
    user_id = str(call.message.chat.id)
    model_id = call.data[len("model_set_"):]
    model = next((m for m in get_available_models() if _model_id(m["name"]) == model_id), None)
    if not model:
        bot.answer_callback_query(call.id, "Модель не найдена — обновите список", show_alert=True)
        return
    set_user_model(user_id, model["name"])
    bot.answer_callback_query(call.id, f"✅ Модель: {model['name']}"[:200])
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    show_settings(call.message)

@bot.callback_query_handler(func=lambda call: call.data == "model_default")
def handle_model_default(call):
    user_id = str(call.message.chat.id)
    set_user_model(user_id, None)
    bot.answer_callback_query(call.id, "Используется модель по умолчанию")
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    show_settings(call.message)

# =========== АПСКЕЙЛ ФОТО, ПРИСЛАННЫХ ПОЛЬЗОВАТЕЛЕМ ===========

MAX_UPSCALE_SIDE = 8192              # максимальная сторона результата, px (защита от нехватки памяти)
MAX_RESULT_FILE_BYTES = 45 * 1024 * 1024   # лимит Telegram для отправки ботом — 50 МБ
PENDING_UPSCALE_LIMIT = 30
pending_upscale = SafeStore(maxsize=PENDING_UPSCALE_LIMIT)  # (chat_id, id сообщения с кнопками) -> байты исходного изображения

def _remember_pending_upscale(key, data):
    pending_upscale[key] = data   # SafeStore потокобезопасен и сам вытесняет самые старые записи

def _offer_photo_upscale(message, file_id):
    """Скачивает присланное изображение и предлагает выбрать коэффициент увеличения."""
    try:
        file_info = bot.get_file(file_id)
        data = bot.download_file(file_info.file_path)
    except Exception as e:
        reason = "файл слишком большой (Telegram даёт ботам скачивать до 20 МБ)" if "too big" in str(e).lower() else str(e)
        bot.reply_to(message, f"❌ Не удалось скачать изображение: {reason}")
        return
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        bot.reply_to(message, "❌ Не удалось прочитать изображение. Пришлите файл JPG, PNG или WEBP.")
        return
    factors = [f for f in (2, 3, 4) if max(width, height) * f <= MAX_UPSCALE_SIDE]
    if not factors:
        bot.reply_to(message, f"❌ Изображение {width}×{height} слишком большое для апскейла "
                              f"(результат не должен превышать {MAX_UPSCALE_SIDE}px по стороне).")
        return
    markup = types.InlineKeyboardMarkup(row_width=3)
    markup.add(*[types.InlineKeyboardButton(f"{f}x → {width * f}×{height * f}", callback_data=f"pup_{f}") for f in factors])
    markup.add(types.InlineKeyboardButton("❌ Отмена", callback_data="pup_cancel"))
    sent = bot.reply_to(message, f"🔍 Получено изображение {width}×{height}.\nВо сколько раз увеличить?", reply_markup=markup)
    _remember_pending_upscale((message.chat.id, sent.message_id), data)

@bot.message_handler(content_types=['photo'])
def handle_photo_for_upscale(message):
    _offer_photo_upscale(message, message.photo[-1].file_id)

@bot.message_handler(content_types=['document'],
                     func=lambda m: (getattr(m.document, "mime_type", None) or "").startswith("image/"))
def handle_image_document_for_upscale(message):
    # Изображение, отправленное «как файл», приходит без сжатия Telegram — лучшее качество для апскейла
    _offer_photo_upscale(message, message.document.file_id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("pup_"))
def handle_photo_upscale_callback(call):
    chat_id = call.message.chat.id
    user_id = str(chat_id)
    key = (chat_id, call.message.message_id)
    action = call.data[len("pup_"):]
    if action == "cancel":
        pending_upscale.pop(key, None)
        bot.answer_callback_query(call.id, "Отменено")
        try:
            bot.delete_message(chat_id, call.message.message_id)
        except Exception:
            pass
        return
    data = pending_upscale.pop(key, None)
    if data is None:
        bot.answer_callback_query(call.id, "Изображение устарело — пришлите его снова", show_alert=True)
        return
    try:
        factor = int(action)
    except ValueError:
        bot.answer_callback_query(call.id)
        return
    bot.answer_callback_query(call.id, f"Увеличиваю в {factor}x...")
    try:
        bot.edit_message_text(f"⏳ Апскейл {factor}x...", chat_id, call.message.message_id)
    except Exception:
        pass
    gen_id = prompt_logger.log_request(user_id, "upscale", original_prompt="", upscale_factor=factor, source="user_photo")
    start_time = time.time()
    try:
        with Image.open(io.BytesIO(data)) as src:
            src_size = src.size
        upscaled = upscale_image(io.BytesIO(data), upscale_factor=float(factor))
        raw = upscaled.getvalue()
        ext = "png"
        with Image.open(io.BytesIO(raw)) as res:
            res_size = res.size
            if len(raw) > MAX_RESULT_FILE_BYTES:      # слишком тяжёлый PNG — пересохраняем в JPEG
                buf = io.BytesIO()
                res.convert("RGB").save(buf, format="JPEG", quality=95)
                raw, ext = buf.getvalue(), "jpg"
        out = io.BytesIO(raw)
        out.name = f"upscaled_{factor}x.{ext}"
        caption = f"✅ {src_size[0]}×{src_size[1]} → {res_size[0]}×{res_size[1]} ({factor}x)"
        # Отправляем файлом: как «фото» Telegram пересжал бы результат и часть выигрыша от апскейла пропала бы
        bot.send_document(chat_id, out, caption=caption)
        prompt_logger.log_result(gen_id, user_id, "upscale", "", duration_sec=time.time() - start_time, upscale_factor=factor)
        try:
            bot.delete_message(chat_id, call.message.message_id)
        except Exception:
            pass
    except Exception as e:
        prompt_logger.log_error(gen_id, user_id, "upscale", "", e, duration_sec=time.time() - start_time)
        bot.send_message(chat_id, f"❌ Ошибка апскейла: {e}")

# =========== ОБРАБОТЧИК ТЕКСТА (ПРОМПТЫ) ===========

@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_quick_generation(message):
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
    settings = get_user_settings(user_id)
    # --- Логирование промпта пользователя (до начала генерации) ---
    gen_id = prompt_logger.log_request(
        user_id, "txt2img", prompt,
        settings=settings,
        negative_prompt=get_user_negative_prompt(user_id),
    )
    start_time = time.time()
    status_msg = bot.reply_to(message, "🎨 **Генерация**\n\n🔄 Подготовка...", parse_mode="Markdown")
    def on_progress(progress, eta):
        update_progress_message(status_msg.message_id, message.chat.id, progress, eta)
    def on_status(text):
        update_status_message(status_msg.message_id, message.chat.id, text)
    try:
        image_file, used_seed, original_prompt, processed_prompt = generate_image(
            prompt, settings, user_id, on_progress, on_status, fixed_seed=None
        )
        final_image_file = image_file
        face_replacement_success = False
        if REACTOR_AVAILABLE and get_face_enabled(user_id):
            active_face = get_active_face(user_id)
            if active_face:
                try:
                    on_status("🎭 Заменяю лицо...")
                    final_image_file = swap_face_with_reactor(
                        source_image=active_face,
                        target_image=image_file,
                        face_model="None",
                        upscaler="None",
                        face_restorer="GFPGAN",
                        restorer_visibility=1.0
                    )
                    face_replacement_success = True
                except Exception as e:
                    print(f"Ошибка замены лица: {e}")
                    final_image_file = image_file
        try:
            bot.delete_message(message.chat.id, status_msg.message_id)
        except:
            pass
        caption = (
            f"✨ <b>Исходный промпт:</b>\n<code>{_h(_cut(original_prompt, 700))}</code>\n\n"
            f"🤖 <b>Обработанный промпт:</b>\n<code>{_h(_cut(processed_prompt, 1200))}</code>\n\n"
            f"📐 <b>Размер:</b> {settings.get('width', 512)}x{settings.get('height', 768)}\n"
            f"⏱️ <b>Шаги:</b> {settings.get('steps', 20)}\n"
            f"🎚️ <b>CFG:</b> {settings.get('cfg_scale', 7.0)}\n"
            f"🎯 <b>Сэмплер:</b> <code>{_h(settings.get('sampler_name'))}</code>\n"
            f"🎲 <b>Seed:</b> <code>{used_seed}</code>\n"
        )
        if face_replacement_success:
            caption += "🎭 <b>Лицо заменено</b>\n"
        # Фото либо файл (если не проходит по весу/разрешению), подпись длиннее 1024 — отдельным сообщением
        sent_msg = send_image_smart(message.chat.id, final_image_file, caption=caption, parse_mode="HTML",
                                    reply_to_message_id=message.id)
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
        prompt_logger.log_result(
            gen_id, user_id, "txt2img", original_prompt,
            processed_prompt=processed_prompt, seed=used_seed,
            settings=settings, duration_sec=time.time() - start_time,
            face_replaced=face_replacement_success,
        )
        last_images[user_id] = final_image_file
        last_message_ids[user_id] = sent_msg.message_id
        last_seeds[user_id] = used_seed
        bot.send_message(message.chat.id, "🔧 **Действия с изображением:**", reply_markup=get_image_actions_keyboard(), reply_to_message_id=sent_msg.message_id)
    except Exception as e:
        prompt_logger.log_error(gen_id, user_id, "txt2img", prompt, e, duration_sec=time.time() - start_time)
        _report_error(message.chat.id, status_msg.message_id, e)

# =========== ОСТАЛЬНЫЕ ОБРАБОТЧИКИ (ЛИЦА, НАСТРОЙКИ, АПСКЕЙЛ и т.д.) ===========

@bot.callback_query_handler(func=lambda call: call.data == "face_toggle_main")
def handle_face_toggle_main(call):
    user_id = str(call.message.chat.id)
    current = get_face_enabled(user_id)
    new_state = not current
    set_face_enabled(user_id, new_state)
    status = "включена ✅" if new_state else "выключена ❌"
    bot.answer_callback_query(call.id, f"Замена лиц {status}")
    face_management(call.message)

@bot.callback_query_handler(func=lambda call: call.data == "face_add")
def handle_face_add(call):
    msg = bot.send_message(call.message.chat.id, "📸 **Добавление нового лица**\n\nОтправьте **четкое фото лица** анфас.\n\n**Рекомендации:**\n• Хорошее освещение, без теней\n• Лицо занимает 70-80% кадра\n• Нейтральное выражение\n• Минимум аксессуаров\n\nПосле отправки фото, вы сможете задать имя для этого лица.", parse_mode="Markdown")
    bot.register_next_step_handler(msg, save_new_face, call.message.chat.id)

def save_new_face(message, original_chat_id):
    user_id = str(message.chat.id)
    # Принимаем и обычное фото, и изображение, отправленное «как файл» (без сжатия — для лица это лучше)
    document = getattr(message, "document", None)
    if message.photo:
        file_id = message.photo[-1].file_id
    elif document and (getattr(document, "mime_type", None) or "").startswith("image/"):
        file_id = document.file_id
    else:
        bot.send_message(message.chat.id, "❌ Пожалуйста, отправьте фото.")
        return
    try:
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        # Лица хранятся как JPEG: приводим формат к единому и заодно проверяем, что это действительно картинка
        try:
            with Image.open(io.BytesIO(downloaded_file)) as face_img:
                converted = io.BytesIO()
                face_img.convert("RGB").save(converted, format="JPEG", quality=95)
        except Exception:
            bot.send_message(message.chat.id, "❌ Не удалось прочитать изображение. Отправьте фото в формате JPG/PNG.")
            return
        temp_face = io.BytesIO(converted.getvalue())
        temp_data.setdefault(user_id, {})['temp_face'] = temp_face
        msg = bot.send_message(message.chat.id, "✏️ **Введите имя для этого лица**\n\nНапример: 'я', 'друг', 'коллега' и т.д.\n\nИли отправьте 'default' для стандартного имени.", parse_mode="Markdown")
        bot.register_next_step_handler(msg, set_face_name, original_chat_id)
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка: {e}")

def set_face_name(message, original_chat_id):
    user_id = str(message.chat.id)
    face_name = (message.text or "").strip().lower()
    if face_name == "default":
        face_name = f"face_{len(get_user_faces(user_id)) + 1}"
    if not _valid_face_name(face_name):
        msg = bot.send_message(message.chat.id, "❌ Некорректное имя (до 32 символов, без / \\ : * ? \" < > |). Введите другое имя:")
        bot.register_next_step_handler(msg, set_face_name, original_chat_id)
        return
    faces = get_user_faces(user_id)
    if any(f["name"] == face_name for f in faces):
        msg = bot.send_message(message.chat.id, f"❌ Лицо с именем '{face_name}' уже существует. Введите другое имя:")
        bot.register_next_step_handler(msg, set_face_name, original_chat_id)
        return
    temp_face = temp_data.get(user_id, {}).get('temp_face')
    if temp_face is not None:
        success = save_user_face(user_id, temp_face, face_name)
        if success:
            bot.send_message(message.chat.id, f"✅ <b>Лицо '{_h(face_name)}' успешно сохранено!</b>\n\nТеперь вы можете выбрать его как активное в меню управления лицами.", parse_mode="HTML")
            preview = get_face_preview(user_id, face_name)
            if preview:
                bot.send_photo(message.chat.id, preview, caption=f"📸 {face_name}")
        else:
            bot.send_message(message.chat.id, "❌ Ошибка при сохранении лица.")
        temp_data.get(user_id, {}).pop('temp_face', None)
    else:
        bot.send_message(message.chat.id, "❌ Ошибка: фото не найдено.")

# --- Вспомогательные функции для меню лиц ---------------------------------
# Причина бага с удалением: после выбора лица бот отправляет ФОТО-сообщение
# (превью), а кнопки «Удалить» / «Назад» пытались сделать edit_message_text
# именно для него. Telegram не позволяет редактировать текст у фото-сообщения
# (400: there is no text in the message to edit), поэтому ничего не происходило.
# Также имя лица в callback_data могло превысить лимит Telegram в 64 байта
# (кириллица = 2 байта на символ), поэтому теперь в кнопках передаётся короткий id.

def _face_id(face_name):
    return hashlib.md5(face_name.encode("utf-8")).hexdigest()[:10]

def _find_face_by_id(user_id, face_id):
    return next((f for f in get_user_faces(user_id) if _face_id(f["name"]) == face_id), None)

def _safe_answer(call, text=None, show_alert=False):
    try:
        bot.answer_callback_query(call.id, text, show_alert=show_alert)
    except Exception:
        pass  # на один callback можно ответить только один раз

def _replace_message(call, text, markup=None):
    """Заменяет сообщение с кнопками на текстовое (работает и для фото-сообщений)."""
    chat_id = call.message.chat.id
    message_id = call.message.message_id
    if getattr(call.message, "content_type", "text") == "text":
        try:
            bot.edit_message_text(text, chat_id, message_id, parse_mode="HTML", reply_markup=markup)
            return
        except Exception as e:
            if "message is not modified" in str(e):
                return
    try:
        bot.delete_message(chat_id, message_id)
    except Exception:
        pass
    bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)

def _valid_face_name(name):
    return bool(name) and len(name) <= 32 and not any(c in name for c in '/\\:*?"<>|') and name not in (".", "..")

# ---------------------------------------------------------------------------

@bot.callback_query_handler(func=lambda call: call.data == "face_list")
def handle_face_list(call):
    user_id = str(call.message.chat.id)
    faces = get_user_faces(user_id)
    markup = types.InlineKeyboardMarkup(row_width=1)
    if not faces:
        markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="face_back"))
        _replace_message(call, "📭 У вас нет сохранённых лиц.", markup)
        _safe_answer(call)
        return
    for face in faces:
        status = "⭐ " if face["is_active"] else "📸 "
        markup.add(types.InlineKeyboardButton(f"{status}{face['name']}", callback_data=f"face_select_{_face_id(face['name'])}"))
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="face_back"))
    _replace_message(call, "📋 <b>Список ваших лиц</b>\n\nВыберите лицо для управления:", markup)
    _safe_answer(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("face_select_"))
def handle_face_select(call):
    user_id = str(call.message.chat.id)
    face = _find_face_by_id(user_id, call.data[len("face_select_"):])
    if not face:
        _safe_answer(call, "Лицо не найдено")
        return
    face_name = face["name"]
    fid = _face_id(face_name)
    markup = types.InlineKeyboardMarkup(row_width=2)
    if not face["is_active"]:
        markup.add(types.InlineKeyboardButton("⭐ Сделать активным", callback_data=f"face_activate_{fid}"))
    markup.add(
        types.InlineKeyboardButton("✏️ Переименовать", callback_data=f"face_rename_{fid}"),
        types.InlineKeyboardButton("🗑️ Удалить", callback_data=f"face_delete_{fid}"),
        types.InlineKeyboardButton("◀️ Назад к списку", callback_data="face_list")
    )
    preview = get_face_preview(user_id, face_name)
    status_text = "⭐ Активно" if face["is_active"] else "📸 Не активно"
    caption = f"<b>{html.escape(face_name)}</b>\n{status_text}"
    if preview:
        bot.send_photo(call.message.chat.id, preview, caption=caption, parse_mode="HTML", reply_markup=markup)
        try:
            bot.delete_message(call.message.chat.id, call.message.message_id)
        except Exception:
            pass
    else:
        _replace_message(call, caption, markup)
    _safe_answer(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("face_activate_"))
def handle_face_activate(call):
    user_id = str(call.message.chat.id)
    face = _find_face_by_id(user_id, call.data[len("face_activate_"):])
    if face and set_active_face(user_id, face["name"]):
        _safe_answer(call, f"Активное лицо: {face['name']}")
    else:
        _safe_answer(call, "Ошибка при активации", show_alert=True)
        return
    handle_face_list(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("face_rename_"))
def handle_face_rename(call):
    user_id = str(call.message.chat.id)
    face = _find_face_by_id(user_id, call.data[len("face_rename_"):])
    if not face:
        _safe_answer(call, "Лицо не найдено")
        return
    old_name = face["name"]
    msg = bot.send_message(call.message.chat.id, f"✏️ <b>Введите новое имя для '{html.escape(old_name)}':</b>", parse_mode="HTML")
    bot.register_next_step_handler(msg, process_face_rename, user_id, old_name, call.message.chat.id)
    _safe_answer(call)

def process_face_rename(message, user_id, old_name, chat_id):
    new_name = (message.text or "").strip().lower()
    if not _valid_face_name(new_name):
        bot.send_message(chat_id, "❌ Некорректное имя (до 32 символов, без / \\ : * ? \" < > |).")
        return
    if new_name == old_name:
        bot.send_message(chat_id, "❌ Имя не изменено.")
        return
    faces = get_user_faces(user_id)
    if any(f["name"] == new_name for f in faces):
        bot.send_message(chat_id, f"❌ Лицо с именем '{new_name}' уже существует.")
        return
    success = rename_face(user_id, old_name, new_name)
    if success:
        bot.send_message(chat_id, f"✅ Лицо переименовано: '{old_name}' → '{new_name}'")
    else:
        bot.send_message(chat_id, "❌ Ошибка при переименовании.")

@bot.callback_query_handler(func=lambda call: call.data.startswith("face_delete_"))
def handle_face_delete(call):
    user_id = str(call.message.chat.id)
    face = _find_face_by_id(user_id, call.data[len("face_delete_"):])
    if not face:
        _safe_answer(call, "Лицо не найдено")
        return
    fid = _face_id(face["name"])
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("✅ Да, удалить", callback_data=f"face_confirm_delete_{fid}"),
        types.InlineKeyboardButton("❌ Нет, отмена", callback_data="face_list")
    )
    _replace_message(call, f"⚠️ <b>Удалить лицо '{html.escape(face['name'])}'?</b>\n\nЭто действие нельзя отменить.", markup)
    _safe_answer(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("face_confirm_delete_"))
def handle_face_confirm_delete(call):
    user_id = str(call.message.chat.id)
    face = _find_face_by_id(user_id, call.data[len("face_confirm_delete_"):])
    if not face:
        _safe_answer(call, "Лицо не найдено или уже удалено")
        handle_face_list(call)
        return
    face_name = face["name"]
    if delete_face(user_id, face_name):
        _safe_answer(call, f"Лицо '{face_name}' удалено")
        handle_face_list(call)
    else:
        _safe_answer(call, "Ошибка при удалении", show_alert=True)

@bot.callback_query_handler(func=lambda call: call.data == "face_back")
def handle_face_back(call):
    face_management(call.message)

# =========== НАСТРОЙКИ (CALLBACK) ===========

@bot.callback_query_handler(func=lambda call: call.data == "toggle_prompt_processing")
def handle_toggle_prompt_processing(call):
    user_id = str(call.message.chat.id)
    current = get_prompt_processing_enabled(user_id)
    new_state = not current
    set_prompt_processing_enabled(user_id, new_state)
    status = "включена ✅" if new_state else "выключена ❌"
    bot.answer_callback_query(call.id, f"Обработка промпта {status}")
    show_settings(call.message)

@bot.callback_query_handler(func=lambda call: call.data == "set_diversity")
def handle_diversity(call):
    user_id = str(call.message.chat.id)
    settings = get_user_settings(user_id)
    modes = ["auto", "high", "low", "off"]
    current = settings.get("diversity_mode", "auto")
    if current not in modes:
        current = "auto"
    next_index = (modes.index(current) + 1) % len(modes)
    new_mode = modes[next_index]
    set_diversity_mode(user_id, new_mode)
    settings = get_user_settings(user_id)
    try:
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=get_settings_inline_keyboard(settings, user_id))
    except Exception as e:
        if "message is not modified" not in str(e):
            raise
    _safe_answer(call, f"Разнообразие: {new_mode}")

@bot.callback_query_handler(func=lambda call: call.data == "set_res")
def handle_resolution_button(call):
    user_id = str(call.message.chat.id)
    settings = get_user_settings(user_id)
    bot.edit_message_text("📐 **Выбери разрешение:**\n\nДоступны размеры от 512x512 до 1024x1280", call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_resolution_keyboard(settings['width'], settings['height']))
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("res_"))
def handle_resolution_choice(call):
    user_id = str(call.message.chat.id)
    _, w, h = call.data.split("_")
    w, h = int(w), int(h)
    update_user_settings(user_id, {"width": w, "height": h})
    bot.answer_callback_query(call.id, f"Размер установлен: {w}x{h}")
    show_settings(call.message)

@bot.callback_query_handler(func=lambda call: call.data == "set_cfg")
def handle_cfg_button(call):
    user_id = str(call.message.chat.id)
    settings = get_user_settings(user_id)
    bot.edit_message_text("🎚️ **Выбери CFG Scale:**\n\n• 5-6: больше креатива\n• 7-8: золотая середина\n• 9-10: строгое следование промпту", call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_cfg_keyboard(settings['cfg_scale']))
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("cfg_"))
def handle_cfg_choice(call):
    user_id = str(call.message.chat.id)
    cfg = float(call.data.split("_")[1])
    update_user_settings(user_id, {"cfg_scale": cfg})
    bot.answer_callback_query(call.id, f"CFG установлен: {cfg}")
    show_settings(call.message)

@bot.callback_query_handler(func=lambda call: call.data == "set_steps")
def handle_steps_button(call):
    user_id = str(call.message.chat.id)
    settings = get_user_settings(user_id)
    bot.edit_message_text("⏱️ **Выбери количество шагов:**\n\n• 15-20: быстро\n• 25-35: оптимально\n• 40-50: максимально", call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_steps_keyboard(settings['steps']))
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("steps_"))
def handle_steps_choice(call):
    user_id = str(call.message.chat.id)
    steps = int(call.data.split("_")[1])
    update_user_settings(user_id, {"steps": steps})
    bot.answer_callback_query(call.id, f"Шагов: {steps}")
    show_settings(call.message)

# =========== НЕГАТИВНЫЙ ПРОМПТ И НЕГАТИВНЫЕ ЭМБЕДДИНГИ ===========

EMBEDDINGS_PER_PAGE = 10

def _emb_id(name):
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:10]

def _edit_or_send(chat_id, message_id, text, markup):
    """Редактирует сообщение меню (если можно), иначе отправляет новое."""
    if message_id is not None:
        try:
            bot.edit_message_text(text, chat_id, message_id, parse_mode="HTML", reply_markup=markup)
            return
        except Exception as e:
            if "message is not modified" in str(e):
                return
    bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)

def _show_negative_menu(chat_id, user_id, message_id=None):
    negative = get_user_negative_prompt(user_id)
    is_default = negative == DEFAULT_NEGATIVE_PROMPT
    embeddings = get_user_negative_embeddings(user_id)
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
    _edit_or_send(chat_id, message_id, text, markup)

@bot.callback_query_handler(func=lambda call: call.data == "set_neg")
def handle_neg_button(call):
    _safe_answer(call)
    _show_negative_menu(call.message.chat.id, str(call.message.chat.id), call.message.message_id)

@bot.callback_query_handler(func=lambda call: call.data == "neg_text")
def handle_neg_text_button(call):
    _safe_answer(call)
    msg = bot.send_message(call.message.chat.id, "✏️ Отправь свой негативный промпт\n\nИли отправь <code>сбросить</code> для стандартного", parse_mode="HTML")
    bot.register_next_step_handler(msg, save_custom_negative, call.message.chat.id, call.message.message_id)

def save_custom_negative(message, chat_id, original_msg_id):
    user_id = str(message.chat.id)
    new_neg = _text(message)
    if not new_neg or new_neg.startswith('/') or new_neg in BUTTON_TEXTS:
        # Фото, стикер, команда или кнопка вместо текста раньше сохранялись/роняли обработчик
        bot.send_message(chat_id, "❌ Отменено: нужен текст негативного промпта.")
        return
    if new_neg.lower() == "сбросить":
        set_user_negative_prompt(user_id, None)
        bot.send_message(chat_id, "✅ Сброшено до стандартного!")
    else:
        set_user_negative_prompt(user_id, new_neg)
        bot.send_message(chat_id, f"✅ Сохранено:\n<code>{_h(_cut(new_neg, 100))}</code>", parse_mode="HTML")
    try:
        bot.delete_message(chat_id, original_msg_id)
    except Exception:
        pass
    show_settings(message)

@bot.callback_query_handler(func=lambda call: call.data == "neg_reset")
def handle_neg_reset(call):
    set_user_negative_prompt(str(call.message.chat.id), None)
    _safe_answer(call, "Сброшено до стандартного")
    _show_negative_menu(call.message.chat.id, str(call.message.chat.id), call.message.message_id)

def _build_embeddings_menu(user_id, page):
    names = get_available_embeddings()
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

    selected = set(get_user_negative_embeddings(user_id))
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

def _show_embeddings_menu(call, page):
    text, markup = _build_embeddings_menu(str(call.message.chat.id), page)
    _edit_or_send(call.message.chat.id, call.message.message_id, text, markup)

@bot.callback_query_handler(func=lambda call: call.data == "neg_emb")
def handle_neg_embeddings_button(call):
    _safe_answer(call)
    _show_embeddings_menu(call, 0)

@bot.callback_query_handler(func=lambda call: call.data.startswith("negemb_p_"))
def handle_negemb_page(call):
    try:
        page = int(call.data[len("negemb_p_"):])
    except ValueError:
        page = 0
    _safe_answer(call)
    _show_embeddings_menu(call, page)

@bot.callback_query_handler(func=lambda call: call.data == "negemb_noop")
def handle_negemb_noop(call):
    _safe_answer(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("negemb_t_"))
def handle_negemb_toggle(call):
    user_id = str(call.message.chat.id)
    try:
        emb_id, page_text = call.data[len("negemb_t_"):].rsplit("_", 1)
        page = int(page_text)
    except ValueError:
        _safe_answer(call)
        return
    name = next((n for n in (get_available_embeddings() or []) if _emb_id(n) == emb_id), None)
    if name is None:
        _safe_answer(call, "Эмбеддинг не найден — обновите список", show_alert=True)
        return
    state = toggle_user_negative_embedding(user_id, name)
    _safe_answer(call, f"{'✅ Добавлен' if state else '➖ Убран'}: {name}"[:200])
    _show_embeddings_menu(call, page)

@bot.callback_query_handler(func=lambda call: call.data == "negemb_clear")
def handle_negemb_clear(call):
    set_user_negative_embeddings(str(call.message.chat.id), [])
    _safe_answer(call, "Выбор снят")
    _show_embeddings_menu(call, 0)

@bot.callback_query_handler(func=lambda call: call.data == "negemb_default")
def handle_negemb_default(call):
    set_user_negative_embeddings(str(call.message.chat.id), None)
    _safe_answer(call, "Возвращены рекомендованные")
    _show_embeddings_menu(call, 0)

@bot.callback_query_handler(func=lambda call: call.data == "negemb_refresh")
def handle_negemb_refresh(call):
    _safe_answer(call, "Обновляю список...")
    refresh_embeddings_on_server()
    get_available_embeddings(force_refresh=True)
    _show_embeddings_menu(call, 0)

# =========== АПСКЕЙЛ, ПОВТОР, РЕДАКТИРОВАНИЕ ===========

@bot.callback_query_handler(func=lambda call: call.data.startswith("upscale_"))
def handle_upscale_callback(call):
    user_id = str(call.message.chat.id)
    source_image = last_images.get(user_id)
    if source_image is None:
        bot.answer_callback_query(call.id, "Нет изображения!")
        return
    factor = int(call.data.split("_")[1])
    bot.answer_callback_query(call.id, f"Увеличиваю в {factor}x...")
    try:
        bot.edit_message_text("⏳ Апскейл...", call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    # --- Логирование апскейла (без промпта, но с историей действий пользователя) ---
    gen_id = prompt_logger.log_request(user_id, "upscale", original_prompt="", upscale_factor=factor)
    start_time = time.time()
    try:
        upscaled = upscale_image(source_image, upscale_factor=float(factor))
        prompt_logger.log_result(gen_id, user_id, "upscale", "", duration_sec=time.time() - start_time, upscale_factor=factor)
        # После апскейла картинка часто весит >10 МБ или шире 10000 px по сумме сторон — тогда уйдёт файлом
        send_image_smart(call.message.chat.id, upscaled, caption=f"✅ {factor}x")
        last_images[user_id] = upscaled
        try:
            bot.delete_message(call.message.chat.id, call.message.message_id)
        except Exception:
            pass
    except Exception as e:
        prompt_logger.log_error(gen_id, user_id, "upscale", "", e, duration_sec=time.time() - start_time)
        bot.send_message(call.message.chat.id, f"❌ Ошибка: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data == "action_upscale")
def handle_action_upscale(call):
    bot.answer_callback_query(call.id)
    bot.send_message(call.message.chat.id, "Выбери коэффициент:", reply_markup=get_upscale_keyboard())

@bot.callback_query_handler(func=lambda call: call.data == "action_repeat")
def handle_repeat_generation(call):
    user_id = str(call.message.chat.id)
    if user_id not in last_configs:
        bot.answer_callback_query(call.id, "Нет сохраненной конфигурации для повтора", show_alert=True)
        return
    config = last_configs[user_id]
    # Если это ControlNet, предлагаем повторить с тем же скелетом?
    # Но кнопка "Повторить" остаётся для обычных генераций, для ControlNet используем отдельную кнопку "тот же скелет"
    # Поэтому здесь обрабатываем только обычные (txt2img)
    if config.get('type') == 'controlnet':
        bot.answer_callback_query(call.id, "Для ControlNet используйте кнопку 'Тот же скелет'", show_alert=True)
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
    bot.send_message(call.message.chat.id, confirm_text, parse_mode="HTML", reply_markup=markup)
    _safe_answer(call)

@bot.callback_query_handler(func=lambda call: call.data == "repeat_confirm")
def repeat_confirm(call):
    user_id = str(call.message.chat.id)
    if user_id not in last_configs:
        bot.answer_callback_query(call.id, "Нет сохраненной конфигурации", show_alert=True)
        return
    config = last_configs[user_id]
    # Если это ControlNet - перенаправляем на "тот же скелет"
    if config.get('type') == 'controlnet':
        bot.answer_callback_query(call.id, "Для ControlNet используйте кнопку 'Тот же скелет'", show_alert=True)
        return
    # Отвечаем на callback СРАЗУ: раньше ответ шёл после многоминутной генерации — кнопка «висела» с индикатором,
    # а запоздалый ответ падал с «query is too old».
    _safe_answer(call, "Повторяю генерацию...")
    chat_id = call.message.chat.id
    try:
        bot.edit_message_text("🔄 Повторяю генерацию...", chat_id, call.message.message_id)
    except Exception:
        pass
    status_msg = bot.send_message(chat_id, "🎨 **Генерация**\n\n`[░░░░░░░░░░░░░░░░░░░░]` 0%\n\n⏳ Подготовка...", parse_mode="Markdown")
    def on_progress(progress, eta):
        update_progress_message(status_msg.message_id, chat_id, progress, eta)
    def on_status(text):
        update_status_message(status_msg.message_id, chat_id, text)
    try:
        # Точный повтор: промпты берутся ровно такими, какие ушли в WebUI в прошлый раз
        # (без повторного перевода через Ollama, без дублирования LoRA и новых «фраз разнообразия»)
        image_file, used_seed, original_prompt, processed_prompt = generate_image(
            config['prompt'], dict(config['settings']), user_id, on_progress, on_status, fixed_seed=config['seed'],
            exact_prompt=config.get('final_prompt'), exact_negative=config.get('final_negative')
        )
        final_image_file = image_file
        face_replacement_success = False
        if REACTOR_AVAILABLE and config.get('face_enabled', False):
            active_face = get_active_face(user_id)
            if active_face:
                try:
                    on_status("🎭 Заменяю лицо...")
                    final_image_file = swap_face_with_reactor(
                        source_image=active_face,
                        target_image=image_file,
                        face_model="None",
                        upscaler="None",
                        face_restorer="GFPGAN",
                        restorer_visibility=1.0
                    )
                    face_replacement_success = True
                except Exception as e:
                    print(f"Ошибка замены лица: {e}")
                    final_image_file = image_file
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except:
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
        if face_replacement_success:
            caption += "🎭 <b>Лицо заменено</b>\n"
        sent_msg = send_image_smart(chat_id, final_image_file, caption=caption, parse_mode="HTML")
        last_images[user_id] = final_image_file
        last_message_ids[user_id] = sent_msg.message_id
        last_seeds[user_id] = used_seed
        bot.send_message(chat_id, "🔧 **Действия с изображением:**", reply_markup=get_image_actions_keyboard(), reply_to_message_id=sent_msg.message_id)
    except Exception as e:
        _report_error(chat_id, status_msg.message_id, e)

@bot.callback_query_handler(func=lambda call: call.data == "repeat_cancel")
def repeat_cancel(call):
    try:
        bot.edit_message_text("❌ Повтор отменен", call.message.chat.id, call.message.message_id)
    except Exception:
        pass
    _safe_answer(call)

# =========== РАСШИРЕННАЯ ГЕНЕРАЦИЯ ===========

def extended_generation_prompt(message):
    text = _text(message)
    if _is_cancel_text(text):
        bot.send_message(message.chat.id, "❌ Отменено.")
        return
    user_id = str(message.chat.id)
    prompt = text
    temp_data.setdefault(user_id, {})['prompt'] = prompt
    current_neg = get_user_negative_prompt(user_id)
    is_custom = current_neg != DEFAULT_NEGATIVE_PROMPT
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        types.InlineKeyboardButton("✅ Стандартный негатив", callback_data="ext_neg_default"),
        types.InlineKeyboardButton(f"✏️ Свой негатив{' ✓' if is_custom else ''}", callback_data="ext_neg_custom"),
        types.InlineKeyboardButton("⚙️ Изменить в настройках", callback_data="ext_neg_settings")
    )
    bot.send_message(message.chat.id, f"📝 <b>Промпт:</b> <i>{_h(_cut(prompt, 100))}</i>\n\nВыбери негативный промпт:", parse_mode="HTML", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("ext_neg_"))
def handle_extended_neg_choice(call):
    user_id = str(call.message.chat.id)
    prompt = temp_data.get(user_id, {}).get('prompt')
    if not prompt:
        bot.answer_callback_query(call.id, "Сессия истекла")
        bot.send_message(call.message.chat.id, "Нажми кнопку генерации заново.")
        return
    if call.data == "ext_neg_default":
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        neg_source = "стандартный"
    elif call.data == "ext_neg_custom":
        negative_prompt = get_user_negative_prompt(user_id)
        if negative_prompt == DEFAULT_NEGATIVE_PROMPT:
            bot.answer_callback_query(call.id, "Сначала задай свой негатив в настройках!")
            return
        neg_source = "свой"
    elif call.data == "ext_neg_settings":
        bot.answer_callback_query(call.id)
        show_settings(call.message)
        return
    else:
        return
    # Сессия использована: убираем промпт сразу, чтобы двойное нажатие кнопки не запустило генерацию дважды
    session = temp_data.get(user_id)
    if session is not None:
        session.pop('prompt', None)
    bot.answer_callback_query(call.id, f"Генерация с {neg_source} негативом...")
    # --- Логирование промпта пользователя (расширенная генерация) ---
    gen_id = prompt_logger.log_request(
        user_id, "txt2img_advanced", prompt,
        settings=get_user_settings(user_id), negative_prompt=negative_prompt,
    )
    start_time = time.time()
    status_msg = bot.send_message(call.message.chat.id, "🎨 **Генерация**\n\n`[░░░░░░░░░░░░░░░░░░░░]` 0%\n\n⏳ Подготовка...", parse_mode="Markdown")
    def on_progress(progress, eta):
        update_progress_message(status_msg.message_id, call.message.chat.id, progress, eta)
    settings = get_user_settings(user_id)
    try:
        images, used_seed = generate_txt2img_advanced(prompt, negative_prompt, settings, on_progress, user_id=user_id)
        if not images:
            raise Exception("WebUI не вернул изображений")
        try:
            bot.delete_message(call.message.chat.id, status_msg.message_id)
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
            sent_msg = send_image_smart(call.message.chat.id, image_file, caption=caption, parse_mode="HTML")
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
        prompt_logger.log_result(
            gen_id, user_id, "txt2img_advanced", prompt,
            processed_prompt=prompt, seed=used_seed, settings=settings,
            duration_sec=time.time() - start_time, negative_prompt=negative_prompt,
        )
        bot.send_message(call.message.chat.id, "🔧 **Действия с изображением:**", reply_markup=get_image_actions_keyboard())
    except Exception as e:
        prompt_logger.log_error(gen_id, user_id, "txt2img_advanced", prompt, e, duration_sec=time.time() - start_time)
        _report_error(call.message.chat.id, status_msg.message_id, e)
    temp_data.pop(user_id, None)  # раньше del по несуществующему ключу падал с KeyError при гонке


# =========== РЕДАКТИРОВАНИЕ (IMG2IMG) ===========

def apply_face_replacement_if_enabled(user_id, image):
    if not REACTOR_AVAILABLE:
        return image
    if not get_face_enabled(user_id):
        return image
    active_face = get_active_face(user_id)
    if not active_face:
        return image
    try:
        print(f"🎭 Применяю замену лица для пользователя {user_id}")
        result = swap_face_with_reactor(
            source_image=active_face,
            target_image=image,
            face_model="None",
            upscaler="None",
            face_restorer="GFPGAN",
            restorer_visibility=1.0
        )
        print("✅ Замена лица выполнена успешно")
        return result
    except Exception as e:
        print(f"❌ Ошибка замены лица: {e}")
        return image

@bot.callback_query_handler(func=lambda call: call.data == "action_img2img")
def show_edit_options(call):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("🎨 Всё изображение", callback_data="edit_all"),
        types.InlineKeyboardButton("🧍 Тело", callback_data="edit_body"),
        types.InlineKeyboardButton("🌄 Фон", callback_data="edit_background"),
        types.InlineKeyboardButton("👗 Одежда", callback_data="edit_clothes"),
        types.InlineKeyboardButton("🚶 Поза (ControlNet)", callback_data="edit_pose")
    )
    bot.edit_message_text("Выберите, что хотите изменить:", call.message.chat.id, call.message.message_id, reply_markup=markup)
    bot.answer_callback_query(call.id)
@bot.callback_query_handler(func=lambda call: call.data == "set_sampler")
def handle_sampler_button(call):
    """Показывает список сэмплеров для выбора"""
    user_id = str(call.message.chat.id)
    from settings import get_available_samplers, get_user_sampler
    samplers = get_available_samplers()
    current = get_user_sampler(user_id)
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    for sampler in samplers[:12]:  # Показываем первые 12 для компактности
        label = f"✅ {sampler}" if sampler == current else sampler
        markup.add(types.InlineKeyboardButton(label, callback_data=f"sampler_set_{sampler}"))
    
    # Если сэмплеров больше 12, добавляем кнопку "Ещё"
    if len(samplers) > 12:
        markup.add(types.InlineKeyboardButton("📋 Показать все", callback_data="sampler_show_all"))
    
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="back_to_settings"))
    
    bot.edit_message_text(
        f"🎯 **Выберите сэмплер**\n\n"
        f"Текущий: `{current}`\n\n"
        f"Доступно: {len(samplers)}",
        call.message.chat.id,
        call.message.message_id,
        parse_mode="Markdown",
        reply_markup=markup
    )
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith("sampler_set_"))
def handle_sampler_choice(call):
    user_id = str(call.message.chat.id)
    sampler_name = call.data[len("sampler_set_"):]
    set_user_sampler(user_id, sampler_name)
    
    bot.answer_callback_query(call.id, f"✅ Сэмплер: {sampler_name}")
    # Возвращаемся к настройкам
    show_settings(call.message)

@bot.callback_query_handler(func=lambda call: call.data == "sampler_show_all")
def handle_sampler_show_all(call):
    user_id = str(call.message.chat.id)
    from settings import get_available_samplers, get_user_sampler
    samplers = get_available_samplers()
    current = get_user_sampler(user_id)
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    for sampler in samplers:
        label = f"✅ {sampler}" if sampler == current else sampler
        markup.add(types.InlineKeyboardButton(label, callback_data=f"sampler_set_{sampler}"))
    
    markup.add(types.InlineKeyboardButton("◀️ Назад", callback_data="set_sampler"))
    
    bot.edit_message_text(
        f"📋 **Все сэмплеры**\nТекущий: `{current}`",
        call.message.chat.id,
        call.message.message_id,
        parse_mode="Markdown",
        reply_markup=markup
    )
    bot.answer_callback_query(call.id)
    
@bot.callback_query_handler(func=lambda call: call.data.startswith("edit_"))
def handle_edit_type(call):
    edit_type = call.data.split("_")[1]
    user_id = str(call.message.chat.id)
    if last_images.get(user_id) is None:
        bot.answer_callback_query(call.id, "Нет изображения для редактирования", show_alert=True)
        return
    # Саму картинку в temp_data больше не кладём (она нигде не читалась, но удерживала память): берётся last_images
    temp_data.setdefault(user_id, {})['edit_type'] = edit_type
    bot.edit_message_text(f"✍️ Введите описание изменений для **{edit_type}**:\n\nНапример: 'добавить мускулы' или 'заменить фон на городской пейзаж'", call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    bot.register_next_step_handler(call.message, process_edit_prompt, edit_type, user_id, call.message.message_id)
    bot.answer_callback_query(call.id)

def process_edit_prompt(message, edit_type, user_id, original_msg_id):
    prompt = _text(message)
    if _is_cancel_text(prompt):
        bot.send_message(message.chat.id, "❌ Нужен текст промпта (не кнопка, команда или файл). Редактирование отменено.")
        return
    original_image = last_images.get(user_id)
    if not original_image:
        bot.send_message(message.chat.id, "❌ Изображение не найдено. Отмена.")
        return
    settings = get_user_settings(user_id)
    # --- Логирование промпта пользователя (редактирование img2img/inpaint) ---
    gen_type = {"all": "img2img", "pose": "controlnet_edit"}.get(edit_type, "inpaint")
    gen_id = prompt_logger.log_request(
        user_id, gen_type, prompt, settings=settings, edit_type=edit_type,
        negative_prompt=get_user_negative_prompt(user_id),
    )
    start_time = time.time()
    status_msg = bot.send_message(message.chat.id, f"🔄 Редактирую <b>{_h(edit_type)}</b>...\nПромпт: <code>{_h(_cut(prompt, 50))}</code>", parse_mode="HTML")
    def on_progress(progress, eta):
        update_progress_message(status_msg.message_id, message.chat.id, progress, eta)
    def on_status(text):
        update_status_message(status_msg.message_id, message.chat.id, text)
    try:
        if edit_type == "all":
            new_image = edit_all_image(prompt, original_image, settings, on_progress, on_status, user_id=user_id)
        elif edit_type == "body":
            new_image = edit_body_part(prompt, original_image, settings, "body", on_progress, on_status, user_id=user_id)
        elif edit_type == "background":
            new_image = edit_body_part(prompt, original_image, settings, "background", on_progress, on_status, user_id=user_id)
        elif edit_type == "clothes":
            new_image = edit_body_part(prompt, original_image, settings, "clothes", on_progress, on_status, user_id=user_id)
        elif edit_type == "pose":
            new_image = edit_pose(prompt, original_image, settings, on_progress, on_status, user_id=user_id)
        else:
            raise ValueError(f"Неизвестный тип редактирования: {edit_type}")
        final_image = apply_face_replacement_if_enabled(user_id, new_image)
        try:
            bot.delete_message(message.chat.id, status_msg.message_id)
        except:
            pass
        sent_msg = send_image_smart(message.chat.id, final_image,
                                    caption=f"✅ Редактирование <b>{_h(edit_type)}</b>\nПромпт: <code>{_h(_cut(prompt, 800))}</code>", parse_mode="HTML")
        last_images[user_id] = final_image
        last_message_ids[user_id] = sent_msg.message_id
        last_configs[user_id] = {
            "type": "txt2img",
            "prompt": prompt,
            "original_prompt": prompt,
            "settings": dict(settings),
            "seed": 0,
            "face_enabled": get_face_enabled(user_id)
        }
        prompt_logger.log_result(
            gen_id, user_id, gen_type, prompt, processed_prompt=prompt,
            settings=settings, duration_sec=time.time() - start_time, edit_type=edit_type,
        )
        bot.send_message(message.chat.id, "🔧 Действия с изображением:", reply_markup=get_image_actions_keyboard())
    except Exception as e:
        prompt_logger.log_error(gen_id, user_id, gen_type, prompt, e, duration_sec=time.time() - start_time)
        _report_error(message.chat.id, status_msg.message_id, e)
    temp_data.get(user_id, {}).pop('edit_type', None)

def edit_all_image(prompt, image, settings, progress_callback=None, status_callback=None, user_id="default"):
    if status_callback:
        status_callback("🔄 Генерация через img2img...")
    images = generate_img2img(prompt, image, settings, progress_callback, user_id=user_id)
    return images[0] if images else image

def edit_body_part(prompt, image, settings, part, progress_callback=None, status_callback=None, user_id="default"):
    from segmentation import get_mask_for_part
    if status_callback:
        status_callback(f"🔄 Выделяю {part}...")
    mask = get_mask_for_part(image, part)
    if mask is None:
        raise Exception(f"Не удалось выделить {part}. Попробуйте другой тип.")
    if status_callback:
        status_callback(f"🔄 Перерисовываю {part}...")
    new_image = generate_inpaint(prompt, image, mask, settings, progress_callback, user_id=user_id)
    return new_image

def edit_pose(prompt, image, settings, progress_callback=None, status_callback=None, user_id="default"):
    try:
        from segmentation import get_pose_control_image
    except ImportError:
        raise Exception("Для позы требуется установить mediapipe и opencv.")
    if status_callback:
        status_callback("🔄 Детектирую позу...")
    control_image = get_pose_control_image(image)
    if control_image is None:
        raise Exception("Не удалось детектировать позу на изображении.")
    if status_callback:
        status_callback("🔄 Генерация с ControlNet...")
    # Используем функцию generate_controlnet из controlnet_manager
    new_image = generate_controlnet(prompt, image, control_image, "openpose", "control_v11p_sd15_openpose", settings, progress_callback, user_id=user_id)
    return new_image
# =========== ОБРАБОТЧИКИ LORA ===========

def _lora_id(lora_name):
    return hashlib.md5(lora_name.encode("utf-8")).hexdigest()[:10]

def handle_manage_lora(call):
    user_id = str(call.message.chat.id)
    available = get_available_loras()
    lora_enabled = get_user_lora_enabled(user_id)
    markup = types.InlineKeyboardMarkup(row_width=2)
    for lora_name, info in available.items():
        status = "✅" if lora_enabled.get(lora_name, True) else "❌"
        btn_text = _cut(f"{status} {info['name']}", 60)
        # callback_data у Telegram ограничена 64 байтами — длинные/русские имена LoRA её превышали и ломали меню
        markup.add(types.InlineKeyboardButton(btn_text, callback_data=f"lora_toggle_{_lora_id(lora_name)}"))
    # Кнопка обновления списка
    markup.add(types.InlineKeyboardButton("🔄 Обновить список LoRA", callback_data="refresh_lora_list"))
    markup.add(types.InlineKeyboardButton("◀️ Назад в настройки", callback_data="back_to_settings"))

    text = f"🎛️ **Управление LoRA**\n\nВыберите LoRA для включения/отключения.\nОни будут автоматически добавляться в промпт при генерации.\n\nДоступно: {len(available)} LoRA"

    try:
        bot.edit_message_text(
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
                bot.delete_message(call.message.chat.id, call.message.message_id)
            except Exception:
                pass
            bot.send_message(
                call.message.chat.id,
                text,
                parse_mode="Markdown",
                reply_markup=markup
            )
    _safe_answer(call)

@bot.callback_query_handler(func=lambda call: call.data == "manage_lora")
def handle_manage_lora_button(call):
    handle_manage_lora(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("lora_toggle_"))
def handle_lora_toggle(call):
    user_id = str(call.message.chat.id)
    lora_id = call.data[len("lora_toggle_"):]
    available = get_available_loras()
    lora_name = next((name for name in available if _lora_id(name) == lora_id), None)
    if lora_name is None:
        _safe_answer(call, "❌ LoRA не найдена — обновите список", show_alert=True)
        return
    current = get_user_lora_enabled(user_id).get(lora_name, True)
    new_state = not current
    set_user_lora_enabled(user_id, lora_name, new_state)
    status = "включена ✅" if new_state else "выключена ❌"
    _safe_answer(call, f"LoRA {available[lora_name]['name']} {status}"[:200])
    # Перерисовываем меню (редактируем то же сообщение)
    handle_manage_lora(call)

@bot.callback_query_handler(func=lambda call: call.data == "refresh_lora_list")
def handle_refresh_lora(call):
    new_loras = get_available_loras(force_refresh=True)
    if new_loras:
        _safe_answer(call, "✅ Список обновлён!")
        handle_manage_lora(call)
    else:
        _safe_answer(call, "❌ Не удалось обновить список", show_alert=True)

@bot.callback_query_handler(func=lambda call: call.data == "back_to_settings")
def handle_back_to_settings(call):
    show_settings(call.message)
    bot.answer_callback_query(call.id)

# =========== ЗАПУСК БОТА ===========

if __name__ == "__main__":
    print("🤖 Бот запущен!")
    # Получаем список LoRA при старте
    try:
        from settings import get_available_loras
        # Принудительно обновляем кэш LoRA
        loras = get_available_loras(force_refresh=True)
        if loras:
            print(f"✅ Загружено {len(loras)} LoRA с сервера")
        else:
            print("⚠️ Не удалось загрузить LoRA с сервера, используем дефолтные")
    except Exception as e:
        print(f"Ошибка при загрузке LoRA: {e}")
    # При запуске просканируем и выведем количество найденных поз
    sources = get_controlnet_sources()
    print(f"📊 Всего поз в базе: {len(sources)}")

    # Предупреждение о недоступном Stable Diffusion API (чтобы не «падать» молча при генерации)
    try:
        if check_api_available():
            print(f"✅ Stable Diffusion API доступен: {config.SD_BASE_URL}")
        else:
            print(f"⚠️ Stable Diffusion API НЕдоступен по адресу {config.SD_BASE_URL}. Проверьте, запущен ли WebUI с --api")
    except Exception as e:
        print(f"⚠️ Не удалось проверить доступность SD API: {e}")

    # Список негативных эмбеддингов из WebUI (кэшируется; обновляется кнопкой в меню)
    try:
        embeddings = get_available_embeddings(force_refresh=True)
        print(f"🧬 Эмбеддингов в WebUI: {len(embeddings) if embeddings is not None else 'не удалось получить'}")
    except Exception as e:
        print(f"⚠️ Не удалось получить список эмбеддингов: {e}")

    # Фиксируем модель по умолчанию (загруженную в WebUI сейчас) — для пользователей без выбора
    try:
        models = get_available_models(force_refresh=True)
        default_model = get_default_model()
        print(f"🧩 Моделей в WebUI: {len(models)}; по умолчанию: {default_model or 'не определена'}")
    except Exception as e:
        print(f"⚠️ Не удалось получить список моделей: {e}")

    # Регистрируем хелпер-команду /prompts в меню Telegram
    try:
        bot.set_my_commands([
            types.BotCommand("start", "Начать / главное меню"),
            types.BotCommand("help", "Справка"),
            types.BotCommand("samplers", "Список сэмплеров"),
            types.BotCommand("setsampler", "Выбрать сэмплер"),
            types.BotCommand("model", "Выбрать модель"),
            types.BotCommand("prompts", "Лог промптов (админ)"),
        ])
    except Exception as e:
        print(f"⚠️ Не удалось установить список команд: {e}")

    print(f"📝 Лог промптов: {config.PROMPT_LOG_FILE} ({'вкл' if config.PROMPT_LOG_ENABLED else 'выкл'})")
    bot.infinity_polling()