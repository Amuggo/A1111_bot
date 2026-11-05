"""
Логирование пользовательских промптов.

Каждая генерация записывается в JSONL-файл (config.PROMPT_LOG_FILE, по умолчанию data/prompt_log.jsonl):
одна строка = один JSON-объект с событиями 'request' (до генерации) и 'result'/'error' (после).

Формат записи:
{
  "ts": "2026-10-29T12:00:00+00:00",
  "event": "request" | "result" | "error",
  "gen_id": "<uuid>",          # связывает request <-> result/error одной генерации
  "user_id": "123456",
  "type": "txt2img" | "controlnet" | "img2img" | "upscale" | "inpaint" | ...,
  "original_prompt": "...",     # промпт как его написал пользователь
  "processed_prompt": "...",    # промпт после LLM/LoRA/embeddings (если есть)
  "negative_prompt": "...",
  "settings": {...},            # width/height/steps/cfg/sampler...
  "seed": 123,
  "face_replaced": true,
  "duration_sec": 12.3,
  "error": "..."                # только для event=error
}

Дополнительно каждый запрос выводится в консоль бота строкой «пользователь : промпт».

Модуль потокобезопасен и никогда не роняет основной процесс: ошибки логируются в stderr.
"""
import asyncio
import json
import os
import threading
import uuid
from datetime import datetime, timezone

try:
    import app_config as config

    LOG_FILE = config.PROMPT_LOG_FILE
    ENABLED = config.PROMPT_LOG_ENABLED
except Exception:  # если config недоступен — используем значение по умолчанию
    LOG_FILE = os.path.join("data", "prompt_log.jsonl")
    ENABLED = os.environ.get("PROMPT_LOG_ENABLED", "1") not in ("0", "false", "no")

_lock = threading.Lock()

# user_id -> отображаемое имя (@username, либо имя/фамилия). Заполняется из main.py.
_user_names = {}


def remember_user(user) -> None:
    """Запоминает имя пользователя Telegram (объект aiogram.types.User) для логов."""
    try:
        if user is None:
            return
        username = getattr(user, "username", None)
        if username:
            name = f"@{username}"
        else:
            full = " ".join(p for p in (getattr(user, "first_name", None), getattr(user, "last_name", None)) if p)
            name = full or str(user.id)
        _user_names[str(user.id)] = name
    except Exception:
        pass


def get_user_name(user_id) -> str:
    """Имя пользователя для вывода; если оно неизвестно — сам user_id."""
    return _user_names.get(str(user_id), str(user_id))


def _console(user_id, gen_type: str, prompt) -> None:
    """Печатает в консоль бота строку «пользователь : промпт»."""
    text = " ".join(str(prompt).split()) if prompt else f"<{gen_type}>"
    print(f"{get_user_name(user_id)} : {text}", flush=True)


def new_gen_id() -> str:
    """Генерирует идентификатор одной генерации для связки request -> result."""
    return uuid.uuid4().hex[:12]


def _write(record: dict) -> None:
    record = dict(record)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
        try:
            directory = os.path.dirname(LOG_FILE)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as file:
                file.write(line + "\n")
        except Exception as error:  # логирование не должно ломать бота
            print(f"⚠️ Не удалось записать в лог промптов: {error}")


async def log_request(user_id, gen_type: str, original_prompt: str, processed_prompt: str = None,
                settings: dict = None, negative_prompt: str = None, gen_id: str = None, **extra) -> str:
    """Пишет событие запроса на генерацию. Возвращает gen_id для последующей связи с результатом."""
    if not ENABLED:
        return gen_id or new_gen_id()
    _console(user_id, gen_type, original_prompt)
    record = {
        "event": "request",
        "gen_id": gen_id or new_gen_id(),
        "user_id": str(user_id),
        "username": get_user_name(user_id),
        "type": gen_type,
        "original_prompt": original_prompt,
    }
    if processed_prompt is not None:
        record["processed_prompt"] = processed_prompt
    if negative_prompt is not None:
        record["negative_prompt"] = negative_prompt
    if settings:
        record["settings"] = {
            k: settings.get(k)
            for k in ("width", "height", "steps", "cfg_scale", "sampler_name", "batch_size", "n_iter")
            if settings.get(k) is not None
        }
    record.update(extra)
    await asyncio.to_thread(_write, record)
    return record["gen_id"]


async def log_result(gen_id: str, user_id, gen_type: str, original_prompt: str, processed_prompt: str = None,
               seed=None, settings: dict = None, duration_sec: float = None, face_replaced: bool = None, **extra) -> None:
    """Пишет успешное завершение генерации."""
    if not ENABLED:
        return
    record = {
        "event": "result",
        "gen_id": gen_id,
        "user_id": str(user_id),
        "type": gen_type,
        "original_prompt": original_prompt,
    }
    if processed_prompt is not None:
        record["processed_prompt"] = processed_prompt
    if seed is not None:
        record["seed"] = seed
    if settings:
        record["settings"] = settings
    if duration_sec is not None:
        record["duration_sec"] = round(duration_sec, 2)
    if face_replaced is not None:
        record["face_replaced"] = face_replaced
    record.update(extra)
    await asyncio.to_thread(_write, record)


async def log_error(gen_id: str, user_id, gen_type: str, original_prompt: str, error, duration_sec: float = None) -> None:
    """Пишет ошибку генерации."""
    if not ENABLED:
        return
    await asyncio.to_thread(_write, {
        "event": "error",
        "gen_id": gen_id,
        "user_id": str(user_id),
        "type": gen_type,
        "original_prompt": original_prompt,
        "error": str(error),
        "duration_sec": round(duration_sec, 2) if duration_sec is not None else None,
    })


def _tail_sync(limit: int = 10) -> list:
    """Возвращает последние N записей лога (для команды /logstats или отладки)."""
    if not os.path.isfile(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
    except Exception:
        return []


async def tail(limit: int = 10) -> list:
    """Последние записи лога (чтение файла — в потоке, цикл событий не блокируется)."""
    return await asyncio.to_thread(_tail_sync, limit)
