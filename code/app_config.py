"""
Централизованная конфигурация проекта.

Все секреты и настройки окружения читаются из переменных окружения
(можно использовать файл .env — см. .env.example).

ВАЖНО: токен Telegram-бота больше НЕ хранится в коде.
Получите его у @BotFather и задайте в переменной окружения TELEGRAM_BOT_TOKEN.
"""
import os
import sys


def _find_dotenv_files():
    """Ищет файл .env в типичных местах: рядом с кодом, в корне проекта и в CWD."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, ".env"),            # code/.env
        os.path.join(os.path.dirname(here), ".env"),  # корень репозитория /.env
        os.path.abspath(".env"),               # текущая рабочая директория
        os.path.abspath(os.path.join("..", ".env")),  # родитель текущей директории
        os.path.expanduser("~/.env"),          # домашняя директория (частый случай)
    ]
    seen = set()
    found = []
    for path in candidates:
        real = os.path.normpath(path)
        if real not in seen:
            seen.add(real)
            if os.path.isfile(real):
                found.append(real)
    return found


def _manual_parse_env(path):
    """Простой парсер .env на случай, если python-dotenv не установлен."""
    values = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                # Поддержка строки вида "export KEY=VALUE"
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if key:
                    values[key] = val
    except OSError:
        pass
    return values


def _load_dotenv() -> None:
    """Загружает .env (рядом с кодом, в корне проекта и из текущей директории).

    Использует python-dotenv, если он установлен; иначе — собственный парсер.
    Значения НЕ перезаписываются, если переменная уже задана через export/systemd.
    """
    files = _find_dotenv_files()
    if not files:
        return
    try:
        from dotenv import load_dotenv
        for path in files:
            load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # Запасной вариант: читаем .env вручную (без зависимости от python-dotenv)
    for path in reversed(files):  # файлы ближе к коду имеют приоритет
        for key, val in _manual_parse_env(path).items():
            os.environ.setdefault(key, val)


_load_dotenv()


def get_env(name: str, default: str = None, required: bool = False) -> str:
    """Читает переменную окружения; при required=True падает с понятной ошибкой."""
    value = os.environ.get(name, default)
    if required and not value:
        print(
            f"❌ Переменная окружения {name} не задана.\n"
            f"   Задайте её:  export {name}=...   (или скопируйте .env.example в .env)\n"
            f"   Для токена бота обратитесь к https://t.me/BotFather",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


# ===== Telegram =====
BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")

# ===== Stable Diffusion WebUI (AUTOMATIC1111) API =====
SD_BASE_URL = get_env("SD_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

# ===== Ollama (LLM-обработка промптов) =====
OLLAMA_BASE_URL = get_env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# ===== Пути данных =====
DATA_DIR = get_env("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Файлы пользовательских настроек (хранятся в DATA_DIR, наружу из git)
SETTINGS_FILE = os.path.join(DATA_DIR, get_env("SETTINGS_FILE_NAME", "user_settings.json"))
FACE_SETTINGS_FILE = os.path.join(DATA_DIR, get_env("FACE_SETTINGS_FILE_NAME", "face_settings.json"))
CONTROLNET_SETTINGS_FILE = os.path.join(DATA_DIR, get_env("CONTROLNET_SETTINGS_FILE_NAME", "controlnet_user_settings.json"))
ADETAILER_SETTINGS_FILE = os.path.join(DATA_DIR, get_env("ADETAILER_SETTINGS_FILE_NAME", "adetailer_user_settings.json"))
SUBSCRIPTIONS_FILE = os.path.join(DATA_DIR, get_env("SUBSCRIPTIONS_FILE_NAME", "subscriptions.json"))
PROMPT_LOG_FILE = os.path.join(DATA_DIR, get_env("PROMPT_LOG_FILE_NAME", "prompt_log.jsonl"))

# ===== Логирование промптов =====
PROMPT_LOG_ENABLED = get_env("PROMPT_LOG_ENABLED", "1") not in ("0", "false", "no")


# ===== Асинхронный стек (aiogram + aiohttp) =====
def _get_int(name: str, default: int, minimum: int = 0) -> int:
    """Целое число из окружения; при мусоре/слишком малом значении — значение по умолчанию/минимум."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, int(float(str(raw).strip())))
    except ValueError:
        print(f"⚠️ {name}={raw!r} не число — использую {default}", file=sys.stderr)
        return default


# Очередь задач генерации
QUEUE_WORKERS = _get_int("QUEUE_WORKERS", 2, minimum=1)          # сколько задач обрабатывается одновременно
QUEUE_MAXSIZE = _get_int("QUEUE_MAXSIZE", 50, minimum=1)         # максимум ожидающих задач
QUEUE_MAX_PER_USER = _get_int("QUEUE_MAX_PER_USER", 3, minimum=1)  # активных задач на одного пользователя
# Сколько одновременных запросов бот шлёт в WebUI. WebUI сам выполняет генерации строго по одной (очередь внутри
# сервера), поэтому по умолчанию 1: так «Отменить» гарантированно прерывает именно ту задачу, что сейчас считается.
SD_MAX_PARALLEL = _get_int("SD_MAX_PARALLEL", 1, minimum=1)

# Telegram: режим работы
BOT_MODE = get_env("BOT_MODE", "polling").strip().lower()        # polling | webhook
POLLING_TIMEOUT = _get_int("POLLING_TIMEOUT", 10, minimum=1)     # long polling, секунд
WEBHOOK_URL = (get_env("WEBHOOK_URL", "") or "").rstrip("/")     # https://example.com  (без пути)
WEBHOOK_SECRET = get_env("WEBHOOK_SECRET", None) or get_env("SECRET_TOKEN", None)
WEBHOOK_PATH = "/" + (get_env("WEBHOOK_PATH", "/webhook") or "/webhook").strip("/")
WEBAPP_HOST = get_env("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = _get_int("WEBAPP_PORT", 8080, minimum=1)

# HTTP-клиент (aiohttp). total не ограничиваем: генерация может идти минуты; ограничиваем только паузу без данных.
HTTP_SOCK_READ_TIMEOUT = _get_int("HTTP_SOCK_READ_TIMEOUT", 300, minimum=10)
HTTP_SOCK_CONNECT_TIMEOUT = _get_int("HTTP_SOCK_CONNECT_TIMEOUT", 10, minimum=1)

# ===== Проверка систем и админ-консоль =====
STARTUP_CHECK = get_env("STARTUP_CHECK", "1") not in ("0", "false", "no")            # отчёт healthcheck при запуске
STARTUP_CHECK_STRICT = get_env("STARTUP_CHECK_STRICT", "0") in ("1", "true", "yes")  # остановить запуск при критичных ошибках
ADMIN_CONSOLE = get_env("ADMIN_CONSOLE", "1") not in ("0", "false", "no")            # терминальная админ-панель (только в TTY)
WATCHDOG_INTERVAL = _get_int("WATCHDOG_INTERVAL", 60)   # секунд между проверками WebUI (0 — выключено); админам приходит оповещение о падении

