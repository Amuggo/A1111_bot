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
PROMPT_LOG_FILE = os.path.join(DATA_DIR, get_env("PROMPT_LOG_FILE_NAME", "prompt_log.jsonl"))

# ===== Логирование промптов =====
PROMPT_LOG_ENABLED = get_env("PROMPT_LOG_ENABLED", "1") not in ("0", "false", "no")
