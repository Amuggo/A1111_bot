import copy
import re
import os
import time
from typing import Any, Callable, Dict, List, Optional

from aio import LazyLock
from http_client import SDError, sd_get_json, sd_post_json
from storage import JsonStore

try:
    import app_config as config

    SETTINGS_FILE = config.SETTINGS_FILE
    SD_BASE_URL = config.SD_BASE_URL
except ImportError:
    SETTINGS_FILE = os.environ.get("SETTINGS_FILE", "user_settings.json")
    SD_BASE_URL = os.environ.get("SD_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

DEFAULT_NEGATIVE_PROMPT = "blurry, watermarks, bad quality, deformed, incorrect anatomy, mutations"

DEFAULT_SETTINGS = {
    "width": 512,
    "height": 768,
    "steps": 40,
    "cfg_scale": 6.0,
    "sampler_name": "DPM++ 2M",
    "batch_size": 1,
    "n_iter": 1,
    "denoising_strength": 0.75,
    "diversity_mode": "auto",
    "prompt_processing": False,
    "lora_enabled": {},  # заполняется динамически
    "lora_weights": {},  # вес каждой LoRA в промпте (по умолчанию 1.0)
    "sd_model": None,  # выбранная модель (None = модель по умолчанию)
    # Негативные эмбеддинги: None = не настраивались (используются рекомендованные, если они есть в WebUI),
    # список = явный выбор пользователя (пустой список = не использовать эмбеддинги).
    "negative_embeddings": None,
    # Hires. fix (двухпроходная детализация): по умолчанию выключен
    "hires_enabled": False,
    "hires_scale": 1.5,
    "hires_upscaler": "Latent",  # "Latent" не требует доп. модели и меньше нагружает VRAM; "ESRGAN_4x" — резче, но дольше
    "hires_denoise": 0.5,
    "presets": {},  # {имя: {настройки}} — см. PRESET_KEYS
}

# Если API недоступен, повторяем запрос не чаще, чем раз в _RETRY_SECONDS (чтобы бот не «подвисал»)
_RETRY_SECONDS = 30

# Все чтения/изменения файла настроек идут через это хранилище: оно потокобезопасно и пишет файл атомарно.
_SETTINGS_STORE = JsonStore(SETTINGS_FILE)


# ===== LORA (кэш списка из API) =====
_loras_lock = LazyLock()
_available_loras: Optional[Dict[str, Dict[str, str]]] = None
_loras_last_attempt = 0.0


async def fetch_loras_from_api() -> Dict[str, Dict[str, str]]:
    """Запрашивает список LoRA из API и возвращает словарь {имя_ключа: {name: человеческое_имя, tag: тег}}"""
    try:
        loras = await sd_get_json("sdapi/v1/loras", timeout=10)
        result = {}
        for lora in loras:
            name = lora.get("name", "")
            base_name = name.rsplit('.', 1)[0] if '.' in name else name
            display_name = lora.get("alias") or base_name
            tag = f"<lora:{base_name}:1>"

            result[base_name] = {"name": display_name, "tag": tag}
        print(f"✅ Загружено {len(result)} LoRA из API")
        return result
    except Exception as e:
        print(f"⚠️ Ошибка при запросе LoRA: {e}")
        return {}


async def get_available_loras(force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
    """Копия списка доступных LoRA. Пустой результат (API недоступен) перезапрашивается не чаще раза в 30 с.
    Параллельные задачи не дёргают API одновременно: запрос выполняется под asyncio.Lock."""
    global _available_loras, _loras_last_attempt
    async with _loras_lock:
        now = time.monotonic()
        if force_refresh or (not _available_loras and now - _loras_last_attempt > _RETRY_SECONDS):
            _loras_last_attempt = now
            fetched = await fetch_loras_from_api()
            if fetched or _available_loras is None:
                _available_loras = fetched
        return dict(_available_loras or {})


def set_available_loras(loras: Dict[str, Dict[str, str]]):
    """Устанавливает список LoRA вручную (для обновления)"""
    global _available_loras
    _available_loras = dict(loras)


# ===== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЕЙ =====
def _fill_defaults(user_settings: dict, available_loras: Dict[str, Any]) -> dict:
    """Дополняет настройки пользователя значениями по умолчанию и синхронизирует список LoRA."""
    if available_loras:
        # Если API с LoRA недоступен (список пуст), выбор пользователя НЕ трогаем,
        # иначе после сбоя API все его LoRA «забылись» бы и включились заново.
        if "lora_enabled" not in user_settings or not isinstance(user_settings["lora_enabled"], dict):
            user_settings["lora_enabled"] = {lora: True for lora in available_loras}
        else:
            user_lora = {k: v for k, v in user_settings["lora_enabled"].items() if k in available_loras}
            for lora in available_loras:
                user_lora.setdefault(lora, True)
            user_settings["lora_enabled"] = user_lora
    for key, value in DEFAULT_SETTINGS.items():
        if key not in user_settings:
            user_settings[key] = copy.deepcopy(value)
    return user_settings


async def user_exists(user_id: str) -> bool:
    """Есть ли уже настройки пользователя (то есть он пользовался ботом раньше)."""
    return str(user_id) in await _SETTINGS_STORE.aread()


async def get_user_settings(user_id: str) -> dict:
    """Копия настроек пользователя (файл перезаписывается только если данные изменились; I/O — в потоке)."""
    user_id = str(user_id)
    available = await get_available_loras()  # сеть — вне файловой блокировки

    def mutator(all_settings: dict) -> dict:
        user_settings = _fill_defaults(all_settings.get(user_id, {}), available)
        all_settings[user_id] = user_settings
        return copy.deepcopy(user_settings)

    return await _SETTINGS_STORE.aupdate(mutator)


async def update_user_settings(user_id: str, updates: Optional[dict] = None,
                               mutator: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Атомарно изменяет настройки пользователя (без потери параллельных изменений).
      updates — словарь {ключ: значение} для записи;
      mutator — синхронная функция, меняющая dict настроек на месте (для удаления ключей, вложенных структур).
    Возвращает новую копию настроек.
    """
    user_id = str(user_id)
    available = await get_available_loras()

    def _mutator(all_settings: dict) -> dict:
        user_settings = _fill_defaults(all_settings.get(user_id, {}), available)
        if updates:
            user_settings.update(copy.deepcopy(updates))
        if mutator:
            mutator(user_settings)
        all_settings[user_id] = user_settings
        return copy.deepcopy(user_settings)

    return await _SETTINGS_STORE.aupdate(_mutator)


async def save_user_settings(user_id: str, settings: dict):
    """Заменяет настройки пользователя целиком (для точечных правок используйте update_user_settings)."""
    user_id = str(user_id)

    def _mutator(all_settings: dict):
        all_settings[user_id] = copy.deepcopy(settings)

    await _SETTINGS_STORE.aupdate(_mutator)


async def get_user_negative_prompt(user_id: str) -> str:
    settings = await get_user_settings(user_id)
    return settings.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)


async def set_user_negative_prompt(user_id: str, neg_prompt: Optional[str]):
    def mutator(settings: dict):
        if neg_prompt is None:
            settings.pop("negative_prompt", None)
        else:
            settings["negative_prompt"] = neg_prompt

    await update_user_settings(user_id, mutator=mutator)


async def get_diversity_mode(user_id: str) -> str:
    settings = await get_user_settings(user_id)
    return settings.get("diversity_mode", "auto")


async def set_diversity_mode(user_id: str, mode: str):
    await update_user_settings(user_id, {"diversity_mode": mode})


async def get_prompt_processing_enabled(user_id: str) -> bool:
    settings = await get_user_settings(user_id)
    return settings.get("prompt_processing", True)


async def set_prompt_processing_enabled(user_id: str, enabled: bool):
    await update_user_settings(user_id, {"prompt_processing": enabled})


async def get_use_advanced_negative(user_id: str) -> bool:
    settings = await get_user_settings(user_id)
    return settings.get("use_advanced_negative", False)


async def get_user_lora_enabled(user_id: str) -> Dict[str, bool]:
    settings = await get_user_settings(user_id)
    available = await get_available_loras()
    user_lora = settings.get("lora_enabled", {})
    filtered = {}
    for lora in available:
        filtered[lora] = user_lora.get(lora, True)
    return filtered


async def set_user_lora_enabled(user_id: str, lora_name: str, enabled: bool):
    def mutator(settings: dict):
        settings.setdefault("lora_enabled", {})[lora_name] = enabled

    await update_user_settings(user_id, mutator=mutator)


LORA_WEIGHT_STEPS = (0.4, 0.6, 0.8, 1.0, 1.2)


async def get_user_lora_weights(user_id: str) -> Dict[str, float]:
    """Вес каждой доступной LoRA у пользователя (по умолчанию 1.0)."""
    raw = (await get_user_settings(user_id)).get("lora_weights") or {}
    available = await get_available_loras()
    weights = {}
    for lora in available:
        try:
            weights[lora] = float(raw.get(lora, 1.0))
        except (TypeError, ValueError):
            weights[lora] = 1.0
    return weights


async def cycle_user_lora_weight(user_id: str, lora_name: str) -> float:
    """Переключает вес LoRA на следующее значение из LORA_WEIGHT_STEPS и возвращает его."""
    result = {}

    def mutator(settings: dict):
        weights = settings.get("lora_weights")
        if not isinstance(weights, dict):
            weights = settings["lora_weights"] = {}
        try:
            current = float(weights.get(lora_name, 1.0))
        except (TypeError, ValueError):
            current = 1.0
        nearest = min(range(len(LORA_WEIGHT_STEPS)), key=lambda i: abs(LORA_WEIGHT_STEPS[i] - current))
        result["w"] = LORA_WEIGHT_STEPS[(nearest + 1) % len(LORA_WEIGHT_STEPS)]
        weights[lora_name] = result["w"]

    await update_user_settings(user_id, mutator=mutator)
    return result["w"]


async def get_active_lora_tags(user_id: str) -> str:
    """Возвращает строку с тегами LoRA для вставки в промпт (включённые, с весом пользователя)"""
    lora_enabled = await get_user_lora_enabled(user_id)
    weights = await get_user_lora_weights(user_id)
    available = await get_available_loras()
    tags = []
    for lora_name, enabled in lora_enabled.items():
        if enabled and lora_name in available:
            tag = available[lora_name]["tag"]
            # <lora:имя:1> -> <lora:имя:вес>
            tags.append(re.sub(r":[^:>]*>$", f":{weights.get(lora_name, 1.0):g}>", tag))
    return " ".join(tags)


# ===== СЭМПЛЕРЫ =====
DEFAULT_SAMPLERS = [
    "Euler a",
    "Euler",
    "LMS",
    "Heun",
    "DPM2",
    "DPM2 a",
    "DPM++ 2S a",
    "DPM++ 2M",
    "DPM++ SDE",
    "DPM++ 2M SDE",
    "DPM++ 2M Karras",
    "DPM++ 2S a Karras",
    "DPM++ SDE Karras",
    "DDIM",
    "PLMS",
    "UniPC"
]

# Кэшируется только УСПЕШНЫЙ ответ API. Запасной список не кэшируется, чтобы после восстановления
# связи с WebUI бот получил настоящие названия (раньше запасной список «залипал» до перезапуска).
_samplers_lock = LazyLock()
_available_samplers: Optional[List[str]] = None
_samplers_last_attempt = 0.0


async def _fetch_samplers_raw() -> Optional[List[str]]:
    """Запрашивает список сэмплеров из API. None — при ошибке."""
    try:
        data = await sd_get_json("sdapi/v1/samplers", timeout=10)
        samplers = [item["name"] for item in data if isinstance(item, dict) and "name" in item]
        if samplers:
            print(f"✅ Загружено {len(samplers)} сэмплеров из API")
            return samplers
    except Exception as e:
        print(f"⚠️ Ошибка при запросе сэмплеров: {e}")
    return None


async def fetch_samplers_from_api() -> list:
    """Запрашивает список сэмплеров из API (при ошибке — запасной список)."""
    return await _fetch_samplers_raw() or list(DEFAULT_SAMPLERS)


async def get_available_samplers(force_refresh: bool = False) -> list:
    """Возвращает список доступных сэмплеров"""
    global _available_samplers, _samplers_last_attempt
    async with _samplers_lock:
        now = time.monotonic()
        if force_refresh or (_available_samplers is None and now - _samplers_last_attempt > _RETRY_SECONDS):
            _samplers_last_attempt = now
            fetched = await _fetch_samplers_raw()
            if fetched:
                _available_samplers = fetched
        return list(_available_samplers) if _available_samplers else list(DEFAULT_SAMPLERS)


async def get_user_sampler(user_id: str) -> str:
    """Возвращает текущий сэмплер пользователя"""
    settings = await get_user_settings(user_id)
    return settings.get("sampler_name", DEFAULT_SETTINGS["sampler_name"])


async def set_user_sampler(user_id: str, sampler_name: str):
    """Устанавливает сэмплер для пользователя"""
    await update_user_settings(user_id, {"sampler_name": sampler_name})


# ===== НЕГАТИВНЫЕ ЭМБЕДДИНГИ (TEXTUAL INVERSION) =====
# Список берётся из API WebUI (GET /sdapi/v1/embeddings -> {"loaded": {имя: {...}}, "skipped": {...}}).
# Имя эмбеддинга добавляется в негативный промпт при генерации (так WebUI применяет textual inversion).
_EMBEDDINGS_TTL = 120  # секунд; после смены модели набор совместимых эмбеддингов может отличаться
_embeddings_lock = LazyLock()
_embeddings_cache: Optional[List[str]] = None
_embeddings_fetched_at = 0.0
_embeddings_last_attempt = 0.0


async def fetch_embeddings_from_api() -> Optional[List[str]]:
    """Имена загруженных эмбеддингов (отсортированы). None — если API недоступен."""
    try:
        data = await sd_get_json("sdapi/v1/embeddings", timeout=15)
        loaded = (data or {}).get("loaded") or {}
        names = sorted((str(name) for name in loaded), key=str.lower)
        print(f"✅ Загружено {len(names)} эмбеддингов из API")
        return names
    except Exception as e:
        print(f"⚠️ Ошибка при запросе эмбеддингов: {e}")
        return None


async def refresh_embeddings_on_server() -> bool:
    """Просит WebUI пересканировать папку embeddings (в части версий эндпоинта нет — тогда это не ошибка)."""
    try:
        await sd_post_json("sdapi/v1/refresh-embeddings", {}, timeout=60)
        return True
    except Exception:
        return False


async def get_available_embeddings(force_refresh: bool = False) -> Optional[List[str]]:
    """Список доступных эмбеддингов (кэш на 2 минуты). None — если список ни разу не удалось получить."""
    global _embeddings_cache, _embeddings_fetched_at, _embeddings_last_attempt
    async with _embeddings_lock:
        now = time.monotonic()
        stale = _embeddings_cache is None or now - _embeddings_fetched_at > _EMBEDDINGS_TTL
        if force_refresh or (stale and now - _embeddings_last_attempt > _RETRY_SECONDS):
            _embeddings_last_attempt = now
            fetched = await fetch_embeddings_from_api()
            if fetched is not None:
                _embeddings_cache = fetched
                _embeddings_fetched_at = now
        return list(_embeddings_cache) if _embeddings_cache is not None else None


def _recommended_embeddings(available: Optional[List[str]]) -> Optional[List[str]]:
    """
    Рекомендованные эмбеддинги (bad-hands-5, badhandv4, easynegative), реально загруженные в WebUI.
    None — если список недоступен (тогда используется прежняя проверка по файлам, см. embeddings_manager).
    """
    if available is None:
        return None
    try:
        from embeddings_manager import RECOMMENDED_EMBEDDINGS
    except Exception:
        return []
    loaded = {name.lower(): name for name in available}
    return [loaded[name.lower()] for name in RECOMMENDED_EMBEDDINGS if name.lower() in loaded]


async def effective_negative_embeddings(settings: dict) -> Optional[List[str]]:
    """
    Эмбеддинги, которые нужно добавить в негативный промпт.
      * пользователь выбирал сам — берём его выбор (без тех, что WebUI сейчас не загрузил);
      * не выбирал — рекомендованные, найденные в WebUI (как и раньше добавлялись автоматически);
      * None — список из API недоступен и выбор не делался: решает embeddings_manager (по файлам).
    """
    stored = (settings or {}).get("negative_embeddings")
    available = await get_available_embeddings()
    if stored is None:
        return _recommended_embeddings(available)
    if available is None:
        return list(stored)
    loaded = set(available)
    return [name for name in stored if name in loaded]


async def get_user_negative_embeddings(user_id: str) -> List[str]:
    """Эмбеддинги, которые сейчас будут применены для пользователя (для отображения в меню)."""
    return await effective_negative_embeddings(await get_user_settings(user_id)) or []


async def toggle_user_negative_embedding(user_id: str, name: str) -> bool:
    """Включает/выключает эмбеддинг. Возвращает новое состояние (True = выбран)."""
    recommended = _recommended_embeddings(await get_available_embeddings()) or []  # сеть — вне блокировки файла
    result = {}

    def mutator(settings: dict):
        current = settings.get("negative_embeddings")
        selected = list(recommended if current is None else current)
        if name in selected:
            selected.remove(name)
            result["state"] = False
        else:
            selected.append(name)
            result["state"] = True
        settings["negative_embeddings"] = selected

    await update_user_settings(user_id, mutator=mutator)
    return result["state"]


async def set_user_negative_embeddings(user_id: str, names: Optional[List[str]]):
    """Явно задаёт список эмбеддингов ([] = не использовать, None = вернуть выбор по умолчанию)."""
    await update_user_settings(user_id, {"negative_embeddings": None if names is None else list(names)})


# ===== МОДЕЛИ (CHECKPOINTS) =====
# Список моделей берётся из API WebUI (/sdapi/v1/sd-models). Выбор хранится у каждого
# пользователя отдельно в ключе "sd_model" (имя модели без расширения и хэша) и
# передаётся в каждый запрос генерации через override_settings (см. Aapi.apply_model_override).

# Кэш списка моделей: [{"title": "...safetensors [abc123]", "name": "имя_модели"}, ...]
_available_models = []
# Модель «по умолчанию» (для пользователей, которые ничего не выбирали) — фиксируется один раз.
_default_model = None
_models_lock = LazyLock()
_default_lock = LazyLock()
_models_last_attempt = 0.0
_default_last_attempt = 0.0


async def fetch_models_from_api() -> list:
    """Запрашивает список моделей из API. Возвращает [{title, name}], отсортированный по имени."""
    try:
        data = await sd_get_json("sdapi/v1/sd-models", timeout=15)
        result = []
        for item in data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            title = item["title"]
            result.append({"title": title, "name": item.get("model_name") or title})
        result.sort(key=lambda m: m["name"].lower())
        print(f"✅ Загружено {len(result)} моделей из API")
        return result
    except Exception as e:
        print(f"⚠️ Ошибка при запросе списка моделей: {e}")
        return []


async def get_available_models(force_refresh: bool = False) -> list:
    """Список доступных моделей (пустой список перезапрашивается при обращении, но не чаще раза в 30 с)."""
    global _available_models, _models_last_attempt
    async with _models_lock:
        if force_refresh or (not _available_models and time.monotonic() - _models_last_attempt > _RETRY_SECONDS):
            _models_last_attempt = time.monotonic()
            fetched = await fetch_models_from_api()
            if fetched or not _available_models:
                _available_models = fetched
        return list(_available_models)


async def refresh_checkpoints_on_server() -> bool:
    """Просит WebUI пересканировать папку с моделями (новые файлы появятся в списке)."""
    try:
        await sd_post_json("sdapi/v1/refresh-checkpoints", {}, timeout=60)
        return True
    except Exception as e:
        print(f"⚠️ Не удалось обновить список моделей на сервере: {e}")
        return False


async def get_default_model() -> Optional[str]:
    """Модель по умолчанию: SD_DEFAULT_MODEL из .env, иначе модель, загруженная в WebUI при запуске бота."""
    global _default_model, _default_last_attempt
    async with _default_lock:
        if _default_model:
            return _default_model
        if _default_last_attempt and time.monotonic() - _default_last_attempt <= _RETRY_SECONDS:
            return None
        _default_last_attempt = time.monotonic()
        wanted = os.environ.get("SD_DEFAULT_MODEL", "").strip()
        if wanted:
            for model in await get_available_models():
                if wanted in (model["title"], model["name"]):
                    _default_model = model["title"]
                    return _default_model
            print(f"⚠️ SD_DEFAULT_MODEL='{wanted}' не найдена среди моделей WebUI")
        try:
            options = await sd_get_json("sdapi/v1/options", timeout=10)
            _default_model = options.get("sd_model_checkpoint") or None
        except Exception as e:
            print(f"⚠️ Не удалось узнать текущую модель WebUI: {e}")
        return _default_model


async def get_default_model_name() -> Optional[str]:
    """Человекочитаемое имя модели по умолчанию."""
    title = await get_default_model()
    if not title:
        return None
    for model in await get_available_models():
        if model["title"] == title:
            return model["name"]
    return title


async def get_user_model(user_id: str) -> Optional[str]:
    """Имя выбранной пользователем модели (None — используется модель по умолчанию)."""
    name = (await get_user_settings(user_id)).get("sd_model")
    if not name:
        return None
    available = await get_available_models()
    if available and not any(m["name"] == name for m in available):
        return None  # модель удалили/переименовали — откатываемся на модель по умолчанию
    return name


async def set_user_model(user_id: str, model_name: Optional[str]):
    """Сохраняет выбор модели пользователя (None — сбросить на модель по умолчанию)."""
    await update_user_settings(user_id, {"sd_model": model_name})


async def resolve_model(settings: dict) -> Optional[str]:
    """Название модели (title) для передачи в API: выбор пользователя либо модель по умолчанию."""
    name = (settings or {}).get("sd_model")
    if name:
        available = await get_available_models()
        for model in available:
            if model["name"] == name:
                return model["title"]
        if not available:
            return name  # список недоступен — пробуем по имени, WebUI понимает и его
    return await get_default_model()


# ===== HIRES. FIX =====
# Итоговая сторона ограничена: на видеокарте с 8 ГБ VRAM (RTX 3050) большие hires-изображения дают нехватку памяти.
HIRES_MAX_SIDE = 1536


def hires_plan(settings: dict) -> Optional[dict]:
    """План hires: {scale, width, height, steps, upscaler, denoise} либо None (выключен / размер уже достаточно велик)."""
    if not settings.get("hires_enabled"):
        return None
    width, height = int(settings.get("width", 512)), int(settings.get("height", 768))
    scale = min(float(settings.get("hires_scale", 1.5)), HIRES_MAX_SIDE / max(width, height))
    if scale < 1.1:
        return None
    steps = int(settings.get("steps", 40))
    return {
        "scale": round(scale, 2),
        "width": int(width * scale), "height": int(height * scale),
        "steps": max(8, min(20, steps // 2)),  # второй проход вдвое короче основного — экономит время на слабой карте
        "upscaler": settings.get("hires_upscaler", "Latent"),
        "denoise": float(settings.get("hires_denoise", 0.5)),
    }


# ===== ПРЕСЕТЫ =====
# Пресет — снимок настроек пользователя под именем. Хранится в его же записи (ключ "presets"), формат файла не менялся.
PRESET_KEYS = ("sd_model", "lora_enabled", "lora_weights", "negative_prompt", "negative_embeddings", "width", "height", "steps",
               "cfg_scale", "sampler_name", "diversity_mode", "hires_enabled", "hires_scale", "hires_upscaler",
               "hires_denoise")
MAX_PRESETS = 10


async def list_presets(user_id: str) -> Dict[str, dict]:
    return dict((await get_user_settings(user_id)).get("presets") or {})


async def save_preset(user_id: str, name: str) -> str:
    """Сохраняет текущие настройки как пресет (одноимённый перезаписывается). Возвращает 'new' | 'updated' | 'limit'."""
    result = {}

    def mutator(settings: dict):
        presets = settings.setdefault("presets", {})
        if name not in presets and len(presets) >= MAX_PRESETS:
            result["r"] = "limit"
            return
        result["r"] = "updated" if name in presets else "new"
        presets[name] = {key: copy.deepcopy(settings.get(key)) for key in PRESET_KEYS}

    await update_user_settings(user_id, mutator=mutator)
    return result["r"]


async def apply_preset(user_id: str, name: str) -> bool:
    """Применяет пресет к текущим настройкам. False — пресета нет."""
    result = {}

    def mutator(settings: dict):
        preset = (settings.get("presets") or {}).get(name)
        if preset is None:
            result["ok"] = False
            return
        for key in PRESET_KEYS:
            if key not in preset:
                continue
            if key == "negative_prompt" and preset[key] is None:
                settings.pop("negative_prompt", None)   # None = стандартный негатив
            else:
                settings[key] = copy.deepcopy(preset[key])
        result["ok"] = True

    await update_user_settings(user_id, mutator=mutator)
    return result["ok"]


async def delete_preset(user_id: str, name: str) -> bool:
    result = {"ok": False}

    def mutator(settings: dict):
        result["ok"] = (settings.get("presets") or {}).pop(name, None) is not None

    await update_user_settings(user_id, mutator=mutator)
    return result["ok"]
