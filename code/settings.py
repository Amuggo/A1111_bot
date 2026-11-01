import json
import os
import time
import requests
from typing import Dict, Any, Optional

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
    "sd_model": None,  # выбранная модель (None = модель по умолчанию)
}

# Глобальный кэш доступных LoRA
_available_loras = None

def fetch_loras_from_api() -> Dict[str, Dict[str, str]]:
    """Запрашивает список LoRA из API и возвращает словарь {имя_ключа: {name: человеческое_имя, tag: тег}}"""
    try:
        response = requests.get(f"{SD_BASE_URL}/sdapi/v1/loras", timeout=10)
        if response.status_code == 200:
            loras = response.json()
            result = {}
            for lora in loras:
                name = lora.get("name", "")
                base_name = name.rsplit('.', 1)[0] if '.' in name else name
                display_name = lora.get("alias") or base_name
                
                # Исключение для armpit_hair: ставим безопасный вес 0.5 вместо 1
                if "armpit_hair" in base_name.lower():
                    tag = f"<lora:{base_name}:0.5>"
                else:
                    tag = f"<lora:{base_name}:1>"
                    
                result[base_name] = {"name": display_name, "tag": tag}
            print(f"✅ Загружено {len(result)} LoRA из API (обработано исключение для armpit_hair)")
            return result
        else:
            print(f"⚠️ Ошибка получения LoRA: {response.status_code}")
            return {}
    except Exception as e:
        print(f"⚠️ Ошибка при запросе LoRA: {e}")
        return {}


def get_available_loras(force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
    """Возвращает список доступных LoRA, при необходимости обновляя из API"""
    global _available_loras
    if _available_loras is None or force_refresh:
        _available_loras = fetch_loras_from_api()
    return _available_loras

def set_available_loras(loras: Dict[str, Dict[str, str]]):
    """Устанавливает список LoRA вручную (для обновления)"""
    global _available_loras
    _available_loras = loras

def _load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_settings(settings: dict):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

def get_user_settings(user_id: str) -> dict:
    all_settings = _load_settings()
    user_settings = all_settings.get(user_id, {})
    default = DEFAULT_SETTINGS.copy()
    available = get_available_loras()
    if "lora_enabled" not in user_settings:
        user_settings["lora_enabled"] = {lora: True for lora in available}
    else:
        # Удаляем устаревшие LoRA, добавляем новые
        user_lora = user_settings["lora_enabled"]
        user_lora = {k: v for k, v in user_lora.items() if k in available}
        for lora in available:
            if lora not in user_lora:
                user_lora[lora] = True
        user_settings["lora_enabled"] = user_lora
    for key, value in default.items():
        if key not in user_settings:
            user_settings[key] = value
    all_settings[user_id] = user_settings
    _save_settings(all_settings)
    return user_settings

def save_user_settings(user_id: str, settings: dict):
    all_settings = _load_settings()
    all_settings[user_id] = settings
    _save_settings(all_settings)

def get_user_negative_prompt(user_id: str) -> str:
    settings = get_user_settings(user_id)
    return settings.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

def set_user_negative_prompt(user_id: str, neg_prompt: str):
    settings = get_user_settings(user_id)
    if neg_prompt is None:
        settings.pop("negative_prompt", None)
    else:
        settings["negative_prompt"] = neg_prompt
    save_user_settings(user_id, settings)


# ===== НЕГАТИВНЫЕ ЭМБЕДДИНГИ ПОЛЬЗОВАТЕЛЯ =====
# Список имён негативных эмбеддингов (полученных из API WebUI), которые пользователь
# добавил в свой негативный промпт. Хранится в настройках пользователя в ключе
# "negative_embeddings" и подмешивается к финальному негативу при генерации.

def get_user_negative_embeddings(user_id: str) -> list:
    """Возвращает список имён негативных эмбеддингов, выбранных пользователем."""
    settings = get_user_settings(user_id)
    embeddings = settings.get("negative_embeddings") or []
    # Фильтруем по актуальному списку из API (если он доступен), чтобы не тянуть удалённые
    try:
        from embeddings_manager import get_available_embeddings
        available = get_available_embeddings()
        if available:
            embeddings = [e for e in embeddings if e in available]
    except Exception:
        pass
    return list(embeddings)


def set_user_negative_embeddings(user_id: str, embeddings: list):
    """Сохраняет список негативных эмбеддингов пользователя."""
    settings = get_user_settings(user_id)
    settings["negative_embeddings"] = list(embeddings or [])
    save_user_settings(user_id, settings)


def add_user_negative_embedding(user_id: str, embedding_name: str) -> bool:
    """Добавляет негативный эмбеддинг в промпт пользователя (без дублей)."""
    embeddings = get_user_negative_embeddings(user_id)
    if embedding_name in embeddings:
        return False
    embeddings.append(embedding_name)
    set_user_negative_embeddings(user_id, embeddings)
    return True


def remove_user_negative_embedding(user_id: str, embedding_name: str) -> bool:
    """Убирает негативный эмбеддинг из промпта пользователя."""
    embeddings = get_user_negative_embeddings(user_id)
    if embedding_name not in embeddings:
        return False
    embeddings = [e for e in embeddings if e != embedding_name]
    set_user_negative_embeddings(user_id, embeddings)
    return True

def get_diversity_mode(user_id: str) -> str:
    settings = get_user_settings(user_id)
    return settings.get("diversity_mode", "auto")

def set_diversity_mode(user_id: str, mode: str):
    settings = get_user_settings(user_id)
    settings["diversity_mode"] = mode
    save_user_settings(user_id, settings)

def get_prompt_processing_enabled(user_id: str) -> bool:
    settings = get_user_settings(user_id)
    return settings.get("prompt_processing", True)

def set_prompt_processing_enabled(user_id: str, enabled: bool):
    settings = get_user_settings(user_id)
    settings["prompt_processing"] = enabled
    save_user_settings(user_id, settings)

def get_use_advanced_negative(user_id: str) -> bool:
    settings = get_user_settings(user_id)
    return settings.get("use_advanced_negative", False)

def get_user_lora_enabled(user_id: str) -> Dict[str, bool]:
    settings = get_user_settings(user_id)
    available = get_available_loras()
    user_lora = settings.get("lora_enabled", {})
    filtered = {}
    for lora in available:
        filtered[lora] = user_lora.get(lora, True)
    return filtered

def set_user_lora_enabled(user_id: str, lora_name: str, enabled: bool):
    settings = get_user_settings(user_id)
    if "lora_enabled" not in settings:
        settings["lora_enabled"] = {}
    settings["lora_enabled"][lora_name] = enabled
    save_user_settings(user_id, settings)

def get_active_lora_tags(user_id: str) -> str:
    """Возвращает строку с тегами LoRA для вставки в промпт (включённые)"""
    lora_enabled = get_user_lora_enabled(user_id)
    available = get_available_loras()
    tags = []
    for lora_name, enabled in lora_enabled.items():
        if enabled and lora_name in available:
            tags.append(available[lora_name]["tag"])
    return " ".join(tags)
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

# Глобальный кэш для сэмплеров
_available_samplers = None

def fetch_samplers_from_api() -> list:
    """Запрашивает список сэмплеров из API"""
    try:
        response = requests.get(f"{SD_BASE_URL}/sdapi/v1/samplers", timeout=10)
        if response.status_code == 200:
            samplers_data = response.json()
            samplers = []
            for item in samplers_data:
                if isinstance(item, dict) and 'name' in item:
                    samplers.append(item['name'])
            print(f"✅ Загружено {len(samplers)} сэмплеров из API")
            return samplers
        else:
            print(f"⚠️ Ошибка получения сэмплеров: {response.status_code}")
            return DEFAULT_SAMPLERS
    except Exception as e:
        print(f"⚠️ Ошибка при запросе сэмплеров: {e}")
        return DEFAULT_SAMPLERS

def get_available_samplers(force_refresh: bool = False) -> list:
    """Возвращает список доступных сэмплеров"""
    global _available_samplers
    if _available_samplers is None or force_refresh:
        _available_samplers = fetch_samplers_from_api()
    return _available_samplers

def get_user_sampler(user_id: str) -> str:
    """Возвращает текущий сэмплер пользователя"""
    settings = get_user_settings(user_id)
    return settings.get("sampler_name", DEFAULT_SETTINGS["sampler_name"])

def set_user_sampler(user_id: str, sampler_name: str):
    """Устанавливает сэмплер для пользователя"""
    settings = get_user_settings(user_id)
    settings["sampler_name"] = sampler_name
    save_user_settings(user_id, settings)

# ===== МОДЕЛИ (CHECKPOINTS) =====
# Список моделей берётся из API WebUI (/sdapi/v1/sd-models). Выбор хранится у каждого
# пользователя отдельно в ключе "sd_model" (имя модели без расширения и хэша) и
# передаётся в каждый запрос генерации через override_settings (см. Aapi.apply_model_override).

# Кэш списка моделей: [{"title": "...safetensors [abc123]", "name": "имя_модели"}, ...]
_available_models = []
# Модель «по умолчанию» (для пользователей, которые ничего не выбирали) — фиксируется один раз.
_default_model = None
# Если API недоступен, повторяем запрос не чаще, чем раз в RETRY_SECONDS (чтобы бот не «подвисал»)
_RETRY_SECONDS = 30
_models_last_attempt = 0.0
_default_last_attempt = 0.0


def fetch_models_from_api() -> list:
    """Запрашивает список моделей из API. Возвращает [{title, name}], отсортированный по имени."""
    try:
        response = requests.get(f"{SD_BASE_URL}/sdapi/v1/sd-models", timeout=15)
        if response.status_code != 200:
            print(f"⚠️ Ошибка получения списка моделей: {response.status_code}")
            return []
        result = []
        for item in response.json():
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


def get_available_models(force_refresh: bool = False) -> list:
    """Возвращает список доступных моделей (пустой список перезапрашивается при каждом обращении)."""
    global _available_models, _models_last_attempt
    if force_refresh or (not _available_models and time.time() - _models_last_attempt > _RETRY_SECONDS):
        _models_last_attempt = time.time()
        _available_models = fetch_models_from_api()
    return _available_models


def refresh_checkpoints_on_server() -> bool:
    """Просит WebUI пересканировать папку с моделями (новые файлы появятся в списке)."""
    try:
        response = requests.post(f"{SD_BASE_URL}/sdapi/v1/refresh-checkpoints", timeout=60)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️ Не удалось обновить список моделей на сервере: {e}")
        return False


def get_default_model() -> Optional[str]:
    """Модель по умолчанию: SD_DEFAULT_MODEL из .env, иначе модель, загруженная в WebUI при запуске бота."""
    global _default_model, _default_last_attempt
    if _default_model:
        return _default_model
    if time.time() - _default_last_attempt <= _RETRY_SECONDS and _default_last_attempt:
        return None
    _default_last_attempt = time.time()
    wanted = os.environ.get("SD_DEFAULT_MODEL", "").strip()
    if wanted:
        for model in get_available_models():
            if wanted in (model["title"], model["name"]):
                _default_model = model["title"]
                return _default_model
        print(f"⚠️ SD_DEFAULT_MODEL='{wanted}' не найдена среди моделей WebUI")
    try:
        response = requests.get(f"{SD_BASE_URL}/sdapi/v1/options", timeout=10)
        if response.status_code == 200:
            _default_model = response.json().get("sd_model_checkpoint") or None
    except Exception as e:
        print(f"⚠️ Не удалось узнать текущую модель WebUI: {e}")
    return _default_model


def get_default_model_name() -> Optional[str]:
    """Человекочитаемое имя модели по умолчанию."""
    title = get_default_model()
    if not title:
        return None
    for model in get_available_models():
        if model["title"] == title:
            return model["name"]
    return title


def get_user_model(user_id: str) -> Optional[str]:
    """Имя выбранной пользователем модели (None — используется модель по умолчанию)."""
    name = get_user_settings(user_id).get("sd_model")
    if not name:
        return None
    available = get_available_models()
    if available and not any(m["name"] == name for m in available):
        return None  # модель удалили/переименовали — откатываемся на модель по умолчанию
    return name


def set_user_model(user_id: str, model_name: Optional[str]):
    """Сохраняет выбор модели пользователя (None — сбросить на модель по умолчанию)."""
    settings = get_user_settings(user_id)
    settings["sd_model"] = model_name
    save_user_settings(user_id, settings)


def resolve_model(settings: dict) -> Optional[str]:
    """Название модели (title) для передачи в API: выбор пользователя либо модель по умолчанию."""
    name = (settings or {}).get("sd_model")
    if name:
        available = get_available_models()
        for model in available:
            if model["name"] == name:
                return model["title"]
        if not available:
            return name  # список недоступен — пробуем по имени, WebUI понимает и его
    return get_default_model()
