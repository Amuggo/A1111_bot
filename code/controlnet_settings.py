# controlnet_settings.py
import copy
import json
import os
from typing import Dict, Any

from storage import JsonStore

try:
    import app_config as config

    CONTROLNET_SETTINGS_FILE = config.CONTROLNET_SETTINGS_FILE
except ImportError:
    CONTROLNET_SETTINGS_FILE = os.environ.get("CONTROLNET_SETTINGS_FILE", "controlnet_user_settings.json")

DEFAULT_CONTROLNET_SETTINGS = {
    "weight": 1.0,
    "control_mode": "Balanced",
    "guidance_start": 0.0,
    "guidance_end": 1.0,
    "resize_mode": "Just Resize",  # Теперь строка!
    "processor_res": 512,
    "low_vram": False,
    "pixel_perfect": True,
    "denoising_strength": 0.7,
}


_STORE = JsonStore(CONTROLNET_SETTINGS_FILE)


def _load_all_settings() -> dict:
    """Загружает все настройки ControlNet пользователей"""
    return _STORE.read()


def _fill_defaults(user_settings: dict) -> dict:
    # Заполняем недостающие значения из дефолтных
    for key, value in DEFAULT_CONTROLNET_SETTINGS.items():
        if key not in user_settings:
            user_settings[key] = copy.deepcopy(value)
    return user_settings


def get_controlnet_settings(user_id: str) -> Dict[str, Any]:
    """Возвращает настройки ControlNet для пользователя (файл пишется только при изменении)"""
    user_id = str(user_id)

    def mutator(all_settings: dict):
        user_settings = _fill_defaults(all_settings.get(user_id, {}))
        all_settings[user_id] = user_settings
        return copy.deepcopy(user_settings)

    return _STORE.update(mutator)


def save_controlnet_settings(user_id: str, settings: Dict[str, Any]):
    """Сохраняет настройки ControlNet для пользователя (замена целиком)"""
    _STORE.update(lambda all_settings: all_settings.__setitem__(str(user_id), copy.deepcopy(settings)))


def set_controlnet_setting(user_id: str, key: str, value: Any):
    """Устанавливает конкретную настройку (атомарно)"""
    user_id = str(user_id)

    def mutator(all_settings: dict):
        user_settings = _fill_defaults(all_settings.get(user_id, {}))
        user_settings[key] = value
        all_settings[user_id] = user_settings

    _STORE.update(mutator)


# Расшифровка значений для отображения
CONTROL_MODE_NAMES = {
    "Balanced": "⚖️ Сбалансированный",
    "Prompt": "📝 Приоритет промпта",
    "Control": "🎯 Приоритет ControlNet",
}

RESIZE_MODE_NAMES = {
    "Just Resize": "📐 Just Resize",
    "Crop and Resize": "✂️ Crop and Resize",
    "Resize and Fill": "🔄 Resize and Fill",
}