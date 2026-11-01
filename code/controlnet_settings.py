# controlnet_settings.py
import json
import os
from typing import Dict, Any

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


def _load_all_settings() -> dict:
    """Загружает все настройки ControlNet пользователей"""
    if os.path.exists(CONTROLNET_SETTINGS_FILE):
        try:
            with open(CONTROLNET_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_all_settings(settings: dict):
    """Сохраняет все настройки ControlNet пользователей"""
    with open(CONTROLNET_SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_controlnet_settings(user_id: str) -> Dict[str, Any]:
    """Возвращает настройки ControlNet для пользователя"""
    all_settings = _load_all_settings()
    user_settings = all_settings.get(user_id, {})
    
    # Заполняем недостающие значения из дефолтных
    for key, value in DEFAULT_CONTROLNET_SETTINGS.items():
        if key not in user_settings:
            user_settings[key] = value
    
    all_settings[user_id] = user_settings
    _save_all_settings(all_settings)
    return user_settings


def save_controlnet_settings(user_id: str, settings: Dict[str, Any]):
    """Сохраняет настройки ControlNet для пользователя"""
    all_settings = _load_all_settings()
    all_settings[user_id] = settings
    _save_all_settings(all_settings)


def set_controlnet_setting(user_id: str, key: str, value: Any):
    """Устанавливает конкретную настройку"""
    settings = get_controlnet_settings(user_id)
    settings[key] = value
    save_controlnet_settings(user_id, settings)


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