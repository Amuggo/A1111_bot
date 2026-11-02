# adetailer_settings.py
"""
Пользовательские настройки ADetailer (расширение для AUTOMATIC1111 WebUI).

Хранит per-user словарь с параметрами, имена которых совпадают с аргументами
ADetailer для API (ad_model, ad_confidence, ...), чтобы его можно было
передать в alwayson_scripts["ADetailer"]["args"] без изменений.

По аналогии с controlnet_settings.py: файл настроек лежит в DATA_DIR.
"""
import copy
import json
import os
from typing import Any, Dict

from storage import JsonStore

try:
    import app_config as config

    ADETAILER_SETTINGS_FILE = config.ADETAILER_SETTINGS_FILE
except ImportError:
    ADETAILER_SETTINGS_FILE = os.environ.get(
        "ADETAILER_SETTINGS_FILE", "adetailer_user_settings.json"
    )

# ===== БАЗОВЫЕ НАСТРОЙКИ (одинаковы для всех вкладок детектора) =====
DEFAULT_ADETAILER_SETTINGS = {
    "ad_enabled": False,               # главный переключатель расширения в боте
    "ad_model": "face_yolov8n.pt",     # None / face_yolov8n.pt / person_yolov8n-seg.pt / mediapipe_face_mesh ...
    "ad_model_classes": "",            # пустая строка = все классы детектора
    "ad_prompt": "",                   # доп. промпт только для области маски
    "ad_negative_prompt": "",          # доп. негатив только для области маски
    "ad_confidence": 0.3,              # порог уверенности детектора (0..1)
    "ad_mask_k": 0,                    # 0 = все маски, >0 = только топ-N масок
    "ad_mask_min_ratio": 0.0,          # отбросить маски меньше N% от площади кадра
    "ad_mask_max_ratio": 1.0,          # отбросить маски больше N% от площади кадра
    "ad_dilate_erode": 4,              # расширение/сужение маски в пикселях
    "ad_x_offset": 0,                  # сдвиг рамки маски по X
    "ad_y_offset": 0,                  # сдвиг рамки маски по Y
    "ad_mask_merge_invert": "None",    # None / Merge / Merge and Invert
    "ad_mask_blur": 4,                 # размытие краёв маски
    "ad_denoising_strength": 0.45,     # сила перерисовки лица (0..1)
    "ad_inpaint_only_masked": True,
    "ad_inpaint_only_masked_padding": 32,
    "ad_use_inpaint_width_height": False,
    "ad_inpaint_width": 512,
    "ad_inpaint_height": 512,
    "ad_use_steps": False,
    "ad_steps": 28,
    "ad_use_cfg_scale": False,
    "ad_cfg_scale": 7.0,
    "ad_use_sampler": False,
    "ad_sampler": "DPM++ 2M Karras",
    "ad_restore_face": False,          # вернуть оригинальные пиксели лиц после обработки
}

# ===== СПИСОК МОДЕЛЕЙ DETECTOR'ОВ =====
# Ключевые модели, доступные «из коробки» (yolov8 + mediapipe). Полный список
# отдаёт сервер: get_adetailer_models() -> /sdapi/v1/scripts (см. Aapi.py).
DEFAULT_ADETAILER_MODELS = [
    ("face_yolov8n.pt", "😀 Лицо (быстрая)"),
    ("face_yolov8s.pt", "😀 Лицо (точная)"),
    ("face_yolov8_medium.pt", "😀 Лицо (средняя)"),
    ("face_yolov8_large.pt", "😀 Лицо (тяжёлая)"),
    ("hand_yolov8n.pt", "✋ Руки (быстрая)"),
    ("hand_yolov8s.pt", "✋ Руки (точная)"),
    ("person_yolov8n-seg.pt", "🧍 Человек (быстрая сегментация)"),
    ("person_yolov8n.pt", "🧍 Человек (быстрая рамка)"),
    ("person_yolov8s-seg.pt", "🧍 Человек (точная сегментация)"),
    ("mediapipe_face_full", "🎭 Mediapipe: всё лицо"),
    ("mediapipe_face_short", "🎭 Mediapipe: верхняя часть лица"),
    ("mediapipe_face_mesh", "🕸 Mediapipe: сетка лица"),
    ("mediapipe_face_mesh_eyes_only", "👁 Mediapipe: только глаза"),
]

MODEL_DISPLAY_NAMES = dict(DEFAULT_ADETAILER_MODELS)

DENOISING_STRENGTHS = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8]
CONFIDENCE_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASK_K_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10]
DILATE_ERODE_VALUES = [-8, -4, 0, 2, 4, 6, 8, 12, 16]
MASK_BLUR_VALUES = [0, 2, 4, 6, 8, 12, 16, 24]
PADDING_VALUES = [0, 16, 32, 48, 64, 96, 128]
MERGE_INVERT_VALUES = ["None", "Merge", "Merge and Invert"]


_STORE = JsonStore(ADETAILER_SETTINGS_FILE)


def _load_all_settings() -> dict:
    """Загружает настройки ADetailer всех пользователей"""
    return _STORE.read()


def _fill_defaults(user_settings: dict) -> dict:
    for key, value in DEFAULT_ADETAILER_SETTINGS.items():
        if key not in user_settings:
            user_settings[key] = copy.deepcopy(value)
    return user_settings


def get_adetailer_settings(user_id: str) -> Dict[str, Any]:
    """Возвращает настройки ADetailer пользователя (с дефолтами); файл пишется только при изменении"""
    user_id = str(user_id)

    def mutator(all_settings: dict):
        user_settings = _fill_defaults(all_settings.get(user_id, {}))
        all_settings[user_id] = user_settings
        return copy.deepcopy(user_settings)

    return _STORE.update(mutator)


def save_adetailer_settings(user_id: str, settings: Dict[str, Any]):
    """Сохраняет настройки ADetailer пользователя (замена целиком)"""
    _STORE.update(lambda all_settings: all_settings.__setitem__(str(user_id), copy.deepcopy(settings)))


def set_adetailer_setting(user_id: str, key: str, value: Any):
    """Устанавливает отдельную настройку (атомарно)"""
    user_id = str(user_id)

    def mutator(all_settings: dict):
        user_settings = _fill_defaults(all_settings.get(user_id, {}))
        user_settings[key] = value
        all_settings[user_id] = user_settings

    _STORE.update(mutator)


def get_adetailer_enabled(user_id: str) -> bool:
    return bool(get_adetailer_settings(user_id).get("ad_enabled", False))


def set_adetailer_enabled(user_id: str, enabled: bool):
    set_adetailer_setting(user_id, "ad_enabled", bool(enabled))


def get_adetailer_model(user_id: str) -> str:
    return get_adetailer_settings(user_id).get("ad_model", "None")


def set_adetailer_model(user_id: str, model: str):
    set_adetailer_setting(user_id, "ad_model", model)


def get_ad_model_display_name(model: str) -> str:
    """Человекочитаемое имя модели детектора"""
    if not model or model == "None":
        return "❌ Выключен"
    return MODEL_DISPLAY_NAMES.get(model, model)


def build_adetailer_args(user_id: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Собирает словарь аргументов ADetailer для передачи в API.

    Возвращает только те ключи, которые реально понимает расширение
    (is_api=True обязателен — иначе WebUI падает при валидации pydantic).
    """
    settings = get_adetailer_settings(user_id)
    if overrides:
        settings = {**settings, **overrides}

    args = {"is_api": True}
    for key in DEFAULT_ADETAILER_SETTINGS:
        if key == "ad_enabled":
            continue
        value = settings.get(key)
        if value is None:
            continue
        args[key] = value

    # В WebUI поле ad_tab_enable отвечает за включение вкладки; при работе через
    # API достаточно ad_model != "None", но явный флаг безопаснее для старых версий.
    args["ad_tab_enable"] = True
    return args


def summarize_adetailer_settings(user_id: str) -> str:
    """Короткое текстовое описание текущих настроек (для капшнов и меню)"""
    s = get_adetailer_settings(user_id)
    if not s.get("ad_enabled"):
        return "🕵️ ADetailer: ❌ выключен"
    lines = [
        f"🕵️ **ADetailer:** ✅ включен",
        f"🔎 Модель: `{s.get('ad_model')}` ({get_ad_model_display_name(s.get('ad_model'))})",
        f"🎚️ Denoising: {s.get('ad_denoising_strength')}",
        f"🎯 Confidence: {s.get('ad_confidence')}",
    ]
    if s.get("ad_prompt"):
        lines.append(f"➕ Промпт лица: `{s['ad_prompt'][:60]}`")
    if s.get("ad_negative_prompt"):
        lines.append(f"➖ Негатив лица: `{s['ad_negative_prompt'][:60]}`")
    if s.get("ad_mask_k"):
        lines.append(f"🔢 Топ масок: {s['ad_mask_k']}")
    if s.get("ad_restore_face"):
        lines.append("🛡 Restore face: включен")
    return "\n".join(lines)
