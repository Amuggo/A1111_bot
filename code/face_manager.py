"""
Управление лицами пользователей для замены лиц через Reactor
Все данные сохраняются локально
"""

import asyncio
import hashlib
import os
import io
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from storage import JsonStore

try:
    import app_config as config

    FACES_DIR = os.path.join(config.DATA_DIR, "user_faces")
    FACE_SETTINGS_FILE = config.FACE_SETTINGS_FILE
except ImportError:
    FACES_DIR = os.environ.get("FACES_DIR", "user_faces")
    FACE_SETTINGS_FILE = os.environ.get("FACE_SETTINGS_FILE", "face_settings.json")
os.makedirs(FACES_DIR, exist_ok=True)

# Настройки лиц читаются/пишутся только через это хранилище (потокобезопасно, запись атомарная),
# а операции с файлами лиц (сохранить/удалить/переименовать) сериализуются общей блокировкой.
_FACE_STORE = JsonStore(FACE_SETTINGS_FILE)
_FACE_FILES_LOCK = threading.RLock()


def _face_path(user_id: str, face_name: str) -> str:
    """Путь к файлу лица. Защита от выхода за пределы папки пользователя (../ и т.п.)."""
    if not face_name or os.path.basename(face_name) != face_name or face_name in (".", ".."):
        raise ValueError(f"Недопустимое имя лица: {face_name!r}")
    return os.path.join(FACES_DIR, str(user_id), f"{face_name}.jpg")


def _load_face_settings_sync() -> dict:
    """Загружает настройки лиц всех пользователей"""
    return _FACE_STORE.read()


def _save_face_settings_sync(settings: dict):
    """Полностью заменяет настройки лиц (оставлено для совместимости; для правок используйте _update_user_face_settings)."""
    _FACE_STORE.write(settings)


def _update_user_face_settings(user_id: str, mutator):
    """Атомарно изменяет запись пользователя; возвращает результат mutator."""
    user_id = str(user_id)

    def _mutator(all_settings: dict):
        entry = all_settings.setdefault(user_id, {"face_enabled": False, "active_face": None})
        entry.setdefault("face_enabled", False)
        entry.setdefault("active_face", None)
        entry.setdefault("face_mode", "swap")   # swap | faceid | both
        entry.setdefault("models", {})          # имя лица -> файл модели ReActor (models/reactor/faces)
        entry.setdefault("faceid_weight", None)       # вес ControlNet-юнита FaceID (None — из .env, по умолчанию 1.0)
        entry.setdefault("faceid_lora_weight", None)  # вес LoRA лица (None — из .env, по умолчанию 0.6; 0 — выкл.)
        return mutator(entry)

    return _FACE_STORE.update(_mutator)


def _get_face_enabled_sync(user_id: str) -> bool:
    """Получает статус включения замены лиц для пользователя"""
    return bool(_update_user_face_settings(user_id, lambda entry: entry.get("face_enabled", False)))


def _set_face_enabled_sync(user_id: str, enabled: bool):
    """Устанавливает статус включения замены лиц"""
    _update_user_face_settings(user_id, lambda entry: entry.__setitem__("face_enabled", bool(enabled)))


def _get_active_face_name_sync(user_id: str) -> Optional[str]:
    """Получает имя активного лица пользователя"""
    return _load_face_settings_sync().get(str(user_id), {}).get("active_face")


def _get_active_face_sync(user_id: str) -> Optional[io.BytesIO]:
    """Получает активное лицо пользователя как BytesIO"""
    active_face_name = _get_active_face_name_sync(user_id)

    if active_face_name:
        try:
            face_path = _face_path(user_id, active_face_name)
            with _FACE_FILES_LOCK:
                if os.path.exists(face_path):
                    with open(face_path, 'rb') as f:
                        return io.BytesIO(f.read())
        except (ValueError, OSError) as e:
            print(f"Не удалось прочитать активное лицо: {e}")
    return None


def _set_active_face_sync(user_id: str, face_name: str) -> bool:
    """Устанавливает активное лицо"""
    try:
        face_path = _face_path(user_id, face_name)
    except ValueError:
        return False
    with _FACE_FILES_LOCK:
        if not os.path.exists(face_path):
            return False
        _update_user_face_settings(user_id, lambda entry: entry.__setitem__("active_face", face_name))
        return True


def _get_user_faces_sync(user_id: str) -> List[Dict]:
    """Получает список лиц пользователя с их статусами"""
    user_dir = os.path.join(FACES_DIR, str(user_id))
    faces = []

    user_entry = _load_face_settings_sync().get(str(user_id), {})
    active_face_name = user_entry.get("active_face")
    models = user_entry.get("models") or {}

    if os.path.exists(user_dir):
        for filename in sorted(os.listdir(user_dir)):
            if filename.endswith('.jpg'):
                name = filename[:-4]  # Убираем .jpg
                faces.append({
                    "name": name,
                    "is_active": (name == active_face_name),
                    "path": os.path.join(user_dir, filename),
                    "model": models.get(name)  # файл модели ReActor или None
                })
    return faces


def _save_user_face_sync(user_id: str, image: io.BytesIO, face_name: str) -> bool:
    """Сохраняет лицо пользователя"""
    try:
        face_path = _face_path(user_id, face_name)  # проверка имени (без ../ и т.п.)
        image.seek(0)
        data = image.read()
        with _FACE_FILES_LOCK:
            os.makedirs(os.path.dirname(face_path), exist_ok=True)
            tmp_path = face_path + ".tmp"
            with open(tmp_path, 'wb') as f:
                f.write(data)
            os.replace(tmp_path, face_path)  # атомарно: файл либо старый, либо новый целиком
        return True
    except Exception as e:
        print(f"Ошибка сохранения лица: {e}")
        return False


def _delete_face_sync(user_id: str, face_name: str) -> bool:
    """Удаляет лицо пользователя"""
    try:
        face_path = _face_path(user_id, face_name)
        with _FACE_FILES_LOCK:
            if not os.path.exists(face_path):
                return False
            os.remove(face_path)

            # Если удалили активное лицо, сбрасываем его и забываем модель ReActor (атомарно)
            def reset_active(entry: dict):
                if entry.get("active_face") == face_name:
                    entry["active_face"] = None
                return (entry.get("models") or {}).pop(face_name, None)

            model_file = _update_user_face_settings(user_id, reset_active)
            _remove_model_file_best_effort(model_file)
            return True
    except Exception as e:
        print(f"Ошибка удаления лица: {e}")
        return False


def _rename_face_sync(user_id: str, old_name: str, new_name: str) -> bool:
    """Переименовывает лицо"""
    try:
        old_path = _face_path(user_id, old_name)
        new_path = _face_path(user_id, new_name)

        with _FACE_FILES_LOCK:
            if not os.path.exists(old_path) or os.path.exists(new_path):
                return False
            os.rename(old_path, new_path)

            # Обновляем активное лицо если нужно (атомарно)
            def follow_rename(entry: dict):
                if entry.get("active_face") == old_name:
                    entry["active_face"] = new_name
                models = entry.get("models") or {}
                if old_name in models:
                    models[new_name] = models.pop(old_name)

            _update_user_face_settings(user_id, follow_rename)
            return True
    except Exception as e:
        print(f"Ошибка переименования: {e}")
        return False


def _get_face_preview_sync(user_id: str, face_name: str) -> Optional[io.BytesIO]:
    """Возвращает превью лица"""
    try:
        face_path = _face_path(user_id, face_name)
        with _FACE_FILES_LOCK:
            if os.path.exists(face_path):
                with open(face_path, 'rb') as f:
                    return io.BytesIO(f.read())
    except (ValueError, OSError) as e:
        print(f"Не удалось прочитать превью лица: {e}")
    return None


def _get_face_stats_sync(user_id: str) -> dict:
    """Возвращает статистику по лицам пользователя"""
    faces = _get_user_faces_sync(user_id)
    active_face = next((f for f in faces if f["is_active"]), None)
    
    return {
        "total_faces": len(faces),
        "active_face": active_face["name"] if active_face else None
    }


# ===== Асинхронный интерфейс: файловый ввод-вывод выполняется в потоке (to_thread), цикл событий не блокируется =====

async def load_face_settings() -> dict:
    return await asyncio.to_thread(_load_face_settings_sync, )

async def save_face_settings(settings: dict):
    return await asyncio.to_thread(_save_face_settings_sync, settings)

async def get_face_enabled(user_id: str) -> bool:
    return await asyncio.to_thread(_get_face_enabled_sync, user_id)

async def set_face_enabled(user_id: str, enabled: bool):
    return await asyncio.to_thread(_set_face_enabled_sync, user_id, enabled)

async def get_active_face_name(user_id: str) -> Optional[str]:
    return await asyncio.to_thread(_get_active_face_name_sync, user_id)

async def get_active_face(user_id: str) -> Optional[io.BytesIO]:
    return await asyncio.to_thread(_get_active_face_sync, user_id)

async def set_active_face(user_id: str, face_name: str) -> bool:
    return await asyncio.to_thread(_set_active_face_sync, user_id, face_name)

async def get_user_faces(user_id: str) -> List[Dict]:
    return await asyncio.to_thread(_get_user_faces_sync, user_id)

async def save_user_face(user_id: str, image: io.BytesIO, face_name: str) -> bool:
    return await asyncio.to_thread(_save_user_face_sync, user_id, image, face_name)

async def delete_face(user_id: str, face_name: str) -> bool:
    return await asyncio.to_thread(_delete_face_sync, user_id, face_name)

async def rename_face(user_id: str, old_name: str, new_name: str) -> bool:
    return await asyncio.to_thread(_rename_face_sync, user_id, old_name, new_name)

async def get_face_preview(user_id: str, face_name: str) -> Optional[io.BytesIO]:
    return await asyncio.to_thread(_get_face_preview_sync, user_id, face_name)

async def get_face_stats(user_id: str) -> dict:
    return await asyncio.to_thread(_get_face_stats_sync, user_id)

async def check_reactor_available() -> bool:
    """Проверяет доступность Reactor API"""
    try:
        from Aapi import check_reactor_available as check
        return await check()
    except Exception:
        return False



# ===== МОДЕЛИ ЛИЦ ReActor, РЕЖИМЫ И КОНТЕКСТ ЛИЦА ДЛЯ ГЕНЕРАЦИИ =====

# swap  — замена лица ПОСЛЕ генерации (ReActor);
# faceid — лицо подаётся в саму генерацию (ControlNet IP-Adapter FaceID), без замены;
# both  — FaceID при генерации + ReActor-замена поверх (максимальное сходство, дольше).
FACE_MODES = ("swap", "faceid", "both")
FACE_MODE_LABELS = {
    "swap": "🔁 Замена после генерации",
    "faceid": "🧠 FaceID в генерации",
    "both": "🧠+🔁 FaceID + замена",
}


def remote_model_name(user_id: str, face_name: str) -> str:
    """Безопасное ASCII-имя модели на стороне WebUI (кириллица и пробелы в имени файла ни к чему)."""
    digest = hashlib.md5(face_name.encode("utf-8")).hexdigest()[:8]
    return f"tg{str(user_id).lstrip('-')}_{digest}"


def _remove_model_file_best_effort(model_file: Optional[str]) -> None:
    """Если WebUI лежит на этой же машине (A1111_BASE_PATH), удаляем файл модели вместе с лицом."""
    base = os.environ.get("A1111_BASE_PATH", "").strip()
    if not base or not model_file:
        return
    try:
        path = os.path.join(base, "models", "reactor", "faces", os.path.basename(model_file))
        if os.path.isfile(path):
            os.remove(path)
    except OSError as e:
        print(f"Не удалось удалить модель лица: {e}")


def _get_face_mode_sync(user_id: str) -> str:
    mode = _load_face_settings_sync().get(str(user_id), {}).get("face_mode", "swap")
    return mode if mode in FACE_MODES else "swap"


def _set_face_mode_sync(user_id: str, mode: str) -> bool:
    if mode not in FACE_MODES:
        return False
    _update_user_face_settings(user_id, lambda entry: entry.__setitem__("face_mode", mode))
    return True


FACEID_WEIGHT_STEPS = (0.6, 0.8, 1.0, 1.2)
FACEID_LORA_STEPS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _get_face_weights_sync(user_id: str) -> Tuple[float, float]:
    """(вес FaceID, вес LoRA лица): настройка пользователя, иначе значения из .env."""
    entry = _load_face_settings_sync().get(str(user_id), {})
    fw = entry.get("faceid_weight")
    lw = entry.get("faceid_lora_weight")
    return (float(fw) if fw is not None else _env_float("FACEID_WEIGHT", 1.0),
            float(lw) if lw is not None else _env_float("FACEID_LORA_WEIGHT", 0.6))


def _set_face_weight_sync(user_id: str, key: str, value: float) -> None:
    _update_user_face_settings(user_id, lambda entry: entry.__setitem__(key, round(float(value), 2)))


async def get_face_weights(user_id: str) -> Tuple[float, float]:
    return await asyncio.to_thread(_get_face_weights_sync, user_id)


async def cycle_face_weight(user_id: str, kind: str) -> float:
    """Переключает вес на следующее значение: kind = 'faceid' | 'lora'. Возвращает новое значение."""
    steps, key, idx = ((FACEID_WEIGHT_STEPS, "faceid_weight", 0) if kind == "faceid"
                       else (FACEID_LORA_STEPS, "faceid_lora_weight", 1))
    current = (await get_face_weights(user_id))[idx]
    nearest = min(range(len(steps)), key=lambda i: abs(steps[i] - current))
    new_value = steps[(nearest + 1) % len(steps)]
    await asyncio.to_thread(_set_face_weight_sync, user_id, key, new_value)
    return new_value


def _get_face_model_file_sync(user_id: str, face_name: str) -> Optional[str]:
    return ((_load_face_settings_sync().get(str(user_id), {}).get("models")) or {}).get(face_name)


def _set_face_model_file_sync(user_id: str, face_name: str, model_file: str) -> None:
    _update_user_face_settings(user_id, lambda entry: entry["models"].__setitem__(face_name, model_file))


async def get_face_mode(user_id: str) -> str:
    return await asyncio.to_thread(_get_face_mode_sync, user_id)


async def set_face_mode(user_id: str, mode: str) -> bool:
    return await asyncio.to_thread(_set_face_mode_sync, user_id, mode)


async def get_face_model_file(user_id: str, face_name: str) -> Optional[str]:
    return await asyncio.to_thread(_get_face_model_file_sync, user_id, face_name)


async def build_face_model(user_id: str, face_name: str, extra_images: Optional[List[bytes]] = None) -> str:
    """
    Создаёт (или пересоздаёт) модель ReActor для лица. Без extra_images берётся сохранённое фото лица,
    с extra_images — все переданные фото (эмбеддинги усредняются: модель получается устойчивее одного фото).
    Возвращает имя файла модели.
    """
    from Aapi import build_reactor_face_model
    images: List[bytes] = list(extra_images or [])
    if not images:
        preview = await get_face_preview(user_id, face_name)
        if preview is None:
            raise ValueError("Фото лица не найдено")
        images.append(preview.getvalue())
    model_file = await build_reactor_face_model(images, remote_model_name(user_id, face_name))
    await asyncio.to_thread(_set_face_model_file_sync, user_id, face_name, model_file)
    return model_file


_REACTOR_CACHE = {"ts": 0.0, "ok": False}


async def reactor_ok(force: bool = False) -> bool:
    """Доступность ReActor с кэшем на минуту (WebUI можно запустить позже бота)."""
    now = time.monotonic()
    if not force and now - _REACTOR_CACHE["ts"] < 60:
        return _REACTOR_CACHE["ok"]
    ok = await check_reactor_available()
    _REACTOR_CACHE.update(ts=now, ok=ok)
    return ok


async def face_features_available() -> bool:
    """Есть ли хоть что-то, чем можно применить лицо: ReActor или FaceID."""
    if await reactor_ok():
        return True
    try:
        from Aapi import get_faceid_setup
        return bool((await get_faceid_setup()).get("ok"))
    except Exception:
        return False


@dataclass
class FaceContext:
    name: str
    image: io.BytesIO
    model_file: Optional[str]
    mode: str
    use_faceid: bool           # лицо идёт в генерацию (ControlNet FaceID)
    use_swap: bool             # после генерации делаем ReActor-замену
    weights: Optional[Tuple[float, float]] = None  # (вес FaceID, вес LoRA лица) пользователя
    note: Optional[str] = None  # почему режим пришлось изменить (для статуса пользователю)


async def get_face_context(user_id: str, *, allow_faceid: bool = True,
                           need_controlnet_slot: bool = False) -> Optional[FaceContext]:
    """
    Единая точка решения «как применять лицо этому пользователю». None — лицо выключено или не выбрано.
    Если выбранный режим невозможен (нет FaceID-модели, мало ControlNet-юнитов, нет ReActor), откатывается
    на доступный и пишет причину в note.
    """
    user_id = str(user_id)
    if not await get_face_enabled(user_id):
        return None
    name = await get_active_face_name(user_id)
    image = await get_active_face(user_id) if name else None
    if not name or image is None:
        return None

    mode = await get_face_mode(user_id)
    model_file = await get_face_model_file(user_id, name)
    use_faceid, note = False, None

    import subscriptions  # функции лиц могут быть ограничены тарифом (тестовые подписки)
    if not await subscriptions.feature_allowed(user_id, "face"):
        return FaceContext(name, image, model_file, mode, False, False,
                           note="функции лиц доступны на тарифе Pro (/plan)")

    if mode in ("faceid", "both") and allow_faceid:
        try:
            from Aapi import get_faceid_setup
            setup = await get_faceid_setup()
        except Exception as e:
            setup = {"ok": False, "reason": str(e)}
        if not setup.get("ok"):
            note = f"FaceID недоступен ({setup.get('reason')}), использую замену после генерации"
        elif need_controlnet_slot and setup.get("unit_count", 1) < 2:
            note = "для FaceID вместе с ControlNet-позой нужно минимум 2 ControlNet-юнита (Settings → ControlNet), использую замену после генерации"
        else:
            use_faceid = True

    use_swap = mode in ("swap", "both") or not use_faceid
    if use_swap and not await reactor_ok():
        use_swap = False
        note = (note + "; " if note else "") + "ReActor недоступен"
    return FaceContext(name, image, model_file, mode, use_faceid, use_swap, weights=await get_face_weights(user_id), note=note)


async def apply_face_swap(ctx: FaceContext, image: io.BytesIO) -> Tuple[io.BytesIO, bool]:
    """ReActor-замена лица на готовой картинке. Сначала по модели лица, при ошибке — по исходному фото. (результат, успех)"""
    from Aapi import swap_face_with_reactor
    attempts = []
    if ctx.model_file:
        attempts.append({"source_image": None, "face_model": ctx.model_file})
    attempts.append({"source_image": ctx.image, "face_model": "None"})
    for attempt in attempts:
        try:
            result = await swap_face_with_reactor(
                target_image=image, upscaler="None", face_restorer="GFPGAN", restorer_visibility=1.0, **attempt)
            return result, True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Ошибка замены лица ({'модель' if attempt['face_model'] != 'None' else 'фото'}): {e}")
    return image, False
