"""
Управление лицами пользователей для замены лиц через Reactor
Все данные сохраняются локально
"""

import asyncio
import os
import io
import threading
from typing import List, Dict, Optional

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

    active_face_name = _get_active_face_name_sync(user_id)

    if os.path.exists(user_dir):
        for filename in sorted(os.listdir(user_dir)):
            if filename.endswith('.jpg'):
                name = filename[:-4]  # Убираем .jpg
                faces.append({
                    "name": name,
                    "is_active": (name == active_face_name),
                    "path": os.path.join(user_dir, filename)
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

            # Если удалили активное лицо, сбрасываем его (атомарно)
            def reset_active(entry: dict):
                if entry.get("active_face") == face_name:
                    entry["active_face"] = None

            _update_user_face_settings(user_id, reset_active)
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
