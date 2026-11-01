"""
Управление лицами пользователей для замены лиц через Reactor
Все данные сохраняются локально
"""

import os
import json
import io
from typing import List, Dict, Optional

try:
    import app_config as config

    FACES_DIR = os.path.join(config.DATA_DIR, "user_faces")
    FACE_SETTINGS_FILE = config.FACE_SETTINGS_FILE
except ImportError:
    FACES_DIR = os.environ.get("FACES_DIR", "user_faces")
    FACE_SETTINGS_FILE = os.environ.get("FACE_SETTINGS_FILE", "face_settings.json")
os.makedirs(FACES_DIR, exist_ok=True)


def _face_path(user_id: str, face_name: str) -> str:
    """Путь к файлу лица. Защита от выхода за пределы папки пользователя (../ и т.п.)."""
    if not face_name or os.path.basename(face_name) != face_name or face_name in (".", ".."):
        raise ValueError(f"Недопустимое имя лица: {face_name!r}")
    return os.path.join(FACES_DIR, str(user_id), f"{face_name}.jpg")


def load_face_settings() -> dict:
    """Загружает настройки лиц всех пользователей"""
    if os.path.exists(FACE_SETTINGS_FILE):
        try:
            with open(FACE_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка загрузки настроек лиц: {e}")
            return {}
    return {}


def save_face_settings(settings: dict):
    """Сохраняет настройки лиц всех пользователей"""
    try:
        with open(FACE_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Ошибка сохранения настроек лиц: {e}")


def get_face_enabled(user_id: str) -> bool:
    """Получает статус включения замены лиц для пользователя"""
    settings = load_face_settings()
    if user_id not in settings:
        settings[user_id] = {"face_enabled": False, "active_face": None}
        save_face_settings(settings)
    return settings[user_id].get("face_enabled", False)


def set_face_enabled(user_id: str, enabled: bool):
    """Устанавливает статус включения замены лиц"""
    settings = load_face_settings()
    if user_id not in settings:
        settings[user_id] = {"face_enabled": False, "active_face": None}
    settings[user_id]["face_enabled"] = enabled
    save_face_settings(settings)


def get_active_face_name(user_id: str) -> Optional[str]:
    """Получает имя активного лица пользователя"""
    settings = load_face_settings()
    return settings.get(user_id, {}).get("active_face")


def get_active_face(user_id: str) -> Optional[io.BytesIO]:
    """Получает активное лицо пользователя как BytesIO"""
    active_face_name = get_active_face_name(user_id)
    
    if active_face_name:
        face_path = _face_path(user_id, active_face_name)
        if os.path.exists(face_path):
            with open(face_path, 'rb') as f:
                return io.BytesIO(f.read())
    return None


def set_active_face(user_id: str, face_name: str) -> bool:
    """Устанавливает активное лицо"""
    settings = load_face_settings()
    if user_id not in settings:
        settings[user_id] = {"face_enabled": False, "active_face": None}
    
    # Проверяем, существует ли такое лицо
    face_path = _face_path(user_id, face_name)
    if os.path.exists(face_path):
        settings[user_id]["active_face"] = face_name
        save_face_settings(settings)
        return True
    return False


def get_user_faces(user_id: str) -> List[Dict]:
    """Получает список лиц пользователя с их статусами"""
    user_dir = os.path.join(FACES_DIR, user_id)
    faces = []
    
    active_face_name = get_active_face_name(user_id)
    
    if os.path.exists(user_dir):
        for filename in os.listdir(user_dir):
            if filename.endswith('.jpg'):
                name = filename[:-4]  # Убираем .jpg
                faces.append({
                    "name": name,
                    "is_active": (name == active_face_name),
                    "path": os.path.join(user_dir, filename)
                })
    return faces


def save_user_face(user_id: str, image: io.BytesIO, face_name: str) -> bool:
    """Сохраняет лицо пользователя"""
    try:
        user_dir = os.path.join(FACES_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        face_path = os.path.join(user_dir, f"{face_name}.jpg")
        image.seek(0)
        
        with open(face_path, 'wb') as f:
            f.write(image.read())
        
        return True
    except Exception as e:
        print(f"Ошибка сохранения лица: {e}")
        return False


def delete_face(user_id: str, face_name: str) -> bool:
    """Удаляет лицо пользователя"""
    try:
        face_path = _face_path(user_id, face_name)
        if os.path.exists(face_path):
            os.remove(face_path)
            
            # Если удалили активное лицо, сбрасываем его
            current_active = get_active_face_name(user_id)
            if current_active == face_name:
                settings = load_face_settings()
                if user_id in settings:
                    settings[user_id]["active_face"] = None
                    save_face_settings(settings)
            
            return True
        return False
    except Exception as e:
        print(f"Ошибка удаления лица: {e}")
        return False


def rename_face(user_id: str, old_name: str, new_name: str) -> bool:
    """Переименовывает лицо"""
    try:
        old_path = _face_path(user_id, old_name)
        new_path = _face_path(user_id, new_name)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            
            # Обновляем активное лицо если нужно
            current_active = get_active_face_name(user_id)
            if current_active == old_name:
                settings = load_face_settings()
                if user_id in settings:
                    settings[user_id]["active_face"] = new_name
                    save_face_settings(settings)
            
            return True
        return False
    except Exception as e:
        print(f"Ошибка переименования: {e}")
        return False


def get_face_preview(user_id: str, face_name: str) -> Optional[io.BytesIO]:
    """Возвращает превью лица"""
    face_path = _face_path(user_id, face_name)
    if os.path.exists(face_path):
        with open(face_path, 'rb') as f:
            return io.BytesIO(f.read())
    return None


def get_face_stats(user_id: str) -> dict:
    """Возвращает статистику по лицам пользователя"""
    faces = get_user_faces(user_id)
    active_face = next((f for f in faces if f["is_active"]), None)
    
    return {
        "total_faces": len(faces),
        "active_face": active_face["name"] if active_face else None
    }


def check_reactor_available() -> bool:
    """Проверяет доступность Reactor API"""
    try:
        from Aapi import check_reactor_available as check
        return check()
    except:
        return False