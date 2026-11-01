import os
import json
import time
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import app_config as config

    SD_BASE_URL = config.SD_BASE_URL
except ImportError:
    SD_BASE_URL = os.environ.get("SD_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

# ===== ПУТИ =====
# Путь к папке A1111 (где лежит stable-diffusion-webui)
A1111_BASE_PATH = os.environ.get("A1111_BASE_PATH", r"N:\A1111")

# Путь к папке embeddings внутри stable-diffusion-webui
STABLE_DIFFUSION_PATH = os.path.join(A1111_BASE_PATH, "stable-diffusion-webui")
EMBEDDINGS_PATH = os.path.join(STABLE_DIFFUSION_PATH, "embeddings")

# Список рекомендуемых эмбеддингов для улучшения анатомии
RECOMMENDED_EMBEDDINGS = {
    "bad-hands-5": {
        "url": "https://huggingface.co/datasets/gsdf/Counterfeit-V2.5/resolve/main/embeddings/bad-hands-5.pt",
        "filename": "bad-hands-5.pt",
        "description": "Исправляет плохие руки, лишние пальцы"
    },
    "badhandv4": {
        "url": "https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/badhandv4.pt",
        "filename": "badhandv4.pt", 
        "description": "Альтернативный эмбеддинг для рук"
    },
    "easynegative": {
        "url": "https://huggingface.co/datasets/gsdf/Counterfeit-V2.5/resolve/main/embeddings/easynegative.pt",
        "filename": "easynegative.pt",
        "description": "Улучшает общее качество, убирает артефакты"
    }
}

# Расширенный негативный промпт для улучшения анатомии
ADVANCED_NEGATIVE_PROMPT = """bad anatomy, bad hands, extra fingers, fewer fingers, missing fingers, fused fingers, 
deformed hands, crooked fingers, extra limbs, missing limbs, mutated hands, six fingers, two left hands, 
bad feet, extra toes, missing toes, deformed feet, bad legs, extra legs, missing legs, 
bad arms, bad proportions, disfigured, mutated, ugly, worst quality, low quality, blurry, 
jpeg artifacts, watermark, signature, text, logo, cropped, out of frame, duplicate, morbid"""


def get_embeddings_path() -> str:
    """Возвращает путь к папке embeddings, создаёт если не существует"""
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    return EMBEDDINGS_PATH


def check_a1111_available() -> bool:
    """Проверяет, доступен ли A1111 по указанному пути"""
    webui_path = os.path.join(STABLE_DIFFUSION_PATH, "webui-user.bat")
    launch_path = os.path.join(STABLE_DIFFUSION_PATH, "launch.py")
    return os.path.exists(webui_path) or os.path.exists(launch_path)


def check_embeddings_installed() -> dict:
    """Проверяет, какие эмбеддинги уже установлены"""
    result = {}
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    
    for name, info in RECOMMENDED_EMBEDDINGS.items():
        file_path = os.path.join(EMBEDDINGS_PATH, info["filename"])
        result[name] = os.path.exists(file_path)
    
    return result


def download_embedding(embedding_name: str, progress_callback=None) -> bool:
    """Скачивает эмбеддинг"""
    if embedding_name not in RECOMMENDED_EMBEDDINGS:
        return False
    
    info = RECOMMENDED_EMBEDDINGS[embedding_name]
    file_path = os.path.join(EMBEDDINGS_PATH, info["filename"])
    
    try:
        os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
        
        if progress_callback:
            progress_callback(f"Скачиваю {info['filename']}...")
        
        urllib.request.urlretrieve(info["url"], file_path)
        
        if progress_callback:
            progress_callback(f"✅ {info['filename']} установлен!")
        
        return True
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Ошибка: {str(e)}")
        return False


def install_all_recommended_embeddings(progress_callback=None) -> dict:
    """Устанавливает все рекомендуемые эмбеддинги"""
    results = {}
    
    for name in RECOMMENDED_EMBEDDINGS:
        if progress_callback:
            progress_callback(f"Устанавливаю {name}...")
        results[name] = download_embedding(name, progress_callback)
    
    return results


def get_negative_prompt_with_embeddings(user_negative: str = "", use_advanced: bool = True,
                                        selected_embeddings: Optional[List[str]] = None) -> str:
    """Формирует финальный негативный промпт с учётом установленных эмбеддингов.

    selected_embeddings — список имён негативных эмбеддингов, выбранных пользователем
    (получен из API WebUI). Если список передан (в т.ч. пустой), используются только
    он и локальные рекомендуемые эмбеддинги; если None — прежнее поведение (все
    установленные рекомендуемые).
    """
    negative_parts = []

    # Добавляем расширенный негативный промпт если включён
    if use_advanced:
        negative_parts.append(ADVANCED_NEGATIVE_PROMPT)

    # Добавляем пользовательский негатив если есть
    if user_negative and user_negative.strip():
        negative_parts.append(user_negative)

    # Эмбеддинги, выбираемые пользователем из API
    embedding_tags: List[str] = []
    if selected_embeddings is not None:
        embedding_tags.extend(e for e in selected_embeddings if e)
    else:
        # Совместимость со старым поведением: добавляем эмбеддинги если они установлены
        installed = check_embeddings_installed()

        if installed.get("bad-hands-5"):
            embedding_tags.append("bad-hands-5")
        if installed.get("badhandv4"):
            embedding_tags.append("badhandv4")
        if installed.get("easynegative"):
            embedding_tags.append("easynegative")

    if embedding_tags:
        negative_parts.append(", ".join(embedding_tags))

    return ", ".join(negative_parts)


# ===== НЕГАТИВНЫЕ ЭМБЕДДИНГИ ИЗ API WEBUI =====
# Кэш ответа /sdapi/v1/embeddings
_available_embeddings_cache: Optional[dict] = None
_embeddings_last_fetch: float = 0.0
_EMBEDDINGS_CACHE_TTL = 30  # секунд


def fetch_embeddings_from_api(force_refresh: bool = False) -> Dict[str, dict]:
    """Запрашивает список эмбеддингов из API WebUI (/sdapi/v1/embeddings).

    Возвращает словарь {имя_эмбеддинга: {"shape": int, "vector_dim": int}}.
    При недоступности API возвращает пустой словарь (кэш не затирается).
    """
    global _available_embeddings_cache, _embeddings_last_fetch

    if _available_embeddings_cache is not None and not force_refresh:
        if time.time() - _embeddings_last_fetch < _EMBEDDINGS_CACHE_TTL:
            return _available_embeddings_cache

    result: Dict[str, dict] = {}
    try:
        if requests is not None:
            response = requests.get(f"{SD_BASE_URL}/sdapi/v1/embeddings", timeout=15)
            if response.status_code != 200:
                print(f"⚠️ Ошибка получения эмбеддингов: {response.status_code}")
                return _available_embeddings_cache or {}
            data = response.json()
        else:
            request = urllib.request.Request(
                f"{SD_BASE_URL}/sdapi/v1/embeddings",
                headers={'Content-Type': 'application/json'},
                method='GET'
            )
            with urllib.request.urlopen(request, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"⚠️ Ошибка при запросе эмбеддингов: {e}")
        return _available_embeddings_cache or {}

    # Формат ответа: {"local_data": {"name": {"shape": N, "vector_dim": M}, ...}, "server_data": [...]}
    local_data = data.get("local_data") or {}
    for name, info in local_data.items():
        if isinstance(info, dict):
            result[name] = {"shape": info.get("shape", 0), "vector_dim": info.get("vector_dim", 0)}
        else:
            result[name] = {"shape": 0, "vector_dim": 0}

    # Fallback: some versions only return server_data list of names
    if not result:
        for name in data.get("server_data") or []:
            if isinstance(name, str):
                result[name] = {"shape": 0, "vector_dim": 0}

    _available_embeddings_cache = result
    _embeddings_last_fetch = time.time()
    print(f"✅ Загружено {len(result)} эмбеддингов из API")
    return result


def refresh_embeddings_on_server() -> bool:
    """Просит WebUI пересканировать папку embeddings (файл .refreshEmbeddings)."""
    try:
        if requests is not None:
            response = requests.post(f"{SD_BASE_URL}/internal/refresh-embeddings", timeout=60)
            return response.status_code == 200
        request = urllib.request.Request(
            f"{SD_BASE_URL}/internal/refresh-embeddings",
            headers={'Content-Type': 'application/json'},
            data=b"{}",
            method='POST'
        )
        with urllib.request.urlopen(request, timeout=60):
            return True
    except Exception as e:
        print(f"⚠️ Не удалось обновить список эмбеддингов на сервере: {e}")
        return False


def get_available_embeddings(force_refresh: bool = False) -> Dict[str, dict]:
    """Возвращает кэшированный список доступных эмбеддингов из API."""
    return fetch_embeddings_from_api(force_refresh=force_refresh)


def is_negative_embedding(name: str, shape: int = 0) -> bool:
    """Эвристическая проверка: является ли эмбеддинг «негативным» (для негативного промпта).

    Учитывает имя (negative/bad/ugly/easynegative и т.п.) и большую форму (>16 токенов —
    типично для «собранных» негативных эмбеддингов вроде easynegative/bad_prompt).
    """
    lowered = name.lower()
    keywords = ("negative", "bad-", "bad_", "badprompt", "bad_prompt", "ugly", "nasty",
                "deformed", "unrealistic", "lowquality", "worst")
    if any(k in lowered for k in keywords):
        return True
    # Эмбеддинги с большой формой обычно описывают «всё плохое» разом
    if shape and shape > 16:
        return True
    return False


def filter_negative_embeddings(embeddings: Dict[str, dict]) -> List[str]:
    """Возвращает имена эмбеддингов, которые выглядят как негативные."""
    negatives = []
    for name, info in embeddings.items():
        if is_negative_embedding(name, int(info.get("shape") or 0)):
            negatives.append(name)
    return sorted(negatives, key=str.lower)


def add_negative_embedding_to_prompt(user_negative: str, embedding_name: str) -> str:
    """Добавляет тег эмбеддинга в пользовательский негативный промпт (без дублей)."""
    user_negative = (user_negative or "").strip()
    tags = [t.strip() for t in user_negative.split(",") if t.strip()]
    if embedding_name not in tags:
        tags.append(embedding_name)
    return ", ".join(tags)


def remove_negative_embedding_from_prompt(user_negative: str, embedding_name: str) -> str:
    """Убирает тег эмбеддинга из пользовательского негативного промпта."""
    user_negative = (user_negative or "").strip()
    tags = [t.strip() for t in user_negative.split(",") if t.strip() and t.strip() != embedding_name]
    return ", ".join(tags)