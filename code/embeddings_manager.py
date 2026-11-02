import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

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
    """Проверяет, какие рекомендованные эмбеддинги уже установлены (файлы в папке embeddings).

    Не создаёт папки и не падает, если путь недоступен (например, не существует диск N:)."""
    result = {}
    for name, info in RECOMMENDED_EMBEDDINGS.items():
        try:
            result[name] = os.path.exists(os.path.join(EMBEDDINGS_PATH, info["filename"]))
        except OSError:
            result[name] = False
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


def _merge_unique_tags(*parts: str) -> str:
    """Склеивает фрагменты негативного промпта через запятую, убирая повторяющиеся теги (без учёта регистра)."""
    seen = set()
    tags = []
    for part in parts:
        for tag in (part or "").replace("\n", " ").split(","):
            tag = " ".join(tag.split())
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                tags.append(tag)
    return ", ".join(tags)


def get_negative_prompt_with_embeddings(user_negative: str = "", use_advanced: bool = True,
                                        embedding_tags=None) -> str:
    """Формирует финальный негативный промпт.

    embedding_tags — список имён эмбеддингов (из API/выбора пользователя). Если None, используется прежняя
    проверка по файлам в папке embeddings (запасной вариант, когда список из API получить не удалось).
    """
    if embedding_tags is None:
        installed = check_embeddings_installed()
        embedding_tags = [name for name in ("bad-hands-5", "badhandv4", "easynegative") if installed.get(name)]

    return _merge_unique_tags(
        ADVANCED_NEGATIVE_PROMPT if use_advanced else "",
        user_negative or "",
        ", ".join(embedding_tags),
    )
