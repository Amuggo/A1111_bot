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


def get_negative_prompt_with_embeddings(user_negative: str = "", use_advanced: bool = True) -> str:
    """Формирует финальный негативный промпт с учётом установленных эмбеддингов"""
    negative_parts = []
    
    # Добавляем расширенный негативный промпт если включён
    if use_advanced:
        negative_parts.append(ADVANCED_NEGATIVE_PROMPT)
    
    # Добавляем пользовательский негатив если есть
    if user_negative and user_negative.strip():
        negative_parts.append(user_negative)
    
    # Добавляем эмбеддинги если они установлены
    installed = check_embeddings_installed()
    embedding_tags = []
    
    if installed.get("bad-hands-5"):
        embedding_tags.append("bad-hands-5")
    if installed.get("badhandv4"):
        embedding_tags.append("badhandv4")
    if installed.get("easynegative"):
        embedding_tags.append("easynegative")
    
    if embedding_tags:
        negative_parts.append(", ".join(embedding_tags))
    
    return ", ".join(negative_parts)