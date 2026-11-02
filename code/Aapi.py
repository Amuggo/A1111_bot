import base64
import io
import re
import json
import random
import time
import threading
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Callable, Tuple, Any

# Импорт из settings и embeddings_manager
from settings import get_user_negative_prompt, get_diversity_mode, get_use_advanced_negative, get_prompt_processing_enabled, get_active_lora_tags, resolve_model, get_user_settings, effective_negative_embeddings
from embeddings_manager import get_negative_prompt_with_embeddings, _merge_unique_tags
from prompt_processor import fast_process

# ===== КОНФИГУРАЦИЯ =====
try:
    import app_config as config

    SD_BASE_URL = config.SD_BASE_URL
except ImportError:
    import os

    SD_BASE_URL = os.environ.get("SD_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

# Настройки ADetailer (расширение WebUI) — модуль может отсутствовать при быстрой сборке
try:
    from adetailer_settings import build_adetailer_args, get_adetailer_enabled
except ImportError:  # pragma: no cover
    build_adetailer_args = None
    get_adetailer_enabled = None


# ===== БАЗОВЫЕ HTTP-ХЕЛПЕРЫ =====
def _get_json(endpoint: str, timeout: int = 10):
    """GET-запрос к API с разбором JSON"""
    request = urllib.request.Request(
        f'{SD_BASE_URL}/{endpoint.lstrip("/")}',
        headers={'Content-Type': 'application/json'},
        method='GET'
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def _post_json(endpoint: str, payload: dict, timeout: int = 30):
    """POST-запрос к API с JSON-телом"""
    request = urllib.request.Request(
        f'{SD_BASE_URL}/{endpoint.lstrip("/")}',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(payload).encode('utf-8'),
        method='POST'
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


# ===== БАЗОВАЯ ФУНКЦИЯ ЗАПРОСА К API =====
def call_api(api_endpoint: str, **payload) -> dict:
    """Универсальная функция вызова API"""
    data = json.dumps(payload).encode('utf-8')
    full_url = f'{SD_BASE_URL}/{api_endpoint}'
    
    request = urllib.request.Request(
        full_url,
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        raise Exception(f"HTTP {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise Exception(f"URL Error: {e.reason}")


# ===== РАЗМЕР ВЫХОДА ДЛЯ img2img / inpaint =====
MAX_EDIT_SIDE = 1280  # максимальная сторона при редактировании (защита от нехватки видеопамяти)


def fit_dims_to_image(image_bytes: bytes, fallback_width: int, fallback_height: int) -> Tuple[int, int]:
    """
    Размер результата для img2img/inpaint: пропорции исходной картинки (кратно 8, не больше MAX_EDIT_SIDE).
    Раньше брался размер из настроек, и картинка с другими пропорциями растягивалась/сжималась.
    """
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Exception:
        return fallback_width, fallback_height
    if width <= 0 or height <= 0:
        return fallback_width, fallback_height
    scale = min(1.0, MAX_EDIT_SIDE / max(width, height))
    width, height = int(width * scale), int(height * scale)
    width, height = max(64, (width // 8) * 8), max(64, (height // 8) * 8)
    return width, height


# ===== ВЫБОР МОДЕЛИ (CHECKPOINT) =====
def apply_model_override(payload: dict, settings: dict) -> dict:
    """
    Добавляет в payload модель пользователя (или модель по умолчанию) через override_settings.
    override_settings_restore_afterwards=False — модель остаётся загруженной, поэтому
    повторные генерации с той же моделью не перезагружают её.
    """
    try:
        model = resolve_model(settings)
    except Exception as e:  # выбор модели не должен ломать генерацию
        print(f"⚠️ Не удалось определить модель: {e}")
        return payload
    if model:
        override = dict(payload.get("override_settings") or {})
        override["sd_model_checkpoint"] = model
        payload["override_settings"] = override
        payload["override_settings_restore_afterwards"] = False
    return payload


# ===== ФУНКЦИЯ ДЛЯ ОТСЛЕЖИВАНИЯ ПРОГРЕССА =====
def get_progress() -> dict:
    """Возвращает текущий прогресс генерации"""
    try:
        request = urllib.request.Request(
            f'{SD_BASE_URL}/sdapi/v1/progress',
            headers={'Content-Type': 'application/json'},
            method='GET'
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return {"progress": 0, "eta_relative": 0, "state": {}}


def generate_with_progress(
    payload: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    endpoint: str = "sdapi/v1/txt2img"
) -> dict:
    """
    Отправляет запрос генерации и пока он выполняется опрашивает прогресс.
    endpoint: "sdapi/v1/txt2img" (по умолчанию) или "sdapi/v1/img2img" — для запросов с init_images/mask.
    """
    data = json.dumps(payload).encode('utf-8')
    full_url = f'{SD_BASE_URL}/{endpoint.lstrip("/")}'

    request = urllib.request.Request(
        full_url,
        headers={'Content-Type': 'application/json'},
        data=data,
        method='POST'
    )

    result_container = {}
    error_container = {}

    def worker():
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result_container['response'] = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            # ← ключевая строка: сохраняем ТЕЛО ошибки
            body = e.read().decode('utf-8', errors='replace')
            error_container['error'] = f"HTTP {e.code}: {body}"
        except Exception as e:
            error_container['error'] = str(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_progress = -1
    while thread.is_alive():
        try:
            progress_data = get_progress()
            progress = progress_data.get('progress', 0)
            eta = progress_data.get('eta_relative', 0)
            if int(progress * 100) > last_progress:
                last_progress = int(progress * 100)
                if progress_callback:
                    progress_callback(progress, eta)
        except Exception:
            pass
        time.sleep(0.5)

    thread.join()

    if 'error' in error_container:
        raise Exception(error_container['error'])
    if 'response' not in result_container:
        raise Exception("WebUI не вернул ответ")

    return result_container['response']


# ===== ПРОВЕРКА ДОСТУПНОСТИ =====
def check_api_available() -> bool:
    """Проверяет доступность API сервера (лёгкий GET-запрос; раньше здесь запускалась пробная генерация)"""
    try:
        _get_json("sdapi/v1/options", timeout=10)
        return True
    except Exception:
        return False


# ===== ADETAILER: ПАРСИНГ ОТВЕТА РАСШИРЕНИЯ =====
AD_START_TAG = "<ADetailer:"
AD_END_TAG = ">"


def parse_adetailer_info(info_text: str) -> List[dict]:
    """
    Разбирает поле info ответа API и возвращает список результатов ADetailer.

    Расширение помечает каждый обработанный фрагмент тегами вида:
        <ADetailer model: face_yolov8n.pt>
        <ADetailer bbox1: 512x512 offset (120, 240)>
        <ADetailer mask1 top1 area: 12.3%>
    Если тегов нет — детектор ничего не нашёл или расширение отключено.
    """
    if not info_text or AD_START_TAG not in info_text:
        return []

    results: List[dict] = []
    for raw_tag in re.findall(r"<ADetailer:.*?>", info_text, re.DOTALL):
        body = raw_tag[len(AD_START_TAG):-len(AD_END_TAG)].strip()
        entry: Dict[str, Any] = {}
        for line in body.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key in ("confidence", "area"):
                try:
                    entry[key] = float(value.rstrip("%"))
                except ValueError:
                    entry[key] = value
            elif key in ("inpaint_width", "inpaint_height", "steps"):
                try:
                    entry[key] = int(float(value))
                except ValueError:
                    entry[key] = value
            else:
                entry[key] = value
        if entry:
            results.append(entry)

    return results


def get_adetailer_status(api_response: dict) -> Dict[str, Any]:
    """
    Возвращает сводку по работе ADetailer для конкретной генерации.

    Использует images[0].info (поле появляется в новых версиях WebUI) либо
    общий info ответа.
    """
    info_text = ""
    images_meta = api_response.get("images") or []
    if images_meta and isinstance(images_meta[0], dict):
        info_text = images_meta[0].get("info", "") or ""
    if not info_text:
        info_text = api_response.get("info", "") or ""

    detections = parse_adetailer_info(info_text)
    return {
        "detected": len(detections),
        "detections": detections,
        "models": sorted({d.get("model", "?") for d in detections}),
        "raw_info": info_text,
    }


# ===== ADETAILER: АВТОДЕТЕКЦИЯ И НАСТРОЙКИ =====
_adetailer_available_cache: Optional[bool] = None


def check_adetailer_available(force_refresh: bool = False) -> bool:
    """Проверяет, что расширение ADetailer установлено и доступно через API"""
    global _adetailer_available_cache
    if _adetailer_available_cache is not None and not force_refresh:
        return _adetailer_available_cache

    available = False
    try:
        scripts = _get_json("sdapi/v1/scripts", timeout=10)
        for tab in scripts.values():
            for script_name, _label in tab.get("txt2img", []) + tab.get("img2img", []):
                if str(script_name).lower() == "adetailer":
                    available = True
    except Exception:
        available = False

    if not available:
        # Запасная проверка: попытка прочитать список детекторов расширения
        try:
            _get_json("adetailer/models", timeout=5)
            available = True
        except Exception:
            pass

    _adetailer_available_cache = available
    return available


def get_adetailer_models(force_refresh: bool = False) -> List[Dict[str, str]]:
    """
    Получает список моделей детекторов ADetailer с сервера.

    Использует эндпоинт /adetailer/models; при недоступности возвращает
    встроенный список из adetailer_settings.DEFAULT_ADETAILER_MODELS.
    """
    fallback: List[Dict[str, str]] = []
    try:
        from adetailer_settings import DEFAULT_ADETAILER_MODELS

        fallback = [
            {"id": name, "name": display, "tag": name}
            for name, display in DEFAULT_ADETAILER_MODELS
        ]
    except Exception:
        fallback = []

    if getattr(get_adetailer_models, "_cache", None) and not force_refresh:
        return get_adetailer_models._cache

    models: List[Dict[str, str]] = []
    try:
        data = _get_json("adetailer/models", timeout=10)
        raw_list = data.get("models") if isinstance(data, dict) else data
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, str):
                    models.append({"id": item, "name": item, "tag": item})
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("id") or item.get("model")
                    if name:
                        models.append({
                            "id": name,
                            "name": item.get("label") or item.get("display_name") or name,
                            "tag": name,
                        })
    except Exception as e:
        print(f"⚠️ Не удалось получить список моделей ADetailer с сервера: {e}")

    result = models or fallback
    if result:
        result = [{"id": "None", "name": "❌ Выключен", "tag": "None"}] + result
    get_adetailer_models._cache = result
    return result


def build_adetailer_script_args(user_id: str, overrides: Optional[dict] = None) -> Optional[dict]:
    """
    Собирает блок alwayson_scripts['ADetailer'] из настроек пользователя.

    Возвращает None, если ADetailer выключен у пользователя или модуль настроек
    недоступен — тогда payload остаётся совместимым со старым поведением.
    """
    if build_adetailer_args is None or get_adetailer_enabled is None:
        return None
    try:
        if not get_adetailer_enabled(user_id):
            return None
        args = build_adetailer_args(user_id, overrides)
    except Exception as e:
        print(f"⚠️ Ошибка сборки настроек ADetailer: {e}")
        return None
    if not args or args.get("ad_model", "None") == "None":
        return None
    return {"args": [args]}


# ===== ОБРАБОТКА ПРОМПТА ЧЕРЕЗ НЕЙРОСЕТЬ =====
def preprocess_prompt(user_id: str, raw_prompt: str, status_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Обрабатывает промпт через локальную нейросеть (если включено в настройках)
    """
    # Проверяем, включена ли обработка
    if not get_prompt_processing_enabled(user_id):
        if status_callback:
            status_callback("⚡ Обработка промпта выключена, использую оригинал")
        return raw_prompt
    
    if status_callback:
        status_callback("🔄 Обработка промпта через нейросеть...")
    
    try:
        processed = fast_process(raw_prompt)
        if status_callback:
            status_callback(f"✅ Промпт обработан")
        return processed
    except Exception as e:
        print(f"Ошибка обработки промпта: {e}")
        if status_callback:
            status_callback("⚠️ Ошибка обработки, использую оригинальный промпт")
        return raw_prompt

# ===== ПОЛУЧЕНИЕ СПИСКА СЭМПЛЕРОВ =====
def get_samplers_list() -> List[str]:
    """Получает список доступных сэмплеров из API Stable Diffusion"""
    try:
        url = f"{SD_BASE_URL}/sdapi/v1/samplers"
        request = urllib.request.Request(url, headers={'Content-Type': 'application/json'}, method='GET')
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if isinstance(data, list):
                # Извлекаем имена сэмплеров
                samplers = []
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        samplers.append(item['name'])
                return samplers
            return []
    except Exception as e:
        print(f"Ошибка получения списка сэмплеров: {e}")
        return []
    
# ===== УЛУЧШЕНИЕ ПРОМПТА ДЛЯ РАЗНООБРАЗИЯ =====
def enhance_prompt_for_diversity(prompt: str, user_id: str, user_settings: dict) -> str:
    """Улучшает промпт для получения разных лиц"""
    diversity_mode = get_diversity_mode(user_id)
    
    if diversity_mode == "off":
        return prompt
    
    enhanced = prompt
    
    has_person = re.search(r'\b(girl|woman|man|boy|person|1girl|1boy)\b', enhanced, re.IGNORECASE)
    
    if has_person:
        if diversity_mode == "high":
            diversity_phrases = [
                "different face", "unique features", "distinct appearance",
                "varied look", "random person", "stranger face"
            ]
            selected = random.sample(diversity_phrases, k=min(3, len(diversity_phrases)))
            enhanced = f"{enhanced}, {', '.join(selected)}"
            
            if user_settings.get("cfg_scale", 7.0) > 6:
                user_settings["cfg_scale"] = 5.5
                
        elif diversity_mode == "auto":
            diversity_phrases = ["different face", "unique features", "distinct appearance"]
            selected = random.sample(diversity_phrases, k=random.randint(1, 2))
            enhanced = f"{enhanced}, {', '.join(selected)}"
    
    quality_tags = ["masterpiece", "best quality"]
    has_quality = any(tag in enhanced.lower() for tag in quality_tags)
    if not has_quality and diversity_mode != "off":
        enhanced = f"{enhanced}, {', '.join(quality_tags)}"
    
    return enhanced


def build_negative_prompt(user_id: str, custom_negative: str = None) -> str:
    """Формирует финальный негативный промпт: базовый текст + анти-мутационные фразы + негативные эмбеддинги."""
    use_advanced = get_use_advanced_negative(user_id)

    if custom_negative:
        user_negative = custom_negative
    else:
        user_negative = get_user_negative_prompt(user_id)

    # Универсальные фразы против мутаций и деформаций (если их ещё нет)
    extra_neg = "bad anatomy, extra limbs, deformed, mutated, disfigured, malformed, gross proportions, missing limbs, fused body, bad hands, bad fingers"
    if extra_neg not in user_negative:
        user_negative = f"{user_negative}, {extra_neg}"

    # Эмбеддинги: выбор пользователя (меню «Негативный промпт → Эмбеддинги») либо рекомендованные из WebUI
    embedding_tags = effective_negative_embeddings(get_user_settings(user_id))
    return get_negative_prompt_with_embeddings(user_negative, use_advanced, embedding_tags)


# Отчёт ADetailer последней генерации в ТЕКУЩЕМ потоке (раньше это был общий атрибут функции,
# который параллельные генерации перезаписывали друг у друга).
_tls = threading.local()
MAX_SEED = 2**32 - 1  # WebUI принимает seed только до 2^32-1


def get_last_adetailer_status() -> Dict[str, Any]:
    return getattr(_tls, "last_adetailer_status", {"detected": 0, "detections": [], "models": []})


def generate_txt2img(
    prompt: str, 
    user_id: str, 
    user_settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    fixed_seed: int = None,
    adetailer_overrides: Optional[dict] = None,
    exact_prompt: Optional[str] = None,
    exact_negative: Optional[str] = None
) -> Tuple[List[io.BytesIO], int, str, str]:
    """
    Генерация изображений
    Возвращает: (images, seed, original_prompt, processed_prompt)

    exact_prompt / exact_negative — «точный повтор»: промпты берутся как есть (без перевода через Ollama,
    без повторного добавления LoRA и случайных фраз разнообразия). Нужны, чтобы повтор с тем же seed
    действительно давал ту же картинку.

    Итоговые промпты, ушедшие в WebUI, сохраняются в атрибутах первого изображения
    (image.final_prompt, image.final_negative); отчёт ADetailer — через get_last_adetailer_status()
    (хранится отдельно для каждого потока, поэтому параллельные генерации не затирают друг друга).
    """

    # Сохраняем оригинальный промпт
    original_prompt = prompt

    if exact_prompt is not None:
        processed_prompt = exact_prompt
        enhanced_prompt = exact_prompt
        if status_callback:
            status_callback("🎨 Генерация изображения...")
    else:
        # Обрабатываем промпт через нейросеть (если включено)
        processed_prompt = preprocess_prompt(user_id, prompt, status_callback)
        # Добавляем активные LoRA в промпт
        lora_tags = get_active_lora_tags(user_id)
        if lora_tags:
            processed_prompt = f"{processed_prompt}, {lora_tags}"
        if status_callback:
            status_callback("🎨 Генерация изображения...")
        # Используем обработанный промпт для дальнейшей обработки
        enhanced_prompt = enhance_prompt_for_diversity(processed_prompt, user_id, user_settings)

    # Используем фиксированный seed если передан, иначе случайный
    if fixed_seed is not None:
        seed = fixed_seed
    else:
        seed = random.randint(1, MAX_SEED)

    negative_prompt = exact_negative if exact_negative is not None else build_negative_prompt(user_id)
    
    # Получаем сэмплер из настроек, если он там есть
    sampler = user_settings.get("sampler_name", "Euler a")
    
    payload = {
        "prompt": enhanced_prompt,
        "negative_prompt": negative_prompt,
        "steps": user_settings.get("steps", 20),
        "width": user_settings.get("width", 512),
        "height": user_settings.get("height", 768),
        "cfg_scale": user_settings.get("cfg_scale", 7.0),
        "sampler_name": sampler,  # Используем сэмплер из настроек
        "seed": seed,
        "batch_size": user_settings.get("batch_size", 1),
        "n_iter": user_settings.get("n_iter", 1),
    }

    # ===== ADETAILER: автодетекция и перерисовка лиц/рук =====
    ad_block = build_adetailer_script_args(user_id, adetailer_overrides)
    if ad_block:
        payload["alwayson_scripts"] = dict(payload.get("alwayson_scripts") or {})
        payload["alwayson_scripts"]["ADetailer"] = ad_block
        model_name = ad_block["args"][0].get("ad_model", "?")
        if status_callback:
            status_callback(f"🕵️ ADetailer: модель `{model_name}`")
        print(f"🕵️ ADetailer txt2img: model={model_name}")

    apply_model_override(payload, user_settings)
    response = generate_with_progress(payload, progress_callback)

    # Сохраняем отчёт ADetailer, чтобы бот мог показать его в капшне
    try:
        _tls.last_adetailer_status = get_adetailer_status(response)
    except Exception as e:  # pragma: no cover - диагностика не должна ломать генерацию
        print(f"⚠️ Не удалось разобрать отчёт ADetailer: {e}")
        _tls.last_adetailer_status = {"detected": 0, "detections": [], "models": []}

    images = []
    for index, img_base64 in enumerate(response.get('images', [])):
        image_bytes = base64.b64decode(img_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = f"generation_{index}.png"
        image_file.final_prompt = enhanced_prompt
        image_file.final_negative = negative_prompt
        images.append(image_file)

    return images, seed, original_prompt, processed_prompt


# ===== ГЕНЕРАЦИЯ С КАСТОМНЫМ НЕГАТИВОМ =====
def generate_txt2img_advanced(
    prompt: str,
    negative_prompt: str,
    user_settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    user_id: str = "default",
    adetailer_overrides: Optional[dict] = None
) -> Tuple[List[io.BytesIO], int]:
    """Генерация с кастомным негативным промптом"""

    seed = random.randint(1, MAX_SEED)

    # Негативные эмбеддинги пользователя добавляются и к «ручному» негативу (без дублей)
    embedding_tags = effective_negative_embeddings(user_settings)
    if embedding_tags:
        negative_prompt = _merge_unique_tags(negative_prompt, ", ".join(embedding_tags))

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": user_settings.get("steps", 20),
        "width": user_settings.get("width", 512),
        "height": user_settings.get("height", 768),
        "cfg_scale": user_settings.get("cfg_scale", 7.0),
        "sampler_name": user_settings.get("sampler_name", "Euler a"),
        "seed": seed,
        "batch_size": user_settings.get("batch_size", 1),
        "n_iter": user_settings.get("n_iter", 1),
    }

    # ===== ADETAILER: автодетекция и перерисовка лиц/рук =====
    ad_block = build_adetailer_script_args(user_id, adetailer_overrides)
    if ad_block:
        payload["alwayson_scripts"] = dict(payload.get("alwayson_scripts") or {})
        payload["alwayson_scripts"]["ADetailer"] = ad_block
        print(f"🕵️ ADetailer txt2img (advanced): model={ad_block['args'][0].get('ad_model', '?')}")

    apply_model_override(payload, user_settings)
    response = generate_with_progress(payload, progress_callback)

    images = []
    for index, img_base64 in enumerate(response.get('images', [])):
        image_bytes = base64.b64decode(img_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = f"generation_{index}.png"
        image_file.final_prompt = prompt
        image_file.final_negative = negative_prompt
        images.append(image_file)

    return images, seed


# ===== REACTOR ФУНКЦИИ ДЛЯ РАБОТЫ С ЛИЦАМИ =====

def swap_face_with_reactor(
    source_image: Optional[io.BytesIO],
    target_image: io.BytesIO,
    face_model: str = "None",
    upscaler: str = "4x_Ultrasharp",
    scale: int = 2,
    face_restorer: str = "CodeFormer",
    restorer_visibility: float = 0.8,
    codeformer_weight: float = 0.5,
    gender_source: int = 0,
    gender_target: int = 0,
    source_faces_index: Optional[list] = None,
    face_index: Optional[list] = None
) -> io.BytesIO:
    """Улучшенная замена лица с расширенными параметрами"""
    source_faces_index = [0] if source_faces_index is None else source_faces_index
    face_index = [0] if face_index is None else face_index

    target_image.seek(0)
    target_base64 = base64.b64encode(target_image.read()).decode('utf-8')
    
    payload = {
        "target_image": target_base64,
        "face_model": face_model,
        "upscaler": upscaler,
        "scale": scale,
        "face_restorer": face_restorer,
        "restorer_visibility": restorer_visibility,
        "codeformer_weight": codeformer_weight,
        "gender_source": gender_source,
        "gender_target": gender_target,
        "source_faces_index": source_faces_index,
        "face_index": face_index
    }
    
    if source_image is not None:
        source_image.seek(0)
        source_base64 = base64.b64encode(source_image.read()).decode('utf-8')
        payload["source_image"] = source_base64
    
    # Пробуем разные возможные эндпоинты
    endpoints = [
        f'{SD_BASE_URL}/reactor/image',
        f'{SD_BASE_URL}/reactor/swap-face',
        f'{SD_BASE_URL}/sdapi/v1/reactor/image'
    ]
    
    last_error = None
    for endpoint in endpoints:
        try:
            request = urllib.request.Request(
                endpoint,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload).encode('utf-8'),
                method='POST'
            )
            
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            # Проверяем формат ответа
            if 'image' in result:
                image_bytes = base64.b64decode(result['image'])
            elif 'images' in result and result['images']:
                image_bytes = base64.b64decode(result['images'][0])
            else:
                raise Exception("Неизвестный формат ответа от Reactor")
            
            swapped_image = io.BytesIO(image_bytes)
            swapped_image.name = "swapped_face.png"
            return swapped_image
            
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            if e.code != 404:  # 404 — это просто «такого эндпоинта нет», настоящую причину не затираем
                last_error = f"HTTP {e.code}: {error_body}"
            elif last_error is None:
                last_error = f"HTTP 404: эндпоинт {endpoint} не найден (установлен ли ReActor?)"
            continue
        except Exception as e:
            last_error = str(e)
            continue

    raise Exception(f"Не удалось заменить лицо. Последняя ошибка: {last_error}")


def get_reactor_models() -> Dict[str, List[str]]:
    """Получает список доступных моделей из Reactor"""
    models_info = {
        "facemodels": [],
        "upscalers": []
    }
    
    # Получаем модели лиц
    endpoints_face = [
        f'{SD_BASE_URL}/reactor/facemodels',
        f'{SD_BASE_URL}/reactor/models',
        f'{SD_BASE_URL}/sdapi/v1/reactor/facemodels'
    ]
    
    for endpoint in endpoints_face:
        try:
            request = urllib.request.Request(
                endpoint,
                headers={'Content-Type': 'application/json'},
                method='GET'
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if isinstance(data, list):
                    models_info["facemodels"] = data
                elif isinstance(data, dict) and 'models' in data:
                    models_info["facemodels"] = data['models']
                elif isinstance(data, dict) and 'facemodels' in data:
                    models_info["facemodels"] = data['facemodels']
                break
        except Exception:
            continue
    
    # Получаем апскейлеры
    endpoints_upscalers = [
        f'{SD_BASE_URL}/reactor/upscalers',
        f'{SD_BASE_URL}/sdapi/v1/reactor/upscalers'
    ]
    
    for endpoint in endpoints_upscalers:
        try:
            request = urllib.request.Request(
                endpoint,
                headers={'Content-Type': 'application/json'},
                method='GET'
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if isinstance(data, list):
                    models_info["upscalers"] = data
                elif isinstance(data, dict) and 'upscalers' in data:
                    models_info["upscalers"] = data['upscalers']
                break
        except Exception:
            continue
    
    return models_info


def check_reactor_available() -> bool:
    """Проверяет доступность Reactor API"""
    endpoints = [
        f'{SD_BASE_URL}/reactor/models',
        f'{SD_BASE_URL}/reactor/facemodels',
        f'{SD_BASE_URL}/reactor/image'
    ]
    
    for endpoint in endpoints:
        try:
            request = urllib.request.Request(
                endpoint,
                headers={'Content-Type': 'application/json'},
                method='GET'
            )
            urllib.request.urlopen(request, timeout=5)
            return True
        except Exception:
            continue
    
    return False


# ===== IMG2IMG =====
def generate_img2img(
    prompt: str,
    init_image: io.BytesIO,
    user_settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    user_id: str = "default",
    adetailer_overrides: Optional[dict] = None
) -> List[io.BytesIO]:
    """Генерация на основе исходного изображения"""
    init_image.seek(0)
    init_bytes = init_image.read()
    init_base64 = base64.b64encode(init_bytes).decode('utf-8')
    width, height = fit_dims_to_image(init_bytes, user_settings.get("width", 512), user_settings.get("height", 512))

    payload = {
        "init_images": [init_base64],
        "prompt": prompt,
        "negative_prompt": build_negative_prompt(user_id),
        "denoising_strength": user_settings.get("denoising_strength", 0.75),
        "steps": user_settings.get("steps", 20),
        "width": width,
        "height": height,
        "cfg_scale": user_settings.get("cfg_scale", 7.0),
        "sampler_name": user_settings.get("sampler_name", "Euler a"),
        "seed": random.randint(1, MAX_SEED),
        "batch_size": user_settings.get("batch_size", 1),
    }

    # ===== ADETAILER: перерисовка лиц/рук поверх img2img =====
    ad_block = build_adetailer_script_args(user_id, adetailer_overrides)
    if ad_block:
        payload["alwayson_scripts"] = dict(payload.get("alwayson_scripts") or {})
        payload["alwayson_scripts"]["ADetailer"] = ad_block
        print(f"🕵️ ADetailer img2img: model={ad_block['args'][0].get('ad_model', '?')}")

    apply_model_override(payload, user_settings)
    response = generate_with_progress(payload, progress_callback, endpoint="sdapi/v1/img2img")

    images = []
    for index, img_base64 in enumerate(response.get('images', [])):
        image_bytes = base64.b64decode(img_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = f"img2img_result_{index}.png"
        images.append(image_file)

    return images


# ===== АПСКЕЙЛ =====
def upscale_image(
    image: io.BytesIO,
    upscale_factor: float = 2,
    upscaler_name: str = "ESRGAN_4x",
    face_restoration: float = 0.5
) -> io.BytesIO:
    """Увеличивает разрешение изображения"""
    image.seek(0)
    image_base64 = base64.b64encode(image.read()).decode('utf-8')
    
    payload = {
        "image": image_base64,
        "upscaling_resize": upscale_factor,
        "upscaler_1": upscaler_name,
        "gfpgan_visibility": face_restoration,
        "codeformer_visibility": face_restoration,
    }

    response = call_api("sdapi/v1/extra-single-image", **payload)
    
    image_bytes = base64.b64decode(response.get('image', ''))
    image_file = io.BytesIO(image_bytes)
    image_file.name = "upscaled.png"
    
    return image_file


# ===== INPAINT =====
def generate_inpaint(
    prompt: str,
    init_image: io.BytesIO,
    mask: io.BytesIO,
    settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    user_id: str = "default"
) -> io.BytesIO:
    """
    Генерация с маской (inpainting)
    mask – изображение, где белое (255) – область для перерисовки, чёрное (0) – сохраняемая область.

    Запрос уходит на img2img (раньше по ошибке — на txt2img, из-за чего исходная картинка и маска игнорировались).
    """
    init_image.seek(0)
    init_bytes = init_image.read()
    init_base64 = base64.b64encode(init_bytes).decode('utf-8')
    mask.seek(0)
    mask_base64 = base64.b64encode(mask.read()).decode('utf-8')
    width, height = fit_dims_to_image(init_bytes, settings.get("width", 512), settings.get("height", 512))

    payload = {
        "init_images": [init_base64],
        "mask": mask_base64,
        "prompt": prompt,
        "negative_prompt": build_negative_prompt(user_id),
        "denoising_strength": settings.get("denoising_strength", 0.75),
        "steps": settings.get("steps", 20),
        "width": width,
        "height": height,
        "cfg_scale": settings.get("cfg_scale", 7.0),
        "sampler_name": settings.get("sampler_name", "Euler a"),
        "seed": random.randint(1, MAX_SEED),
        "batch_size": 1,
        "inpainting_fill": 1,  # 1 = original, 0 = fill with color
        "mask_blur": 4,
    }

    apply_model_override(payload, settings)
    response = generate_with_progress(payload, progress_callback, endpoint="sdapi/v1/img2img")
    images = response.get('images', [])
    if not images:
        raise Exception("API не вернул изображений")
    img_base64 = images[0]
    image_bytes = base64.b64decode(img_base64)
    image_file = io.BytesIO(image_bytes)
    image_file.name = "inpainted.png"
    return image_file


def get_loras_list() -> List[Dict[str, str]]:
    """Получает список доступных LoRA из API Stable Diffusion"""
    try:
        url = f"{SD_BASE_URL}/sdapi/v1/loras"
        request = urllib.request.Request(url, headers={'Content-Type': 'application/json'}, method='GET')
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            # Ожидаем список объектов с полями name, filename и т.д.
            if isinstance(data, list):
                result = []
                for item in data:
                    # Попробуем извлечь имя
                    if isinstance(item, dict):
                        name = item.get('name') or item.get('filename') or item.get('alias')
                        if name:
                            # Создаём понятный тег
                            # Имя обычно заканчивается на .safetensors или .pt, убираем
                            clean_name = name.replace('.safetensors', '').replace('.pt', '').replace('.bin', '')
                            tag = f"<lora:{clean_name}:1>"
                            result.append({
                                "id": clean_name,
                                "name": clean_name.replace('_', ' ').title(),  # человеческий вид
                                "tag": tag,
                                "filename": name
                            })
                return result
            else:
                return []
    except Exception as e:
        print(f"Ошибка получения списка LoRA: {e}")
        return []

# ===== СОВМЕСТИМОСТЬ СО СТАРЫМ КОДОМ =====
def generate_image(
    prompt: str, 
    user_settings: dict, 
    user_id: str = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    fixed_seed: int = None,
    exact_prompt: Optional[str] = None,
    exact_negative: Optional[str] = None
) -> Tuple[io.BytesIO, int, str, str]:
    """Генерация с возвратом изображения, seed, оригинального и обработанного промпта"""
    if user_id is None:
        user_id = "default"

    images, seed, original_prompt, processed_prompt = generate_txt2img(
        prompt, user_id, user_settings, progress_callback, status_callback, fixed_seed,
        exact_prompt=exact_prompt, exact_negative=exact_negative
    )

    if not images:
        # Раньше возвращался пустой BytesIO, и бот падал позже с невнятной ошибкой отправки
        raise Exception("WebUI не вернул изображений")
    return images[0], seed, original_prompt, processed_prompt
