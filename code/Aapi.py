import asyncio
import base64
import contextvars
import io
import logging
import re
import json
import os
import random
import time
from typing import Awaitable, Dict, List, Optional, Callable, Tuple, Any

import aiohttp

from http_client import SDError, get_session, make_timeout, sd_gate, sd_get_json, sd_post_json

# Импорт из settings и embeddings_manager
from settings import get_user_negative_prompt, get_diversity_mode, get_use_advanced_negative, get_prompt_processing_enabled, get_active_lora_tags, resolve_model, get_user_settings, effective_negative_embeddings, hires_plan
from embeddings_manager import get_negative_prompt_with_embeddings, _merge_unique_tags
from prompt_processor import fast_process

log = logging.getLogger(__name__)

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


# ===== BASE64: чистые утилиты (CPU-лёгкие) и обёртки для больших изображений =====
def decode_base64_to_bytes(data: str) -> bytes:
    """Чистая синхронная утилита: base64-строка → bytes (картинки из ответов WebUI)."""
    return base64.b64decode(data)


def encode_bytes_to_base64(data: bytes) -> str:
    """Чистая синхронная утилита: bytes → base64-строка."""
    return base64.b64encode(data).decode('utf-8')


async def _b64encode(data: bytes) -> str:
    """Кодирование в base64 в потоке: для больших изображений это заметная CPU-работа, цикл событий не блокируем."""
    return await asyncio.to_thread(encode_bytes_to_base64, data)


# ===== БАЗОВЫЕ HTTP-ХЕЛПЕРЫ =====
async def _get_json(endpoint: str, timeout: int = 10):
    """GET-запрос к API с разбором JSON"""
    return await sd_get_json(endpoint, timeout=timeout)


async def _post_json(endpoint: str, payload: dict, timeout: int = 30):
    """POST-запрос к API с JSON-телом"""
    return await sd_post_json(endpoint, payload, timeout=timeout)


# ===== БАЗОВАЯ ФУНКЦИЯ ЗАПРОСА К API =====
async def call_api(api_endpoint: str, **payload) -> dict:
    """Универсальная функция вызова API (долгий запрос: без общего таймаута, пауза без данных ≤ HTTP_SOCK_READ_TIMEOUT)"""
    return await sd_post_json(api_endpoint, payload)


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


# ===== HIRES. FIX =====
def apply_hires(payload: dict, settings: dict) -> dict:
    """Включает двухпроходную генерацию (hires fix), если она включена у пользователя и укладывается в лимит VRAM."""
    plan = hires_plan(settings)
    if plan:
        payload.update({
            "enable_hr": True,
            "hr_scale": plan["scale"],
            "hr_upscaler": plan["upscaler"],
            "hr_second_pass_steps": plan["steps"],
            "denoising_strength": plan["denoise"],  # в txt2img это денойз именно второго прохода
        })
    return payload


# ===== ВЫБОР МОДЕЛИ (CHECKPOINT) =====
async def apply_model_override(payload: dict, settings: dict) -> dict:
    """
    Добавляет в payload модель пользователя (или модель по умолчанию) через override_settings.
    override_settings_restore_afterwards=False — модель остаётся загруженной, поэтому
    повторные генерации с той же моделью не перезагружают её.
    """
    try:
        model = await resolve_model(settings)
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
async def get_progress() -> dict:
    """Возвращает текущий прогресс генерации"""
    try:
        return await sd_get_json("sdapi/v1/progress", timeout=5)
    except Exception:
        return {"progress": 0, "eta_relative": 0, "state": {}}


PROGRESS_POLL_INTERVAL = 1.0  # как часто опрашиваем /sdapi/v1/progress, секунд

# Коллбэк прогресса: async (progress: 0..1, eta_seconds, state: dict с sampling_step/sampling_steps)
ProgressCallback = Callable[..., Awaitable[None]]


async def generate_with_progress(
    payload: dict,
    progress_callback: Optional[ProgressCallback] = None,
    endpoint: str = "sdapi/v1/txt2img"
) -> dict:
    """
    Отправляет запрос генерации и пока он выполняется опрашивает прогресс.
    endpoint: "sdapi/v1/txt2img" (по умолчанию) или "sdapi/v1/img2img" — для запросов с init_images/mask.

    Всё неблокирующее: сам запрос идёт отдельной asyncio-задачей, а цикл событий свободен для других
    пользователей. При отмене (CancelledError) запрос закрывается; прерывание самой генерации в WebUI
    (/sdapi/v1/interrupt) делает вызывающая сторона — см. task_queue.
    """
    async with sd_gate.slot():  # WebUI считает по одной генерации; ожидание места не блокирует цикл
        request_task = asyncio.create_task(sd_post_json(endpoint, payload))
        last_key = None
        try:
            while True:
                done, _ = await asyncio.wait({request_task}, timeout=PROGRESS_POLL_INTERVAL)
                if done:
                    break
                if progress_callback is None:
                    continue
                try:
                    progress_data = await get_progress()
                    progress = progress_data.get('progress', 0)
                    eta = progress_data.get('eta_relative', 0)
                    state = progress_data.get('state') or {}
                    key = (int(progress * 100), state.get('sampling_step'))
                    if key != last_key:
                        last_key = key
                        await progress_callback(progress, eta, state)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # прогресс — «косметика», он не должен ронять генерацию
                    log.debug("Ошибка обновления прогресса: %s", e)
            return request_task.result()  # SDError с телом ответа WebUI, если запрос не удался
        except asyncio.CancelledError:
            request_task.cancel()
            try:
                await request_task
            except BaseException:
                pass
            raise
        finally:
            if not request_task.done():
                request_task.cancel()


# ===== ПРОВЕРКА ДОСТУПНОСТИ =====
async def check_api_available() -> bool:
    """Проверяет доступность API сервера (лёгкий GET-запрос; раньше здесь запускалась пробная генерация)"""
    try:
        await _get_json("sdapi/v1/options", timeout=10)
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


async def check_adetailer_available(force_refresh: bool = False) -> bool:
    """Проверяет, что расширение ADetailer установлено и доступно через API"""
    global _adetailer_available_cache
    if _adetailer_available_cache is not None and not force_refresh:
        return _adetailer_available_cache

    available = False
    try:
        scripts = await _get_json("sdapi/v1/scripts", timeout=10)
        for tab in scripts.values():
            for script_name, _label in tab.get("txt2img", []) + tab.get("img2img", []):
                if str(script_name).lower() == "adetailer":
                    available = True
    except Exception:
        available = False

    if not available:
        # Запасная проверка: попытка прочитать список детекторов расширения
        try:
            await _get_json("adetailer/models", timeout=5)
            available = True
        except Exception:
            pass

    _adetailer_available_cache = available
    return available


async def get_adetailer_models(force_refresh: bool = False) -> List[Dict[str, str]]:
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
        data = await _get_json("adetailer/models", timeout=10)
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


async def build_adetailer_script_args(user_id: str, overrides: Optional[dict] = None) -> Optional[dict]:
    """
    Собирает блок alwayson_scripts['ADetailer'] из настроек пользователя.

    Возвращает None, если ADetailer выключен у пользователя или модуль настроек
    недоступен — тогда payload остаётся совместимым со старым поведением.
    """
    if build_adetailer_args is None or get_adetailer_enabled is None:
        return None
    try:
        if not await get_adetailer_enabled(user_id):
            return None
        args = await build_adetailer_args(user_id, overrides)
    except Exception as e:
        print(f"⚠️ Ошибка сборки настроек ADetailer: {e}")
        return None
    if not args or args.get("ad_model", "None") == "None":
        return None
    return {"args": [args]}


# ===== ОБРАБОТКА ПРОМПТА ЧЕРЕЗ НЕЙРОСЕТЬ =====
async def preprocess_prompt(user_id: str, raw_prompt: str, status_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Обрабатывает промпт через локальную нейросеть (если включено в настройках)
    """
    # Проверяем, включена ли обработка
    if not await get_prompt_processing_enabled(user_id):
        if status_callback:
            await status_callback("⚡ Обработка промпта выключена, использую оригинал")
        return raw_prompt
    
    if status_callback:
        await status_callback("🔄 Обработка промпта через нейросеть...")
    
    try:
        processed = await fast_process(raw_prompt)
        if status_callback:
            await status_callback(f"✅ Промпт обработан")
        return processed
    except Exception as e:
        print(f"Ошибка обработки промпта: {e}")
        if status_callback:
            await status_callback("⚠️ Ошибка обработки, использую оригинальный промпт")
        return raw_prompt

# ===== ПОЛУЧЕНИЕ СПИСКА СЭМПЛЕРОВ =====
async def get_samplers_list() -> List[str]:
    """Получает список доступных сэмплеров из API Stable Diffusion"""
    try:
        data = await sd_get_json("sdapi/v1/samplers", timeout=10)
        if isinstance(data, list):
            return [item['name'] for item in data if isinstance(item, dict) and 'name' in item]
        return []
    except Exception as e:
        print(f"Ошибка получения списка сэмплеров: {e}")
        return []
    
# ===== УЛУЧШЕНИЕ ПРОМПТА ДЛЯ РАЗНООБРАЗИЯ =====
async def enhance_prompt_for_diversity(prompt: str, user_id: str, user_settings: dict) -> str:
    """Улучшает промпт для получения разных лиц"""
    diversity_mode = await get_diversity_mode(user_id)
    
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


async def build_negative_prompt(user_id: str, custom_negative: str = None) -> str:
    """Формирует финальный негативный промпт: базовый текст + анти-мутационные фразы + негативные эмбеддинги."""
    use_advanced = await get_use_advanced_negative(user_id)

    if custom_negative:
        user_negative = custom_negative
    else:
        user_negative = await get_user_negative_prompt(user_id)

    # Универсальные фразы против мутаций и деформаций (если их ещё нет)
    extra_neg = "bad anatomy, extra limbs, deformed, mutated, disfigured, malformed, gross proportions, missing limbs, fused body, bad hands, bad fingers"
    if extra_neg not in user_negative:
        user_negative = f"{user_negative}, {extra_neg}"

    # Эмбеддинги: выбор пользователя (меню «Негативный промпт → Эмбеддинги») либо рекомендованные из WebUI
    embedding_tags = await effective_negative_embeddings(await get_user_settings(user_id))
    # чистая функция (при embedding_tags=None проверяет файлы на диске) — выполняем в потоке
    return await asyncio.to_thread(get_negative_prompt_with_embeddings, user_negative, use_advanced, embedding_tags)


# Отчёт ADetailer последней генерации в ТЕКУЩЕМ потоке (раньше это был общий атрибут функции,
# который параллельные генерации перезаписывали друг у друга).
# ContextVar вместо threading.local: в asyncio все задачи живут в одном потоке, а у каждой задачи свой контекст.
_last_adetailer_status: contextvars.ContextVar = contextvars.ContextVar("last_adetailer_status", default=None)
MAX_SEED = 2**32 - 1  # WebUI принимает seed только до 2^32-1


def get_last_adetailer_status() -> Dict[str, Any]:
    return _last_adetailer_status.get() or {"detected": 0, "detections": [], "models": []}


# ===== FACEID (IP-Adapter через ControlNet): лицо участвует в самой генерации =====
# В отличие от ReActor (замена лица ПОСЛЕ генерации), здесь эталонное лицо подаётся в ControlNet-юнит
# IP-Adapter FaceID, и модель рисует картинку сразу с нужным лицом. Требуется: sd-webui-controlnet >= 1.1.428,
# пакет insightface в venv WebUI, модель ip-adapter-faceid-plusv2_* в models/ControlNet и (желательно) её LoRA.

class FaceIDUnavailable(Exception):
    """В WebUI не хватает компонентов для FaceID (см. README, раздел «Лица»)."""


_faceid_cache: Dict[str, Any] = {"ts": 0.0, "data": None}
_FACEID_TTL = 60.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _faceid_kind(name: str) -> str:
    low = name.lower()
    return "plusv2" if "plusv2" in low else ("plus" if "plus" in low else "base")


async def get_faceid_setup(force: bool = False) -> Dict[str, Any]:
    """
    Проверяет, что WebUI готов к FaceID. Возвращает
    {"ok", "reason", "model", "module", "lora", "unit_count"}. Результат кэшируется на минуту.
    Переменные .env: FACEID_BASE (sd15|sdxl), FACEID_CN_MODEL, FACEID_MODULE, FACEID_LORA, FACEID_UNIT_COUNT.
    """
    now = time.monotonic()
    if not force and _faceid_cache["data"] is not None and now - _faceid_cache["ts"] < _FACEID_TTL:
        return _faceid_cache["data"]

    info: Dict[str, Any] = {"ok": False, "reason": "", "model": None, "module": None, "lora": None, "unit_count": 1}

    def _done() -> Dict[str, Any]:
        _faceid_cache["ts"] = now
        _faceid_cache["data"] = info
        return info

    try:
        data = await sd_get_json("controlnet/model_list", timeout=10)
        models = data.get("model_list", []) if isinstance(data, dict) else list(data or [])
    except Exception as e:
        info["reason"] = f"ControlNet API недоступен ({e}); установите расширение sd-webui-controlnet"
        return _done()

    base = os.environ.get("FACEID_BASE", "sd15").strip().lower()
    model = os.environ.get("FACEID_CN_MODEL", "").strip()
    if not model:
        candidates = [m for m in models if "faceid" in m.lower() and "lora" not in m.lower() and base in m.lower()]
        if not candidates:
            info["reason"] = f"в models/ControlNet нет модели ip-adapter-faceid-plusv2_{base}"
            return _done()
        model = max(candidates, key=lambda m: (_faceid_kind(m) == "plusv2", _faceid_kind(m) == "plus"))

    module = os.environ.get("FACEID_MODULE", "").strip() or (
        "ip-adapter_face_id_plus" if "plus" in model.lower() else "ip-adapter_face_id")

    lora = os.environ.get("FACEID_LORA", "").strip()
    if not lora:
        try:
            loras = [item["id"] for item in await get_loras_list() if "faceid" in item["id"].lower() and base in item["id"].lower()]
            loras.sort(key=lambda n: _faceid_kind(n) == _faceid_kind(model), reverse=True)
            lora = loras[0] if loras else ""
        except Exception as e:
            print(f"⚠️ FaceID: не удалось получить список LoRA: {e}")

    try:
        unit_count = int(os.environ.get("FACEID_UNIT_COUNT", "").strip() or 0)
        if not unit_count:
            cn_settings = await sd_get_json("controlnet/settings", timeout=10)
            unit_count = int(cn_settings.get("control_net_unit_count", 1))
        info["unit_count"] = unit_count
    except Exception:
        pass  # оставляем 1: для обычной генерации этого достаточно

    info.update(ok=True, model=model, module=module, lora=lora or None)
    return _done()


def inject_controlnet_unit(payload: dict, unit: dict) -> dict:
    """Добавляет ControlNet-юнит в payload, не затирая уже существующие (например, позу)."""
    scripts = dict(payload.get("alwayson_scripts") or {})
    cn = dict(scripts.get("ControlNet") or {})
    args = list(cn.get("args") or [])
    args.append(unit)
    cn["args"] = args
    scripts["ControlNet"] = cn
    payload["alwayson_scripts"] = scripts
    return payload


async def apply_faceid_to_payload(payload: dict, face_image: io.BytesIO,
                                  status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
                                  weight: Optional[float] = None, lora_weight: Optional[float] = None) -> bool:
    """Добавляет в payload ControlNet-юнит IP-Adapter FaceID и LoRA лица. Бросает FaceIDUnavailable, если WebUI не готов."""
    setup = await get_faceid_setup()
    if not setup["ok"]:
        raise FaceIDUnavailable(setup["reason"])

    face_image.seek(0)
    face_b64 = await _b64encode(face_image.read())
    unit = {
        "input_image": face_b64,
        "module": setup["module"],
        "model": setup["model"],
        "weight": weight if weight is not None else _env_float("FACEID_WEIGHT", 1.0),
        "resize_mode": "Crop and Resize",
        "low_vram": False,
        "processor_res": 512,
        "guidance_start": 0.0,
        "guidance_end": 1.0,
        "pixel_perfect": True,
        "control_mode": "Balanced",
    }
    inject_controlnet_unit(payload, unit)

    lw = lora_weight if lora_weight is not None else _env_float("FACEID_LORA_WEIGHT", 0.6)
    # lw == 0 — LoRA лица выключена пользователем; уже стоящий в промпте тег (точный повтор) не трогаем
    if setup["lora"] and lw > 0 and f"<lora:{setup['lora']}" not in payload.get("prompt", ""):
        payload["prompt"] = f"{payload['prompt']}, <lora:{setup['lora']}:{lw}>"

    print(f"🧠 FaceID: module={setup['module']}, model={setup['model']}, lora={setup['lora']}")
    if status_callback:
        await status_callback("🧠 FaceID: лицо учитывается прямо в генерации")
    return True


def _soften_adetailer_for_faceid(ad_block: dict) -> None:
    """ADetailer перерисовывает лицо без IP-Adapter и может «сломать» сходство — ограничиваем силу перерисовки."""
    try:
        args = ad_block["args"][0]
        cap = _env_float("FACEID_AD_DENOISE", 0.25)
        if float(args.get("ad_denoising_strength", 0.4)) > cap:
            args["ad_denoising_strength"] = cap
    except Exception:
        pass


async def generate_txt2img(
    prompt: str, 
    user_id: str, 
    user_settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    fixed_seed: int = None,
    adetailer_overrides: Optional[dict] = None,
    exact_prompt: Optional[str] = None,
    exact_negative: Optional[str] = None,
    face_image: Optional[io.BytesIO] = None,
    face_weights: Optional[Tuple[float, float]] = None
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
            await status_callback("🎨 Генерация изображения...")
    else:
        # Обрабатываем промпт через нейросеть (если включено)
        processed_prompt = await preprocess_prompt(user_id, prompt, status_callback)
        # Добавляем активные LoRA в промпт
        lora_tags = await get_active_lora_tags(user_id)
        if lora_tags:
            processed_prompt = f"{processed_prompt}, {lora_tags}"
        if status_callback:
            await status_callback("🎨 Генерация изображения...")
        # Используем обработанный промпт для дальнейшей обработки
        # Фразы «разнообразия» («different face» и т.п.) противоречат FaceID — с эталонным лицом их не добавляем
        enhanced_prompt = processed_prompt if face_image is not None else await enhance_prompt_for_diversity(processed_prompt, user_id, user_settings)

    # Используем фиксированный seed если передан, иначе случайный
    if fixed_seed is not None:
        seed = fixed_seed
    else:
        seed = random.randint(1, MAX_SEED)

    negative_prompt = exact_negative if exact_negative is not None else await build_negative_prompt(user_id)
    
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

    # ===== FACEID: эталонное лицо идёт в саму генерацию (ControlNet IP-Adapter) =====
    faceid_applied = False
    if face_image is not None:
        fw, lw = face_weights if face_weights else (None, None)
        faceid_applied = await apply_faceid_to_payload(payload, face_image, status_callback, fw, lw)

    # ===== ADETAILER: автодетекция и перерисовка лиц/рук =====
    ad_block = await build_adetailer_script_args(user_id, adetailer_overrides)
    if ad_block:
        if faceid_applied:
            _soften_adetailer_for_faceid(ad_block)
        payload["alwayson_scripts"] = dict(payload.get("alwayson_scripts") or {})
        payload["alwayson_scripts"]["ADetailer"] = ad_block
        model_name = ad_block["args"][0].get("ad_model", "?")
        if status_callback:
            await status_callback(f"🕵️ ADetailer: модель `{model_name}`")
        print(f"🕵️ ADetailer txt2img: model={model_name}")

    apply_hires(payload, user_settings)
    await apply_model_override(payload, user_settings)
    response = await generate_with_progress(payload, progress_callback)

    # Сохраняем отчёт ADetailer, чтобы бот мог показать его в капшне
    try:
        _last_adetailer_status.set(get_adetailer_status(response))
    except Exception as e:  # pragma: no cover - диагностика не должна ломать генерацию
        print(f"⚠️ Не удалось разобрать отчёт ADetailer: {e}")
        _last_adetailer_status.set({"detected": 0, "detections": [], "models": []})

    images = []
    for index, img_base64 in enumerate(response.get('images', [])):
        image_bytes = base64.b64decode(img_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = f"generation_{index}.png"
        image_file.final_prompt = payload["prompt"]  # с LoRA лица, если FaceID включён — для точного повтора
        image_file.final_negative = negative_prompt
        image_file.faceid_used = faceid_applied
        images.append(image_file)

    return images, seed, original_prompt, processed_prompt


# ===== ГЕНЕРАЦИЯ С КАСТОМНЫМ НЕГАТИВОМ =====
async def generate_txt2img_advanced(
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
    embedding_tags = await effective_negative_embeddings(user_settings)
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
    ad_block = await build_adetailer_script_args(user_id, adetailer_overrides)
    if ad_block:
        payload["alwayson_scripts"] = dict(payload.get("alwayson_scripts") or {})
        payload["alwayson_scripts"]["ADetailer"] = ad_block
        print(f"🕵️ ADetailer txt2img (advanced): model={ad_block['args'][0].get('ad_model', '?')}")

    apply_hires(payload, user_settings)
    await apply_model_override(payload, user_settings)
    response = await generate_with_progress(payload, progress_callback)

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

async def swap_face_with_reactor(
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
    target_base64 = await _b64encode(target_image.read())
    
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
    
    # select_source: 0 — исходное фото, 1 — сохранённая модель лица (models/reactor/faces/<имя>.safetensors)
    use_model = bool(face_model) and face_model != "None"
    payload["select_source"] = 1 if use_model else 0

    if source_image is not None and not use_model:
        source_image.seek(0)
        source_base64 = await _b64encode(source_image.read())
        payload["source_image"] = source_base64
    
    # Пробуем разные возможные эндпоинты
    paths = ['reactor/image', 'reactor/swap-face', 'sdapi/v1/reactor/image']

    last_error = None
    session = await get_session()
    async with sd_gate.slot():  # ReActor работает на том же GPU, что и генерация
        for path in paths:
            url = f'{SD_BASE_URL}/{path}'
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status >= 400:
                        error_body = (await resp.text(errors='replace'))[:2000]
                        if resp.status != 404:  # 404 — это просто «такого эндпоинта нет», настоящую причину не затираем
                            last_error = f"HTTP {resp.status}: {error_body}"
                        elif last_error is None:
                            last_error = f"HTTP 404: эндпоинт {path} не найден (установлен ли ReActor?)"
                        continue
                    result = await resp.json(content_type=None)

                # Проверяем формат ответа
                if 'image' in result:
                    encoded = result['image']
                elif 'images' in result and result['images']:
                    encoded = result['images'][0]
                else:
                    raise Exception("Неизвестный формат ответа от Reactor")

                image_bytes = await asyncio.to_thread(decode_base64_to_bytes, encoded)
                swapped_image = io.BytesIO(image_bytes)
                swapped_image.name = "swapped_face.png"
                return swapped_image

            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = str(e)
                continue

    raise Exception(f"Не удалось заменить лицо. Последняя ошибка: {last_error}")


async def get_reactor_models() -> Dict[str, List[str]]:
    """Получает список доступных моделей из Reactor"""
    models_info = {
        "facemodels": [],
        "upscalers": []
    }

    # Получаем модели лиц
    for path in ('reactor/facemodels', 'reactor/models', 'sdapi/v1/reactor/facemodels'):
        try:
            data = await sd_get_json(path, timeout=10)
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
    for path in ('reactor/upscalers', 'sdapi/v1/reactor/upscalers'):
        try:
            data = await sd_get_json(path, timeout=10)
            if isinstance(data, list):
                models_info["upscalers"] = data
            elif isinstance(data, dict) and 'upscalers' in data:
                models_info["upscalers"] = data['upscalers']
            break
        except Exception:
            continue

    return models_info


async def check_reactor_available() -> bool:
    """Проверяет доступность Reactor API"""
    for path in ('reactor/models', 'reactor/facemodels', 'reactor/image'):
        try:
            await sd_get_json(path, timeout=5)
            return True
        except Exception:
            continue
    return False


async def build_reactor_face_model(images: List[bytes], name: str, compute_method: int = 0) -> str:
    """
    Создаёт модель лица ReActor из одного или нескольких фото (POST /reactor/facemodels).
    Расширение усредняет эмбеддинги лиц (compute_method: 0 — Mean, 1 — Median, 2 — Mode) и сохраняет
    models/reactor/faces/<name>.safetensors на машине WebUI. Возвращает имя файла модели.
    """
    if not images:
        raise ValueError("Нет изображений для модели лица")
    encoded = ["data:image/jpeg;base64," + await _b64encode(raw) for raw in images]
    payload = {"source_images": encoded, "name": name, "compute_method": int(compute_method)}
    async with sd_gate.slot():  # ReActor считает эмбеддинги на том же GPU/CPU, что и генерация
        await sd_post_json("reactor/facemodels", payload, timeout=180)

    # Проверяем по списку моделей, под каким именем файла расширение сохранило результат
    try:
        info = await get_reactor_models()
        for entry in info.get("facemodels", []):
            base_name = os.path.basename(str(entry))
            if os.path.splitext(base_name)[0] == name:
                return base_name
    except Exception as e:
        print(f"⚠️ Не удалось проверить список моделей лиц: {e}")
    return f"{name}.safetensors"


# ===== IMG2IMG =====
async def generate_img2img(
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
        "negative_prompt": await build_negative_prompt(user_id),
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
    ad_block = await build_adetailer_script_args(user_id, adetailer_overrides)
    if ad_block:
        payload["alwayson_scripts"] = dict(payload.get("alwayson_scripts") or {})
        payload["alwayson_scripts"]["ADetailer"] = ad_block
        print(f"🕵️ ADetailer img2img: model={ad_block['args'][0].get('ad_model', '?')}")

    await apply_model_override(payload, user_settings)
    response = await generate_with_progress(payload, progress_callback, endpoint="sdapi/v1/img2img")

    images = []
    for index, img_base64 in enumerate(response.get('images', [])):
        image_bytes = base64.b64decode(img_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = f"img2img_result_{index}.png"
        images.append(image_file)

    return images


# ===== АПСКЕЙЛ =====
async def upscale_image(
    image: io.BytesIO,
    upscale_factor: float = 2,
    upscaler_name: str = "ESRGAN_4x",
    face_restoration: float = 0.5
) -> io.BytesIO:
    """Увеличивает разрешение изображения (base64 больших картинок кодируется/декодируется в потоке)"""
    image.seek(0)
    image_base64 = await _b64encode(image.read())

    payload = {
        "image": image_base64,
        "upscaling_resize": upscale_factor,
        "upscaler_1": upscaler_name,
        "gfpgan_visibility": face_restoration,
        "codeformer_visibility": face_restoration,
    }

    async with sd_gate.slot():
        response = await call_api("sdapi/v1/extra-single-image", **payload)

    image_bytes = await asyncio.to_thread(decode_base64_to_bytes, response.get('image', ''))
    image_file = io.BytesIO(image_bytes)
    image_file.name = "upscaled.png"

    return image_file


# ===== INPAINT =====
async def generate_inpaint(
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
        "negative_prompt": await build_negative_prompt(user_id),
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

    await apply_model_override(payload, settings)
    response = await generate_with_progress(payload, progress_callback, endpoint="sdapi/v1/img2img")
    images = response.get('images', [])
    if not images:
        raise Exception("API не вернул изображений")
    img_base64 = images[0]
    image_bytes = base64.b64decode(img_base64)
    image_file = io.BytesIO(image_bytes)
    image_file.name = "inpainted.png"
    return image_file


async def get_loras_list() -> List[Dict[str, str]]:
    """Получает список доступных LoRA из API Stable Diffusion"""
    try:
        data = await sd_get_json("sdapi/v1/loras", timeout=10)
        result = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get('name') or item.get('filename') or item.get('alias')
                    if name:
                        # Имя обычно заканчивается на .safetensors или .pt, убираем
                        clean_name = name.replace('.safetensors', '').replace('.pt', '').replace('.bin', '')
                        result.append({
                            "id": clean_name,
                            "name": clean_name.replace('_', ' ').title(),  # человеческий вид
                            "tag": f"<lora:{clean_name}:1>",
                            "filename": name
                        })
        return result
    except Exception as e:
        print(f"Ошибка получения списка LoRA: {e}")
        return []

# ===== СОВМЕСТИМОСТЬ СО СТАРЫМ КОДОМ =====
async def generate_image(
    prompt: str, 
    user_settings: dict, 
    user_id: str = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    fixed_seed: int = None,
    exact_prompt: Optional[str] = None,
    exact_negative: Optional[str] = None,
    face_image: Optional[io.BytesIO] = None,
    face_weights: Optional[Tuple[float, float]] = None
) -> Tuple[io.BytesIO, int, str, str]:
    """Генерация с возвратом изображения, seed, оригинального и обработанного промпта"""
    if user_id is None:
        user_id = "default"

    images, seed, original_prompt, processed_prompt = await generate_txt2img(
        prompt, user_id, user_settings, progress_callback, status_callback, fixed_seed,
        exact_prompt=exact_prompt, exact_negative=exact_negative, face_image=face_image,
        face_weights=face_weights
    )

    if not images:
        # Раньше возвращался пустой BytesIO, и бот падал позже с невнятной ошибкой отправки
        raise Exception("WebUI не вернул изображений")
    return images[0], seed, original_prompt, processed_prompt
