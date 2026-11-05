import asyncio
import os
import io
import re
import PIL
import base64
import json
import random
import time
from typing import Awaitable, Optional, Callable, List, Dict, Any
from aio import LazyLock
from settings import get_active_lora_tags

import pandas as pd

# Импортируем общие утилиты из Aapi (прогресс, негативный промпт)
from Aapi import generate_with_progress, build_negative_prompt, apply_model_override, fit_dims_to_image, MAX_SEED, _b64encode, decode_base64_to_bytes

# ===== КОНФИГУРАЦИЯ =====
CONTROLNET_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controlnet_images")

# Кэш для Parquet-файлов
_parquet_cache = {}

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

def _safe_dims(settings: dict, fallback: tuple = (512, 768)) -> tuple:
    """
    Возвращает (w, h) — положительные, ≤ 2048, кратные 8.
    Нужно потому, что SD API возвращает 422 на width/height не кратные 8.
    """
    try:
        w = int(settings.get("width", fallback[0]))
        h = int(settings.get("height", fallback[1]))
    except (TypeError, ValueError):
        return fallback
    if not (64 <= w <= 2048 and 64 <= h <= 2048):
        print(f"⚠️ Размеры вне допустимого диапазона ({w}x{h}), использую {fallback}")
        return fallback
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return w, h


def _validate_control_image(control_image: io.BytesIO) -> bytes:
    """
    Проверяет, что картинка не пустая и открывается PIL.
    Возвращает нормализованный PNG-байт-стрим (RGB/RGBA/L).
    """
    if control_image is None:
        raise ValueError("control_image is None")
    control_image.seek(0)
    data = control_image.read()
    if not data:
        raise ValueError("control_image пуст (0 байт)")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode not in ('RGB', 'RGBA', 'L'):
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except ImportError:
        # PIL не установлен — отправляем как есть
        return data
    except Exception as e:
        raise ValueError(f"invalid control_image ({len(data)} байт): {e}")

def _load_parquet_item(filename: str, index: int, col: str) -> Optional[io.BytesIO]:
    """Загружает изображение из parquet по индексу и колонке"""
    df = _parquet_cache.get(filename)
    if df is None:
        return None
    raw = df.iloc[index][col]
    if isinstance(raw, dict):
        if 'bytes' in raw and isinstance(raw['bytes'], bytes):
            return io.BytesIO(raw['bytes'])
        for key in ['data', 'image', 'img']:
            if key in raw:
                data = raw[key]
                if isinstance(data, bytes):
                    return io.BytesIO(data)
                if isinstance(data, str):
                    try:
                        return io.BytesIO(base64.b64decode(data))
                    except:
                        pass
    elif isinstance(raw, bytes):
        return io.BytesIO(raw)
    elif isinstance(raw, str):
        try:
            return io.BytesIO(base64.b64decode(raw))
        except:
            pass
    return None


def _get_controlnet_sources_sync() -> List[Dict[str, Any]]:
    """
    Сканирует папку controlnet_images:
    - рекурсивно ищет все *bone_structure.png (или .jpg)
    - извлекает категорию (имя папки на уровень выше папки разрешения)
    - извлекает разрешение (имя папки с разрешением, например 512768)
    - извлекает номер из имени файла (цифры)
    - формирует label = "Категория Номер" (без NSFW_)
    - full_label = "Категория Номер (Разрешение)"
    - сортирует по категории и номеру
    - также собирает Parquet-файлы (если есть)
    """
    global _parquet_cache
    if not os.path.exists(CONTROLNET_IMAGES_DIR):
        os.makedirs(CONTROLNET_IMAGES_DIR)
        return []

    sources = []

    # 1. Сбор Parquet
    for root, dirs, files in os.walk(CONTROLNET_IMAGES_DIR):
        for f in files:
            if f.lower().endswith('.parquet'):
                file_path = os.path.join(root, f)
                try:
                    df = pd.read_parquet(file_path)
                    _parquet_cache[f] = df
                    img_col = None
                    priority = ['condtioning_image', 'conditioning_image', 'control', 'skeleton', 'pose', 'openpose', 'image']
                    for col in priority:
                        if col in df.columns:
                            img_col = col
                            break
                    if img_col is None:
                        for col in df.columns:
                            if 'object' in str(df[col].dtype) or 'bytes' in str(df[col].dtype):
                                img_col = col
                                break
                    if img_col is None:
                        continue
                    has_name = 'pose_name' in df.columns
                    for idx in range(len(df)):
                        if has_name and df.at[idx, 'pose_name']:
                            label = str(df.at[idx, 'pose_name'])
                            full_label = label
                        else:
                            label = f"{os.path.splitext(f)[0]}#{idx+1}"
                            full_label = label
                        if len(label) > 30:
                            label = label[:27] + '...'
                        sources.append({
                            'type': 'parquet',
                            'file': f,
                            'index': idx,
                            'label': label,
                            'full_label': full_label,
                            'img_col': img_col,
                            'category': 'parquet',
                            'number': idx,
                            'resolution': 'unknown'
                        })
                except Exception as e:
                    print(f"❌ Ошибка чтения Parquet {f}: {e}")

    # 2. Сбор PNG bone_structure
    bone_pattern = re.compile(r'.*_bone_structure\.(png|jpg|jpeg)$', re.IGNORECASE)
    number_pattern = re.compile(r'(\d+)')

    for root, dirs, files in os.walk(CONTROLNET_IMAGES_DIR):
        for f in files:
            if bone_pattern.match(f):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(root, CONTROLNET_IMAGES_DIR)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    category_raw = parts[-2]
                    resolution_raw = parts[-1]
                else:
                    category_raw = "unknown"
                    resolution_raw = "unknown"

                numbers = number_pattern.findall(f)
                number = numbers[0] if numbers else "0"

                category_clean = category_raw.replace('NSFW_', '').replace('_', ' ').strip()
                label = f"{category_clean} {number}"
                if len(label) > 30:
                    label = label[:27] + '...'
                full_label = f"{category_clean} {number} ({resolution_raw})"

                sources.append({
                    'type': 'image',
                    'path': full_path,
                    'label': label,
                    'full_label': full_label,
                    'category': category_raw,
                    'resolution': resolution_raw,
                    'number': number,
                })

    # Сортировка: сначала по категории, потом по номеру (как число)
    def sort_key(s):
        cat = s.get('category', '')
        num = s.get('number', 0)
        try:
            num = int(num)
        except:
            num = 0
        return (cat, num)

    sources.sort(key=sort_key)

    print(f"🔍 Найдено источников: {len(sources)} (Parquet + PNG)")
    return sources


def _get_controlnet_preview_sync(source: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Возвращает BytesIO с изображением для источника"""
    try:
        if source['type'] == 'parquet':
            df = _parquet_cache.get(source['file'])
            if df is None:
                file_path = os.path.join(CONTROLNET_IMAGES_DIR, source['file'])
                df = pd.read_parquet(file_path)
                _parquet_cache[source['file']] = df
            img_col = source.get('img_col')
            if img_col is None:
                priority = ['condtioning_image', 'conditioning_image', 'control', 'skeleton', 'pose', 'openpose', 'image']
                for col in priority:
                    if col in df.columns:
                        img_col = col
                        break
                if img_col is None:
                    for col in df.columns:
                        if 'object' in str(df[col].dtype) or 'bytes' in str(df[col].dtype):
                            img_col = col
                            break
            if img_col is None:
                return None
            return _load_parquet_item(source['file'], source['index'], img_col)
        elif source['type'] == 'image':
            with open(source['path'], 'rb') as f:
                return io.BytesIO(f.read())
    except Exception as e:
        print(f"Ошибка загрузки превью: {e}")
    return None


# ===== АСИНХРОННЫЙ ДОСТУП К ДАННЫМ (тяжёлое чтение parquet/папок — в потоке) =====
_sources_cache: Optional[List[Dict[str, Any]]] = None
_sources_lock = LazyLock()


async def get_controlnet_sources(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Список источников ControlNet. Индекс (os.walk + чтение parquet) строится ОДИН раз в потоке
    (при старте бота) и дальше берётся из кэша; force_refresh=True — пересобрать.
    """
    global _sources_cache
    async with _sources_lock:
        if force_refresh or _sources_cache is None:
            _sources_cache = await asyncio.to_thread(_get_controlnet_sources_sync)
        return list(_sources_cache)


async def get_controlnet_preview_by_source(source: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Изображение источника (выборка из parquet/файла — в потоке)."""
    return await asyncio.to_thread(_get_controlnet_preview_sync, source)


# ===== ГЕНЕРАЦИЯ С CONTROLNET =====

async def generate_controlnet(
    prompt: str,
    init_image: io.BytesIO,
    control_image: io.BytesIO,
    module: str,
    model: str,
    settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    user_id: str = "default"
) -> io.BytesIO:
    """
    Генерация с ControlNet (img2img).
    """
    # --- Валидация init_image ---
    init_image.seek(0)
    init_bytes = init_image.read()
    if not init_bytes:
        raise Exception("init_image пуст")
    init_base64 = await _b64encode(init_bytes)

    # --- Валидация control_image (PIL — CPU-работа, уходит в поток) ---
    try:
        control_bytes = await asyncio.to_thread(_validate_control_image, control_image)
    except ValueError as e:
        raise Exception(f"ControlNet: {e}")
    control_base64 = await _b64encode(control_bytes)

    # --- Валидация размеров: результат повторяет пропорции исходной картинки ---
    w, h = _safe_dims(settings)
    w, h = fit_dims_to_image(init_bytes, w, h)

    controlnet_args = [{
        "input_image": control_base64,
        "module": module,
        "model": model,
        "weight": 1.0,
        "resize_mode": "Crop and Resize",           # int
        "low_vram": False,
        "processor_res": 512,
        "threshold_a": 0.5,
        "threshold_b": 0.5,
        "guidance_start": 0.0,
        "guidance_end": 1.0,
        "pixel_perfect": True,
        "control_mode": "Balanced",
    }]

    payload = {
        "init_images": [init_base64],
        "prompt": prompt,
        "negative_prompt": await build_negative_prompt(user_id),
        "denoising_strength": settings.get("denoising_strength", 0.7),
        "steps": settings.get("steps", 20),
        "width": w,
        "height": h,
        "cfg_scale": settings.get("cfg_scale", 7.0),
        "sampler_name": settings.get("sampler_name", "Euler a"),
        "seed": random.randint(1, MAX_SEED),
        "batch_size": 1,
        "alwayson_scripts": {
            "ControlNet": {
                "args": controlnet_args
            }
        }
    }

    print(
        f"🎮 ControlNet img2img: {w}x{h}, "
        f"module={module}, model={model}, "
        f"control_img={len(control_bytes)}B"
    )

    apply_model_override(payload, settings)
    # img2img-запрос должен идти на img2img (раньше по ошибке — на txt2img, и init_images игнорировались)
    response = await generate_with_progress(payload, progress_callback, endpoint="sdapi/v1/img2img")
    images = response.get('images', [])
    if not images:
        raise Exception("API не вернул изображений")
    img_base64 = images[0]
    image_bytes = await asyncio.to_thread(decode_base64_to_bytes, img_base64)
    image_file = io.BytesIO(image_bytes)
    image_file.name = "controlnet.png"
    return image_file


async def generate_txt2img_controlnet(
    prompt: str,
    control_image: io.BytesIO,
    module: str,
    model: str,
    settings: dict,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    user_id: str = "default"
) -> io.BytesIO:
    """
    Генерация txt2img с ControlNet (без исходного изображения).
    """
    lora_tags = await get_active_lora_tags(user_id)
    if lora_tags:
        prompt = f"{prompt}, {lora_tags}"
        if status_callback:
            await status_callback(f"🎛️ Добавлены LoRA: {lora_tags}")

    # --- Валидация control_image ---
    try:
        control_bytes = await asyncio.to_thread(_validate_control_image, control_image)
    except ValueError as e:
        raise Exception(f"ControlNet: {e}")
    control_base64 = await _b64encode(control_bytes)

    # --- Валидация размеров ---
    w, h = _safe_dims(settings)

    # --- ЕДИНЫЙ аргумент ControlNet ---
    # resize_mode: ЦЕЛОЕ число (0=Just Resize, 1=Crop and Resize, 2=Resize and Fill).
    # Строка "Resize and Fill" отвергается частью версий расширения → 422.
    # Поле "enabled" убрано — его нет в схеме ControlNetUnit.
    controlnet_args = {
        "input_image": control_base64,
        "module": module,           # "none"
        "model": model,
        "weight": 1.0,
        "resize_mode": "Crop and Resize",
        "low_vram": False,
        "processor_res": 512,
        "threshold_a": 0.5,
        "threshold_b": 0.5,
        "guidance_start": 0.0,
        "guidance_end": 0.9,
        "pixel_perfect": True,
        "control_mode": "Balanced",
    }

    seed = random.randint(1, MAX_SEED)
    payload = {
        "prompt": prompt,
        "negative_prompt": await build_negative_prompt(user_id),
        "steps": settings.get("steps", 40),
        "width": w,
        "height": h,
        "cfg_scale": settings.get("cfg_scale", 6.0),
        "sampler_name": settings.get("sampler_name", "DPM++ 2M"),
        "seed": seed,
        "batch_size": 1,
        "n_iter": 1,
        "alwayson_scripts": {
            "ControlNet": {
                "args": [controlnet_args]
            }
        }
    }

    # Диагностика — что реально уходит
    print(
        f"🎮 ControlNet txt2img: {w}x{h}, "
        f"module={module}, model={model}, "
        f"resize_mode={controlnet_args['resize_mode']}, "
        f"control_img={len(control_bytes)}B"
    )

    apply_model_override(payload, settings)
    response = await generate_with_progress(payload, progress_callback)
    images = response.get('images', [])
    if not images:
        raise Exception("API не вернул изображений")
    img_base64 = images[0]
    image_bytes = await asyncio.to_thread(decode_base64_to_bytes, img_base64)
    image_file = io.BytesIO(image_bytes)
    image_file.name = "controlnet_txt2img.png"
    image_file.seed = seed  # main.py показывает seed в подписи
    return image_file