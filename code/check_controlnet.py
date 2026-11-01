# check_controlnet.py
import os, io, json, base64, urllib.request, urllib.error
from controlnet_manager import (
    get_controlnet_sources,
    get_controlnet_preview_by_source,
    CONTROLNET_IMAGES_DIR,
)

SD = os.environ.get("SD_BASE_URL", "http://127.0.0.1:7860").rstrip("/")


def parse_resolution(resolution):
    """Повторяет логику из main.process_controlnet_prompt"""
    if not resolution or resolution == "unknown":
        return None
    try:
        if "x" in resolution:
            parts = resolution.split("x")
        elif len(resolution) == 6:
            parts = [resolution[:3], resolution[3:]]
        else:
            parts = [resolution[:4], resolution[4:]]
        return int(parts[0]), int(parts[1])
    except Exception as e:
        return f"PARSE_ERR: {e}"


def check_image(img):
    if img is None:
        return "None (нет картинки)"
    img.seek(0)
    data = img.read()
    if len(data) < 100:
        return f"СЛИШКОМ МАЛО: {len(data)} байт"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data)); im.load()
            return f"PNG {im.size} {im.mode}"
        except ImportError:
            return f"PNG (без PIL, {len(data)} байт)"
        except Exception as e:
            return f"PNG БИТЫЙ: {e}"
    if data[:2] == b"\xff\xd8":
        return f"JPEG {len(data)} байт (PIL не проверял)"
    return f"НЕ PNG/JPEG: {data[:8]!r}"


def main():
    print(f"📁 {CONTROLNET_IMAGES_DIR}")
    sources = get_controlnet_sources()
    print(f"📊 Источников: {len(sources)}\n")

    bad = []
    for i, src in enumerate(sources):
        res = src.get("resolution")
        dims = parse_resolution(res)
        img_status = check_image(get_controlnet_preview_by_source(src))

        warnings = []
        if isinstance(dims, tuple):
            w, h = dims
            if w % 8 or h % 8:
                warnings.append(f"w/h не кратны 8: {w}x{h}")
            if not (64 <= w <= 2048 and 64 <= h <= 2048):
                warnings.append(f"размеры вне [64..2048]: {w}x{h}")
        elif isinstance(dims, str):
            warnings.append(dims)
        elif dims is None and res not in (None, "unknown"):
            warnings.append(f"res={res!r} → None")

        if "None" in img_status or "СЛИШКОМ" in img_status \
           or "БИТЫЙ" in img_status or "НЕ PNG" in img_status:
            warnings.append(f"картинка: {img_status}")

        if warnings:
            bad.append((i, src, warnings))

    print(f"❌ Проблемных: {len(bad)}\n")
    for i, src, warns in bad[:40]:
        print(f"[{i}] {src.get('full_label', src.get('label'))}  ({src['type']})")
        print(f"     category={src.get('category')!r}  resolution={src.get('resolution')!r}")
        for w in warns:
            print(f"     ⚠️  {w}")
    if len(bad) > 40:
        print(f"... и ещё {len(bad) - 40}")

    if not bad:
        print("\n✅ Все источники выглядят валидными.")
        print("   Если 422 всё равно есть — нужно ТЕЛО ответа (шаг 2).")
        return

    # Отправим первый проблемный в API и покажем тело 422
    i, src, _ = bad[0]
    img = get_controlnet_preview_by_source(src)
    if img is None:
        print("\n❌ Первый проблемный источник не загрузился вовсе.")
        return

    img.seek(0)
    b64 = base64.b64encode(img.read()).decode()

    dims = parse_resolution(src.get("resolution"))
    w, h = dims if isinstance(dims, tuple) else (512, 768)

    payload = {
        "prompt": "test", "negative_prompt": "", "steps": 1,
        "width": w, "height": h, "cfg_scale": 7.0,
        "sampler_name": "Euler a", "seed": 1, "batch_size": 1, "n_iter": 1,
        "alwayson_scripts": {"ControlNet": {"args": [{
            "input_image": b64, "module": "none",
            "model": "control_v11p_sd15_openpose",
            "weight": 1.0, "resize_mode": 0,
            "guidance_start": 0.0, "guidance_end": 0.9,
            "pixel_perfect": True, "control_mode": "Balanced",
        }]}}
    }
    req = urllib.request.Request(
        f"{SD}/sdapi/v1/txt2img",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"\n🧪 Тест на источнике [{i}]: {w}x{h}")
    try:
        r = urllib.request.urlopen(req, timeout=120)
        print(f"✅ 200 OK, {len(r.read())} байт")
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP {e.code}")
        print("ТЕЛО:", e.read().decode("utf-8", "replace")[:3000])
    except Exception as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    main()