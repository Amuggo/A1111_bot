"""
Проверка фото лица перед сохранением: ровно одно лицо, достаточный размер кадра и лица.
Детектор — MediaPipe (если установлен), иначе каскад Хаара из OpenCV. Если нет ни того ни другого,
проверка пропускается (фото принимается), чтобы не блокировать добавление лиц.
"""

import asyncio
import io
from typing import Tuple

MIN_SIDE = 256          # минимальная сторона фото, px
MIN_FACE_RATIO = 0.04   # лицо должно занимать не меньше 4% площади кадра


def _detect_faces_sync(jpeg: bytes):
    """Возвращает (ширина, высота, [(w_лица, h_лица), ...]) или None, если детекторов нет."""
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(jpeg)) as img:
        rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    try:
        import mediapipe as mp
        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as det:
            result = det.process(rgb)
        boxes = []
        for d in (result.detections or []):
            bb = d.location_data.relative_bounding_box
            boxes.append((max(bb.width, 0) * w, max(bb.height, 0) * h))
        return w, h, boxes
    except ImportError:
        pass
    try:
        import cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48))
        return w, h, [(float(fw), float(fh)) for (_, _, fw, fh) in found]
    except ImportError:
        return None


def _check_sync(jpeg: bytes) -> Tuple[bool, str]:
    try:
        res = _detect_faces_sync(jpeg)
    except Exception as e:  # детектор не должен ломать сценарий добавления лица
        print(f"⚠️ Проверка лица пропущена: {e}")
        return True, ""
    if res is None:
        return True, ""
    w, h, faces = res
    if min(w, h) < MIN_SIDE:
        return False, f"Фото слишком маленькое ({w}×{h}), нужна сторона от {MIN_SIDE} px."
    if not faces:
        return False, "Лицо не найдено. Нужно чёткое фото анфас или в три четверти, без сильных теней и очков-масок."
    if len(faces) > 1:
        return False, f"На фото {len(faces)} лица — нужно ровно одно. Обрежьте кадр по нужному человеку."
    fw, fh = faces[0]
    if (fw * fh) / (w * h) < MIN_FACE_RATIO:
        return False, "Лицо слишком мелкое в кадре. Приблизьтесь или обрежьте фото, чтобы лицо занимало заметную часть."
    return True, ""


async def check_face_photo(jpeg: bytes) -> Tuple[bool, str]:
    """(годится, причина отказа). Работа в потоке — не блокирует цикл событий."""
    return await asyncio.to_thread(_check_sync, jpeg)
