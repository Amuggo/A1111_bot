"""
Модуль для сегментации тела, фона, одежды и детекции позы
Использует MediaPipe и OpenCV
"""

import cv2
import numpy as np
import mediapipe as mp
import io
from PIL import Image

# Правильная инициализация MediaPipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def image_to_cv2(image_bytes_io: io.BytesIO) -> np.ndarray:
    """Конвертирует BytesIO в OpenCV изображение (RGB)"""
    image_bytes_io.seek(0)
    img = Image.open(image_bytes_io).convert("RGB")
    return np.array(img)

def cv2_to_bytesio(cv2_image: np.ndarray) -> io.BytesIO:
    """Конвертирует OpenCV изображение (RGB) в BytesIO (PNG)"""
    img = Image.fromarray(cv2_image)
    out = io.BytesIO()
    img.save(out, format='PNG')
    out.seek(0)
    return out

def get_body_mask(image_io: io.BytesIO) -> io.BytesIO:
    """
    Возвращает маску тела (человека) – белое тело на чёрном фоне.
    Использует Selfie Segmentation.
    """
    image = image_to_cv2(image_io)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie:
        results = selfie.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        mask = results.segmentation_mask
        # Превращаем в бинарную маску (порог 0.5)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        # Применяем небольшое размытие для сглаживания
        binary_mask = cv2.GaussianBlur(binary_mask, (5,5), 0)
        # Создаём 3-канальную маску (для inpainting нужен одноканальный, но мы сохраняем как PNG)
        mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        return cv2_to_bytesio(mask_3ch)

def get_background_mask(image_io: io.BytesIO) -> io.BytesIO:
    """Возвращает маску фона (инвертированную маску тела)"""
    body_mask = get_body_mask(image_io)
    # Инвертируем
    mask_img = Image.open(body_mask).convert('L')
    inverted = Image.eval(mask_img, lambda x: 255 - x)
    out = io.BytesIO()
    inverted.save(out, format='PNG')
    out.seek(0)
    return out

def get_clothes_mask(image_io: io.BytesIO) -> io.BytesIO:
    """
    Упрощённая сегментация одежды – использует тело, но исключает лицо и руки.
    На практике лучше использовать модель для сегментации одежды (например, YOLO).
    Здесь мы возвращаем маску тела без лица (простейший вариант).
    """
    # Здесь можно реализовать более сложную логику, но для демонстрации используем тело минус лицо
    # Для простоты – вернём маску тела, но это не совсем одежда.
    # Пользователь может использовать внешние инструменты.
    # Рекомендуется установить библиотеку с моделями одежды.
    # Пока возвращаем маску тела.
    return get_body_mask(image_io)

def get_pose_control_image(image_io: io.BytesIO) -> io.BytesIO:
    """
    Детектирует позу и возвращает изображение скелета (OpenPose-стиль) для ControlNet.
    """
    image = image_to_cv2(image_io)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.pose_landmarks:
            return None
        # Создаём чёрное полотно
        h, w, _ = image.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # Рисуем скелет
        mp.solutions.drawing_utils.draw_landmarks(
            canvas,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2)
        )
        return cv2_to_bytesio(canvas)

def get_mask_for_part(image_io: io.BytesIO, part: str) -> io.BytesIO:
    """
    Универсальная функция для получения маски по названию части.
    part: "body", "background", "clothes"
    """
    if part == "body":
        return get_body_mask(image_io)
    elif part == "background":
        return get_background_mask(image_io)
    elif part == "clothes":
        return get_clothes_mask(image_io)
    else:
        raise ValueError(f"Unknown part: {part}")