import bootstrap_env  # noqa: F401,E402  — ДОЛЖЕН идти первым: переключает скрипт на библиотеки из папки env/
import os
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import io
from PIL import Image
import base64
from collections import Counter

class RealPoseRenamer:
    def __init__(self):
        print("🔧 Инициализация MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, image):
        """Извлекает ключевые точки позы"""
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark

    def point(self, lm, idx):
        return (lm[idx].x, lm[idx].y, lm[idx].z)

    def angle(self, a, b, c):
        a = np.array([a[0], a[1]])
        b = np.array([b[0], b[1]])
        c = np.array([c[0], c[1]])
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def analyze_pose(self, landmarks):
        if not landmarks:
            return None
        # Основные точки
        l_sh = self.point(landmarks, 11)
        r_sh = self.point(landmarks, 12)
        l_el = self.point(landmarks, 13)
        r_el = self.point(landmarks, 14)
        l_wr = self.point(landmarks, 15)
        r_wr = self.point(landmarks, 16)
        l_hip = self.point(landmarks, 23)
        r_hip = self.point(landmarks, 24)
        l_knee = self.point(landmarks, 25)
        r_knee = self.point(landmarks, 26)
        l_ank = self.point(landmarks, 27)
        r_ank = self.point(landmarks, 28)

        # Определяем положение тела
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2
        knee_mid_y = (l_knee[1] + r_knee[1]) / 2
        ankle_mid_y = (l_ank[1] + r_ank[1]) / 2
        shoulder_mid_y = (l_sh[1] + r_sh[1]) / 2

        # Углы в коленях
        l_k_ang = self.angle(l_hip, l_knee, l_ank)
        r_k_ang = self.angle(r_hip, r_knee, r_ank)
        avg_k_ang = (l_k_ang + r_k_ang) / 2

        # Положение
        if abs(shoulder_mid_y - hip_mid_y) < 0.05:
            pose_type = 'лежа'
        elif 80 < avg_k_ang < 140 and hip_mid_y > knee_mid_y:
            pose_type = 'сидя'
        elif 80 < avg_k_ang < 110 and knee_mid_y > ankle_mid_y:
            pose_type = 'на коленях'
        elif 110 < avg_k_ang < 150:
            pose_type = 'присед'
        else:
            pose_type = 'стоя'

        # Руки
        arms = []
        # Проверяем, подняты ли руки
        if l_wr[1] < l_sh[1] - 0.1 and r_wr[1] < r_sh[1] - 0.1:
            arms.append('с поднятыми руками')
        elif l_wr[1] < l_sh[1] - 0.1 or r_wr[1] < r_sh[1] - 0.1:
            arms.append('одна рука вверх')
        elif l_wr[0] > l_sh[0] + 0.1 and r_wr[0] > r_sh[0] + 0.1:
            arms.append('с вытянутыми руками')
        else:
            arms.append('с опущенными руками')

        # Ноги
        legs = []
        ankle_dist = abs(l_ank[0] - r_ank[0])
        if ankle_dist > 0.2:
            legs.append('ноги врозь')
        elif ankle_dist < 0.05:
            legs.append('ноги вместе')
        else:
            legs.append('ноги нейтрально')

        # Дополнительно: согнуты ли ноги
        if avg_k_ang < 100:
            legs.append('ноги согнуты')

        # Сборка названия
        name_parts = [pose_type] + arms + legs
        # Убираем дубли
        unique_parts = []
        for p in name_parts:
            if p not in unique_parts:
                unique_parts.append(p)
        return ' '.join(unique_parts)

    def process_image(self, image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = np.array(img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            landmarks = self.get_landmarks(img)
            if landmarks is None:
                return None
            return self.analyze_pose(landmarks)
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            return None

    def rename_parquet(self, parquet_path):
        print(f"📁 Загружаем {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        # Находим колонку с изображением
        img_col = None
        for col in df.columns:
            if 'image' in col.lower() or 'control' in col.lower() or 'skeleton' in col.lower():
                img_col = col
                break
        if img_col is None:
            for col in df.columns:
                if 'object' in str(df[col].dtype) or 'bytes' in str(df[col].dtype):
                    img_col = col
                    break
        if img_col is None:
            print("❌ Не найдена колонка с изображениями")
            return

        print(f"🔄 Обрабатываем {len(df)} поз...")
        name_counts = {}
        renamed = 0
        for i in range(len(df)):
            raw = df.iloc[i][img_col]
            # Извлекаем байты
            if isinstance(raw, dict):
                img_data = raw.get('bytes') or raw.get('data') or raw.get('image')
                if isinstance(img_data, str):
                    img_data = base64.b64decode(img_data)
            elif isinstance(raw, bytes):
                img_data = raw
            elif isinstance(raw, str):
                img_data = base64.b64decode(raw)
            else:
                img_data = None

            if img_data:
                name = self.process_image(img_data)
                if name:
                    if name in name_counts:
                        name_counts[name] += 1
                        name = f"{name} {name_counts[name]}"
                    else:
                        name_counts[name] = 1
                    df.at[i, 'pose_name'] = name
                    renamed += 1
                else:
                    df.at[i, 'pose_name'] = f"Поза {i+1}"
            else:
                df.at[i, 'pose_name'] = f"Поза {i+1}"
            if (i+1) % 50 == 0:
                print(f"  Обработано: {i+1}/{len(df)} (распознано: {renamed})")

        # Сохраняем
        output = parquet_path.replace('.parquet', '_real_names.parquet')
        df.to_parquet(output, index=False)
        print(f"\n✅ Сохранено в {output}")
        print(f"   Всего: {len(df)}, распознано: {renamed}, не распознано: {len(df)-renamed}")
        print(f"   Уникальных названий: {len(name_counts)}")
        if name_counts:
            print("\n🏆 Топ-10 названий:")
            for name, count in Counter(df['pose_name']).most_common(10):
                print(f"  {name}: {count} шт.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Использование: python real_pose_renamer.py путь_к_файлу.parquet")
        sys.exit(1)
    renamer = RealPoseRenamer()
    renamer.rename_parquet(sys.argv[1])