"""
Локальный процессор промптов через Ollama (aya:8b)
Автоматически загружает и выгружает модель для экономии VRAM
"""

import requests
import json
import subprocess
import threading
import time
import sys
import os

class PromptProcessor:
    def __init__(self):
        try:
            import app_config as config

            self.ollama_url = config.OLLAMA_BASE_URL
        except ImportError:
            self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        # Модель можно переопределить переменной окружения OLLAMA_MODEL
        self.model = os.environ.get("OLLAMA_MODEL", "aya:8b")  # или "phi3:mini" для экономии VRAM
        self.ollama_process = None
        self.model_loaded = False
        # Запросы к Ollama выполняются по одному: иначе один поток выгружает модель (keep_alive=0),
        # пока другой ещё ждёт ответа, а оба потока могли одновременно запустить `ollama serve`.
        self._lock = threading.Lock()
    
    def _is_ollama_running(self) -> bool:
        """Проверяет, запущен ли Ollama сервер"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель в память"""
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(self.model in m.get("name", "") for m in models)
            return False
        except:
            return False
    
    def _start_ollama_server(self):
        """Запускает Ollama сервер в фоне"""
        if self._is_ollama_running():
            print("✅ Ollama сервер уже запущен")
            return
        
        print("🚀 Запускаю Ollama сервер...")
        try:
            # Запускаем Ollama в фоне
            if sys.platform == "win32":
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Ждём запуска сервера
            for _ in range(10):
                time.sleep(1)
                if self._is_ollama_running():
                    print("✅ Ollama сервер запущен")
                    return
            
            print("⚠️ Не удалось запустить Ollama сервер")
        except Exception as e:
            print(f"❌ Ошибка запуска Ollama: {e}")
    
    def _load_model(self):
        """Загружает модель в память (если не загружена)"""
        if self._is_model_loaded():
            print(f"✅ Модель {self.model} уже загружена")
            return
        
        print(f"📥 Загружаю модель {self.model} в память...")
        try:
            # Пустой запрос для загрузки модели
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": " ",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.model_loaded = True
                print("✅ Модель загружена")
            else:
                print(f"⚠️ Ошибка загрузки модели: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
    
    def _unload_model(self):
        """Выгружает модель из памяти"""
        if not self._is_model_loaded():
            return
        
        print("🔄 Выгружаю модель из памяти...")
        try:
            # Отправляем запрос с keep_alive=0 для немедленной выгрузки
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": " ",
                    "stream": False,
                    "options": {"num_predict": 1},
                    "keep_alive": 0
                },
                timeout=10
            )
            self.model_loaded = False
            print("✅ Модель выгружена")
        except Exception as e:
            print(f"⚠️ Ошибка выгрузки модели: {e}")
    
    def _stop_ollama_server(self):
        """Останавливает Ollama сервер"""
        if not self._is_ollama_running():
            return
        
        print("🛑 Останавливаю Ollama сервер...")
        try:
            # Вежливая остановка
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], capture_output=True)
            else:
                subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            
            self.ollama_process = None
            self.model_loaded = False
            print("✅ Ollama сервер остановлен")
        except Exception as e:
            print(f"⚠️ Ошибка остановки Ollama: {e}")
    
    def process(self, prompt: str) -> str:
        """
        Обрабатывает промпт:
        1. Запускает Ollama (если не запущен)
        2. Загружает модель (если не загружена)
        3. Отправляет запрос
        4. Выгружает модель
        """
        if not prompt or len(prompt.strip()) < 2:
            return prompt

        with self._lock:
            return self._process_locked(prompt)

    def _process_locked(self, prompt: str) -> str:
        try:
            # 1. Запускаем сервер
            self._start_ollama_server()
            
            # 2. Загружаем модель
            self._load_model()
            
            # 3. Обрабатываем промпт
            print(f"🔄 Обрабатываю промпт: {prompt[:50]}...")
            
            system_prompt = """Ты — помощник для генерации изображений в Stable Diffusion.

Твоя задача: преобразовать русский промпт в строку тегов на английском.

ПРАВИЛА:
1. Переведи всё на английский
2. Разбей на отдельные теги через запятую
3. Отсортируй теги в таком порядке: СТИЛЬ → СУБЪЕКТ → ДЕЙСТВИЕ → ОКРУЖЕНИЕ → ОСВЕЩЕНИЕ → ЦВЕТА → ДЕТАЛИ → КАЧЕСТВО
4. НЕ ДОБАВЛЯЙ теги, которых нет в запросе
5. НЕ ПРИДУМЫВАЙ новые объекты
6. Ответь ТОЛЬКО строкой с тегами, без пояснений

ПРИМЕРЫ:
Вход: "девочка"
Выход: girl

Вход: "киберпанк девушка с синими волосами бежит по ночному городу"
Выход: cyberpunk, girl with blue hair, running, through the night city

Вход: "закат над морем, высокое качество, масляная живопись"
Выход: oil painting, sunset over the sea, high quality

Выполняй строго по правилам."""

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                translated = (result.get("response") or "").strip()
                translated = translated.strip('"\'').strip()
                if not translated:  # пустой ответ модели — лучше исходный промпт, чем пустая генерация
                    print("⚠️ Ollama вернул пустой ответ, использую исходный промпт")
                    return prompt
                print(f"✅ Результат: {translated}")
                return translated
            else:
                print(f"⚠️ Ошибка: HTTP {response.status_code}")
                return prompt
                
        except Exception as e:
            print(f"❌ Ошибка обработки: {e}")
            return prompt
        
        finally:
            # 4. Выгружаем модель (освобождаем VRAM)
            self._unload_model()
    
    def __del__(self):
        """Деструктор — выгружаем модель при удалении объекта"""
        try:
            self._unload_model()
        except:
            pass


# Глобальный экземпляр
_processor = None

def get_processor():
    global _processor
    if _processor is None:
        _processor = PromptProcessor()
    return _processor

def translate_prompt(prompt: str) -> str:
    """Основная функция для вызова из бота"""
    processor = get_processor()
    return processor.process(prompt)

def fast_process(prompt: str) -> str:
    """Быстрая обработка промпта для вызова из бота"""
    try:
        processor = get_processor()
        return processor.process(prompt)
    except Exception as e:
        print(f"Ошибка обработки промпта: {e}")
        return prompt


# Интерактивный режим
if __name__ == "__main__":
    print("🤖 Процессор промптов с автоматической выгрузкой модели")
    print("-" * 50)
    
    processor = PromptProcessor()
    
    while True:
        user_input = input("\n📝 Введите промпт (или 'exit' для выхода): ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if not user_input:
            continue
        
        result = processor.process(user_input)
        print(f"\n✅ Результат:\n{result}")
    
    print("👋 До свидания!")