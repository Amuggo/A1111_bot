"""
Асинхронный процессор промптов через Ollama (aya:8b).
Автоматически загружает и выгружает модель для экономии VRAM.

Все запросы идут через общий aiohttp-сеанс (http_client). Если Ollama недоступна или отвечает ошибкой —
возвращается исходный промпт без обработки (как и раньше).
"""

import asyncio
import logging
import os
import subprocess
import sys

import aiohttp

import http_client
from aio import LazyLock

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — помощник для генерации изображений в Stable Diffusion.

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
        # Запросы к Ollama выполняются по одному (asyncio.Lock): иначе одна задача выгружает модель
        # (keep_alive=0), пока другая ещё ждёт ответа, а обе могли одновременно запустить `ollama serve`.
        # Остальные задачи при этом НЕ блокируются — они просто ждут своей очереди на этом замке.
        self._lock = LazyLock()

    async def _get(self, path: str, timeout: float):
        session = await http_client.get_session()
        async with session.get(f"{self.ollama_url}{path}", timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)

    async def _post(self, path: str, payload: dict, timeout: float):
        """POST к Ollama. Возвращает (статус, json|None)."""
        session = await http_client.get_session()
        async with session.post(f"{self.ollama_url}{path}", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            data = await resp.json(content_type=None) if resp.status == 200 else None
            return resp.status, data

    async def _is_ollama_running(self) -> bool:
        """Проверяет, запущен ли Ollama сервер"""
        try:
            return await self._get("/api/tags", 2) is not None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def _is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель в память"""
        try:
            data = await self._get("/api/ps", 2)
            if data is None:
                return False
            return any(self.model in m.get("name", "") for m in data.get("models", []))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def _start_ollama_server(self):
        """Запускает Ollama сервер в фоне"""
        if await self._is_ollama_running():
            print("✅ Ollama сервер уже запущен")
            return

        print("🚀 Запускаю Ollama сервер...")
        try:
            # Popen только создаёт процесс и возвращается сразу (не блокирует), но выносим в поток на всякий случай
            def spawn():
                kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                return subprocess.Popen(["ollama", "serve"], **kwargs)

            self.ollama_process = await asyncio.to_thread(spawn)

            # Ждём запуска сервера (asyncio.sleep — другие задачи бота продолжают работать)
            for _ in range(10):
                await asyncio.sleep(1)
                if await self._is_ollama_running():
                    print("✅ Ollama сервер запущен")
                    return

            print("⚠️ Не удалось запустить Ollama сервер")
        except Exception as e:
            print(f"❌ Ошибка запуска Ollama: {e}")

    async def _load_model(self):
        """Загружает модель в память (если не загружена)"""
        if await self._is_model_loaded():
            print(f"✅ Модель {self.model} уже загружена")
            return

        print(f"📥 Загружаю модель {self.model} в память...")
        try:
            # Пустой запрос для загрузки модели
            status, _ = await self._post(
                "/api/generate",
                {"model": self.model, "prompt": " ", "stream": False, "options": {"num_predict": 1}},
                timeout=30,
            )
            if status == 200:
                self.model_loaded = True
                print("✅ Модель загружена")
            else:
                print(f"⚠️ Ошибка загрузки модели: {status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"❌ Ошибка загрузки модели: {e}")

    async def _unload_model(self):
        """Выгружает модель из памяти"""
        if not await self._is_model_loaded():
            return

        print("🔄 Выгружаю модель из памяти...")
        try:
            # Запрос с keep_alive=0 для немедленной выгрузки
            await self._post(
                "/api/generate",
                {"model": self.model, "prompt": " ", "stream": False, "options": {"num_predict": 1}, "keep_alive": 0},
                timeout=10,
            )
            self.model_loaded = False
            print("✅ Модель выгружена")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"⚠️ Ошибка выгрузки модели: {e}")

    async def _stop_ollama_server(self):
        """Останавливает Ollama сервер"""
        if not await self._is_ollama_running():
            return

        print("🛑 Останавливаю Ollama сервер...")
        try:
            def stop():
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], capture_output=True)
                else:
                    subprocess.run(["pkill", "-f", "ollama"], capture_output=True)

            await asyncio.to_thread(stop)
            self.ollama_process = None
            self.model_loaded = False
            print("✅ Ollama сервер остановлен")
        except Exception as e:
            print(f"⚠️ Ошибка остановки Ollama: {e}")

    async def process(self, prompt: str) -> str:
        """
        Обрабатывает промпт:
        1. Запускает Ollama (если не запущен)
        2. Загружает модель (если не загружена)
        3. Отправляет запрос
        4. Выгружает модель
        При недоступности Ollama возвращает исходный промпт.
        """
        if not prompt or len(prompt.strip()) < 2:
            return prompt

        async with self._lock:
            return await self._process_locked(prompt)

    async def _process_locked(self, prompt: str) -> str:
        try:
            # 1. Запускаем сервер
            await self._start_ollama_server()

            # 2. Загружаем модель
            await self._load_model()

            # 3. Обрабатываем промпт
            print(f"🔄 Обрабатываю промпт: {prompt[:50]}...")

            status, result = await self._post(
                "/api/generate",
                {
                    "model": self.model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 256},
                },
                timeout=45,
            )

            if status == 200 and result is not None:
                translated = (result.get("response") or "").strip()
                translated = translated.strip('"\'').strip()
                if not translated:  # пустой ответ модели — лучше исходный промпт, чем пустая генерация
                    print("⚠️ Ollama вернул пустой ответ, использую исходный промпт")
                    return prompt
                print(f"✅ Результат: {translated}")
                return translated
            print(f"⚠️ Ошибка: HTTP {status}")
            return prompt

        except asyncio.CancelledError:
            raise  # отмена задачи пользователем — не глотаем (модель выгрузится в finally)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as e:
            # Ollama недоступна/ответила мусором — продолжаем с «сырым» промптом, как и раньше
            print(f"❌ Ошибка обработки (Ollama недоступна?): {e}")
            return prompt

        finally:
            # 4. Выгружаем модель (освобождаем VRAM); щит — чтобы отмена задачи не оставила её в памяти
            try:
                await asyncio.shield(self._unload_model())
            except Exception as e:
                print(f"⚠️ Не удалось выгрузить модель: {e}")


# Глобальный экземпляр
_processor = None


def get_processor() -> PromptProcessor:
    global _processor
    if _processor is None:
        _processor = PromptProcessor()
    return _processor


async def translate_prompt(prompt: str) -> str:
    """Основная функция для вызова из бота"""
    return await get_processor().process(prompt)


async def fast_process(prompt: str) -> str:
    """Быстрая обработка промпта для вызова из бота (при любой ошибке — исходный промпт)"""
    try:
        return await get_processor().process(prompt)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("Ошибка обработки промпта: %s", e)
        return prompt


# Интерактивный режим
if __name__ == "__main__":
    async def _interactive() -> None:
        print("🤖 Процессор промптов с автоматической выгрузкой модели")
        print("-" * 50)
        processor = PromptProcessor()
        try:
            while True:
                user_input = (await asyncio.to_thread(input, "\n📝 Введите промпт (или 'exit' для выхода): ")).strip()
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                if not user_input:
                    continue
                result = await processor.process(user_input)
                print(f"\n✅ Результат:\n{result}")
        finally:
            await http_client.close_session()

    asyncio.run(_interactive())
