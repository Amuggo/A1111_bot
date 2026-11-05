"""
Потокобезопасное хранилище JSON-файлов (настройки пользователей, лица и т.д.).

Зачем это нужно. telebot обрабатывает сообщения в нескольких потоках, а раньше каждый
модуль делал «прочитать файл → изменить → записать файл» без блокировок:
  * два параллельных обработчика теряли изменения друг друга (гонка read-modify-write);
  * запись шла прямо в рабочий файл — если процесс обрывался посреди записи, JSON оказывался
    битым, а `except: return {}` молча превращал это в пустые настройки ВСЕХ пользователей,
    которые затем перезаписывали файл.

Что делает JsonStore:
  * все операции чтения/изменения/записи одного файла выполняются под одной блокировкой;
  * запись атомарная: во временный файл рядом, fsync, затем os.replace;
  * если файл повреждён, он не затирается, а сохраняется рядом как *.corrupt-<время>,
    после чего хранилище начинает с пустых данных.
"""

import asyncio
import copy
import json
import os
import tempfile
import threading
import time
from typing import Any, Callable


class JsonStore:
    def __init__(self, path: str, default_factory: Callable[[], Any] = dict):
        self.path = path
        self._default_factory = default_factory
        self._lock = threading.RLock()

    # ---- низкоуровневое чтение/запись (вызывать под self._lock) ----
    def _read_unlocked(self) -> Any:
        if not os.path.exists(self.path):
            return self._default_factory()
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                content = file.read()
            if not content.strip():
                return self._default_factory()
            return json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            backup = f"{self.path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
            try:
                os.replace(self.path, backup)
                print(f"⚠️ Файл {self.path} повреждён ({error}); сохранён как {backup}")
            except OSError as move_error:
                print(f"⚠️ Файл {self.path} повреждён ({error}) и не может быть перемещён: {move_error}")
            return self._default_factory()
        except OSError as error:
            print(f"⚠️ Не удалось прочитать {self.path}: {error}")
            return self._default_factory()

    def _write_unlocked(self, data: Any) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(self.path) + ".", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            # На Windows os.replace может кратко падать, если файл открыт антивирусом/индексатором
            for attempt in range(6):
                try:
                    os.replace(tmp_path, self.path)
                    return
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ---- публичный API ----
    def read(self) -> Any:
        """Возвращает копию данных файла."""
        with self._lock:
            return copy.deepcopy(self._read_unlocked())

    def write(self, data: Any) -> None:
        """Полностью заменяет содержимое файла."""
        with self._lock:
            self._write_unlocked(data)

    def update(self, mutator: Callable[[Any], Any]) -> Any:
        """
        Атомарно: читает данные, вызывает mutator(data) (он меняет data на месте и/или
        возвращает результат), записывает файл, если данные изменились. Возвращает результат mutator.
        """
        with self._lock:
            data = self._read_unlocked()
            before = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
            result = mutator(data)
            after = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
            if before != after or not os.path.exists(self.path):
                self._write_unlocked(data)
            return result

    # ---- асинхронные обёртки: блокирующий ввод-вывод уходит в поток, цикл событий не блокируется ----
    async def aread(self) -> Any:
        return await asyncio.to_thread(self.read)

    async def awrite(self, data: Any) -> None:
        await asyncio.to_thread(self.write, data)

    async def aupdate(self, mutator: Callable[[Any], Any]) -> Any:
        """Атомарное «прочитать → изменить → записать» в потоке (mutator должен быть быстрым и без await)."""
        return await asyncio.to_thread(self.update, mutator)
