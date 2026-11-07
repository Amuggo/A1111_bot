"""
Перехват stdout/stderr: последние строки лога хранятся в памяти (команда /logs в Telegram и `logs` в консоли)
и пишутся в файл data/logs/bot.log с ротацией (команда «Лог-файл»). Вывод в терминал при этом не меняется.
Токены бота маскируются. LOG_TO_FILE=0 отключает файл, LOG_BUFFER_LINES задаёт размер буфера (по умолчанию 3000).
"""

import collections
import logging.handlers
import os
import re
import sys
import threading
import time
from typing import Deque, List, Optional

import app_config as config

_TOKEN_RE = re.compile(r"\d{6,}:[A-Za-z0-9_-]{30,}")
_ERROR_RE = re.compile(r"❌|⚠️|🚨|Traceback|Error|ERROR|Exception|CRITICAL|WARNING", re.IGNORECASE)

_lock = threading.Lock()
_lines: Deque[str] = collections.deque(maxlen=3000)
_file_handler: Optional[logging.handlers.RotatingFileHandler] = None
_file_path: Optional[str] = None
_installed = False


def _record(line: str) -> None:
    line = _TOKEN_RE.sub("<токен скрыт>", line.rstrip("\r\n"))
    if not line.strip():
        return
    entry = f"{time.strftime('%d.%m %H:%M:%S')} {line}"
    with _lock:
        _lines.append(entry)
        if _file_handler is not None:
            try:
                _file_handler.emit(logging.makeLogRecord({"msg": entry}))  # ротация выполняется самим обработчиком
            except Exception:
                pass  # логирование не должно ронять бота


class _Tee:
    """Пишет в исходный поток и копит строки; остальные атрибуты (isatty, encoding, fileno…) отдаёт исходному потоку."""

    def __init__(self, stream):
        self._stream = stream
        self._partial = ""

    def write(self, text):
        written = self._stream.write(text)
        try:
            self._partial += text if isinstance(text, str) else text.decode("utf-8", "replace")
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                _record(line)
        except Exception:
            self._partial = ""
        return written

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    """Подключает перехват; безопасно вызывать повторно."""
    global _installed, _file_handler, _file_path, _lines
    if _installed:
        return
    _installed = True
    try:
        size = int(os.environ.get("LOG_BUFFER_LINES", "").strip() or 3000)
        _lines = collections.deque(maxlen=max(100, size))
    except ValueError:
        pass
    if os.environ.get("LOG_TO_FILE", "1").strip().lower() not in ("0", "false", "no"):
        try:
            log_dir = os.path.join(config.DATA_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)
            _file_path = os.path.join(log_dir, "bot.log")
            _file_handler = logging.handlers.RotatingFileHandler(
                _file_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            _file_handler.handleError = lambda record: None
        except Exception as e:
            print(f"⚠️ Файл лога недоступен: {e}")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and not isinstance(stream, _Tee):
            setattr(sys, name, _Tee(stream))


def tail(count: int = 40, errors_only: bool = False) -> List[str]:
    with _lock:
        lines = list(_lines)
    if errors_only:
        lines = [line for line in lines if _ERROR_RE.search(line)]
    return lines[-max(1, count):]


def log_file_path() -> Optional[str]:
    return _file_path if _file_path and os.path.exists(_file_path) else None
