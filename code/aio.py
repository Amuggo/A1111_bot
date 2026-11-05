"""
Мелкие asyncio-утилиты, общие для проекта.

* LazyLock — asyncio.Lock, который создаётся при первом использовании уже внутри работающего цикла
  (на Python < 3.10 Lock, созданный на уровне модуля, привязывается не к тому циклу).
* SDGate — ограничитель числа одновременных запросов к WebUI и учёт «какая задача сейчас реально
  считается на сервере». Это нужно для безопасной отмены: /sdapi/v1/interrupt прерывает ТЕКУЩУЮ генерацию
  WebUI, поэтому вызывать его можно только для задачи, которая её и выполняет, а не ждёт в очереди сервера.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Set


class LazyLock:
    def __init__(self) -> None:
        self._lock: Optional[asyncio.Lock] = None

    def _get(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def __aenter__(self) -> "LazyLock":
        await self._get().acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._get().release()


class SDGate:
    def __init__(self, parallel: int = 1) -> None:
        self._parallel = max(1, parallel)
        self._sem: Optional[asyncio.Semaphore] = None
        self._owners: Set[asyncio.Task] = set()

    @property
    def parallel(self) -> int:
        return self._parallel

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Занимает «место» в WebUI на время запроса (ожидание своей очереди не блокирует другие корутины)."""
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._parallel)
        await self._sem.acquire()
        owner = asyncio.current_task()
        if owner is not None:
            self._owners.add(owner)
        try:
            yield
        finally:
            if owner is not None:
                self._owners.discard(owner)
            self._sem.release()

    def is_running(self, task: Optional[asyncio.Task]) -> bool:
        """True, если эта задача сейчас держит место в WebUI (то есть её запрос реально выполняется)."""
        return task is not None and task in self._owners
