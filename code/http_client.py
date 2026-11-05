"""
Общий асинхронный HTTP-клиент (aiohttp) для Stable Diffusion WebUI и Ollama.

Один переиспользуемый ClientSession на всё приложение (создаётся лениво, закрывается при остановке бота).
Таймауты: total=None (генерация может идти минуты), sock_connect=10, sock_read=300 (HTTP_SOCK_READ_TIMEOUT) —
ограничивается только пауза «WebUI молчит», а не общая длительность запроса.
"""
import asyncio
import logging
from typing import Any, Optional

import aiohttp

import app_config as config
from aio import LazyLock, SDGate

log = logging.getLogger(__name__)

SD_BASE_URL: str = config.SD_BASE_URL

# Одно общее «окно» доступа к WebUI для всех задач бота (см. aio.SDGate)
sd_gate = SDGate(config.SD_MAX_PARALLEL)

_session: Optional[aiohttp.ClientSession] = None
_session_lock = LazyLock()


class SDError(Exception):
    """Ошибка обращения к WebUI/Ollama: HTTP-статус, недоступность или таймаут (текст безопасен для показа пользователю)."""


def make_timeout(total: Optional[float] = None, sock_read: Optional[float] = None,
                 sock_connect: Optional[float] = None) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=total,
        sock_connect=sock_connect if sock_connect is not None else config.HTTP_SOCK_CONNECT_TIMEOUT,
        sock_read=sock_read if sock_read is not None else config.HTTP_SOCK_READ_TIMEOUT,
    )


async def get_session() -> aiohttp.ClientSession:
    """Возвращает общий ClientSession (создаёт при первом обращении)."""
    global _session
    if _session is not None and not _session.closed:
        return _session
    async with _session_lock:
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession(timeout=make_timeout())
    return _session


async def close_session() -> None:
    """Закрывает общий сеанс (вызывается при остановке бота)."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


async def request_json(method: str, url: str, *, json: Any = None,
                       timeout: Optional[aiohttp.ClientTimeout] = None) -> Any:
    """
    Запрос с разбором JSON-ответа. HTTP >= 400 → SDError с телом ответа (WebUI кладёт причину в тело);
    недоступность и таймаут тоже превращаются в SDError с понятным текстом.
    """
    session = await get_session()
    try:
        kwargs = {"json": json} if json is not None else {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        async with session.request(method, url, **kwargs) as resp:
            if resp.status >= 400:
                body = (await resp.text(errors="replace"))[:2000]
                raise SDError(f"HTTP {resp.status}: {body}")
            return await resp.json(content_type=None)
    except SDError:
        raise
    except asyncio.TimeoutError as e:
        raise SDError(f"Превышено время ожидания ответа от {url}") from e
    except aiohttp.ClientError as e:
        raise SDError(f"Не удалось связаться с {url}: {e}") from e


async def sd_get_json(path: str, timeout: float = 10) -> Any:
    """GET к WebUI (короткий таймаут для справочных запросов)."""
    return await request_json("GET", f"{SD_BASE_URL}/{path.lstrip('/')}", timeout=aiohttp.ClientTimeout(total=timeout))


async def sd_post_json(path: str, payload: Optional[dict] = None, timeout: Optional[float] = None) -> Any:
    """POST к WebUI. timeout=None — «долгий» режим для генерации (без общего лимита, sock_read=300)."""
    to = aiohttp.ClientTimeout(total=timeout) if timeout else make_timeout()
    return await request_json("POST", f"{SD_BASE_URL}/{path.lstrip('/')}", json=payload if payload is not None else {}, timeout=to)


async def sd_interrupt() -> bool:
    """Просит WebUI прервать текущую генерацию (POST /sdapi/v1/interrupt). Возвращает True при успехе."""
    try:
        await sd_post_json("sdapi/v1/interrupt", {}, timeout=10)
        return True
    except Exception as e:
        log.warning("Не удалось прервать генерацию в WebUI: %s", e)
        return False
