"""
Очередь задач генерации.

* ``asyncio.Queue`` + N воркеров (QUEUE_WORKERS). Хендлер кладёт задачу в очередь и сразу отвечает
  «⏳ Задача №N в очереди» — приём сообщений других пользователей при этом не блокируется.
* Реестр активных задач ``dict[task_id, asyncio.Task]``: отмена кнопкой «❌ Отменить» или командой /cancel
  (а) прерывает генерацию в WebUI (POST /sdapi/v1/interrupt) — но только если именно эта задача сейчас
  считается на сервере, чтобы не оборвать чужую; (б) отменяет asyncio-задачу; (в) убирает её из реестра.
* Пока задача выполняется, фоновая задача шлёт «печатает…» раз в ~4 секунды.
* Воркер не умирает от исключений: тело цикла обёрнуто в try/except с logging.exception.
"""
import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Awaitable, Callable, Dict, List, Optional

from http_client import sd_gate, sd_interrupt
from tg import BotAPI, InlineKeyboardButton, InlineKeyboardMarkup

log = logging.getLogger(__name__)

TYPING_INTERVAL = 4.0        # секунд между «печатает…»
STATUS_MIN_INTERVAL = 1.5    # не чаще одного редактирования статус-сообщения за это время
MAX_POSITION_UPDATES = 10    # сколько ожидающих сообщений обновляем после каждого завершения

TaskFactory = Callable[["Task"], Awaitable[None]]


class QueueFull(Exception):
    """Очередь переполнена или у пользователя слишком много активных задач (текст безопасен для показа)."""


@dataclass
class Task:
    id: int
    chat_id: int
    user_id: str
    kind: str
    factory: TaskFactory
    queue: "TaskQueue"
    status_message_id: Optional[int] = None
    state: str = "queued"  # queued | running | done | failed | cancelled
    cancel_requested: bool = False
    runner: Optional["asyncio.Task[None]"] = None
    created: float = field(default_factory=time.monotonic)
    _last_edit: float = 0.0
    _last_text: str = ""

    async def edit_status(self, text: str, parse_mode: Optional[str] = None, force: bool = False,
                          keep_cancel_button: bool = True) -> None:
        """Обновляет статус-сообщение задачи (с кнопкой «Отменить»). Не чаще раза в 1.5 с, кроме force=True."""
        if self.status_message_id is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_edit < STATUS_MIN_INTERVAL or text == self._last_text):
            return
        self._last_edit, self._last_text = now, text
        await self.queue._safe_edit(self.chat_id, self.status_message_id, text, parse_mode,
                                    self.queue.cancel_markup(self.id) if keep_cancel_button else None)

    async def delete_status(self) -> None:
        if self.status_message_id is not None:
            with suppress(Exception):
                await self.queue.bot.delete_message(self.chat_id, self.status_message_id)


class TaskQueue:
    def __init__(self, bot: BotAPI, workers: int = 2, maxsize: int = 50, max_per_user: int = 3):
        self.bot = bot
        self.workers = max(1, workers)
        self.max_per_user = max(1, max_per_user)
        self._queue: "asyncio.Queue[Task]" = asyncio.Queue(maxsize=maxsize)
        self._ids = count(1)
        self._tasks: Dict[int, Task] = {}                 # все незавершённые задачи (в очереди + выполняются)
        self.active: Dict[int, "asyncio.Task[None]"] = {}  # id -> выполняющаяся asyncio-задача (реестр из ТЗ)
        self._worker_tasks: List["asyncio.Task[None]"] = []
        self._closed = False

    # ---------- жизненный цикл ----------
    async def start(self) -> None:
        if self._worker_tasks:
            return
        self._worker_tasks = [asyncio.create_task(self._worker(i + 1), name=f"queue-worker-{i + 1}")
                              for i in range(self.workers)]
        log.info("Очередь запущена: воркеров %d", self.workers)

    async def shutdown(self) -> None:
        """Остановка: прерываем выполняющиеся задачи (и генерацию в WebUI), останавливаем воркеров."""
        self._closed = True
        for task in list(self._tasks.values()):
            if task.state == "running":
                await self._cancel_running(task)
            elif task.state == "queued":
                task.state = "cancelled"
        for worker in self._worker_tasks:
            worker.cancel()
        for worker in self._worker_tasks:
            with suppress(asyncio.CancelledError, Exception):
                await worker
        self._worker_tasks.clear()
        self._tasks.clear()
        self.active.clear()

    # ---------- постановка в очередь ----------
    def cancel_markup(self, task_id: int) -> InlineKeyboardMarkup:
        markup = InlineKeyboardMarkup(row_width=1)
        markup.add(InlineKeyboardButton("❌ Отменить", callback_data=f"task_cancel_{task_id}"))
        return markup

    def user_tasks(self, chat_id: int) -> List[Task]:
        return [t for t in self._tasks.values() if t.chat_id == chat_id]

    def _queued(self) -> List[Task]:
        return [t for t in self._tasks.values() if t.state == "queued"]

    def _position(self, task: Task) -> int:
        queued = sorted(self._queued(), key=lambda t: t.id)
        return queued.index(task) + 1 if task in queued else 0

    async def submit(self, chat_id: int, user_id: str, kind: str, factory: TaskFactory,
                     reply_to_message_id: Optional[int] = None) -> Task:
        """Кладёт задачу в очередь и сразу отвечает пользователю. Бросает QueueFull, если места нет."""
        if self._closed:
            raise QueueFull("Бот останавливается, попробуйте позже.")
        if len(self.user_tasks(chat_id)) >= self.max_per_user:
            raise QueueFull(f"У вас уже {self.max_per_user} активных задач. Дождитесь их завершения или нажмите /cancel.")
        task = Task(id=next(self._ids), chat_id=chat_id, user_id=str(user_id), kind=kind, factory=factory, queue=self)
        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            raise QueueFull("Очередь переполнена, попробуйте через минуту.")
        self._tasks[task.id] = task
        position = self._position(task)
        running = sum(1 for t in self._tasks.values() if t.state == "running")
        text = (f"⏳ Задача №{task.id} в очереди (позиция {position})" if running >= self.workers or position > 1
                else f"⏳ Задача №{task.id} принята")
        try:
            sent = await self.bot.send_message(chat_id, text, reply_markup=self.cancel_markup(task.id),
                                               reply_to_message_id=reply_to_message_id)
            task.status_message_id = sent.message_id
            task._last_text = text
        except Exception:
            log.exception("Не удалось отправить статус задачи №%s", task.id)
        return task

    # ---------- отмена ----------
    async def cancel(self, task_id: int, chat_id: Optional[int] = None) -> str:
        """
        Отменяет задачу. Возвращает: 'queued' | 'running' | 'unknown' | 'forbidden'.
        chat_id — владелец запроса (чужие задачи отменять нельзя).
        """
        task = self._tasks.get(task_id)
        if task is None:
            return "unknown"
        if chat_id is not None and task.chat_id != chat_id:
            return "forbidden"
        if task.state == "queued":
            task.state = "cancelled"
            self._tasks.pop(task.id, None)
            await self._safe_edit(task.chat_id, task.status_message_id, f"❌ Задача №{task.id} отменена", None, None)
            await self._refresh_positions()
            return "queued"
        if task.state == "running":
            await self._cancel_running(task)
            return "running"
        return "unknown"

    async def cancel_user(self, chat_id: int) -> int:
        """Отменяет все задачи пользователя (команда /cancel). Возвращает, сколько отменено."""
        done = 0
        for task in self.user_tasks(chat_id):
            if await self.cancel(task.id, chat_id) in ("queued", "running"):
                done += 1
        return done

    async def _cancel_running(self, task: Task) -> None:
        task.cancel_requested = True
        runner = task.runner
        if runner is None or runner.done():
            return
        # (а) прерываем генерацию на стороне WebUI — только если она считается именно для этой задачи
        if sd_gate.is_running(runner):
            await sd_interrupt()
        # (б) отменяем asyncio-задачу, (в) убираем из реестра
        runner.cancel()
        self.active.pop(task.id, None)

    # ---------- воркеры ----------
    async def _worker(self, number: int) -> None:
        while True:
            task = await self._queue.get()
            try:
                if task.state == "cancelled":
                    continue
                await self._run_task(task)
            except asyncio.CancelledError:
                raise  # остановка воркера при shutdown
            except Exception:
                # воркер не должен умирать из-за ошибки одной задачи
                log.exception("Воркер %d: необработанная ошибка в задаче №%s", number, task.id)
            finally:
                self._queue.task_done()
                self._tasks.pop(task.id, None)
                self.active.pop(task.id, None)
                if not self._closed:
                    with suppress(Exception):
                        await self._refresh_positions()

    async def _run_task(self, task: Task) -> None:
        task.state = "running"
        await task.edit_status(f"🎨 Задача №{task.id}: запускаю…", force=True)
        typing = asyncio.create_task(self._typing_loop(task.chat_id))
        task.runner = asyncio.create_task(task.factory(task), name=f"task-{task.id}")
        self.active[task.id] = task.runner
        try:
            await task.runner
            task.state = "done"
        except asyncio.CancelledError:
            if task.cancel_requested or task.runner.cancelled():
                # отмена пользователем: воркер продолжает жить
                task.state = "cancelled"
                await self._safe_edit(task.chat_id, task.status_message_id, f"❌ Задача №{task.id} отменена", None, None)
            else:
                # отменён сам воркер (shutdown) — гасим задачу и пробрасываем отмену
                task.runner.cancel()
                with suppress(BaseException):
                    await task.runner
                raise
        except Exception as error:
            task.state = "failed"
            log.exception("Задача №%s (%s) завершилась ошибкой", task.id, task.kind)
            await self._safe_edit(task.chat_id, task.status_message_id, f"❌ Ошибка: {str(error)[:3500]}", None, None)
        finally:
            typing.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await typing
            self.active.pop(task.id, None)

    async def _typing_loop(self, chat_id: int) -> None:
        """Раз в ~4 секунды показывает «печатает…», пока задача выполняется."""
        try:
            while True:
                with suppress(Exception):
                    await self.bot.send_chat_action(chat_id, "typing")
                await asyncio.sleep(TYPING_INTERVAL)
        except asyncio.CancelledError:
            pass

    async def _refresh_positions(self) -> None:
        """После завершения/отмены обновляет номера позиций у ожидающих (первые MAX_POSITION_UPDATES)."""
        for task in sorted(self._queued(), key=lambda t: t.id)[:MAX_POSITION_UPDATES]:
            text = f"⏳ Задача №{task.id} в очереди (позиция {self._position(task)})"
            if text != task._last_text and task.status_message_id is not None:
                task._last_text = text
                await self._safe_edit(task.chat_id, task.status_message_id, text, None, self.cancel_markup(task.id))

    async def _safe_edit(self, chat_id: int, message_id: Optional[int], text: str,
                         parse_mode: Optional[str], markup: Any) -> None:
        """edit_message_text, который не падает («message is not modified», удалённое сообщение и т.п.)."""
        if message_id is None:
            return
        try:
            await self.bot.edit_message_text(text, chat_id, message_id, parse_mode=parse_mode, reply_markup=markup)
        except Exception as error:
            if "message is not modified" not in str(error):
                log.debug("Не удалось изменить сообщение %s: %s", message_id, error)
