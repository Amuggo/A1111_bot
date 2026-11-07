"""
Админ-консоль в терминале: управление запущенным ботом без Telegram.
Включается автоматически, если бот запущен из интерактивного терминала (ADMIN_CONSOLE=0 — выключить).
Команды: help, check, status, queue, cancel, interrupt, reload, users, loglevel, quit.
"""

import asyncio
import json
import logging
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from typing import Awaitable, Callable, List, Optional

import app_config as config

HELP = """\
Команды админ-консоли:
  help                  — эта справка
  check [--no-telegram] — полная проверка систем (окружение, Telegram, WebUI, расширения, Ollama)
  status                — аптайм, режим, очередь, число пользователей
  queue                 — список задач в очереди и выполняемых
  cancel <chat_id|all>  — отменить задачи пользователя или все задачи
  interrupt             — прервать текущую генерацию в WebUI (если «завис»)
  reload                — обновить кэши LoRA/моделей/эмбеддингов/поз/ReActor/FaceID
  users                 — сколько пользователей и лиц сохранено
  loglevel <уровень>    — DEBUG | INFO | WARNING | ERROR
  sub status            — подписки: режим, оплата, число пользователей по тарифам
  sub on | off          — включить / полностью выключить систему подписок на лету (выключена — ограничений нет)
  sub grant <id> <free|pro> [дней]  — выдать тариф (без срока — бессрочно), пользователь получит уведомление
  sub revoke <id>       — снять подписку (вернётся Free)
  sub info <id>         — тариф и расход лимита пользователя
  sub list              — все выданные тарифы
  sub reset <id>        — обнулить дневной счётчик генераций пользователя
  sub pay on | off      — включить / выключить приём оплаты Telegram Stars (кнопки в /plan)
  sub payments [N]      — последние платежи (id платежа нужен для возврата)
  sub refund <charge_id> — вернуть Stars пользователю и откатить дни подписки
  sub testpay <id> <товар> — имитировать успешную оплату (без Stars) вместе с реферальным бонусом
  sub credits <id> <N>  — добавить (или отнять, если N<0) бонусные генерации
  sub ref <id>          — реферальная статистика пользователя
  admin add|remove <id> — выдать / забрать права администратора; admin list — список
  broadcast <all|pro|free> <текст> — рассылка: показывает предпросмотр и число получателей; затем
                        broadcast send (запустить) | broadcast cancel (отменить/остановить) | broadcast status
  uimode simple|auto [id] — предпросмотр интерфейса глазами клиента Free (auto — вернуть обычный вид)
  logs [N] | logs errors — последние строки лога / только ошибки и предупреждения (в Telegram: logs file — файл)
  quit                  — корректно остановить бота (в режиме webhook — Ctrl+C)

Права: владелец (ADMIN_USER_ID) и терминал — всё. Администраторы (ADMIN_USER_IDS) — просмотр и безопасные действия:
status, queue, users, check, logs, cancel <id>, interrupt, reload, sub status|info|list|payments|ref, admin list.
Деньги, тарифы, права, остановка — только владельцу.
"""


@dataclass
class ConsoleContext:
    get_queue: Callable[[], Optional[object]]
    reload_caches: Callable[[], Awaitable[None]]
    stop: Callable[[], Awaitable[None]]
    started_at: float
    mode: str
    notify: Optional[Callable[[int, str], Awaitable[object]]] = None  # отправка сообщения пользователю в Telegram
    refund_stars: Optional[Callable[[int, str], Awaitable[object]]] = None  # возврат Stars (user_id, charge_id)
    send_user: Optional[Callable[[int, str], Awaitable[str]]] = None  # обычное сообщение пользователю → "ok" | "blocked" | "error"
    reply: Optional[Callable[[str], Awaitable[object]]] = None        # сообщение инициатору команды (итог фоновых задач)
    actor: Optional[str] = None        # id администратора, выполняющего команду (None — терминал)
    say: Callable[..., None] = print   # куда выводить результат: терминал или сбор текста для Telegram
    color: bool = False                # ANSI-цвета только в терминале


def _start_reader(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Optional[str]]") -> None:
    """stdin читается в демон-потоке: блокирующий input() в обычном потоке не давал бы процессу завершиться по Ctrl+C."""
    def post(item):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # цикл событий уже закрыт
            pass

    def run():
        while True:
            line = sys.stdin.readline()
            if line == "":
                post(None)  # EOF — консоль закрыта
                return
            post(line.rstrip("\n"))

    threading.Thread(target=run, daemon=True, name="admin-console-stdin").start()


def _fmt_age(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}ч {m}м" if h else (f"{m}м {s}с" if m else f"{s}с")


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def _cmd_check(ctx, args):
    from healthcheck import format_report, run_health_check
    ctx.say("⏳ Проверяю системы...")
    results = await run_health_check(telegram="--no-telegram" not in args)
    ctx.say(format_report(results, color=ctx.color))


async def _cmd_status(ctx, args):
    queue = ctx.get_queue()
    tasks = queue.snapshot() if queue else []
    running = sum(t["state"] == "running" for t in tasks)
    queued = sum(t["state"] == "queued" for t in tasks)
    users = len(await asyncio.to_thread(_read_json, config.SETTINGS_FILE))
    ctx.say(f"Аптайм: {_fmt_age(time.time() - ctx.started_at)}")
    ctx.say(f"Режим: {ctx.mode}; WebUI: {config.SD_BASE_URL}")
    ctx.say(f"Очередь: выполняется {running}, ждёт {queued} (воркеров {config.QUEUE_WORKERS}, запросов к WebUI {config.SD_MAX_PARALLEL})")
    ctx.say(f"Пользователей с настройками: {users}")
    import subscriptions as subs
    st = await subs.stats()
    ctx.say(f"Подписки: {'включены' if st['enabled'] else 'выключены'}; оплата: {'вкл.' if st['payments'] else 'выкл.'}; "
          f"активных Pro: {st['active'].get('pro', 0)}; платежей: {st['payments_count']} ({st['stars_total']}⭐)")


async def _cmd_queue(ctx, args):
    queue = ctx.get_queue()
    tasks = queue.snapshot() if queue else []
    if not tasks:
        ctx.say("Очередь пуста.")
        return
    ctx.say(f"{'№':>4}  {'чат':>12}  {'тип':<12} {'состояние':<10} возраст")
    for t in tasks:
        ctx.say(f"{t['id']:>4}  {t['chat_id']:>12}  {t['kind']:<12} {t['state']:<10} {_fmt_age(t['age'])}")


async def _cmd_cancel(ctx, args):
    queue = ctx.get_queue()
    if not queue or not args:
        ctx.say("Использование: cancel <chat_id|all>")
        return
    if args[0] == "all":
        chats = {t["chat_id"] for t in queue.snapshot()}
    elif args[0].lstrip("-").isdigit():
        chats = {int(args[0])}
    else:
        ctx.say("Использование: cancel <chat_id|all>")
        return
    total = 0
    for chat_id in chats:
        total += await queue.cancel_user(chat_id)
    ctx.say(f"Отменено задач: {total}")


async def _cmd_interrupt(ctx, args):
    from http_client import sd_interrupt
    ctx.say("✅ Запрос на прерывание отправлен" if await sd_interrupt() else "❌ Не удалось прервать (WebUI недоступен?)")


async def _cmd_reload(ctx, args):
    ctx.say("⏳ Обновляю кэши...")
    await ctx.reload_caches()
    ctx.say("✅ Кэши обновлены")


async def _cmd_users(ctx, args):
    settings = await asyncio.to_thread(_read_json, config.SETTINGS_FILE)
    faces = await asyncio.to_thread(_read_json, config.FACE_SETTINGS_FILE)
    models = sum(len((e or {}).get("models") or {}) for e in faces.values() if isinstance(e, dict))
    ctx.say(f"Пользователей с настройками: {len(settings)}; с лицами: {len(faces)}; моделей ReActor: {models}")


async def _cmd_loglevel(ctx, args):
    level = (args[0] if args else "").upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        ctx.say("Использование: loglevel DEBUG|INFO|WARNING|ERROR")
        return
    logging.getLogger().setLevel(level)
    ctx.say(f"Уровень логов: {level}")


async def _notify(ctx, user_id: str, text: str) -> None:
    if ctx.notify is None or not user_id.lstrip("-").isdigit():
        return
    try:
        await ctx.notify(int(user_id), text)
        ctx.say(f"   ✉️ Пользователь {user_id} уведомлён")
    except Exception as e:
        ctx.say(f"   ⚠️ Не удалось уведомить {user_id}: {e}")


async def _cmd_sub(ctx, args):
    import subscriptions as subs
    usage = ("Использование: sub status | on | off | grant <id> <free|pro> [дней] | revoke <id> | info <id> | list | reset <id> | "
             "pay on|off | payments [N] | refund <charge_id> | testpay <id> <товар> | credits <id> <N> | ref <id>")
    if not args:
        ctx.say(usage)
        return
    action, rest = args[0].lower(), args[1:]
    if action == "status":
        st = await subs.stats()
        master = "" if st["master"] else " [SUBSCRIPTIONS_ENABLED=0 в .env — выключены полностью]"
        ctx.say(f"Подписки: {'ВКЛЮЧЕНЫ' if st['enabled'] else 'выключены'}{master}; выдано тарифов: {st['granted']}")
        ctx.say(f"Оплата Stars: {'ВКЛЮЧЕНА' if st['payments'] else 'выключена'}; платежей: {st['payments_count']} на {st['stars_total']}⭐; рефералов: {st['referrals']}")
        ctx.say("Активные тарифы: " + ", ".join(f"{name}: {n}" for name, n in st["active"].items()))
        ctx.say(f"Лимит Free: {subs.PLANS['free']['daily']} генераций в день; Pro — без лимита, лица только на Pro")
    elif action in ("on", "off"):
        await subs.set_enabled(action == "on")
        if action == "on" and not subs._master_on():
            ctx.say("⚠️ Флаг включён, но в .env стоит SUBSCRIPTIONS_ENABLED=0 — система остаётся выключенной, пока вы его не уберёте.")
        else:
            ctx.say("✅ Подписки включены: лимиты и разделение интерфейса действуют" if action == "on"
                    else "✅ Подписки выключены полностью: ни лимитов, ни тарифов, ни разделения интерфейса")
    elif action == "grant":
        if len(rest) < 2 or not rest[0].lstrip("-").isdigit():
            ctx.say("Использование: sub grant <id> <free|pro> [дней]")
            return
        days = None
        if len(rest) > 2:
            try:
                days = float(rest[2])
                if days <= 0:
                    raise ValueError
            except ValueError:
                ctx.say("Срок — положительное число дней")
                return
        try:
            expires = await subs.grant(rest[0], rest[1].lower(), days)
        except ValueError as e:
            ctx.say(f"❌ {e}")
            return
        ctx.say(f"✅ {rest[0]} → {rest[1].lower()} ({subs.fmt_expires(expires)})")
        await _notify(ctx, rest[0], f"🎁 Вам выдан тариф <b>{subs.PLANS[rest[1].lower()]['title']}</b> ({subs.fmt_expires(expires)}). Подробнее: /plan")
    elif action == "revoke":
        if not rest:
            ctx.say("Использование: sub revoke <id>")
            return
        if await subs.revoke(rest[0]):
            ctx.say(f"✅ Подписка {rest[0]} снята (теперь Free)")
            await _notify(ctx, rest[0], "ℹ️ Ваша подписка снята, действует тариф Free. Подробнее: /plan")
        else:
            ctx.say("У пользователя нет выданной подписки")
    elif action == "info":
        if not rest:
            ctx.say("Использование: sub info <id>")
            return
        plan, expires = await subs.get_plan(rest[0])
        ctx.say(f"{rest[0]}: тариф {subs.PLANS[plan]['title']} ({subs.fmt_expires(expires) if plan != 'free' else '—'})")
        ctx.say((await subs.describe(rest[0])).replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    elif action == "list":
        rows = await subs.list_users()
        if not rows:
            ctx.say("Выданных тарифов нет.")
        for r in rows:
            flag = " (истекла)" if r["expired"] else ""
            ctx.say(f"{r['user_id']:>14}  {r['plan']:<5} {subs.fmt_expires(r['expires'])}{flag}  {r['note']}")
    elif action == "pay":
        if not rest or rest[0] not in ("on", "off"):
            ctx.say("Использование: sub pay on|off")
            return
        await subs.set_payments(rest[0] == "on")
        ctx.say("✅ Оплата включена (нужны и включённые подписки: sub on)" if rest[0] == "on" else "✅ Оплата выключена")
    elif action == "payments":
        limit = int(rest[0]) if rest and rest[0].isdigit() else 15
        rows = await subs.list_payments(limit)
        if not rows:
            ctx.say("Платежей нет.")
        for p in rows:
            flags = (" [тест]" if p.get("test") else "") + (" [возврат]" if p.get("refunded") else "")
            when = time.strftime("%d.%m %H:%M", time.localtime(p["ts"]))
            ctx.say(f"{when}  {p['user']:>12}  {p['product']:<7} {p['stars']:>5}⭐{flags}  {p['charge_id']}")
    elif action == "testpay":
        if len(rest) < 2 or not rest[0].lstrip("-").isdigit():
            ctx.say(f"Использование: sub testpay <id> <товар>; товары: {', '.join(subs.PRODUCTS)}")
            return
        result = await subs.process_payment(rest[0], rest[1], 0, f"test-{int(time.time() * 1000)}", test=True)
        if not result["ok"]:
            ctx.say(f"❌ {result.get('reason')}")
            return
        ctx.say(f"✅ Тестовая оплата: {rest[0]} → Pro {subs.fmt_expires(result['expires'])}")
        await _notify(ctx, rest[0], f"✅ (тест) Оплата получена. Тариф <b>Pro</b> {subs.fmt_expires(result['expires'])}")
        reward = result.get("reward")
        if reward:
            ctx.say(f"   🎁 Реферер {reward['referrer']} получил +{reward['days']} дн. Pro ({subs.fmt_expires(reward['expires'])})")
            await _notify(ctx, str(reward["referrer"]), f"🎁 Ваш приглашённый оформил подписку — вам +{reward['days']} дн. Pro")
    elif action == "refund":
        if not rest:
            ctx.say("Использование: sub refund <charge_id>")
            return
        payment = next((p for p in await subs.list_payments(2000) if p["charge_id"] == rest[0]), None)
        if payment is None or payment.get("refunded"):
            ctx.say("Платёж не найден или уже возвращён.")
            return
        if not payment.get("test"):
            if ctx.refund_stars is None:
                ctx.say("❌ Возврат недоступен в этом режиме")
                return
            try:
                await ctx.refund_stars(int(payment["user"]), payment["charge_id"])
            except Exception as e:
                ctx.say(f"❌ Telegram не вернул Stars: {e}")
                return
        await subs.reverse_payment(rest[0])
        ctx.say(f"✅ Платёж возвращён, у {payment['user']} убавлено {payment['days']} дн. (реферальный бонус не отзывается)")
        await _notify(ctx, payment["user"], "ℹ️ Платёж возвращён, срок подписки скорректирован. /plan")
    elif action == "credits":
        if len(rest) < 2 or not rest[0].lstrip("-").isdigit() or not rest[1].lstrip("-").isdigit():
            ctx.say("Использование: sub credits <id> <N>")
            return
        ctx.say(f"✅ Бонусных генераций у {rest[0]}: {await subs.add_credits(rest[0], int(rest[1]))}")
    elif action == "ref":
        if not rest:
            ctx.say("Использование: sub ref <id>")
            return
        info = await subs.referral_info(rest[0])
        ctx.say(f"{rest[0]}: приглашено {info['invited']}, оплатили {info['paid']}, бонусных генераций {info['credits']}"
              + (f", пришёл от {info['referrer']}" if info["referrer"] else ""))
    elif action == "reset":
        if not rest:
            ctx.say("Использование: sub reset <id>")
            return
        await subs.reset_usage(rest[0])
        ctx.say(f"✅ Дневной счётчик {rest[0]} обнулён")
    else:
        ctx.say(usage)


async def _cmd_admin(ctx, args):
    import subscriptions as subs
    action = args[0].lower() if args else ""
    if action == "list":
        owner, env_ids, stored = await subs.list_admins()
        ctx.say(f"Владелец (ADMIN_USER_ID): {owner or 'не задан'}")
        ctx.say("Админы из .env (ADMIN_USER_IDS): " + (", ".join(env_ids) if env_ids else "нет"))
        ctx.say("Админы, выданные из консоли: " + (", ".join(stored) if stored else "нет"))
    elif action in ("add", "remove") and len(args) > 1 and args[1].lstrip("-").isdigit():
        if action == "add":
            changed = await subs.add_admin(args[1])
            ctx.say(f"✅ {args[1]} теперь администратор" if changed else "Уже администратор")
            if changed:
                await _notify(ctx, args[1], "🛡️ Вам выданы права администратора бота (лог промптов /prompts, без лимитов).")
        else:
            owner, env_ids, _ = await subs.list_admins()
            if args[1] == owner or args[1] in env_ids:
                ctx.say("Этого администратора нельзя снять командой — он указан в .env (ADMIN_USER_ID / ADMIN_USER_IDS).")
            else:
                ctx.say(f"✅ Права {args[1]} сняты" if await subs.remove_admin(args[1]) else "Такого администратора нет")
    else:
        ctx.say("Использование: admin list | admin add <id> | admin remove <id>")


async def _cmd_logs(ctx, args):
    import logbuffer
    if args and args[0].lower() == "file":
        ctx.say(f"Файл лога: {logbuffer.log_file_path() or 'выключен (LOG_TO_FILE=0)'}")
        return
    errors_only = bool(args) and args[0].lower() in ("errors", "err")
    count = next((int(a) for a in args if a.isdigit()), 40)
    lines = logbuffer.tail(min(max(count, 1), 300), errors_only=errors_only)
    ctx.say("\n".join(lines) if lines else "Лог пуст.")


# ---------- рассылка ----------
BROADCAST_RATE = 15         # сообщений в секунду (лимит Telegram — около 30/с на бота)
BROADCAST_TTL = 600         # сколько секунд предпросмотр ждёт подтверждения
AUDIENCES = {"all": "все пользователи", "pro": "пользователи Pro", "free": "пользователи Free"}
_pending_broadcast: Optional[dict] = None
_broadcast_task: Optional["asyncio.Task"] = None
_broadcast_state: dict = {}


def has_pending_broadcast() -> bool:
    return _pending_broadcast is not None and time.time() - _pending_broadcast["ts"] < BROADCAST_TTL


async def _audience_ids(audience: str):
    import subscriptions as subs
    users = await asyncio.to_thread(_read_json, config.SETTINGS_FILE)
    ids = [u for u in users if str(u).lstrip("-").isdigit()]
    blocked = set(await subs.blocked_ids())
    return await subs.filter_by_plan([u for u in ids if u not in blocked], audience), len(blocked)


async def _print_report(text: str) -> None:
    print(text)


async def _run_broadcast(ctx, ids, text, report):
    import subscriptions as subs
    counts = {"ok": 0, "blocked": 0, "error": 0}
    newly_blocked = []
    _broadcast_state.update(total=len(ids), sent=0, stop=False, counts=counts)
    for uid in ids:
        if _broadcast_state["stop"]:
            break
        status = await ctx.send_user(int(uid), text)
        counts[status if status in counts else "error"] += 1
        if status == "blocked":
            newly_blocked.append(uid)
        _broadcast_state["sent"] += 1
        await asyncio.sleep(1 / BROADCAST_RATE)
    if newly_blocked:
        await subs.mark_blocked(newly_blocked)   # в следующих рассылках этих пользователей не трогаем
    stopped = " (остановлена)" if _broadcast_state["stop"] else ""
    await report(f"📣 Рассылка завершена{stopped}: доставлено {counts['ok']}, заблокировали бота {counts['blocked']}, ошибок {counts['error']}")


async def _cmd_broadcast(ctx, args):
    global _pending_broadcast, _broadcast_task
    sub = args[0].lower() if args else ""
    running = _broadcast_task is not None and not _broadcast_task.done()
    if sub == "status":
        if running:
            ctx.say(f"Идёт рассылка: {_broadcast_state['sent']}/{_broadcast_state['total']}")
        else:
            ctx.say("Рассылка не идёт." + (" Есть неподтверждённый предпросмотр." if has_pending_broadcast() else ""))
    elif sub == "cancel":
        if running:
            _broadcast_state["stop"] = True
            ctx.say("⏹ Останавливаю рассылку, итог придёт отдельным сообщением.")
        else:
            had = has_pending_broadcast()
            _pending_broadcast = None
            ctx.say("Предпросмотр отменён." if had else "Нечего отменять.")
    elif sub == "send":
        if running:
            ctx.say("Рассылка уже идёт: broadcast status")
        elif not has_pending_broadcast():
            ctx.say("Нет предпросмотра (или он устарел, 10 минут). Сначала: broadcast <all|pro|free> <текст>")
        elif ctx.send_user is None:
            ctx.say("❌ Отправка сообщений недоступна в этом режиме")
        else:
            pending, _pending_broadcast = _pending_broadcast, None
            report = ctx.reply or _print_report
            _broadcast_task = asyncio.create_task(_run_broadcast(ctx, pending["ids"], pending["text"], report), name="broadcast")
            ctx.say(f"🚀 Рассылка запущена: {len(pending['ids'])} получателей (~{max(1, len(pending['ids']) // BROADCAST_RATE)} с). Итог придёт отдельным сообщением.")
    elif sub in AUDIENCES:
        text = args[1].strip() if len(args) > 1 else ""
        if not text:
            ctx.say("Использование: broadcast <all|pro|free> <текст>")
            return
        if len(text) > 3500:
            ctx.say("❌ Текст слишком длинный (максимум 3500 символов)")
            return
        if running:
            ctx.say("Сначала дождитесь окончания текущей рассылки (broadcast status) или остановите её (broadcast cancel).")
            return
        ids, blocked = await _audience_ids(sub)
        if not ids:
            ctx.say("Получателей нет.")
            return
        _pending_broadcast = {"ids": ids, "text": text, "ts": time.time()}
        ctx.say(f"📣 Рассылка: {AUDIENCES[sub]}\nПолучателей: {len(ids)} (не учтены заблокировавшие бота: {blocked})\n"
                f"Текст уйдёт как есть, без форматирования:\n──────────\n{text}\n──────────\n"
                "Подтвердить: broadcast send · отменить: broadcast cancel")
    else:
        ctx.say("Использование: broadcast <all|pro|free> <текст> | send | cancel | status")


async def _cmd_uimode(ctx, args):
    import subscriptions as subs
    mode = args[0].lower() if args else ""
    target = args[1] if len(args) > 1 else ctx.actor
    if mode not in ("simple", "auto") or not target or not str(target).lstrip("-").isdigit():
        ctx.say("Использование: uimode simple|auto [id] (в терминале id обязателен)")
        return
    if not await subs.is_admin(target):
        ctx.say("Предпросмотр доступен только администраторам — остальные и так видят интерфейс по тарифу.")
        return
    await subs.set_ui_preview(target, mode == "simple")
    ctx.say("👁 Теперь вы видите интерфейс как клиент Free (простые настройки). Вернуть: /adm uimode auto" if mode == "simple"
            else "✅ Обычный вид интерфейса возвращён")


async def _cmd_quit(ctx, args):
    ctx.say("🛑 Останавливаю бота...")
    await ctx.stop()


COMMANDS = {
    "check": _cmd_check, "status": _cmd_status, "queue": _cmd_queue, "cancel": _cmd_cancel,
    "interrupt": _cmd_interrupt, "reload": _cmd_reload, "users": _cmd_users, "loglevel": _cmd_loglevel,
    "sub": _cmd_sub, "admin": _cmd_admin, "logs": _cmd_logs, "broadcast": _cmd_broadcast, "uimode": _cmd_uimode, "quit": _cmd_quit, "exit": _cmd_quit, "stop": _cmd_quit,
}


# ---------- права доступа и выполнение команд ----------
VIEWER_COMMANDS = {"help", "status", "queue", "users", "check", "logs", "cancel", "interrupt", "reload"}
SUB_VIEW_ACTIONS = {"status", "info", "list", "payments", "ref"}


def required_level(name: str, args: List[str]) -> str:
    """'admin' — достаточно роли администратора (ADMIN_USER_IDS), 'owner' — только владелец или терминал."""
    first = args[0].lower() if args else ""
    if name == "cancel" and first == "all":
        return "owner"
    if name in VIEWER_COMMANDS:
        return "admin"
    if name == "sub" and first in SUB_VIEW_ACTIONS:
        return "admin"
    if name == "admin" and first == "list":
        return "admin"
    if name == "uimode" and len(args) < 2:
        return "admin"
    return "owner"


async def dispatch(line: str, ctx: ConsoleContext, level: str = "owner") -> None:
    """Разбирает и выполняет одну команду; результат выводится через ctx.say."""
    if line.lstrip().lower().startswith("broadcast"):
        parts = ["broadcast"] + line.split(None, 2)[1:]   # текст рассылки берём как есть: кавычки и апострофы допустимы
    else:
        try:
            parts = shlex.split(line)
        except ValueError as e:
            ctx.say(f"Ошибка разбора команды: {e}")
            return
    if not parts:
        return
    name, args = parts[0].lower(), parts[1:]
    if name in ("help", "?", "h"):
        ctx.say(HELP)
        return
    handler = COMMANDS.get(name)
    if handler is None:
        ctx.say(f"Неизвестная команда «{name}». help — список команд.")
        return
    if required_level(name, args) == "owner" and level != "owner":
        ctx.say("⛔ Это действие доступно только владельцу (ADMIN_USER_ID).")
        return
    try:
        await handler(ctx, args)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        ctx.say(f"❌ Ошибка команды: {type(e).__name__}: {e}")


async def run_capture(line: str, ctx: ConsoleContext, level: str) -> str:
    """Выполняет команду и возвращает её вывод текстом (для админ-панели в Telegram)."""
    sink: List[str] = []
    local = replace(ctx, say=lambda *parts: sink.append(" ".join(str(p) for p in parts)), color=False)
    await dispatch(line, local, level)
    return "\n".join(sink)


async def run(ctx: ConsoleContext) -> None:
    """Интерактивный терминал: владельцу доступно всё."""
    loop = asyncio.get_running_loop()
    lines: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    _start_reader(loop, lines)
    print("🛠️ Админ-консоль: введите help (отключить: ADMIN_CONSOLE=0)")
    while True:
        line = await lines.get()
        if line is None:
            return
        await dispatch(line, ctx, "owner")
