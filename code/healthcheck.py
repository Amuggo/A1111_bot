"""
Проверка всех систем бота: окружение, Telegram, Stable Diffusion WebUI и его расширения, Ollama, данные.

Запуск отдельно (бот не стартует):
    python code/healthcheck.py            # отчёт в консоль, код возврата 1 при критичных ошибках
    python code/healthcheck.py --json     # то же в JSON
    python code/healthcheck.py --no-telegram   # без запроса к api.telegram.org

При запуске бота отчёт печатается автоматически (STARTUP_CHECK=1), а в админ-консоли доступна команда `check`.
Статусы: ok — работает, warn — работает частично / необязательная часть недоступна, fail — бот не сможет
выполнить свою основную задачу, info — справка.
"""

try:
    import bootstrap_env  # noqa: F401  — подключает библиотеки из папки env/, как и в main.py
except Exception:  # pragma: no cover
    pass

import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, List

import app_config as config

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
ICONS = {OK: "✅", WARN: "⚠️", FAIL: "❌", INFO: "ℹ️"}
_COLORS = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", INFO: "\033[36m"}
_RESET = "\033[0m"

RECOMMENDED_NEGATIVE_EMBEDDINGS = ("bad-hands-5", "badhandv4", "easynegative")
CHECK_TIMEOUT = 20


@dataclass
class CheckResult:
    group: str
    name: str
    status: str
    detail: str = ""
    hint: str = ""


def _r(group: str, name: str, status: str, detail: str = "", hint: str = "") -> CheckResult:
    return CheckResult(group, name, status, detail, hint)


async def _guard(group: str, name: str, coro: Awaitable[Any], timeout: float = CHECK_TIMEOUT) -> List[CheckResult]:
    """Любая проверка выполняется с таймаутом, а её исключение превращается в строку отчёта, а не роняет всю проверку."""
    try:
        res = await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        return [_r(group, name, FAIL, f"нет ответа за {timeout:g} с")]
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return [_r(group, name, FAIL, f"{type(e).__name__}: {e}")]
    return res if isinstance(res, list) else [res]


# ===================== ОКРУЖЕНИЕ =====================
G_ENV = "Окружение"
G_PAY = "Подписки, оплата и рефералы"
G_TEST = "Самотест подписок и платежей"


def check_python() -> CheckResult:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 10):
        return _r(G_ENV, "Python", WARN, text, "Рекомендуется Python 3.10+")
    return _r(G_ENV, "Python", OK, text)


def check_packages() -> List[CheckResult]:
    required = {"aiogram": "aiogram", "aiohttp": "aiohttp", "dotenv": "python-dotenv", "pandas": "pandas",
                "pyarrow": "pyarrow", "PIL": "Pillow", "numpy": "numpy"}
    missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    results = []
    if missing:
        results.append(_r(G_ENV, "Обязательные пакеты", FAIL, "не установлены: " + ", ".join(missing),
                          "pip install -r requirements.txt"))
    else:
        results.append(_r(G_ENV, "Обязательные пакеты", OK, "все на месте"))
        try:
            from importlib.metadata import version
            aiogram_v = version("aiogram")
            major_minor = tuple(int(x) for x in re.findall(r"\d+", aiogram_v)[:2])
            if major_minor < (3, 13):
                results.append(_r(G_ENV, "aiogram и Telegram Stars", WARN, f"версия {aiogram_v} без API Stars (нужна ≥3.13)",
                                  "pip install -U aiogram — иначе оплата и возвраты Stars работать не будут"))
            else:
                results.append(_r(G_ENV, "aiogram и Telegram Stars", OK, f"версия {aiogram_v}"))
        except Exception:
            pass

    if importlib.util.find_spec("cv2") is None:
        results.append(_r(G_ENV, "OpenCV", WARN, "не установлен",
                          "pip install opencv-python — нужен для редактирования тела/фона/позы и проверки фото лиц"))
    else:
        results.append(_r(G_ENV, "OpenCV", OK))
    if importlib.util.find_spec("mediapipe") is None:
        results.append(_r(G_ENV, "MediaPipe", WARN, "не установлен",
                          "pip install mediapipe — нужен для сегментации и позы; проверка лиц обойдётся каскадом OpenCV"))
    else:
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                results.append(_r(G_ENV, "MediaPipe", OK, getattr(mp, "__version__", "")))
            else:
                results.append(_r(G_ENV, "MediaPipe", WARN, "в этой версии нет mp.solutions",
                                  "pip install mediapipe==0.10.14 — иначе сегментация и поза недоступны"))
        except Exception as e:
            results.append(_r(G_ENV, "MediaPipe", WARN, f"не импортируется: {e}"))
    return results


def check_env() -> List[CheckResult]:
    results = []
    token = config.BOT_TOKEN or ""
    if not token:
        results.append(_r(G_ENV, "TELEGRAM_BOT_TOKEN", FAIL, "не задан", "Скопируйте .env.example в .env и вставьте токен от @BotFather"))
    elif not re.fullmatch(r"\d{6,}:[A-Za-z0-9_-]{30,}", token):
        results.append(_r(G_ENV, "TELEGRAM_BOT_TOKEN", WARN, "формат токена необычный", "Проверьте, что скопировали токен целиком"))
    else:
        results.append(_r(G_ENV, "TELEGRAM_BOT_TOKEN", OK, f"задан (бот id {token.split(':')[0]})"))

    if config.BOT_MODE == "webhook":
        if not config.WEBHOOK_URL.startswith("https://"):
            results.append(_r(G_ENV, "Webhook", FAIL, f"WEBHOOK_URL={config.WEBHOOK_URL or '—'}", "Нужен публичный https-адрес без пути"))
        elif not config.WEBHOOK_SECRET:
            results.append(_r(G_ENV, "Webhook", WARN, "WEBHOOK_SECRET не задан", "Без секрета запросы к вебхуку не проверяются"))
        else:
            results.append(_r(G_ENV, "Webhook", OK, config.WEBHOOK_URL + config.WEBHOOK_PATH))
    else:
        results.append(_r(G_ENV, "Режим бота", INFO, "long polling"))

    admin = os.environ.get("ADMIN_USER_ID", "").strip()
    if not admin:
        results.append(_r(G_ENV, "ADMIN_USER_ID (владелец)", WARN, "не задан",
                          "Без владельца из Telegram нельзя выдавать тарифы, делать возвраты и останавливать бота; /prompts закрыт для всех"))
    elif not admin.lstrip("-").isdigit():
        results.append(_r(G_ENV, "ADMIN_USER_ID (владелец)", WARN, "должен быть числовым id чата (узнать: @userinfobot)"))
    else:
        results.append(_r(G_ENV, "ADMIN_USER_ID (владелец)", OK, admin))
    raw_ids = os.environ.get("ADMIN_USER_IDS", "")
    tokens = [t for t in re.split(r"[,\s;]+", raw_ids.strip()) if t]
    bad_ids = [t for t in tokens if not t.lstrip("-").isdigit()]
    if bad_ids:
        results.append(_r(G_ENV, "ADMIN_USER_IDS", WARN, "не числа, будут пропущены: " + ", ".join(bad_ids),
                          "Формат: ADMIN_USER_IDS=111111111,222222222"))
    else:
        results.append(_r(G_ENV, "ADMIN_USER_IDS (админ-панель)", INFO if tokens else INFO,
                          f"{len(tokens)} админ(ов): просмотр и безопасные действия" if tokens else "не заданы"))
    results.append(_r(G_ENV, "Оповещения о падении WebUI", INFO,
                      f"каждые {config.WATCHDOG_INTERVAL} с" if config.WATCHDOG_INTERVAL > 0 else "выключены (WATCHDOG_INTERVAL=0)"))
    return results


def check_data() -> List[CheckResult]:
    results = []
    data_dir = config.DATA_DIR
    try:
        os.makedirs(data_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=data_dir):
            pass
        free_gb = shutil.disk_usage(data_dir).free / 1e9
        status = WARN if free_gb < 1 else OK
        results.append(_r(G_ENV, "Папка данных", status, f"{data_dir} (свободно {free_gb:.1f} ГБ)",
                          "Мало места на диске" if status == WARN else ""))
    except Exception as e:
        results.append(_r(G_ENV, "Папка данных", FAIL, f"{data_dir}: {e}", "Проверьте права на запись (DATA_DIR)"))
        return results

    broken = []
    stats = {}
    for label, path in (("настройки", config.SETTINGS_FILE), ("лица", config.FACE_SETTINGS_FILE),
                        ("ControlNet", config.CONTROLNET_SETTINGS_FILE), ("ADetailer", config.ADETAILER_SETTINGS_FILE),
                        ("подписки", config.SUBSCRIPTIONS_FILE)):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                stats[label] = json.load(f)
        except Exception as e:
            broken.append(f"{os.path.basename(path)} ({e})")
    if broken:
        results.append(_r(G_ENV, "Файлы настроек", FAIL, "повреждены: " + "; ".join(broken),
                          "Восстановите из копии или удалите файл (будут созданы заново, настройки сбросятся)"))
    else:
        users = len(stats.get("настройки", {}) or {})
        faces = stats.get("лица", {}) or {}
        models = sum(len((e or {}).get("models") or {}) for e in faces.values() if isinstance(e, dict))
        results.append(_r(G_ENV, "Файлы настроек", OK, f"пользователей: {users}; с лицами: {len(faces)}; моделей ReActor: {models}"))
        subs = stats.get("подписки") or {}
        log = [p for p in subs.get("payment_log") or [] if not p.get("refunded") and not p.get("test")]
        master = os.environ.get("SUBSCRIPTIONS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
        on = master and bool(subs.get("enabled", True))
        state = "ВЫКЛЮЧЕНЫ ПОЛНОСТЬЮ (SUBSCRIPTIONS_ENABLED=0)" if not master else ("включены" if on else "выключены командой `sub off`")
        results.append(_r(G_PAY, "Подписки", INFO if on or not master else WARN,
                          f"{state}; выдано тарифов: {len(subs.get('users') or {})}; администраторов: {len(subs.get('admins') or [])}"))
        if subs.get("payments") and not on:
            results.append(_r(G_PAY, "Оплата", INFO, "приём оплаты включён, но подписки выключены — кнопки оплаты не показываются"))
        else:
            results.append(_r(G_PAY, "Оплата", INFO,
                              f"{'включена' if subs.get('payments') else 'выключена'}; платежей: {len(log)} на {sum(p.get('stars', 0) for p in log)}⭐; "
                              f"рефералов: {len(subs.get('referrals') or {})}"))
    return results


def check_payments_config() -> List[CheckResult]:
    import subscriptions as S
    results = []
    bad = [p["title"] for p in S.PRODUCTS.values() if not 1 <= p["stars"] <= 10000]
    results.append(_r(G_PAY, "Товары Stars", FAIL if bad else OK,
                      ("некорректная цена: " + ", ".join(bad)) if bad else "; ".join(f"{p['title']} — {p['stars']}⭐" for p in S.PRODUCTS.values()),
                      "Цена задаётся PRO_30_STARS / PRO_90_STARS (целое от 1 до 10000)" if bad else ""))
    results.append(_r(G_PAY, "Платёжный провайдер", INFO, "не нужен: оплата через Telegram Stars (XTR)"))
    if not os.environ.get("ADMIN_USER_ID", "").strip():
        results.append(_r(G_PAY, "Обработка платежей", WARN, "ADMIN_USER_ID не задан",
                          "Без него вы не получите уведомлений об оплатах и ошибках платежей"))
    if not os.environ.get("SUPPORT_CONTACT", "").strip():
        results.append(_r(G_PAY, "Контакт поддержки", INFO, "SUPPORT_CONTACT не задан — /paysupport покажет общий текст"))
    results.append(_r(G_PAY, "Рефералы", INFO,
                      f"+{S.REF_SIGNUP_CREDITS} генераций обоим при входе; +{S.REF_PURCHASE_DAYS} дн. Pro за первую оплату друга; "
                      f"до {S.REF_MAX_INVITES} приглашений на человека"))
    return results


# ===================== САМОТЕСТ ЛОГИКИ ПОДПИСОК, ОПЛАТЫ И РЕФЕРАЛОВ =====================
async def selftest_subscriptions() -> List[CheckResult]:
    """Прогон бизнес-логики на ВРЕМЕННОМ хранилище: рабочие данные не затрагиваются, сеть не нужна."""
    import subscriptions as S
    from storage import JsonStore
    results: List[CheckResult] = []
    key = next(iter(S.PRODUCTS))
    prod = S.PRODUCTS[key]
    day = 86400

    with tempfile.TemporaryDirectory() as tmp:
        svc = S.SubscriptionService(JsonStore(os.path.join(tmp, "selftest.json"), default_factory=S._default_data))

        async def limits():
            if not await svc.is_enabled() or not await svc.payments_enabled():
                return "подписки и оплата должны быть включены по умолчанию"
            await svc.set_enabled(False)
            if await svc.check_and_consume("900001") is not None:
                return "при выключенных подписках лимит применился"
            await svc.set_enabled(True)
            await svc.add_credits("900001", 3)
            for i in range(3):
                if await svc.check_and_consume("900001") is not None:
                    return f"отказ при хватающих токенах на {i + 1}-й генерации"
            if await svc.check_and_consume("900001") is None:
                return "генерация прошла без токенов"
            await svc.grant("900002", "pro", days=1)
            if await svc.check_and_consume("900002") is not None:
                return "Pro получил отказ по токенам"
            if not await svc.feature_allowed("900002", "face"):
                return "доступ к функциям лиц выдан неверно"

        async def expiry():
            await svc.grant("900003", "pro", days=1)
            await svc._store.aupdate(lambda d: d["users"]["900003"].__setitem__("expires", time.time() - 5))
            if (await svc.get_plan("900003"))[0] != "free":
                return "истёкшая подписка не сброшена до Free"

        async def payment():
            r1 = await svc.process_payment("900004", key, prod["stars"], "st-1")
            if not r1["ok"] or abs(r1["expires"] - (time.time() + prod["days"] * day)) > 60:
                return f"первая оплата обработана неверно: {r1}"
            dup = await svc.process_payment("900004", key, prod["stars"], "st-1")
            if not dup.get("duplicate") or (await svc.get_plan("900004"))[1] != r1["expires"]:
                return "повтор того же платежа продлил подписку (нет идемпотентности)"
            if (await svc.process_payment("900004", key, prod["stars"] + 1, "st-2"))["ok"]:
                return "платёж с неверной суммой принят"
            if (await svc.process_payment("900004", "no_such", 1, "st-2b"))["ok"]:
                return "платёж за неизвестный товар принят"
            r3 = await svc.process_payment("900004", key, prod["stars"], "st-3")
            if abs(r3["expires"] - (r1["expires"] + prod["days"] * day)) > 5:
                return "вторая оплата не продлила срок от текущего"

        async def payment_refund():
            before = (await svc.get_plan("900004"))[1]
            rev = await svc.reverse_payment("st-3")
            after = (await svc.get_plan("900004"))[1]
            if rev is None or abs(after - (before - prod["days"] * day)) > 5:
                return "возврат не убавил дни подписки"
            if await svc.reverse_payment("st-3") is not None:
                return "повторный возврат того же платежа прошёл"

        async def referrals():
            if not (await svc.register_referral("900005", "900002", True))["ok"]:
                return "валидное приглашение отклонено"
            if (await svc.register_referral("900005", "900002", True))["ok"]:
                return "повторное приглашение принято"
            if (await svc.register_referral("900006", "900006", True))["ok"]:
                return "самоприглашение принято"
            if (await svc.register_referral("900007", "900002", False))["ok"]:
                return "бонус выдан не новому пользователю"
            if (await svc.referral_info("900005"))["credits"] != S.REF_SIGNUP_CREDITS:
                return "бонусные генерации приглашённому не начислены"
            before = (await svc.get_plan("900002"))[1]
            pay = await svc.process_payment("900005", key, prod["stars"], "st-4")
            if S.REF_PURCHASE_DAYS > 0:
                reward = pay.get("reward")
                if not reward or reward["referrer"] != "900002" or abs(reward["expires"] - (before + S.REF_PURCHASE_DAYS * day)) > 5:
                    return "бонус за первую оплату приглашённого не начислен"
            if (await svc.process_payment("900005", key, prod["stars"], "st-5")).get("reward"):
                return "реферальный бонус начислен повторно"

        async def credits():
            u = "900010"
            if await svc.claim_welcome(u) != S.WELCOME_TOKENS or await svc.claim_welcome(u) != 0:
                return "приветственные токены выданы неверно (должны один раз)"
            await svc.add_credits(u, S.COST_FACE - S.WELCOME_TOKENS)       # итого ровно на одну генерацию с лицом
            if await svc.check_and_consume(u, S.COST_FACE) is not None or await svc.balance(u) != 0:
                return "генерация с лицом списала неверную сумму"
            if await svc.check_and_consume(u, S.COST_UPSCALE) is None:
                return "апскейл прошёл без токенов"
            await svc.add_credits(u, 1)
            await svc.check_and_consume(u, S.COST_UPSCALE)
            await svc.refund(u)
            if await svc.balance(u) != 1:
                return "возврат не восстановил токены"

        async def access_levels():
            import admin_console
            for name, args in (("status", []), ("logs", ["errors"]), ("sub", ["list"]), ("sub", ["payments"]), ("admin", ["list"]),
                               ("cancel", ["123"]), ("interrupt", []), ("reload", [])):
                if admin_console.required_level(name, args) != "admin":
                    return f"«{name} {' '.join(args)}» должна быть доступна администратору"
            for name, args in (("sub", ["grant", "1", "pro"]), ("sub", ["refund", "x"]), ("sub", ["on"]), ("sub", ["pay", "on"]),
                               ("admin", ["add", "1"]), ("quit", []), ("cancel", ["all"]), ("loglevel", ["DEBUG"]), ("sub", [])):
                if admin_console.required_level(name, args) != "owner":
                    return f"«{name} {' '.join(args)}» должна быть только у владельца"
            saved = {k: os.environ.get(k) for k in ("ADMIN_USER_ID", "ADMIN_USER_IDS")}
            try:  # без await между подменой и восстановлением env — другие задачи ничего не заметят
                os.environ["ADMIN_USER_ID"], os.environ["ADMIN_USER_IDS"] = "910001", "910002, 910003;x,910002"
                ids = S.env_admin_ids()
                if ids != ["910001", "910002", "910003"]:
                    return f"список админов из .env разобран неверно: {ids}"
                data = {"admins": ["910004"]}
                if not {"910001", "910002", "910004"} <= S._admins(data) or "910009" in S._admins(data):
                    return "состав администраторов вычислен неверно"
                if not S.is_owner("910001") or S.is_owner("910002"):
                    return "владелец определён неверно"
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        async def master_switch():
            saved = os.environ.get("SUBSCRIPTIONS_ENABLED")
            try:  # без await между подменой и восстановлением env — другие задачи ничего не заметят
                os.environ["SUBSCRIPTIONS_ENABLED"] = "0"
                data = await svc._store.aread()
                master_off = (not S._on(data), not S._master_on())
            finally:
                if saved is None:
                    os.environ.pop("SUBSCRIPTIONS_ENABLED", None)
                else:
                    os.environ["SUBSCRIPTIONS_ENABLED"] = saved
            if master_off != (True, True):
                return "SUBSCRIPTIONS_ENABLED=0 не выключил систему целиком"
            if not S._on(await svc._store.aread()):
                return "после снятия главного выключателя система не вернулась"

        async def ui_roles():
            await svc.set_enabled(True)
            await svc.grant("900010", "pro", days=1)
            await svc.add_admin("900011")
            checks = ((("900012", False), "Free видит расширенные настройки"), (("900010", True), "Pro не видит расширенные настройки"),
                      (("900011", True), "администратор не видит расширенные настройки"))
            for (uid, expected), message in checks:
                if await svc.advanced_allowed(uid) != expected:
                    return message
            await svc.set_ui_preview("900011", True)
            if await svc.advanced_allowed("900011"):
                return "предпросмотр простого вида не сработал"
            await svc.set_ui_preview("900011", False)
            if not await svc.advanced_allowed("900011"):
                return "обычный вид администратора не вернулся"
            await svc.set_enabled(False)
            if not await svc.advanced_allowed("900012"):
                return "при выключенных подписках расширенные настройки должны быть у всех"
            await svc.set_enabled(True)

        async def broadcast_audience():
            await svc.grant("900013", "pro", days=1)
            ids = ["900013", "900014", "900015"]
            if await svc.filter_by_plan(ids, "pro") != ["900013"] or await svc.filter_by_plan(ids, "free") != ["900014", "900015"] \
                    or await svc.filter_by_plan(ids, "all") != ids:
                return "фильтр аудитории рассылки по тарифу работает неверно"
            await svc.mark_blocked(["900014", "900014"])
            if await svc.blocked_ids() != ["900014"]:
                return "список заблокировавших бота ведётся неверно"

        async def admins():
            if not await svc.add_admin("900008") or not await svc.is_admin("900008"):
                return "администратор не добавлен"
            if await svc.add_admin("900008"):
                return "дубль администратора добавлен"
            if await svc.check_and_consume("900008") is not None:
                return "администратор попал под лимит"
            if not await svc.remove_admin("900008") or await svc.is_admin("900008"):
                return "права администратора не сняты"

        for name, case in (("Лимиты и доступ к функциям", limits), ("Истечение подписки", expiry),
                           ("Оплата (идемпотентность, сумма, продление)", payment), ("Возврат платежа", payment_refund),
                           ("Реферальные бонусы", referrals), ("Бонусные генерации", credits), ("Роли администраторов", admins),
                           ("Права админ-панели и ADMIN_USER_IDS", access_levels), ("Главный выключатель подписок", master_switch),
                           ("Вид интерфейса по роли", ui_roles), ("Аудитория рассылки", broadcast_audience)):
            try:
                error = await case()
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            results.append(_r(G_TEST, name, FAIL if error else OK, error or "",
                              "Ошибка в subscriptions.py — оплату и подписки включать нельзя" if error else ""))
    return results


# ===================== TELEGRAM =====================
G_TG = "Telegram"


async def _check_telegram():
    import aiohttp
    token = config.BOT_TOKEN
    if not token:
        return _r(G_TG, "getMe", FAIL, "нет токена")
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"https://api.telegram.org/bot{token}/getMe") as resp:
                data = await resp.json(content_type=None)
                if resp.status == 401:
                    return _r(G_TG, "Токен", FAIL, "Telegram отклонил токен (401)", "Перевыпустите токен в @BotFather и обновите .env")
                if not data.get("ok"):
                    return _r(G_TG, "Токен", FAIL, str(data.get("description", resp.status)))
                username = data["result"].get("username")
                return [_r(G_TG, "Токен", OK, f"@{username}"),
                        _r(G_TG, "Реферальная ссылка", INFO, f"https://t.me/{username}?start=ref_<id пользователя>")]
    except aiohttp.ClientError as e:
        return _r(G_TG, "Связь с Telegram", WARN, f"нет связи с api.telegram.org: {e}",
                  "Проверьте интернет; при блокировке Telegram нужен прокси/VPN")


# ===================== STABLE DIFFUSION WEBUI =====================
G_SD = "Stable Diffusion WebUI"
G_EXT = "Расширения WebUI"
G_OPT = "Опционально"


async def _check_webui():
    """Возвращает (WebUI доступен, результаты)."""
    from Aapi import check_api_available
    from http_client import sd_get_json
    results: List[CheckResult] = []
    if not await check_api_available():
        return False, [_r(G_SD, "API", FAIL, f"недоступен: {config.SD_BASE_URL}", "Запустите WebUI с флагом --api (и проверьте SD_BASE_URL)")]
    results.append(_r(G_SD, "API", OK, config.SD_BASE_URL))

    try:
        models = await sd_get_json("sdapi/v1/sd-models", timeout=15)
        if models:
            results.append(_r(G_SD, "Чекпоинты", OK, f"{len(models)} шт."))
        else:
            results.append(_r(G_SD, "Чекпоинты", FAIL, "список пуст", "Положите модель в models/Stable-diffusion"))
    except Exception as e:
        results.append(_r(G_SD, "Чекпоинты", FAIL, str(e)))
    try:
        options = await sd_get_json("sdapi/v1/options", timeout=15)
        current = options.get("sd_model_checkpoint")
        results.append(_r(G_SD, "Загруженная модель", OK if current else WARN, current or "не загружена"))
    except Exception as e:
        results.append(_r(G_SD, "Загруженная модель", WARN, str(e)))
    try:
        mem = await sd_get_json("sdapi/v1/memory", timeout=10)
        system = (mem.get("cuda") or {}).get("system") or {}
        if system.get("total"):
            total_gb, free_gb = system["total"] / 1024 ** 3, system.get("free", 0) / 1024 ** 3
            status = WARN if total_gb < 6 else INFO
            results.append(_r(G_SD, "Видеопамять", status, f"всего {total_gb:.1f} ГБ, свободно {free_gb:.1f} ГБ",
                              "Мало VRAM: Hires/SDXL могут не поместиться" if status == WARN else ""))
        else:
            results.append(_r(G_SD, "Видеопамять", WARN, "CUDA не обнаружена — генерация пойдёт на CPU (очень медленно)"))
    except Exception:
        pass  # эндпоинта может не быть — не критично
    return True, results


async def _check_adetailer() -> CheckResult:
    from Aapi import check_adetailer_available
    if await check_adetailer_available(force_refresh=True):
        return _r(G_EXT, "ADetailer", OK)
    return _r(G_EXT, "ADetailer", WARN, "не найден", "Установите расширение adetailer — иначе не будет автодоработки лиц и рук")


async def _check_controlnet() -> List[CheckResult]:
    from http_client import sd_get_json
    results = []
    try:
        ver = await sd_get_json("controlnet/version", timeout=10)
        results.append(_r(G_EXT, "ControlNet", OK, f"версия API {ver.get('version', '?')}"))
    except Exception:
        return [_r(G_EXT, "ControlNet", WARN, "не найден", "Установите sd-webui-controlnet — нужен для поз и FaceID")]
    try:
        data = await sd_get_json("controlnet/model_list", timeout=10)
        models = data.get("model_list", []) if isinstance(data, dict) else list(data or [])
        if any("openpose" in m.lower() for m in models):
            results.append(_r(G_EXT, "ControlNet: openpose", OK))
        else:
            results.append(_r(G_EXT, "ControlNet: openpose", WARN, "модель control_v11p_sd15_openpose не найдена",
                              "Положите её в models/ControlNet — без неё генерация по позам не работает"))
    except Exception as e:
        results.append(_r(G_EXT, "ControlNet: модели", WARN, str(e)))
    return results


async def _check_reactor() -> List[CheckResult]:
    from Aapi import check_reactor_available, get_reactor_models
    from http_client import sd_get_json
    if not await check_reactor_available():
        return [_r(G_EXT, "ReActor", WARN, "не найден", "Установите sd-webui-reactor — нужен для замены лиц и моделей лиц")]
    results = [_r(G_EXT, "ReActor", OK)]
    try:
        info = await get_reactor_models()
        results.append(_r(G_EXT, "ReActor: модели лиц", INFO, f"на сервере: {len([m for m in info.get('facemodels', []) if str(m) != 'None'])}"))
    except Exception:
        pass
    try:
        swap = await sd_get_json("reactor/models", timeout=10)
        entries = swap if isinstance(swap, list) else (swap.get("models", []) if isinstance(swap, dict) else [])
        if not any("inswapper" in str(m).lower() for m in entries):
            results.append(_r(G_EXT, "ReActor: inswapper", WARN, "inswapper_128.onnx не найден", "Положите файл в models/insightface"))
    except Exception:
        pass
    return results


async def _check_faceid() -> List[CheckResult]:
    from Aapi import get_faceid_setup
    setup = await get_faceid_setup(force=True)
    if not setup.get("ok"):
        return [_r(G_EXT, "FaceID (лицо в генерации)", WARN, setup.get("reason", ""),
                   "Нужны sd-webui-controlnet ≥1.1.428, insightface и ip-adapter-faceid-plusv2 (см. README, «Лица»)")]
    results = [_r(G_EXT, "FaceID (лицо в генерации)", OK, f"{setup['model']}; юнитов ControlNet: {setup.get('unit_count', 1)}")]
    if not setup.get("lora"):
        results.append(_r(G_EXT, "FaceID: LoRA", WARN, "LoRA faceid не найдена", "Положите ip-adapter-faceid-plusv2_*_lora.safetensors в models/Lora — иначе сходство слабее"))
    return results


async def _check_assets() -> List[CheckResult]:
    from http_client import sd_get_json
    results = []
    try:
        loras = await sd_get_json("sdapi/v1/loras", timeout=15)
        results.append(_r(G_SD, "LoRA", OK if loras else WARN, f"{len(loras)} шт.",
                          "" if loras else "Список пуст — проверьте models/Lora и обновите список в WebUI"))
    except Exception as e:
        results.append(_r(G_SD, "LoRA", WARN, str(e)))
    try:
        emb = await sd_get_json("sdapi/v1/embeddings", timeout=15)
        loaded = {k.lower() for k in (emb.get("loaded") or {})}
        results.append(_r(G_SD, "Эмбеддинги", INFO, f"загружено {len(loaded)}"))
        missing = [e for e in RECOMMENDED_NEGATIVE_EMBEDDINGS if e not in loaded]
        if missing:
            results.append(_r(G_SD, "Негативные эмбеддинги", WARN, "нет: " + ", ".join(missing),
                              "Необязательно: бот использует их, если загружены в WebUI"))
    except Exception as e:
        results.append(_r(G_SD, "Эмбеддинги", WARN, str(e)))
    return results


async def _check_ollama() -> CheckResult:
    import aiohttp
    model = os.environ.get("OLLAMA_MODEL", "aya:8b")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{config.OLLAMA_BASE_URL}/api/tags") as resp:
                data = await resp.json(content_type=None)
    except Exception as e:
        return _r(G_OPT, "Ollama", WARN, f"недоступна ({config.OLLAMA_BASE_URL}): {type(e).__name__}",
                  "Необязательно: без неё перевод и улучшение промптов выключены")
    names = [m.get("name", "") for m in data.get("models", [])]
    if any(n == model or n.split(":")[0] == model.split(":")[0] and ":" not in model for n in names):
        return _r(G_OPT, "Ollama", OK, f"модель {model}")
    return _r(G_OPT, "Ollama", WARN, f"работает, но модели {model} нет", f"ollama pull {model}")


async def _check_poses() -> CheckResult:
    from controlnet_manager import get_controlnet_sources
    sources = await get_controlnet_sources(force_refresh=True)
    if sources:
        return _r(G_OPT, "База поз ControlNet", OK, f"{len(sources)} шт.")
    return _r(G_OPT, "База поз ControlNet", WARN, "пуста", "Положите parquet-датасеты в code/data/controlnet_images/<категория>/")


# ===================== ЗАПУСК И ОТЧЁТ =====================
async def run_health_check(telegram: bool = True) -> List[CheckResult]:
    results: List[CheckResult] = [check_python()]
    packages = await asyncio.to_thread(check_packages)
    results += packages
    results += check_env()
    results += await asyncio.to_thread(check_data)
    results += await asyncio.to_thread(check_payments_config)
    results += await _guard(G_TEST, "Самотест", selftest_subscriptions())

    if any(r.status == FAIL and r.name == "Обязательные пакеты" for r in packages):
        # Без aiohttp/aiogram сетевые проверки невозможны — не засыпаем отчёт вторичными ошибками
        results.append(_r(G_SD, "Сетевые проверки", WARN, "пропущены: не установлены обязательные пакеты",
                          "Сначала pip install -r requirements.txt, затем повторите проверку"))
        return results

    if telegram:
        results += await _guard(G_TG, "Токен", _check_telegram())

    webui = await _guard(G_SD, "API", _check_webui())
    up = False
    if webui and isinstance(webui[0], tuple):  # _guard вернул не список результатов, а кортеж (up, results)
        up, items = webui[0]
        results += items
    else:
        results += webui

    if up:
        parts = await asyncio.gather(
            _guard(G_EXT, "ADetailer", _check_adetailer()),
            _guard(G_EXT, "ControlNet", _check_controlnet()),
            _guard(G_EXT, "ReActor", _check_reactor()),
            _guard(G_EXT, "FaceID", _check_faceid()),
            _guard(G_SD, "LoRA/эмбеддинги", _check_assets()),
        )
        for part in parts:
            results += part
    else:
        results.append(_r(G_EXT, "Расширения WebUI", WARN, "проверка пропущена: WebUI недоступен"))

    results += await _guard(G_OPT, "Ollama", _check_ollama())
    results += await _guard(G_OPT, "База поз", _check_poses())
    return results


def summarize(results: List[CheckResult]):
    return (sum(r.status == OK for r in results), sum(r.status == WARN for r in results),
            sum(r.status == FAIL for r in results))


def has_failures(results: List[CheckResult]) -> bool:
    return any(r.status == FAIL for r in results)


def format_report(results: List[CheckResult], color: bool = False) -> str:
    lines = ["", "════════ ПРОВЕРКА СИСТЕМ ════════"]
    group = None
    for r in results:
        if r.group != group:
            group = r.group
            lines.append(f"\n▸ {group}")
        icon = ICONS[r.status]
        if color:
            icon = f"{_COLORS[r.status]}{icon}{_RESET}"
        detail = f" — {r.detail}" if r.detail else ""
        lines.append(f"  {icon} {r.name}{detail}")
        if r.hint and r.status in (WARN, FAIL):
            lines.append(f"      ↳ {r.hint}")
    ok, warn, fail = summarize(results)
    verdict = "всё готово" if not fail and not warn else ("есть замечания" if not fail else "есть критичные ошибки")
    lines.append(f"\nИтог: ✅ {ok}  ⚠️ {warn}  ❌ {fail} — {verdict}\n")
    return "\n".join(lines)


def telegram_summary(results: List[CheckResult]) -> str:
    """Короткая HTML-сводка для администратора: только предупреждения и ошибки."""
    import html
    ok, warn, fail = summarize(results)
    lines = [f"🤖 <b>Бот запущен</b>: ✅ {ok}  ⚠️ {warn}  ❌ {fail}"]
    for r in results:
        if r.status in (WARN, FAIL):
            lines.append(f"{ICONS[r.status]} {html.escape(r.name)}: {html.escape(r.detail or r.hint)[:200]}")
    return "\n".join(lines)


def use_color() -> bool:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        os.system("")  # включает ANSI-цвета в консоли Windows 10+
    return True


async def _amain(args) -> int:
    try:
        results = await run_health_check(telegram=not args.no_telegram)
    finally:
        try:
            from http_client import close_session
            await close_session()
        except Exception:
            pass  # http_client не импортируется без aiohttp — закрывать нечего
    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        print(format_report(results, color=use_color()))
    return 1 if has_failures(results) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка всех систем бота")
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    parser.add_argument("--no-telegram", action="store_true", help="не обращаться к api.telegram.org")
    sys.exit(asyncio.run(_amain(parser.parse_args())))
