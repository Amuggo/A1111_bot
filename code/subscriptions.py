"""
Подписки, оплата Telegram Stars, реферальные бонусы и роли администраторов.

Токены:  валюта генераций. 3 токена при первом входе (WELCOME_TOKENS), обычная генерация — 1, с лицом — 4,
         апскейл — 0.5 (цены в .env). Токены тратят все, кроме Pro и администраторов.
Тарифы:  free — генерации за токены, простые настройки; pro  — генерации без токенов, расширенные настройки.
Оплата:  Telegram Stars (валюта XTR, платёжный провайдер не нужен). Товары — PRODUCTS (цены в .env).
Рефералы: приглашённый по ссылке t.me/<бот>?start=ref_<id> и пригласивший получают бонусные генерации;
         когда приглашённый впервые оплачивает подписку, пригласивший получает дни Pro.
Система работает штатно (по умолчанию включена вместе с приёмом оплаты) и выключается полностью двумя способами:
  • SUBSCRIPTIONS_ENABLED=0 в .env — главный выключатель: лимитов, тарифов, кнопок оплаты и разделения интерфейса нет;
  • `sub off` в админ-консоли / `/adm sub off` — то же на лету, без перезапуска.
Вид интерфейса зависит от роли: администраторы и Pro видят расширенные настройки, Free — простые (ADVANCED_FOR_FREE=1 открывает всем).
Администраторы: владелец (ADMIN_USER_ID) и добавленные командой `admin add <id>`; они не ограничиваются.
Данные — data/subscriptions.json. Логика собрана в SubscriptionService, чтобы её можно было прогнать
на временном хранилище (самотест в healthcheck) и не трогать рабочие данные.
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import app_config as config
from storage import JsonStore


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default




def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, "").strip().replace(",", ".") or default))
    except ValueError:
        return default


WELCOME_TOKENS = _float_env("WELCOME_TOKENS", 3)            # токенов при первом входе
COST_GENERATION = _float_env("COST_GENERATION", 1)          # обычная генерация
COST_FACE = _float_env("COST_FACE", 4)                      # генерация с лицом
COST_UPSCALE = _float_env("COST_UPSCALE", 0.5)              # апскейл


def fmt_tokens(value) -> str:
    """5 → «5», 0.5 → «0.5», 2.50 → «2.5»."""
    value = round(float(value), 2)
    return str(int(value)) if value == int(value) else f"{value:g}"

REF_SIGNUP_CREDITS = _int_env("REF_SIGNUP_CREDITS", 5)     # бонусных генераций обоим при регистрации по ссылке
REF_PURCHASE_DAYS = _int_env("REF_PURCHASE_DAYS", 7)       # дней Pro пригласившему за первую оплату приглашённого
REF_MAX_INVITES = _int_env("REF_MAX_INVITES", 30, 1)       # сколько приглашений одного человека приносят бонусы

ADVANCED_FOR_FREE = os.environ.get("ADVANCED_FOR_FREE", "0").strip().lower() in ("1", "true", "yes")

PLANS: Dict[str, dict] = {
    "free": {"title": "Free", "daily": None, "face": True, "advanced": ADVANCED_FOR_FREE},   # лица для Free — за токены
    "pro": {"title": "Pro", "daily": None, "face": True, "advanced": True},   # daily=None — без лимита
}

PRODUCTS: Dict[str, dict] = {
    "pro_30": {"plan": "pro", "days": 30, "stars": _int_env("PRO_30_STARS", 150, 1), "title": "Pro на 30 дней"},
    "pro_90": {"plan": "pro", "days": 90, "stars": _int_env("PRO_90_STARS", 400, 1), "title": "Pro на 90 дней"},
}


def _default_data() -> dict:
    return {"enabled": True, "payments": True, "admins": [], "users": {}, "usage": {}, "credits": {},
            "referrals": {}, "ref_rewarded": {}, "payment_log": [], "blocked": [], "ui_preview": {}, "terms": {}, "welcomed": []}


def _master_on() -> bool:
    """Главный выключатель из .env: SUBSCRIPTIONS_ENABLED=0 отключает систему подписок целиком."""
    return os.environ.get("SUBSCRIPTIONS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def _on(data: dict) -> bool:
    return _master_on() and bool(data.get("enabled"))


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _owner_id() -> str:
    return os.environ.get("ADMIN_USER_ID", "").strip()


def parse_ids(raw: str) -> List[str]:
    """'123, 456;789' -> ['123', '456', '789'] (нечисловые значения отбрасываются)."""
    return [x for x in re.split(r"[,\s;]+", (raw or "").strip()) if x.lstrip("-").isdigit()]


def env_admin_ids() -> List[str]:
    """Администраторы из .env: владелец (ADMIN_USER_ID) и ADMIN_USER_IDS — доступ к админ-панели. Владелец первый."""
    ids = ([_owner_id()] if _owner_id() else []) + parse_ids(os.environ.get("ADMIN_USER_IDS", ""))
    return list(dict.fromkeys(ids))


def _admins(data: dict) -> set:
    return {str(a) for a in data.get("admins") or []} | set(env_admin_ids())


def is_owner(user_id: str) -> bool:
    return bool(_owner_id()) and str(user_id) == _owner_id()


def _active_plan(data: dict, user_id: str) -> Tuple[str, Optional[float]]:
    """(тариф, срок окончания). Истёкшая подписка считается Free — файл при этом не переписываем."""
    entry = (data.get("users") or {}).get(str(user_id))
    if not entry or entry.get("plan") not in PLANS:
        return "free", None
    expires = entry.get("expires")
    if expires is not None and expires <= time.time():
        return "free", None
    return entry["plan"], expires


def _extend(data: dict, user_id: str, plan: str, days: float, note: str = "") -> Optional[float]:
    """Продлевает тариф от текущего срока (если он ещё действует). Возвращает новый срок; None — бессрочно."""
    cur_plan, cur_exp = _active_plan(data, user_id)
    if cur_plan == plan:
        expires = None if cur_exp is None else cur_exp + days * 86400
    else:
        expires = time.time() + days * 86400
    data.setdefault("users", {})[str(user_id)] = {"plan": plan, "expires": expires, "note": note, "granted": time.time()}
    return expires


def fmt_expires(expires: Optional[float]) -> str:
    return "бессрочно" if expires is None else "до " + time.strftime("%d.%m.%Y %H:%M", time.localtime(expires))


class SubscriptionService:
    def __init__(self, store: JsonStore):
        self._store = store

    # ---------- режимы ----------
    async def is_enabled(self) -> bool:
        return _on(await self._store.aread())

    async def set_enabled(self, enabled: bool) -> None:
        await self._store.aupdate(lambda d: d.__setitem__("enabled", bool(enabled)))

    async def payments_enabled(self) -> bool:
        data = await self._store.aread()
        return _on(data) and bool(data.get("payments"))

    async def set_payments(self, enabled: bool) -> None:
        await self._store.aupdate(lambda d: d.__setitem__("payments", bool(enabled)))

    # ---------- администраторы ----------
    async def is_admin(self, user_id: str) -> bool:
        return str(user_id) in _admins(await self._store.aread())

    async def admin_level(self, user_id: str) -> Optional[str]:
        """'owner' (ADMIN_USER_ID), 'admin' (ADMIN_USER_IDS или выданные из консоли) или None."""
        if is_owner(user_id):
            return "owner"
        return "admin" if await self.is_admin(user_id) else None

    async def all_admin_ids(self) -> List[str]:
        data = await self._store.aread()
        return list(dict.fromkeys(env_admin_ids() + [str(a) for a in data.get("admins") or []]))

    async def add_admin(self, user_id: str) -> bool:
        def mutate(data: dict):
            admins = [str(a) for a in data.setdefault("admins", [])]
            if str(user_id) in admins:
                return False
            admins.append(str(user_id))
            data["admins"] = admins
            return True
        return bool(await self._store.aupdate(mutate))

    async def remove_admin(self, user_id: str) -> bool:
        def mutate(data: dict):
            admins = [str(a) for a in data.get("admins") or []]
            if str(user_id) not in admins:
                return False
            data["admins"] = [a for a in admins if a != str(user_id)]
            return True
        return bool(await self._store.aupdate(mutate))

    async def list_admins(self) -> Tuple[str, List[str], List[str]]:
        """(владелец, администраторы из .env, администраторы, выданные из консоли)."""
        data = await self._store.aread()
        env_ids = [a for a in env_admin_ids() if a != _owner_id()]
        return _owner_id(), env_ids, [str(a) for a in data.get("admins") or []]

    # ---------- тарифы ----------
    async def get_plan(self, user_id: str) -> Tuple[str, Optional[float]]:
        return _active_plan(await self._store.aread(), user_id)

    async def grant(self, user_id: str, plan: str, days: Optional[float] = None, note: str = "") -> Optional[float]:
        """Выдаёт тариф «с нуля» (заменяя текущий). days=None — бессрочно. Возвращает срок окончания."""
        if plan not in PLANS:
            raise ValueError(f"Неизвестный тариф «{plan}». Доступны: {', '.join(PLANS)}")
        expires = None if days is None else time.time() + float(days) * 86400

        def mutate(data: dict):
            data.setdefault("users", {})[str(user_id)] = {"plan": plan, "expires": expires, "note": note, "granted": time.time()}

        await self._store.aupdate(mutate)
        return expires

    async def revoke(self, user_id: str) -> bool:
        return bool(await self._store.aupdate(lambda d: (d.get("users") or {}).pop(str(user_id), None) is not None))

    async def reset_usage(self, user_id: str) -> None:
        await self._store.aupdate(lambda d: (d.get("usage") or {}).pop(str(user_id), None))

    async def add_credits(self, user_id: str, amount: float) -> float:
        """Начисляет (или списывает, если amount < 0) токены. Возвращает новый баланс."""
        def mutate(data: dict):
            credits = data.setdefault("credits", {})
            credits[str(user_id)] = max(0.0, round(float(credits.get(str(user_id), 0)) + float(amount), 2))
            return credits[str(user_id)]
        return await self._store.aupdate(mutate)

    # ---------- ограничения ----------
    async def claim_welcome(self, user_id: str) -> float:
        """Один раз выдаёт приветственные токены. Возвращает выданное количество (0 — уже выдавали / не положено)."""
        uid = str(user_id)

        def mutate(data: dict) -> float:
            welcomed = data.setdefault("welcomed", [])
            if uid in welcomed:
                return 0.0
            welcomed.append(uid)
            if not _on(data) or uid in _admins(data) or WELCOME_TOKENS <= 0:
                return 0.0
            credits = data.setdefault("credits", {})
            credits[uid] = round(float(credits.get(uid, 0)) + WELCOME_TOKENS, 2)
            return WELCOME_TOKENS

        return await self._store.aupdate(mutate)

    async def balance(self, user_id: str) -> float:
        return float(((await self._store.aread()).get("credits") or {}).get(str(user_id), 0))

    async def check_and_consume(self, user_id: str, cost: float = None) -> Optional[str]:
        """Перед постановкой задачи в очередь. None — можно (токены списаны), иначе — текст отказа.
        Администраторы и Pro токены не тратят."""
        uid = str(user_id)
        cost = COST_GENERATION if cost is None else float(cost)

        def mutate(data: dict) -> Optional[str]:
            welcomed = data.setdefault("welcomed", [])
            credits = data.setdefault("credits", {})
            if uid not in welcomed:                      # первый вход: приветственные токены
                welcomed.append(uid)
                if _on(data) and uid not in _admins(data) and WELCOME_TOKENS > 0:
                    credits[uid] = round(float(credits.get(uid, 0)) + WELCOME_TOKENS, 2)
            usage = data.setdefault("usage", {}).setdefault(uid, {})
            usage["charged"] = 0.0
            if not _on(data) or uid in _admins(data):
                return None
            plan, _ = _active_plan(data, uid)
            if plan == "pro":
                return None
            have = float(credits.get(uid, 0))
            if have + 1e-9 < cost:
                return (f"🪙 Не хватает токенов: нужно {fmt_tokens(cost)}, у вас {fmt_tokens(have)}.\n"
                        "Токены можно получить, пригласив друзей, или оформить Pro без трат токенов: /plan")
            credits[uid] = round(have - cost, 2)
            usage["charged"] = cost
            if usage.get("day") != _today():
                usage.update(day=_today(), count=0)
            usage["count"] = usage.get("count", 0) + 1
            return None

        return await self._store.aupdate(mutate)

    async def refund(self, user_id: str) -> None:
        """Возвращает токены за последнюю постановку (если задачу не удалось поставить в очередь)."""
        uid = str(user_id)

        def mutate(data: dict):
            usage = (data.get("usage") or {}).get(uid)
            charged = float((usage or {}).get("charged", 0))
            if charged <= 0:
                return
            credits = data.setdefault("credits", {})
            credits[uid] = round(float(credits.get(uid, 0)) + charged, 2)
            usage["charged"] = 0.0
            usage["count"] = max(0, usage.get("count", 0) - 1)

        await self._store.aupdate(mutate)

    async def feature_allowed(self, user_id: str, feature: str) -> bool:
        data = await self._store.aread()
        if not _on(data) or str(user_id) in _admins(data):
            return True
        return bool(PLANS[_active_plan(data, user_id)[0]].get(feature))

    # ---------- вид интерфейса по роли ----------
    async def advanced_allowed(self, user_id: str) -> bool:
        """Расширенные настройки генерации: администраторы (если не включён предпросмотр простого вида), Pro, все — при выключенных подписках."""
        data = await self._store.aread()
        uid = str(user_id)
        if not _on(data):
            return True
        if uid in _admins(data):
            return (data.get("ui_preview") or {}).get(uid) != "simple"
        return bool(PLANS[_active_plan(data, uid)[0]].get("advanced"))

    async def set_ui_preview(self, user_id: str, simple: bool) -> None:
        """Администратор может увидеть интерфейс глазами клиента Free (simple=True) и вернуться (False)."""
        def mutate(data: dict):
            preview = data.setdefault("ui_preview", {})
            if simple:
                preview[str(user_id)] = "simple"
            else:
                preview.pop(str(user_id), None)
        await self._store.aupdate(mutate)

    # ---------- аудитория рассылки ----------
    async def blocked_ids(self) -> List[str]:
        return [str(x) for x in (await self._store.aread()).get("blocked") or []]

    async def mark_blocked(self, user_ids: List[str]) -> None:
        def mutate(data: dict):
            data["blocked"] = list(dict.fromkeys([str(x) for x in data.get("blocked") or []] + [str(u) for u in user_ids]))
        await self._store.aupdate(mutate)

    async def filter_by_plan(self, user_ids: List[str], audience: str) -> List[str]:
        """audience: all | pro | free (по активному на сегодня тарифу)."""
        data = await self._store.aread()
        if audience == "all":
            return list(user_ids)
        return [u for u in user_ids if (_active_plan(data, u)[0] == "pro") == (audience == "pro")]

    # ---------- оплата ----------
    async def process_payment(self, user_id: str, product_key: str, stars: int, charge_id: str, test: bool = False) -> dict:
        """
        Обрабатывает успешную оплату (идемпотентно по charge_id). Возвращает
        {"ok", "duplicate", "expires", "product", "reward": {"referrer", "days", "expires"} | None, "reason"}.
        Реферальный бонус начисляется один раз — за первую оплату приглашённого.
        """
        uid = str(user_id)

        def mutate(data: dict) -> dict:
            log = data.setdefault("payment_log", [])
            if any(p["charge_id"] == charge_id for p in log):
                return {"ok": True, "duplicate": True}
            product = PRODUCTS.get(product_key)
            if product is None:
                return {"ok": False, "reason": f"неизвестный товар «{product_key}»"}
            if not test and int(stars) != product["stars"]:
                return {"ok": False, "reason": f"сумма {stars}⭐ не совпадает с ценой {product['stars']}⭐"}
            expires = _extend(data, uid, product["plan"], product["days"], note="оплата")
            log.append({"charge_id": charge_id, "user": uid, "product": product_key, "stars": int(stars),
                        "days": product["days"], "ts": time.time(), "test": bool(test), "refunded": False})
            del log[:-2000]
            reward = None
            referrer = (data.get("referrals") or {}).get(uid)
            rewarded = data.setdefault("ref_rewarded", {})
            if referrer and uid not in rewarded and REF_PURCHASE_DAYS > 0:
                ref_expires = _extend(data, referrer, "pro", REF_PURCHASE_DAYS, note="реферальный бонус")
                rewarded[uid] = time.time()
                reward = {"referrer": referrer, "days": REF_PURCHASE_DAYS, "expires": ref_expires}
            return {"ok": True, "duplicate": False, "expires": expires, "product": product, "reward": reward}

        return await self._store.aupdate(mutate)

    async def reverse_payment(self, charge_id: str) -> Optional[dict]:
        """Откатывает покупку после возврата Stars: помечает платёж и убавляет дни. Реферальный бонус не отзывается."""
        def mutate(data: dict):
            for p in data.get("payment_log") or []:
                if p["charge_id"] == charge_id and not p.get("refunded"):
                    p["refunded"] = True
                    entry = (data.get("users") or {}).get(p["user"])
                    if entry and entry.get("expires") is not None:
                        new_exp = entry["expires"] - p["days"] * 86400
                        if new_exp <= time.time():
                            data["users"].pop(p["user"], None)
                        else:
                            entry["expires"] = new_exp
                    return dict(p)
            return None

        return await self._store.aupdate(mutate)

    async def list_payments(self, limit: int = 15) -> List[dict]:
        return list(reversed((await self._store.aread()).get("payment_log") or []))[:limit]

    # ---------- согласие с условиями ----------
    async def terms_accepted(self, user_id: str, version: str) -> bool:
        """Принял ли пользователь актуальную версию условий. Администраторы и выключенные подписки — без проверки."""
        data = await self._store.aread()
        if str(user_id) in _admins(data):
            return True
        entry = (data.get("terms") or {}).get(str(user_id))
        return bool(entry) and entry.get("version") == str(version)

    async def accept_terms(self, user_id: str, version: str) -> None:
        def mutate(data: dict):
            data.setdefault("terms", {})[str(user_id)] = {"version": str(version), "ts": time.time()}
        await self._store.aupdate(mutate)

    # ---------- управление подпиской пользователем ----------
    async def usage_today(self, user_id: str) -> int:
        usage = ((await self._store.aread()).get("usage") or {}).get(str(user_id), {})
        return int(usage.get("count", 0)) if usage.get("day") == _today() else 0

    async def user_payments(self, user_id: str, limit: int = 5) -> List[dict]:
        log = (await self._store.aread()).get("payment_log") or []
        return [p for p in reversed(log) if p.get("user") == str(user_id) and not p.get("test")][:limit]

    async def cancel_subscription(self, user_id: str) -> Optional[dict]:
        """Досрочно завершает платную подписку (остаток срока сгорает). Возвращает {'plan', 'expires'} или None, если подписки нет."""
        uid = str(user_id)

        def mutate(data: dict):
            plan, expires = _active_plan(data, uid)
            if plan == "free":
                return None
            (data.get("users") or {}).pop(uid, None)
            return {"plan": plan, "expires": expires}

        return await self._store.aupdate(mutate)

    # ---------- рефералы ----------
    async def register_referral(self, new_user: str, referrer: str, is_new: bool = True) -> dict:
        """Привязывает нового пользователя к пригласившему и начисляет бонусные генерации обоим."""
        new_user, referrer = str(new_user), str(referrer)

        def mutate(data: dict) -> dict:
            if new_user == referrer:
                return {"ok": False, "reason": "нельзя пригласить самого себя"}
            if not is_new:
                return {"ok": False, "reason": "бонус только для новых пользователей"}
            refs = data.setdefault("referrals", {})
            if new_user in refs:
                return {"ok": False, "reason": "приглашение уже использовано"}
            if sum(1 for v in refs.values() if v == referrer) >= REF_MAX_INVITES:
                return {"ok": False, "reason": "лимит приглашений достигнут"}
            refs[new_user] = referrer
            if REF_SIGNUP_CREDITS > 0:
                credits = data.setdefault("credits", {})
                for uid in (new_user, referrer):
                    credits[uid] = round(float(credits.get(uid, 0)) + REF_SIGNUP_CREDITS, 2)
            return {"ok": True, "credits": REF_SIGNUP_CREDITS}

        return await self._store.aupdate(mutate)

    async def referral_info(self, user_id: str) -> dict:
        data = await self._store.aread()
        uid = str(user_id)
        invited = [u for u, r in (data.get("referrals") or {}).items() if r == uid]
        return {"invited": len(invited), "paid": sum(1 for u in invited if u in (data.get("ref_rewarded") or {})),
                "credits": float((data.get("credits") or {}).get(uid, 0)), "referrer": (data.get("referrals") or {}).get(uid)}

    # ---------- тексты и статистика ----------
    async def describe(self, user_id: str, bot_username: Optional[str] = None) -> str:
        """Текст команды /plan (HTML)."""
        data = await self._store.aread()
        if not _on(data):
            return "ℹ️ Подписки сейчас отключены — ограничений нет."
        uid = str(user_id)
        plan, expires = _active_plan(data, uid)
        info = PLANS[plan]
        ref = await self.referral_info(uid)
        lines = [
            f"💳 <b>Ваш тариф: {info['title']}</b>" + (f" ({fmt_expires(expires)})" if plan != "free" else ""),
            "🪙 Токены: не тратятся на Pro" if plan == "pro" else f"🪙 Токенов: {fmt_tokens(ref['credits'])}",
            f"Цены: генерация — {fmt_tokens(COST_GENERATION)}, с лицом — {fmt_tokens(COST_FACE)}, апскейл — {fmt_tokens(COST_UPSCALE)}",
            f"🎛️ Тонкая настройка генерации: {'доступна' if info['advanced'] else 'только на Pro'}",
        ]
        if data.get("payments"):
            lines += ["", "⭐ <b>Оформить Pro</b> (оплата Telegram Stars):"]
            lines += [f"• {p['title']} — {p['stars']}⭐" for p in PRODUCTS.values()]
        lines += await self._referral_lines(uid, bot_username, ref)
        if not data.get("payments"):
            lines += ["", "ℹ️ Онлайн-оплата сейчас недоступна — тариф выдаёт администратор."]
        return "\n".join(lines)

    async def _referral_lines(self, uid: str, bot_username: Optional[str], ref: Optional[dict] = None) -> List[str]:
        ref = ref or await self.referral_info(uid)
        lines = ["", "👥 <b>Приглашайте друзей</b>",
                 f"Вам и другу — по {fmt_tokens(REF_SIGNUP_CREDITS)} токенов; когда друг оформит подписку, вы получите {REF_PURCHASE_DAYS} дн. Pro."]
        if bot_username:
            lines.append(f"<code>https://t.me/{bot_username}?start=ref_{uid}</code>")
        lines.append(f"Приглашено: {ref['invited']}, из них оплатили: {ref['paid']}")
        return lines

    async def describe_referral(self, user_id: str, bot_username: Optional[str] = None) -> str:
        return "\n".join((await self._referral_lines(str(user_id), bot_username))[1:])

    async def stats(self) -> dict:
        data = await self._store.aread()
        counts = {name: 0 for name in PLANS}
        for uid in (data.get("users") or {}):
            counts[_active_plan(data, uid)[0]] += 1
        paid = [p for p in data.get("payment_log") or [] if not p.get("refunded") and not p.get("test")]
        return {"enabled": _on(data), "master": _master_on(), "payments": bool(data.get("payments")),
                "granted": len(data.get("users") or {}), "active": counts, "admins": len(data.get("admins") or []),
                "payments_count": len(paid), "stars_total": sum(p["stars"] for p in paid),
                "referrals": len(data.get("referrals") or {})}

    async def list_users(self) -> List[dict]:
        data = await self._store.aread()
        rows = []
        for uid, entry in (data.get("users") or {}).items():
            plan, expires = _active_plan(data, uid)
            rows.append({"user_id": uid, "plan": plan, "expires": expires, "note": entry.get("note", ""),
                         "expired": entry.get("plan") != plan})
        return rows


# ---------- рабочий экземпляр и прежний модульный интерфейс ----------
service = SubscriptionService(JsonStore(config.SUBSCRIPTIONS_FILE, default_factory=_default_data))

is_enabled = service.is_enabled
set_enabled = service.set_enabled
payments_enabled = service.payments_enabled
set_payments = service.set_payments
is_admin = service.is_admin
advanced_allowed = service.advanced_allowed
set_ui_preview = service.set_ui_preview
blocked_ids = service.blocked_ids
mark_blocked = service.mark_blocked
filter_by_plan = service.filter_by_plan
admin_level = service.admin_level
all_admin_ids = service.all_admin_ids
add_admin = service.add_admin
remove_admin = service.remove_admin
list_admins = service.list_admins
get_plan = service.get_plan
grant = service.grant
revoke = service.revoke
reset_usage = service.reset_usage
add_credits = service.add_credits
check_and_consume = service.check_and_consume
claim_welcome = service.claim_welcome
balance = service.balance
refund = service.refund
feature_allowed = service.feature_allowed
process_payment = service.process_payment
reverse_payment = service.reverse_payment
list_payments = service.list_payments
register_referral = service.register_referral
referral_info = service.referral_info
describe = service.describe
describe_referral = service.describe_referral
stats = service.stats
list_users = service.list_users
terms_accepted = service.terms_accepted
accept_terms = service.accept_terms
user_payments = service.user_payments
usage_today = service.usage_today
cancel_subscription = service.cancel_subscription
