import os
import requests
import json

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8646586971:AAFfMjrcJpzxWdCgven6MbNmzAsDsWB70m0")
TELEGRAM_AGENT_CHAT_ID = os.environ.get("TELEGRAM_AGENT_CHAT_ID", "7202298356")
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_telegram_message(text, reply_markup=None):
    """Sends a message to the agent's personal Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_AGENT_CHAT_ID:
        print("[Telegram] Bot token or Chat ID missing.")
        return None

    url = f"{TELEGRAM_API_BASE}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_AGENT_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        res = requests.post(url, json=payload, timeout=10)
        data = res.json()
        if not data.get("ok"):
            print(f"[Telegram] Error sending message: {data}")
            return None
        return data.get("result", {}).get("message_id")
    except Exception as e:
        print(f"[Telegram] Request failed: {e}")
        return None


def notify_live_request(site_name, session_id, user_message):
    """
    Alerts you on Telegram that a visitor wants live support.
    Includes session ID tag so replies can be mapped back.
    """
    text = (
        f"🚨 <b>Live Chat Request</b>\n"
        f"🏢 <b>Site:</b> {site_name}\n"
        f"🆔 <b>Session:</b> <code>{session_id}</code>\n"
        f"💬 <b>Visitor:</b> <i>\"{user_message}\"</i>\n\n"
        f"👉 <i>Swipe right to Reply directly to this message to send your response to the visitor!</i>"
    )
    return send_telegram_message(text)


def notify_visitor_reply(site_name, session_id, user_message):
    """Alerts you when an ongoing live chat visitor sends another message."""
    text = (
        f"💬 <b>[{site_name}] Session <code>{session_id}</code>:</b>\n"
        f"\"{user_message}\"\n\n"
        f"👉 <i>Reply to this message to answer.</i>"
    )
    return send_telegram_message(text)
