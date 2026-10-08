"""
Multi-Tenant Live Chatbot Backend with Telegram Mobile Agent Bridge
Built for Salavai Laundry and multi-website business integrations.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import io
import time
import uuid
import re
from datetime import datetime
from email_utils import send_lead_notification
from document_utils import extract_text_from_file, chunk_text_into_qa_pairs
from telegram_utils import notify_live_request, notify_visitor_reply, send_telegram_message

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "salavai-multi-site-secret-2026")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "salavai123")
TELEGRAM_AGENT_CHAT_ID = os.environ.get("TELEGRAM_AGENT_CHAT_ID", "7202298356")

# ---------------------------------------------------------------------------
# Multi-Site Configurations (Tenants)
# ---------------------------------------------------------------------------
SITES = {
    "salavai": {
        "name": "The Salavai Laundry — Main Website",
        "primary_color": "#a41e22",
        "dark_color": "#14324f",
        "greeting": "👋 Welcome to The Salavai Laundry! How can we assist you with your laundry or dry cleaning today?",
        "quick_replies": [
            "What is KG Laundry pricing?",
            "How does pickup work?",
            "I want to become a partner",
            "💬 Talk to Human"
        ]
    },
    "pos": {
        "name": "Salavai POS / Billing Counter",
        "primary_color": "#2563eb",
        "dark_color": "#0f172a",
        "greeting": "👋 Hi! Need help with billing, invoices, or a customer order lookup at the counter?",
        "quick_replies": ["Order status lookup", "Billing / invoice issue", "💬 Talk to Human"]
    },
    "ecommerce": {
        "name": "Salavai Online Store (E-commerce)",
        "primary_color": "#059669",
        "dark_color": "#064e3b",
        "greeting": "👋 Welcome to our online store! Ask about orders, delivery, or returns.",
        "quick_replies": ["Track my order", "Delivery & returns policy", "💬 Talk to Human"]
    },
    "franchise": {
        "name": "Salavai Franchise / Partner Portal",
        "primary_color": "#7c3aed",
        "dark_color": "#3b0764",
        "greeting": "👋 Welcome! Interested in starting your own Salavai franchise?",
        "quick_replies": ["Franchise investment details", "I want to become a partner", "💬 Talk to Human"]
    }
}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Knowledge Base & NLP Search (per-site, with a shared "core" layer)
# ---------------------------------------------------------------------------
# CORE_KNOWLEDGE_BASE applies to every site automatically (hours, address,
# core services, SIGP program, etc.) -- these facts are true business-wide.
# SITE_KNOWLEDGE_BASE holds additional entries uploaded for ONE specific site
# only (e.g. a POS billing FAQ doc, or an e-commerce delivery policy doc).
CORE_KNOWLEDGE_BASE = [
    {"question": "What is KG Laundry and how is it priced?",
     "answer": "KG Laundry is our everyday laundry service priced by weight, not per item. "
               "It's best for daily wear, bedsheets, towels, and everyday loads -- just bag it, "
               "weigh it, and hand it over. We wash, dry, and fold everything with care."},
    {"question": "How does doorstep pickup and delivery work?",
     "answer": "You can schedule a pickup directly from our website or WhatsApp. Our team collects "
               "your laundry from your doorstep, processes it at our store, and delivers it back "
               "to you -- no need to visit the store in person."},
    {"question": "How do you clean delicate silk sarees and suits?",
     "answer": "Silk sarees and delicate garments are handled under our Silk & Delicates program, "
               "using specialized wet cleaning and precise temperature control to protect fine fabrics."},
    {"question": "What is the standard turnaround delivery time?",
     "answer": "Standard turnaround is typically 24-48 hours depending on the service type and load. "
               "KG Laundry is usually fastest; delicate silk/specialty items may take a bit longer."},
    {"question": "Where are your store locations situated?",
     "answer": "You can search by your pincode or area name on our website to find your nearest active "
               "store and its pickup coverage area."},
    {"question": "What services do you offer?",
     "answer": "We offer KG Laundry (priced by weight), Retail Laundry (per item), Silk & Delicates "
               "(specialized care), and Pickup & Delivery across all service types."},
    {"question": "What equipment and technology do you use?",
     "answer": "We use modern commercial laundry machines, LG wet cleaning technology, and eco-friendly "
               "cleaning chemicals -- the same standard trusted by hospitality and garment-care businesses."},
    {"question": "How can I contact support? What is your WhatsApp number, phone number, or email?",
     "answer": "You can reach us via WhatsApp or phone at +91 6385550203, or email us at "
               "contact@thesalavailaundry.com. Our main store is located at 2/658, East Coast Rd, "
               "Ranga Reddy Gardens, Neelankarai, Chennai, Tamil Nadu 600115. "
               "Store hours are typically Mon-Sat 8 AM-8 PM, Sun 9 AM-5 PM."},
    {"question": "What is the Self Income Generation Program (SIGP)?",
     "answer": "The Self Income Generation Program (SIGP) empowers communities through professional "
               "laundry entrepreneurship across Tamil Nadu. It's designed for dhobi families, "
               "homemakers, and differently-abled individuals -- no prior business experience required. "
               "You can submit an SIGP enquiry directly on our website with your background and "
               "questions about capital, equipment, and training support."},
]

# Per-site extra knowledge, uploaded via /admin. Starts empty for every site.
SITE_KNOWLEDGE_BASE = {site_id: [] for site_id in SITES}

# Per-site search index -- rebuilt whenever that site's KB changes.
_vectorizers = {}
_kb_vectors = {}
_kb_entries = {}  # site_id -> combined list of entries actually indexed (core + site-specific)


def rebuild_search_index(site_id):
    """Rebuilds the TF-IDF search index for one site from CORE + its own uploads."""
    entries = CORE_KNOWLEDGE_BASE + SITE_KNOWLEDGE_BASE.get(site_id, [])
    _kb_entries[site_id] = entries
    vectorizer = TfidfVectorizer(stop_words="english")
    _vectorizers[site_id] = vectorizer
    _kb_vectors[site_id] = vectorizer.fit_transform(
        [item["question"] + " " + item["answer"] for item in entries]
    )


def rebuild_all_search_indexes():
    for site_id in SITES:
        rebuild_search_index(site_id)


rebuild_all_search_indexes()

# ---------------------------------------------------------------------------
# Live Sessions & State Management
# ---------------------------------------------------------------------------
# SESSIONS = {
#    session_id: {
#        "site_id": "salavai",
#        "mode": "bot" | "live" | "franchise",
#        "messages": [ {"sender": "user"|"bot"|"agent", "text": "...", "time": timestamp} ],
#        "telegram_msg_id": 12345,
#        "pending_agent_replies": ["..."]
#    }
# }
SESSIONS = {}
TELEGRAM_MSG_TO_SESSION = {}  # telegram_msg_id -> session_id
franchise_leads = []
live_chat_leads = []  # one entry per "Talk to Human" request -- a separate billable lead type
notify_emails = [e.strip() for e in os.environ.get("NOTIFY_EMAIL", "").split(",") if e.strip()]


def get_faq_answer(user_question, site_id, threshold=0.15):
    site_id = site_id if site_id in SITES else "salavai"
    vectorizer = _vectorizers.get(site_id)
    kb_vectors = _kb_vectors.get(site_id)
    entries = _kb_entries.get(site_id, [])
    if not vectorizer or not entries:
        return None, 0.0

    question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vector, kb_vectors)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    if best_score < threshold:
        return None, best_score
    return entries[best_idx]["answer"], best_score


# ---------------------------------------------------------------------------
# Routes: Core Chat & Widget
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/widget.js")
def widget_js():
    response = send_from_directory("static", "widget.js", mimetype="application/javascript")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/embed-test")
def embed_test():
    return render_template("embed_test.html")


@app.route("/demo/<site_id>")
def demo_site(site_id):
    """Generic mock website preview for any tenant — used to demo the widget
    live on 4 separate 'websites' (POS, E-commerce, Franchise, Main site)."""
    cfg = SITES.get(site_id)
    if not cfg:
        return "Unknown site_id. Try one of: " + ", ".join(SITES.keys()), 404
    return render_template("mock_site.html", site_id=site_id, cfg=cfg)


@app.route("/api/site-config")
def site_config():
    site_id = request.args.get("site_id", "salavai")
    cfg = SITES.get(site_id, SITES["salavai"])
    return jsonify(cfg)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id")
    site_id = data.get("site_id", "salavai")
    site_cfg = SITES.get(site_id, SITES["salavai"])

    if not session_id or session_id not in SESSIONS:
        session_id = uuid.uuid4().hex[:8]
        SESSIONS[session_id] = {
            "site_id": site_id,
            "mode": "bot",
            "messages": [],
            "pending_agent_replies": [],
            "state": {}
        }

    session_obj = SESSIONS[session_id]
    session_obj["messages"].append({"sender": "user", "text": message, "time": time.time()})

    # Check if user specifically requested live human agent
    live_triggers = ["human", "agent", "support", "person", "representative", "talk to human", "live chat"]
    is_live_request = any(trig in message.lower() for trig in live_triggers)

    # 1. User is currently in LIVE agent mode
    if session_obj["mode"] == "live":
        msg_id = notify_visitor_reply(site_cfg["name"], session_id, message)
        if msg_id:
            TELEGRAM_MSG_TO_SESSION[msg_id] = session_id
        return jsonify({
            "reply": None,  # no bot reply; user is waiting for agent
            "session_id": session_id,
            "mode": "live",
            "status": "waiting_for_agent"
        })

    # 2. User just requested to switch to LIVE agent
    if is_live_request:
        session_obj["mode"] = "live"
        msg_id = notify_live_request(site_cfg["name"], session_id, message)
        if msg_id:
            TELEGRAM_MSG_TO_SESSION[msg_id] = session_id
            session_obj["telegram_msg_id"] = msg_id

        live_chat_leads.append({
            "type": "live_chat",
            "site_id": site_id,
            "site_name": site_cfg["name"],
            "session_id": session_id,
            "message": message,
            "time": time.time()
        })

        return jsonify({
            "reply": "Connecting you with our support team... An agent has been notified and will reply here in just a moment!",
            "session_id": session_id,
            "mode": "live"
        })

    # 3. Franchise Lead Flow
    session_state = session_obj.get("state", {})
    if session_state.get("mode") == "franchise":
        return handle_franchise_flow(message, session_obj, session_id)

    franchise_triggers = ["franchise", "partner", "business opportunity", "start my own", "own laundry business"]
    if any(trigger in message.lower() for trigger in franchise_triggers):
        session_state["mode"] = "franchise"
        session_state["step"] = "name"
        session_state["data"] = {}
        session_obj["state"] = session_state
        return jsonify({
            "reply": "That's great! I'd love to help you explore becoming a business partner. First, what's your full name?",
            "session_id": session_id,
            "mode": "bot"
        })

    # 4. Standard FAQ answer
    answer, score = get_faq_answer(message, site_id)
    if answer:
        reply = answer
    else:
        # Bot couldn't answer confidently -> offer live human handoff
        reply = ("I'm not fully sure about that yet! Would you like to chat with a live agent? "
                 "Just tap '💬 Talk to Human' below or let me know!")

    session_obj["messages"].append({"sender": "bot", "text": reply, "time": time.time()})
    return jsonify({
        "reply": reply,
        "session_id": session_id,
        "mode": session_obj["mode"]
    })


def handle_franchise_flow(message, session_obj, session_id):
    session_state = session_obj.get("state", {})
    step = session_state.get("step", "name")
    data = session_state.get("data", {})

    if step == "name":
        data["name"] = message
        session_state["step"] = "city"
        reply = f"Nice to meet you, {message}! Which city or area are you interested in operating in?"
    elif step == "city":
        data["city"] = message
        session_state["step"] = "experience"
        reply = "Got it. Do you have any prior experience running a business or service operation? (Yes/No is fine)"
    elif step == "experience":
        data["experience"] = message
        session_state["step"] = "budget"
        reply = "Thanks! And roughly what investment budget are you considering for this opportunity?"
    elif step == "budget":
        data["budget"] = message
        session_state["step"] = "done"

        lead_record = dict(data)
        lead_record["site_id"] = session_obj.get("site_id", "salavai")
        lead_record["site_name"] = SITES.get(lead_record["site_id"], {}).get("name", lead_record["site_id"])
        lead_record["time"] = time.time()
        lead_record["type"] = "franchise"
        franchise_leads.append(lead_record)
        send_lead_notification(data, notify_emails)

        reply = (f"Perfect, thank you {data.get('name', '')}! Here's a summary of what you shared:\n"
                 f"📍 City: {data.get('city')}\n"
                 f"💼 Experience: {data.get('experience')}\n"
                 f"💰 Budget: {data.get('budget')}\n\n"
                 f"Our partnership team will reach out to you directly with next steps.")
        session_state["mode"] = None
    else:
        reply = "Thanks again for your interest! Feel free to ask me anything else."
        session_state["mode"] = None

    session_state["data"] = data
    session_obj["state"] = session_state
    session_obj["messages"].append({"sender": "bot", "text": reply, "time": time.time()})
    return jsonify({"reply": reply, "session_id": session_id, "mode": "bot"})


@app.route("/api/chat/poll")
def poll_messages():
    """Poll endpoint for visitor widget to receive agent replies in live mode."""
    session_id = request.args.get("session_id")
    if not session_id or session_id not in SESSIONS:
        return jsonify({"new_messages": []})

    session_obj = SESSIONS[session_id]
    pending = list(session_obj.get("pending_agent_replies", []))
    session_obj["pending_agent_replies"] = []  # Clear after reading

    return jsonify({
        "new_messages": pending,
        "mode": session_obj["mode"]
    })


# ---------------------------------------------------------------------------
# Telegram Webhook: Receives your replies on Telegram & routes to visitor
# ---------------------------------------------------------------------------
@app.route("/api/telegram/webhook", methods=["POST"])
def telegram_webhook():
    update = request.get_json() or {}
    message = update.get("message", {})
    text = message.get("text", "").strip()
    reply_to = message.get("reply_to_message", {})
    from_user = message.get("from", {})
    agent_name = from_user.get("first_name") or from_user.get("username") or "An agent"

    if not text:
        return jsonify({"ok": True})

    # ---------------------------------------------------------------------
    # /claim command: lets an agent explicitly take over a session, even if
    # another agent already claimed it (e.g. shift handoff). Usage:
    #   /claim #a1b2c3d4
    # ---------------------------------------------------------------------
    claim_match = re.match(r"^/claim\b\s*#?([a-f0-9]{8})?\s*$", text, re.IGNORECASE)
    if claim_match:
        claim_session_id = claim_match.group(1).lower() if claim_match.group(1) else None

        if not claim_session_id and reply_to:
            claim_session_id = TELEGRAM_MSG_TO_SESSION.get(reply_to.get("message_id"))

        if not claim_session_id:
            active_live = [(sid, sess) for sid, sess in SESSIONS.items() if sess.get("mode") == "live"]
            if len(active_live) == 1:
                claim_session_id = active_live[0][0]
            elif len(active_live) > 1:
                listing = "\n".join(
                    f"• <code>{sid}</code> — {SITES.get(sess.get('site_id'), {}).get('name', sess.get('site_id'))}"
                    for sid, sess in active_live
                )
                send_telegram_message(
                    "⚠️ Multiple live chats are active. Specify which to claim, e.g.:\n"
                    "<code>/claim #a1b2c3d4</code>\n\n"
                    f"Active sessions:\n{listing}"
                )
                return jsonify({"ok": True})

        if claim_session_id and claim_session_id in SESSIONS:
            session_obj = SESSIONS[claim_session_id]
            previous = session_obj.get("claimed_by")
            session_obj["claimed_by"] = agent_name
            site_name = SITES.get(session_obj.get("site_id"), {}).get("name", session_obj.get("site_id"))
            if previous and previous != agent_name:
                send_telegram_message(f"🔁 <b>{agent_name}</b> has taken over session <code>{claim_session_id}</code> ({site_name}) from {previous}.")
            else:
                send_telegram_message(f"✋ <b>{agent_name}</b> is now handling session <code>{claim_session_id}</code> ({site_name}).")
        else:
            send_telegram_message("⚠️ Could not find that session to claim. Use <code>/claim #session_id</code>.")

        return jsonify({"ok": True})

    # ---------------------------------------------------------------------
    # /close command: ends live-agent mode and hands the session back to the
    # bot, so it resumes answering FAQs automatically. Usage:
    #   /close #a1b2c3d4     -> close that specific session
    #   /close                (as a reply to an alert) -> closes that session
    # ---------------------------------------------------------------------
    close_match = re.match(r"^/close\b\s*#?([a-f0-9]{8})?\s*$", text, re.IGNORECASE)
    if close_match:
        close_session_id = close_match.group(1).lower() if close_match.group(1) else None

        if not close_session_id and reply_to:
            close_session_id = TELEGRAM_MSG_TO_SESSION.get(reply_to.get("message_id"))

        if not close_session_id:
            active_live = [(sid, sess) for sid, sess in SESSIONS.items() if sess.get("mode") == "live"]
            if len(active_live) == 1:
                close_session_id = active_live[0][0]
            elif len(active_live) > 1:
                listing = "\n".join(
                    f"• <code>{sid}</code> — {SITES.get(sess.get('site_id'), {}).get('name', sess.get('site_id'))}"
                    for sid, sess in active_live
                )
                send_telegram_message(
                    "⚠️ Multiple live chats are active. Specify which to close, e.g.:\n"
                    "<code>/close #a1b2c3d4</code>\n\n"
                    f"Active sessions:\n{listing}"
                )
                return jsonify({"ok": True})

        if close_session_id and close_session_id in SESSIONS:
            session_obj = SESSIONS[close_session_id]
            session_obj["mode"] = "bot"
            session_obj["claimed_by"] = None
            site_name = SITES.get(session_obj.get("site_id"), {}).get("name", session_obj.get("site_id"))
            session_obj["pending_agent_replies"].append({
                "text": "Thanks for chatting with our support team! I'm back to help with any other questions. 😊",
                "sender": "bot",
                "time": time.time()
            })
            send_telegram_message(f"🔒 Closed live chat for <b>{site_name}</b> (session <code>{close_session_id}</code>). Bot has resumed answering FAQs.")
        else:
            send_telegram_message("⚠️ Could not find that session to close. Use <code>/close #session_id</code>.")

        return jsonify({"ok": True})

    target_session_id = None

    # Option A: Check if this was a direct reply to a notification
    if reply_to:
        reply_to_id = reply_to.get("message_id")
        target_session_id = TELEGRAM_MSG_TO_SESSION.get(reply_to_id)

    # Option B: Check if agent explicitly started message with session id, e.g. "#a1b2 Hello!"
    match = re.match(r"^#?([a-f0-9]{8})\s+(.*)$", text, re.IGNORECASE)
    if match:
        target_session_id = match.group(1).lower()
        text = match.group(2)

    # Option C: Only auto-fallback if there is EXACTLY ONE active live session.
    # If there are 0 or 2+, guessing risks sending a reply to the wrong customer/site
    # (this caused Main Website replies to land in the POS chat) -- so we refuse to guess.
    if not target_session_id:
        active_live = [(sid, sess) for sid, sess in SESSIONS.items() if sess.get("mode") == "live"]
        if len(active_live) == 1:
            target_session_id = active_live[0][0]
        elif len(active_live) > 1:
            listing = "\n".join(
                f"• <code>{sid}</code> — {SITES.get(sess.get('site_id'), {}).get('name', sess.get('site_id'))}"
                for sid, sess in active_live
            )
            send_telegram_message(
                "⚠️ Multiple live chats are active right now. I can't guess which one you meant.\n\n"
                "Please <b>swipe-reply</b> to the original alert, or prefix your message with the session id, e.g.:\n"
                "<code>#8ec1815f Hello!</code>\n\n"
                f"Active sessions:\n{listing}"
            )
            return jsonify({"ok": True})

    if target_session_id and target_session_id in SESSIONS:
        session_obj = SESSIONS[target_session_id]
        site_name = SITES.get(session_obj.get("site_id"), {}).get("name", session_obj.get("site_id"))

        # ---------------------------------------------------------------
        # Claim check: first agent to reply "owns" this session. If a
        # DIFFERENT agent tries to reply afterwards, warn instead of
        # silently delivering -- prevents two agents double-replying to
        # the same customer. Use /claim #session_id to override.
        # ---------------------------------------------------------------
        claimed_by = session_obj.get("claimed_by")
        if claimed_by and claimed_by != agent_name:
            send_telegram_message(
                f"⚠️ Session <code>{target_session_id}</code> ({site_name}) is already being handled by "
                f"<b>{claimed_by}</b>. Your message was NOT sent to avoid double-replying.\n\n"
                f"If you need to take over, reply with <code>/claim #{target_session_id}</code> first."
            )
            return jsonify({"ok": True})

        if not claimed_by:
            session_obj["claimed_by"] = agent_name

        session_obj["pending_agent_replies"].append({
            "text": text,
            "sender": "agent",
            "time": time.time()
        })
        session_obj["messages"].append({
            "text": text,
            "sender": "agent",
            "time": time.time()
        })
        send_telegram_message(f"✅ Delivered to <b>{site_name}</b> (session <code>{target_session_id}</code>) — handled by <b>{session_obj['claimed_by']}</b>: \"{text}\"")
    else:
        send_telegram_message("⚠️ Could not match this reply to an active customer session. Use reply-to or start message with <code>#session_id message</code>.")

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Admin Console
# ---------------------------------------------------------------------------
def require_admin_login(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Incorrect password."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@require_admin_login
def admin_dashboard():
    kb_counts = {
        site_id: len(CORE_KNOWLEDGE_BASE) + len(SITE_KNOWLEDGE_BASE.get(site_id, []))
        for site_id in SITES
    }

    # ---------------------------------------------------------------------
    # Lead reporting: combine franchise leads + live-chat ("Talk to Human")
    # leads into one timeline, filterable by range (week/month/custom) and
    # broken down per site -- supports per-lead billing conversations.
    # ---------------------------------------------------------------------
    range_key = request.args.get("range", "all")  # all | week | month | custom
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")

    now = time.time()
    range_start, range_end = None, None
    if range_key == "week":
        range_start = now - 7 * 86400
    elif range_key == "month":
        range_start = now - 30 * 86400
    elif range_key == "custom" and start_date:
        try:
            range_start = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
            if end_date:
                range_end = datetime.strptime(end_date, "%Y-%m-%d").timestamp() + 86400
        except ValueError:
            pass

    def in_range(lead):
        t = lead.get("time", 0)
        if range_start and t < range_start:
            return False
        if range_end and t > range_end:
            return False
        return True

    all_leads = (
        [dict(l, type=l.get("type", "franchise")) for l in franchise_leads] +
        [dict(l) for l in live_chat_leads]
    )
    filtered_leads = [l for l in all_leads if in_range(l)]
    filtered_leads.sort(key=lambda l: l.get("time", 0), reverse=True)
    for lead in filtered_leads:
        lead["date_str"] = datetime.fromtimestamp(lead.get("time", 0)).strftime("%d %b %Y, %I:%M %p")

    site_breakdown = {}
    for lead in filtered_leads:
        sid = lead.get("site_id", "salavai")
        site_breakdown.setdefault(sid, {"franchise": 0, "live_chat": 0})
        site_breakdown[sid][lead.get("type", "franchise")] += 1

    return render_template(
        "admin_dashboard.html",
        sites=SITES,
        kb_counts=kb_counts,
        core_count=len(CORE_KNOWLEDGE_BASE),
        emails=notify_emails,
        leads=franchise_leads,
        filtered_leads=filtered_leads,
        total_leads=len(filtered_leads),
        site_breakdown=site_breakdown,
        range_key=range_key,
        start_date=start_date,
        end_date=end_date,
    )


@app.route("/admin/upload", methods=["POST"])
@require_admin_login
def admin_upload():
    file = request.files.get("document")
    site_id = request.form.get("site_id", "salavai")
    if site_id not in SITES:
        site_id = "salavai"

    if not file or file.filename == "":
        return redirect(url_for("admin_dashboard"))

    filename = file.filename.lower()
    file_bytes = file.read()
    try:
        text = extract_text_from_file(file_bytes, filename)
        new_entries = chunk_text_into_qa_pairs(text)
        SITE_KNOWLEDGE_BASE[site_id].extend(new_entries)
        rebuild_search_index(site_id)
    except Exception as e:
        print(f"[admin_upload] Failed to process document: {e}")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/emails", methods=["POST"])
@require_admin_login
def admin_update_emails():
    global notify_emails
    raw = request.form.get("emails", "")
    notify_emails = [e.strip() for e in raw.split(",") if e.strip()]
    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5008))
    app.run(debug=True, host="0.0.0.0", port=port)
