"""
Multi-Tenant Live Chatbot Backend with Telegram Mobile Agent Bridge
Built for Salavai Laundry and multi-website business integrations.

Persistence: Postgres (via SQLAlchemy, see db.py) backs sessions, messages,
leads, and the knowledge base. The TF-IDF search index itself stays in
memory as a derived cache -- it's rebuilt from DB rows on startup and
whenever the knowledge base changes, so a redeploy never loses real data,
only this cheap-to-rebuild index.
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
import db as dbm
from db import ChatSession, ChatMessage, TelegramMsgMap, FranchiseLead, LiveChatLead, KnowledgeEntry, NotifyEmail, Site

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "salavai-multi-site-secret-2026")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "salavai123")
TELEGRAM_AGENT_CHAT_ID = os.environ.get("TELEGRAM_AGENT_CHAT_ID", "7202298356")

# ---------------------------------------------------------------------------
# Default/seed site configs. These are ONLY used to populate the `sites`
# table the first time the app runs against an empty database. After that,
# the DB is the source of truth -- new tenants/websites can be added via
# /admin/sites without any code change or redeploy.
# ---------------------------------------------------------------------------
DEFAULT_SITES = {
    "salavai": {
        "name": "The Salavai Laundry — Main Website",
        "primary_color": "#a41e22",
        "dark_color": "#14324f",
        "greeting": "👋 Welcome to The Salavai Laundry! How can we assist you with your laundry or dry cleaning today?",
        "quick_replies": [
            "What is KG Laundry pricing?",
            "How does pickup work?",
            "I want to become a partner",
            "💬 Chat with Human"
        ]
    },
    "pos": {
        "name": "Salavai POS / Billing Counter",
        "primary_color": "#2563eb",
        "dark_color": "#0f172a",
        "greeting": "👋 Hi! Need help with billing, invoices, or a customer order lookup at the counter?",
        "quick_replies": ["Order status lookup", "Billing / invoice issue", "💬 Chat with Human"]
    },
    "ecommerce": {
        "name": "Salavai Online Store (E-commerce)",
        "primary_color": "#059669",
        "dark_color": "#064e3b",
        "greeting": "👋 Welcome to our online store! Ask about orders, delivery, or returns.",
        "quick_replies": ["Track my order", "Delivery & returns policy", "💬 Chat with Human"]
    },
    "franchise": {
        "name": "Salavai Franchise / Partner Portal",
        "primary_color": "#7c3aed",
        "dark_color": "#3b0764",
        "greeting": "👋 Welcome! Interested in starting your own Salavai franchise?",
        "quick_replies": ["Franchise investment details", "I want to become a partner", "💬 Chat with Human"]
    }
}

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


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Sites (Tenants) -- DB-backed, with an in-memory cache. New websites can be
# added via /admin/sites without any code change or redeploy.
# ---------------------------------------------------------------------------
_sites_cache = {}


def ensure_default_sites_seeded():
    """One-time: insert DEFAULT_SITES into the DB if the sites table is empty."""
    with dbm.get_db() as s:
        existing = s.query(Site).count()
        if existing == 0:
            for site_id, cfg in DEFAULT_SITES.items():
                s.add(Site(
                    site_id=site_id, name=cfg["name"], primary_color=cfg["primary_color"],
                    dark_color=cfg["dark_color"], greeting=cfg["greeting"],
                    quick_replies=cfg["quick_replies"]
                ))
            s.commit()


def reload_sites_cache():
    global _sites_cache
    with dbm.get_db() as s:
        rows = s.query(Site).all()
        _sites_cache = {
            r.site_id: {
                "name": r.name, "primary_color": r.primary_color, "dark_color": r.dark_color,
                "greeting": r.greeting, "quick_replies": r.quick_replies or []
            }
            for r in rows
        }


def get_sites():
    """Returns the current site config dict, e.g. {'salavai': {...}, 'pos': {...}}."""
    return _sites_cache


def get_site(site_id, default_id="salavai"):
    return _sites_cache.get(site_id) or _sites_cache.get(default_id) or {}


# ---------------------------------------------------------------------------
# Knowledge Base & NLP Search (per-site, with a shared "core" layer)
# ---------------------------------------------------------------------------
# Core entries live in CORE_KNOWLEDGE_BASE (code, not DB -- they're part of
# the product, not tenant data). Site-specific uploaded entries live in the
# knowledge_entries table (site_id set). The TF-IDF index itself is a cheap
# in-memory cache rebuilt from these two sources.
_vectorizers = {}
_kb_vectors = {}
_kb_entries = {}  # site_id -> combined list of entries actually indexed


def ensure_core_kb_seeded():
    """One-time: insert CORE_KNOWLEDGE_BASE into the DB (site_id=NULL) if empty."""
    with dbm.get_db() as s:
        existing = s.query(KnowledgeEntry).filter(KnowledgeEntry.site_id.is_(None)).count()
        if existing == 0:
            for item in CORE_KNOWLEDGE_BASE:
                s.add(KnowledgeEntry(site_id=None, question=item["question"], answer=item["answer"]))
            s.commit()


def rebuild_search_index(site_id):
    """Rebuilds the TF-IDF search index for one site from DB (core + site uploads)."""
    with dbm.get_db() as s:
        core_rows = s.query(KnowledgeEntry).filter(KnowledgeEntry.site_id.is_(None)).all()
        site_rows = s.query(KnowledgeEntry).filter(KnowledgeEntry.site_id == site_id).all()
        entries = [{"question": r.question, "answer": r.answer} for r in (core_rows + site_rows)]

    _kb_entries[site_id] = entries
    vectorizer = TfidfVectorizer(stop_words="english")
    _vectorizers[site_id] = vectorizer
    _kb_vectors[site_id] = vectorizer.fit_transform(
        [item["question"] + " " + item["answer"] for item in entries]
    )


def rebuild_all_search_indexes():
    for site_id in get_sites():
        rebuild_search_index(site_id)


def get_faq_answer(user_question, site_id, threshold=0.15):
    if site_id not in get_sites():
        site_id = "salavai"
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
# App startup: initialize DB, seed sites + core KB, build search indexes
# ---------------------------------------------------------------------------
dbm.init_db()
ensure_default_sites_seeded()
reload_sites_cache()
ensure_core_kb_seeded()
rebuild_all_search_indexes()


def get_notify_emails():
    with dbm.get_db() as s:
        return [row.email for row in s.query(NotifyEmail).all()]


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
    live on any registered site (POS, E-commerce, Franchise, Main site, or
    any new tenant added via /admin/sites)."""
    cfg = get_sites().get(site_id)
    if not cfg:
        return "Unknown site_id. Try one of: " + ", ".join(get_sites().keys()), 404
    return render_template("mock_site.html", site_id=site_id, cfg=cfg)


@app.route("/api/site-config")
def site_config():
    site_id = request.args.get("site_id", "salavai")
    cfg = get_site(site_id)
    return jsonify(cfg)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id")
    site_id = data.get("site_id", "salavai")
    site_cfg = get_site(site_id)

    with dbm.get_db() as s:
        sess = s.get(ChatSession, session_id) if session_id else None
        if not sess:
            session_id = uuid.uuid4().hex[:8]
            sess = ChatSession(id=session_id, site_id=site_id, mode="bot", state={})
            s.add(sess)
            s.commit()

        s.add(ChatMessage(session_id=session_id, sender="user", text=message, time=time.time()))
        s.commit()

        live_triggers = ["human", "agent", "support", "person", "representative", "talk to human", "live chat"]
        is_live_request = any(trig in message.lower() for trig in live_triggers)

        # 1. User is currently in LIVE agent mode
        if sess.mode == "live":
            msg_id = notify_visitor_reply(site_cfg["name"], session_id, message)
            if msg_id:
                s.merge(TelegramMsgMap(telegram_msg_id=msg_id, session_id=session_id))
                s.commit()
            return jsonify({
                "reply": None,
                "session_id": session_id,
                "mode": "live",
                "status": "waiting_for_agent"
            })

        # 2. User just requested to switch to LIVE agent
        if is_live_request:
            sess.mode = "live"
            s.commit()
            msg_id = notify_live_request(site_cfg["name"], session_id, message)
            if msg_id:
                s.merge(TelegramMsgMap(telegram_msg_id=msg_id, session_id=session_id))
                sess.telegram_msg_id = msg_id
                s.commit()

            s.add(LiveChatLead(
                site_id=site_id, site_name=site_cfg["name"], session_id=session_id,
                message=message, time=time.time()
            ))
            s.commit()

            return jsonify({
                "reply": "Connecting you with our support team... An agent has been notified and will reply here in just a moment!",
                "session_id": session_id,
                "mode": "live"
            })

        # 3. Franchise Lead Flow
        session_state = sess.state or {}
        if session_state.get("mode") == "franchise":
            return handle_franchise_flow(message, sess, session_id, s)

        franchise_triggers = ["franchise", "partner", "business opportunity", "start my own", "own laundry business"]
        if any(trigger in message.lower() for trigger in franchise_triggers):
            session_state["mode"] = "franchise"
            session_state["step"] = "name"
            session_state["data"] = {}
            sess.state = session_state
            s.commit()
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
            reply = ("I'm not fully sure about that yet! Would you like to chat with a live agent? "
                     "Just tap '💬 Chat with Human' below or let me know!")

        s.add(ChatMessage(session_id=session_id, sender="bot", text=reply, time=time.time()))
        s.commit()
        return jsonify({
            "reply": reply,
            "session_id": session_id,
            "mode": sess.mode
        })


def handle_franchise_flow(message, sess, session_id, s):
    session_state = sess.state or {}
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

        site_id = sess.site_id or "salavai"
        site_name = get_site(site_id).get("name", site_id)
        s.add(FranchiseLead(
            site_id=site_id, site_name=site_name,
            name=data.get("name"), city=data.get("city"),
            experience=data.get("experience"), budget=data.get("budget"),
            time=time.time()
        ))
        send_lead_notification(data, get_notify_emails())

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
    sess.state = session_state
    s.add(ChatMessage(session_id=session_id, sender="bot", text=reply, time=time.time()))
    s.commit()
    return jsonify({"reply": reply, "session_id": session_id, "mode": "bot"})


@app.route("/api/chat/poll")
def poll_messages():
    """Poll endpoint for visitor widget to receive agent replies in live mode."""
    session_id = request.args.get("session_id")
    with dbm.get_db() as s:
        sess = s.get(ChatSession, session_id) if session_id else None
        if not sess:
            return jsonify({"new_messages": []})

        pending = (
            s.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id, ChatMessage.delivered.is_(False))
            .filter(ChatMessage.sender.in_(["agent", "bot"]))
            .order_by(ChatMessage.time.asc())
            .all()
        )
        new_messages = [{"sender": m.sender, "text": m.text, "time": m.time} for m in pending]
        for m in pending:
            m.delivered = True
        s.commit()

        return jsonify({
            "new_messages": new_messages,
            "mode": sess.mode
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

    with dbm.get_db() as s:

        def get_active_live():
            return s.query(ChatSession).filter(ChatSession.mode == "live").all()

        def site_name_for(sess):
            return get_site(sess.site_id).get("name", sess.site_id)

        # -------------------------------------------------------------
        # /claim command
        # -------------------------------------------------------------
        claim_match = re.match(r"^/claim\b\s*#?([a-f0-9]{8})?\s*$", text, re.IGNORECASE)
        if claim_match:
            claim_session_id = claim_match.group(1).lower() if claim_match.group(1) else None

            if not claim_session_id and reply_to:
                m = s.get(TelegramMsgMap, reply_to.get("message_id"))
                claim_session_id = m.session_id if m else None

            if not claim_session_id:
                active_live = get_active_live()
                if len(active_live) == 1:
                    claim_session_id = active_live[0].id
                elif len(active_live) > 1:
                    listing = "\n".join(f"• <code>{sess.id}</code> — {site_name_for(sess)}" for sess in active_live)
                    send_telegram_message(
                        "⚠️ Multiple live chats are active. Specify which to claim, e.g.:\n"
                        "<code>/claim #a1b2c3d4</code>\n\n"
                        f"Active sessions:\n{listing}"
                    )
                    return jsonify({"ok": True})

            sess = s.get(ChatSession, claim_session_id) if claim_session_id else None
            if sess:
                previous = sess.claimed_by
                sess.claimed_by = agent_name
                s.commit()
                site_name = site_name_for(sess)
                if previous and previous != agent_name:
                    send_telegram_message(f"🔁 <b>{agent_name}</b> has taken over session <code>{claim_session_id}</code> ({site_name}) from {previous}.")
                else:
                    send_telegram_message(f"✋ <b>{agent_name}</b> is now handling session <code>{claim_session_id}</code> ({site_name}).")
            else:
                send_telegram_message("⚠️ Could not find that session to claim. Use <code>/claim #session_id</code>.")

            return jsonify({"ok": True})

        # -------------------------------------------------------------
        # /close command
        # -------------------------------------------------------------
        close_match = re.match(r"^/close\b\s*#?([a-f0-9]{8})?\s*$", text, re.IGNORECASE)
        if close_match:
            close_session_id = close_match.group(1).lower() if close_match.group(1) else None

            if not close_session_id and reply_to:
                m = s.get(TelegramMsgMap, reply_to.get("message_id"))
                close_session_id = m.session_id if m else None

            if not close_session_id:
                active_live = get_active_live()
                if len(active_live) == 1:
                    close_session_id = active_live[0].id
                elif len(active_live) > 1:
                    listing = "\n".join(f"• <code>{sess.id}</code> — {site_name_for(sess)}" for sess in active_live)
                    send_telegram_message(
                        "⚠️ Multiple live chats are active. Specify which to close, e.g.:\n"
                        "<code>/close #a1b2c3d4</code>\n\n"
                        f"Active sessions:\n{listing}"
                    )
                    return jsonify({"ok": True})

            sess = s.get(ChatSession, close_session_id) if close_session_id else None
            if sess:
                sess.mode = "bot"
                sess.claimed_by = None
                site_name = site_name_for(sess)
                s.add(ChatMessage(
                    session_id=close_session_id, sender="bot",
                    text="Thanks for chatting with our support team! I'm back to help with any other questions. 😊",
                    time=time.time()
                ))
                s.commit()
                send_telegram_message(f"🔒 Closed live chat for <b>{site_name}</b> (session <code>{close_session_id}</code>). Bot has resumed answering FAQs.")
            else:
                send_telegram_message("⚠️ Could not find that session to close. Use <code>/close #session_id</code>.")

            return jsonify({"ok": True})

        # -------------------------------------------------------------
        # Normal reply routing
        # -------------------------------------------------------------
        target_session_id = None

        if reply_to:
            m = s.get(TelegramMsgMap, reply_to.get("message_id"))
            target_session_id = m.session_id if m else None

        match = re.match(r"^#?([a-f0-9]{8})\s+(.*)$", text, re.IGNORECASE)
        if match:
            target_session_id = match.group(1).lower()
            text = match.group(2)

        if not target_session_id:
            active_live = get_active_live()
            if len(active_live) == 1:
                target_session_id = active_live[0].id
            elif len(active_live) > 1:
                listing = "\n".join(f"• <code>{sess.id}</code> — {site_name_for(sess)}" for sess in active_live)
                send_telegram_message(
                    "⚠️ Multiple live chats are active right now. I can't guess which one you meant.\n\n"
                    "Please <b>swipe-reply</b> to the original alert, or prefix your message with the session id, e.g.:\n"
                    "<code>#8ec1815f Hello!</code>\n\n"
                    f"Active sessions:\n{listing}"
                )
                return jsonify({"ok": True})

        sess = s.get(ChatSession, target_session_id) if target_session_id else None
        if sess:
            site_name = site_name_for(sess)
            claimed_by = sess.claimed_by
            if claimed_by and claimed_by != agent_name:
                send_telegram_message(
                    f"⚠️ Session <code>{target_session_id}</code> ({site_name}) is already being handled by "
                    f"<b>{claimed_by}</b>. Your message was NOT sent to avoid double-replying.\n\n"
                    f"If you need to take over, reply with <code>/claim #{target_session_id}</code> first."
                )
                return jsonify({"ok": True})

            if not claimed_by:
                sess.claimed_by = agent_name

            s.add(ChatMessage(session_id=target_session_id, sender="agent", text=text, time=time.time()))
            s.commit()
            send_telegram_message(f"✅ Delivered to <b>{site_name}</b> (session <code>{target_session_id}</code>) — handled by <b>{sess.claimed_by}</b>: \"{text}\"")
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
    with dbm.get_db() as s:
        core_count = s.query(KnowledgeEntry).filter(KnowledgeEntry.site_id.is_(None)).count()
        kb_counts = {}
        for site_id in get_sites():
            site_count = s.query(KnowledgeEntry).filter(KnowledgeEntry.site_id == site_id).count()
            kb_counts[site_id] = core_count + site_count

        # -----------------------------------------------------------------
        # Uploaded document batches, per site -- lets the agent see what
        # was uploaded and remove a specific document's knowledge later
        # without affecting the shared core FAQs.
        # -----------------------------------------------------------------
        uploaded_rows = (
            s.query(KnowledgeEntry)
            .filter(KnowledgeEntry.batch_id.isnot(None))
            .order_by(KnowledgeEntry.uploaded_at.desc())
            .all()
        )
        batches_by_site = {}
        seen_batches = set()
        for row in uploaded_rows:
            key = (row.site_id, row.batch_id)
            if key in seen_batches:
                batches_by_site[row.site_id][-1]["chunk_count"] += 1
                continue
            seen_batches.add(key)
            batches_by_site.setdefault(row.site_id, []).append({
                "batch_id": row.batch_id,
                "filename": row.filename,
                "uploaded_at": row.uploaded_at,
                "uploaded_at_str": datetime.fromtimestamp(row.uploaded_at).strftime("%d %b %Y, %I:%M %p") if row.uploaded_at else "",
                "chunk_count": 1,
            })

        emails = [row.email for row in s.query(NotifyEmail).all()]

        # -----------------------------------------------------------------
        # Lead reporting: combine franchise leads + live-chat ("Talk to
        # Human") leads into one timeline, filterable by range and broken
        # down per site -- supports per-lead billing conversations.
        # -----------------------------------------------------------------
        range_key = request.args.get("range", "all")
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

        franchise_rows = s.query(FranchiseLead).all()
        live_chat_rows = s.query(LiveChatLead).all()

        all_leads = [
            {"type": "franchise", "site_id": r.site_id, "site_name": r.site_name,
             "name": r.name, "city": r.city, "experience": r.experience,
             "budget": r.budget, "time": r.time}
            for r in franchise_rows
        ] + [
            {"type": "live_chat", "site_id": r.site_id, "site_name": r.site_name,
             "session_id": r.session_id, "message": r.message, "time": r.time}
            for r in live_chat_rows
        ]

        def in_range(lead):
            t = lead.get("time", 0)
            if range_start and t < range_start:
                return False
            if range_end and t > range_end:
                return False
            return True

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
            sites=get_sites(),
            kb_counts=kb_counts,
            core_count=core_count,
            batches_by_site=batches_by_site,
            emails=emails,
            leads=[l for l in all_leads if l["type"] == "franchise"],
            filtered_leads=filtered_leads,
            total_leads=len(filtered_leads),
            site_breakdown=site_breakdown,
            range_key=range_key,
            start_date=start_date,
            end_date=end_date,
            new_site=request.args.get("new_site"),
            base_url=request.url_root.rstrip("/"),
            protected_site_ids=PROTECTED_SITE_IDS,
        )


@app.route("/admin/upload", methods=["POST"])
@require_admin_login
def admin_upload():
    file = request.files.get("document")
    site_id = request.form.get("site_id", "salavai")
    if site_id not in get_sites():
        site_id = "salavai"

    if not file or file.filename == "":
        return redirect(url_for("admin_dashboard"))

    filename = file.filename.lower()
    file_bytes = file.read()
    try:
        text = extract_text_from_file(file_bytes, filename)
        new_entries = chunk_text_into_qa_pairs(text)
        batch_id = uuid.uuid4().hex[:12]
        with dbm.get_db() as s:
            for entry in new_entries:
                s.add(KnowledgeEntry(
                    site_id=site_id, question=entry["question"], answer=entry["answer"],
                    batch_id=batch_id, filename=file.filename, uploaded_at=time.time()
                ))
            s.commit()
        rebuild_search_index(site_id)
    except Exception as e:
        print(f"[admin_upload] Failed to process document: {e}")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/kb/delete", methods=["POST"])
@require_admin_login
def admin_delete_kb_batch():
    """Remove a previously uploaded document's knowledge base entries
    (identified by batch_id) from one site. Core/shared FAQs are never
    affected -- only ever touches site-specific uploaded entries."""
    site_id = request.form.get("site_id", "")
    batch_id = request.form.get("batch_id", "")

    if site_id and batch_id:
        with dbm.get_db() as s:
            s.query(KnowledgeEntry).filter(
                KnowledgeEntry.site_id == site_id,
                KnowledgeEntry.batch_id == batch_id
            ).delete()
            s.commit()
        if site_id in get_sites():
            rebuild_search_index(site_id)

    return redirect(url_for("admin_dashboard", tab="kb"))


@app.route("/admin/emails", methods=["POST"])
@require_admin_login
def admin_update_emails():
    raw = request.form.get("emails", "")
    new_emails = [e.strip() for e in raw.split(",") if e.strip()]
    with dbm.get_db() as s:
        s.query(NotifyEmail).delete()
        for email in new_emails:
            s.add(NotifyEmail(email=email))
        s.commit()
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/sites/add", methods=["POST"])
@require_admin_login
def admin_add_site():
    """Add a brand-new tenant/website -- no code change or redeploy needed.
    Generates a site_id from the name, stores config in the DB, builds a
    fresh (core-only) search index for it, and shows the embed snippet."""
    name = (request.form.get("name") or "").strip()
    primary_color = request.form.get("primary_color") or "#a41e22"
    dark_color = request.form.get("dark_color") or "#14324f"
    greeting = (request.form.get("greeting") or "👋 Hello! How can we assist you today?").strip()
    quick_replies_raw = request.form.get("quick_replies") or ""
    quick_replies = [q.strip() for q in quick_replies_raw.split(",") if q.strip()]
    if not quick_replies or not any("human" in q.lower() for q in quick_replies):
        quick_replies.append("💬 Chat with Human")

    if not name:
        return redirect(url_for("admin_dashboard"))

    base_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "site"
    site_id = base_slug
    with dbm.get_db() as s:
        suffix = 1
        while s.get(Site, site_id):
            suffix += 1
            site_id = f"{base_slug}-{suffix}"

        s.add(Site(
            site_id=site_id, name=name, primary_color=primary_color,
            dark_color=dark_color, greeting=greeting, quick_replies=quick_replies
        ))
        s.commit()

    reload_sites_cache()
    rebuild_search_index(site_id)

    return redirect(url_for("admin_dashboard", new_site=site_id))


# NOTE: Previously the 4 default Salavai sites were hard-blocked from
# deletion. That was overly restrictive for someone managing all tenants
# themselves -- any site can now be removed, with a confirm() prompt in the
# UI as the safety net. The only hard rule left: never delete the very last
# remaining site, so the system always has at least one tenant configured.
PROTECTED_SITE_IDS = set()


@app.route("/admin/sites/delete", methods=["POST"])
@require_admin_login
def admin_delete_site():
    """Remove a tenant/website. Also cleans up its site-specific knowledge
    base entries (core KB is untouched). Refuses to delete the last
    remaining site so the system is never left with zero tenants."""
    site_id = request.form.get("site_id", "")

    with dbm.get_db() as s:
        total_sites = s.query(Site).count()
        site = s.get(Site, site_id)
        if site and total_sites > 1:
            s.query(KnowledgeEntry).filter(KnowledgeEntry.site_id == site_id).delete()
            s.delete(site)
            s.commit()

    _vectorizers.pop(site_id, None)
    _kb_vectors.pop(site_id, None)
    _kb_entries.pop(site_id, None)
    reload_sites_cache()

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5008))
    app.run(debug=True, host="0.0.0.0", port=port)
