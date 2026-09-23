"""
Laundry Business Chatbot Demo -- built as a proof-of-concept for The Salavai Laundry
(and reusable for similar service/franchise businesses).

Capabilities:
1. FAQ/Pricing Assistant -- answers common customer questions using free,
   local TF-IDF retrieval (no paid LLM API).
2. Franchise Lead Qualifier -- a guided, rule-based multi-step chat flow that
   captures and qualifies franchise/partner inquiries, with email notification.
3. Admin Console -- password-protected page where the business owner can:
   - Upload a Word/PDF document to retrain the bot's knowledge base anytime
   - Manage which email addresses receive lead notifications
   - View captured leads in a readable table

This is intentionally built the same way as Gritbi's RAG chatbot project --
proving the same free/local retrieval technique has real commercial value,
not just educational value.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import io
from email_utils import send_lead_notification
from document_utils import extract_text_from_file, chunk_text_into_qa_pairs

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "gritbi-laundry-demo-dev-key")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "salavai123")  # change via Render env var in production


@app.after_request
def add_cors_headers(response):
    """
    Allow this API to be called from any origin -- required so the embeddable
    widget.js can run on a completely different domain (e.g. thesalavailaundry.com)
    while calling this Flask backend hosted elsewhere (e.g. onrender.com).
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Knowledge base -- starts with sample FAQs, can be expanded/replaced via the
# admin console by uploading a Word/PDF document. Stored in memory, so it
# resets on server restart (see README for notes on persisting this properly).
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
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

vectorizer = None
KB_VECTORS = None


def rebuild_search_index():
    """(Re)fits the TF-IDF vectorizer over the current knowledge base. Call this
    whenever KNOWLEDGE_BASE changes (startup, or after an admin upload)."""
    global vectorizer, KB_VECTORS
    vectorizer = TfidfVectorizer(stop_words="english")
    KB_VECTORS = vectorizer.fit_transform(
        [item["question"] + " " + item["answer"] for item in KNOWLEDGE_BASE]
    )


rebuild_search_index()

# In-memory storage for franchise leads (would be a real database in production)
franchise_leads = []

# Notification emails -- seeded from env var, editable live via the admin console
notify_emails = [e.strip() for e in os.environ.get("NOTIFY_EMAIL", "").split(",") if e.strip()]


def get_faq_answer(user_question, threshold=0.15):
    question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vector, KB_VECTORS)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return None, best_score

    return KNOWLEDGE_BASE[best_idx]["answer"], best_score


# ---------------------------------------------------------------------------
# Public chatbot routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/widget.js")
def widget_js():
    response = send_from_directory("static", "widget.js", mimetype="application/javascript")
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


@app.route("/embed-test")
def embed_test():
    return render_template("embed_test.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    session_state = data.get("state", {})

    if session_state.get("mode") == "franchise":
        return handle_franchise_flow(message, session_state)

    franchise_triggers = ["franchise", "partner", "business opportunity", "start my own", "own laundry business"]
    if any(trigger in message.lower() for trigger in franchise_triggers):
        session_state["mode"] = "franchise"
        session_state["step"] = "name"
        session_state["data"] = {}
        return jsonify({
            "reply": "That's great! I'd love to help you explore becoming a Salavai Laundry business partner. "
                      "First, what's your full name?",
            "state": session_state
        })

    answer, score = get_faq_answer(message)
    if answer:
        reply = answer
    else:
        reply = ("I don't have a confident answer for that yet. You can WhatsApp us directly, "
                  "or ask me about pricing, pickup, turnaround time, or becoming a business partner!")

    return jsonify({"reply": reply, "state": session_state})


def handle_franchise_flow(message, session_state):
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

        franchise_leads.append(dict(data))
        send_lead_notification(data, notify_emails)

        reply = (f"Perfect, thank you {data.get('name', '')}! Here's a summary of what you shared:\n"
                  f"📍 City: {data.get('city')}\n"
                  f"💼 Experience: {data.get('experience')}\n"
                  f"💰 Budget: {data.get('budget')}\n\n"
                  f"Our partnership team will reach out to you directly with next steps. "
                  f"You can also WhatsApp us anytime for immediate questions!")
        session_state["mode"] = None

    else:
        reply = "Thanks again for your interest! Feel free to ask me anything else."
        session_state["mode"] = None

    session_state["data"] = data
    return jsonify({"reply": reply, "state": session_state})


@app.route("/api/leads")
def view_leads():
    return jsonify({"total_leads": len(franchise_leads), "leads": franchise_leads})


# ---------------------------------------------------------------------------
# Admin console -- password-protected
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
    return render_template(
        "admin_dashboard.html",
        kb_count=len(KNOWLEDGE_BASE),
        emails=notify_emails,
        leads=franchise_leads,
    )


@app.route("/admin/upload", methods=["POST"])
@require_admin_login
def admin_upload():
    file = request.files.get("document")
    if not file or file.filename == "":
        return redirect(url_for("admin_dashboard"))

    filename = file.filename.lower()
    file_bytes = file.read()

    try:
        text = extract_text_from_file(file_bytes, filename)
        new_entries = chunk_text_into_qa_pairs(text)

        global KNOWLEDGE_BASE
        KNOWLEDGE_BASE.extend(new_entries)
        rebuild_search_index()
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
    app.run(debug=True, host="0.0.0.0", port=5008)
