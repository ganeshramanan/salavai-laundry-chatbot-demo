"""
Laundry Business Chatbot Demo -- built as a proof-of-concept for The Salavai Laundry
(and reusable for similar service/franchise businesses).

Two capabilities:
1. FAQ/Pricing Assistant -- answers common customer questions using free,
   local TF-IDF retrieval (no paid LLM API).
2. Franchise Lead Qualifier -- a guided, rule-based multi-step chat flow that
   captures and qualifies franchise/partner inquiries.

This is intentionally built the same way as Gritbi's RAG chatbot project --
proving the same free/local retrieval technique has real commercial value,
not just educational value.
"""

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Knowledge base built from the business's actual site content (FAQs + pricing info).
# In a real deployment, this would be edited by the business owner or scraped from their site.
KNOWLEDGE_BASE = [
    {
        "question": "What is KG Laundry and how is it priced?",
        "answer": "KG Laundry is our everyday laundry service priced by weight, not per item. "
                   "It's best for daily wear, bedsheets, towels, and everyday loads -- just bag it, "
                   "weigh it, and hand it over. We wash, dry, and fold everything with care."
    },
    {
        "question": "How does doorstep pickup and delivery work?",
        "answer": "You can schedule a pickup directly from our website or WhatsApp. Our team collects "
                   "your laundry from your doorstep, processes it at our store, and delivers it back "
                   "to you -- no need to visit the store in person."
    },
    {
        "question": "How do you clean delicate silk sarees and suits?",
        "answer": "Silk sarees and delicate garments are handled under our Silk & Delicates program, "
                   "using specialized wet cleaning and precise temperature control to protect fine fabrics."
    },
    {
        "question": "What is the standard turnaround delivery time?",
        "answer": "Standard turnaround is typically 24-48 hours depending on the service type and load. "
                   "KG Laundry is usually fastest; delicate silk/specialty items may take a bit longer."
    },
    {
        "question": "Where are your store locations situated?",
        "answer": "You can search by your pincode or area name on our website to find your nearest active "
                   "store and its pickup coverage area."
    },
    {
        "question": "What services do you offer?",
        "answer": "We offer KG Laundry (priced by weight), Retail Laundry (per item), Silk & Delicates "
                   "(specialized care), and Pickup & Delivery across all service types."
    },
    {
        "question": "What equipment and technology do you use?",
        "answer": "We use modern commercial laundry machines, LG wet cleaning technology, and eco-friendly "
                   "cleaning chemicals -- the same standard trusted by hospitality and garment-care businesses."
    },
    {
        "question": "How can I contact support?",
        "answer": "You can reach us via WhatsApp/hotline, email, or the contact form on our website. "
                   "Store hours are typically Mon-Sat 8 AM-8 PM, Sun 9 AM-5 PM."
    },
]

vectorizer = TfidfVectorizer(stop_words="english")
KB_VECTORS = vectorizer.fit_transform([item["question"] + " " + item["answer"] for item in KNOWLEDGE_BASE])

# In-memory storage for franchise leads captured during the demo (would be a database/CRM in production)
franchise_leads = []

# Franchise qualifier flow -- a simple state machine, not AI-generated
FRANCHISE_STEPS = ["name", "city", "experience", "budget", "done"]


def get_faq_answer(user_question, threshold=0.15):
    question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vector, KB_VECTORS)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return None, best_score

    return KNOWLEDGE_BASE[best_idx]["answer"], best_score


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    session_state = data.get("state", {})

    # Franchise qualifier flow takes over once triggered
    if session_state.get("mode") == "franchise":
        return handle_franchise_flow(message, session_state)

    # Detect intent to start franchise flow
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

    # Otherwise, treat as an FAQ question
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

        # Save the qualified lead
        franchise_leads.append(dict(data))

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
    """Simple internal view of captured franchise leads (demo purposes)."""
    return jsonify({"total_leads": len(franchise_leads), "leads": franchise_leads})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5008)
