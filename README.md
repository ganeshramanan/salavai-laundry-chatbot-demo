# Salavai Laundry — Chatbot Demo (Proof of Concept)

A lightweight, embeddable-style chat widget demo built for **The Salavai Laundry** — showing how an AI assistant could handle common customer questions and qualify franchise/business partner leads, without any paid LLM API costs.

## Why this exists
The Salavai Laundry's website has a comprehensive FAQ section, a detailed pricing catalog (109+ garment categories), and an active franchise/partner recruitment program — but customers/prospects still need to scroll, search, or wait for a WhatsApp reply to get answers. This demo shows how a chat widget could reduce that friction.

## Two capabilities in this demo

### 1. FAQ / Pricing Assistant
Answers common questions (KG Laundry pricing, pickup process, turnaround time, silk care, etc.) using **TF-IDF + cosine similarity retrieval** — the same free, local technique used in Gritbi's RAG chatbot project. No OpenAI/Anthropic API calls, no per-message cost.

### 2. Franchise Lead Qualifier
Detects when a visitor expresses interest in becoming a business partner (keywords like "franchise," "partner," "start my own business") and switches into a **guided, rule-based conversation flow** — collecting name, city, experience, and budget — then logs it as a qualified lead. This pre-screens serious inquiries before a human follows up, which is valuable given their expansion program targets a broad, often non-technical audience (dhobi families, homemakers, differently-abled individuals).

## Tech Stack
- **Backend**: Python, Flask
- **Retrieval**: scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- **Frontend**: HTML, CSS, vanilla JS — floating chat widget styled to match Salavai Laundry's brand colors (deep red/navy)
- **Deployment**: Render (gunicorn + Procfile)

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5008` — click the chat bubble in the bottom-right corner.

## How this would become a real embeddable widget
Right now this demo is a full mock page with the widget included. To make it a true drop-in widget for the real site:
1. Extract the chat widget HTML/CSS/JS into a single self-contained `<script>` snippet
2. Host that script + the Flask backend on a small server (e.g., Render, same as this demo)
3. The business owner adds one `<script src="...">` tag to their existing website — no code changes to their site needed
4. The widget calls the hosted `/api/chat` endpoint via `fetch()`, same as this demo

## Extending the knowledge base
The `KNOWLEDGE_BASE` list in `app.py` currently contains 8 sample FAQ entries based on the real site content. In a production version, this would be:
- Editable by the business owner (e.g., a simple admin form), or
- Periodically synced from their actual FAQ page

## Extending lead capture
Currently, franchise leads are stored in memory (`franchise_leads` list) and viewable at `/api/leads` for demo purposes. **Email notification is now implemented** (see below) — in a fuller production version, this should also:
- Save to a real database (not just in-memory)
- Possibly integrate with a CRM or Google Sheet for easy follow-up

## Setting up free email notifications
This project sends an email automatically whenever a franchise lead completes the guided chat flow, using **Gmail's free SMTP relay** — no third-party service, no signup, no cost.

### One-time setup
1. Use a Gmail account (a dedicated one like `salavaibot@gmail.com` is recommended, but any Gmail works)
2. Enable **2-Step Verification** on that Google account (required for App Passwords)
3. Generate an **App Password**: go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords), create one for "Mail"
4. On Render, add these environment variables to the service:
   - `SMTP_EMAIL` — the Gmail address sending notifications
   - `SMTP_APP_PASSWORD` — the 16-character App Password from step 3 (NOT the normal Gmail password)
   - `NOTIFY_EMAIL` — the email address that should receive lead notifications (can be the business owner's real inbox)
5. Redeploy — new leads will now trigger an automatic email

If these environment variables aren't set, the chatbot still works normally — it just skips sending the email (fails silently, logs a message instead of crashing).

## Business case for the owner
- **Faster answers** for the most common question (pricing) without waiting for WhatsApp replies
- **Better franchise lead quality** — pre-qualified with city, experience, and budget before a human spends time following up
- **24/7 availability** — answers questions even outside store hours
- **No per-message API cost** — since retrieval is local, this can scale to any traffic volume for free

---
Built as a business proof-of-concept demo, using the same free/local AI techniques taught in Gritbi's project catalog.
