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

## Setting up email notifications (via Resend)
This project sends an email automatically whenever a franchise lead completes the guided chat flow, using **[Resend](https://resend.com)** — an HTTPS-based transactional email API. We switched from Gmail SMTP to Resend because Render's free tier blocks outbound SMTP ports (587/25), causing hung connections and worker crashes. Resend sends via a normal HTTPS API call, which is never blocked.

### One-time setup
1. Sign up at [resend.com](https://resend.com) (free tier: 100 emails/day, 3,000/month — plenty for lead notifications)
2. Get an API key from the Resend dashboard
3. Verify a sending domain in Resend (or use their default `onboarding@resend.dev` sender for quick testing, no domain verification needed)
4. On Render, add these environment variables to the service:
   - `RESEND_API_KEY` — your Resend API key
   - `RESEND_FROM` — the verified "from" address, e.g. `"Salavai Bot <notifications@yourdomain.com>"` (or leave unset to use the default test sender)
   - `NOTIFY_EMAIL` — comma-separated recipient email addresses that should receive lead notifications
5. Redeploy — new leads will now trigger an automatic email

Email sending runs in a background thread with a timeout, so even if it fails or is slow, it can never block or crash the lead-capture flow. If these environment variables aren't set, the chatbot still works normally — it just skips sending the email (fails silently, logs a message instead of crashing).

## Admin Console

A password-protected admin page lets the business owner manage the bot without touching any code:

**URL**: `/admin` (redirects to `/admin/login` if not logged in)

**Default password**: `salavai123` — **change this immediately** by setting the `ADMIN_PASSWORD` environment variable on Render.

### What the admin console can do
1. **Upload a Word (.docx) or PDF document** to add new knowledge to the bot — it automatically extracts the text, splits it into chunks, and makes it searchable immediately (no redeploy needed)
2. **Manage notification emails** — add/remove which email addresses receive franchise lead alerts, directly from the browser (no need to edit Render environment variables each time)
3. **View captured leads** — a readable table of every franchise inquiry captured so far

## Business case for the owner
- **Faster answers** for the most common question (pricing) without waiting for WhatsApp replies
- **Better franchise lead quality** — pre-qualified with city, experience, and budget before a human spends time following up
- **24/7 availability** — answers questions even outside store hours
- **No per-message API cost** — since retrieval is local, this can scale to any traffic volume for free

---
Built as a business proof-of-concept demo, using the same free/local AI techniques taught in Gritbi's project catalog.
