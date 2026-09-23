"""
Free email notification helper using Gmail SMTP.

Setup required (one-time, by whoever owns the notification inbox):
1. Use a Gmail account (can be a new one dedicated to this, e.g. salavaibot@gmail.com)
2. Enable 2-Step Verification on that Google account
3. Generate an "App Password": https://myaccount.google.com/apppasswords
   (This is a 16-character password specifically for apps like this -- NOT your normal Gmail password)
4. Set these as environment variables on Render (never hardcode them in code):
   - SMTP_EMAIL      = the Gmail address sending notifications
   - SMTP_APP_PASSWORD = the 16-character app password from step 3
   - NOTIFY_EMAIL    = the email address that should RECEIVE lead notifications
     (can be the same as SMTP_EMAIL, or the business owner's actual inbox)

This uses Gmail's free SMTP relay -- no third-party service, no signup, no cost,
well within Gmail's free sending limits (500 emails/day) for this use case.
"""

import smtplib
import os
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_lead_notification(lead_data, notify_emails=None):
    """
    Sends an email notification when a new franchise lead is captured.
    Runs in a background thread so it can NEVER block or crash the request
    that captures the lead -- even if SMTP hangs or times out, the lead is
    already saved and the user already has their reply before this even runs.
    """
    thread = threading.Thread(target=_send_email_sync, args=(lead_data, notify_emails), daemon=True)
    thread.start()


def _send_email_sync(lead_data, notify_emails=None):
    """
    Sends an email notification when a new franchise lead is captured.
    Fails silently (logs to console) if SMTP env vars aren't configured --
    so the chatbot itself never breaks even if email isn't set up yet.

    notify_emails: list of recipient email addresses. If not provided, falls
    back to the NOTIFY_EMAIL environment variable (comma-separated).
    """
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_APP_PASSWORD")

    if notify_emails is None:
        notify_email_raw = os.environ.get("NOTIFY_EMAIL", smtp_email or "")
        notify_emails = [e.strip() for e in notify_email_raw.split(",") if e.strip()]

    if not smtp_email or not smtp_password or not notify_emails:
        print("[email_utils] SMTP not configured -- skipping email notification. "
              "Set SMTP_EMAIL, SMTP_APP_PASSWORD env vars and add recipient emails to enable.")
        return False

    subject = f"🧺 New Franchise Lead: {lead_data.get('name', 'Unknown')}"
    body = f"""A new franchise/business partner lead came in via the chatbot:

Name: {lead_data.get('name', '-')}
City/Area: {lead_data.get('city', '-')}
Prior Experience: {lead_data.get('experience', '-')}
Budget: {lead_data.get('budget', '-')}

Follow up with them soon!
"""

    msg = MIMEMultipart()
    msg["From"] = smtp_email
    msg["To"] = ", ".join(notify_emails)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=8) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, notify_emails, msg.as_string())
        print(f"[email_utils] Lead notification sent to {notify_emails}")
        return True
    except Exception as e:
        print(f"[email_utils] Failed to send email notification: {e}")
        return False
