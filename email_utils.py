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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_lead_notification(lead_data):
    """
    Sends an email notification when a new franchise lead is captured.
    Fails silently (logs to console) if SMTP env vars aren't configured --
    so the chatbot itself never breaks even if email isn't set up yet.
    """
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_APP_PASSWORD")
    notify_email = os.environ.get("NOTIFY_EMAIL", smtp_email)

    if not smtp_email or not smtp_password:
        print("[email_utils] SMTP not configured -- skipping email notification. "
              "Set SMTP_EMAIL and SMTP_APP_PASSWORD env vars to enable.")
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
    msg["To"] = notify_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, notify_email, msg.as_string())
        print(f"[email_utils] Lead notification sent to {notify_email}")
        return True
    except Exception as e:
        print(f"[email_utils] Failed to send email notification: {e}")
        return False
