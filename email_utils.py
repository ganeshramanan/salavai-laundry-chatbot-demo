"""
Email notification helper using Resend (https://resend.com) -- an HTTPS-based
transactional email API, not raw SMTP. This avoids Render's outbound SMTP
port blocking issues entirely, since it just makes a normal HTTPS API call
(same as calling any other REST API), which is never blocked.

Setup required (one-time):
1. Sign up at https://resend.com (free tier: 100 emails/day, 3,000/month)
2. Get an API key from the Resend dashboard
3. Verify a sending domain (or use Resend's default onboarding domain for testing)
4. Set these environment variables on Render:
   - RESEND_API_KEY   = your Resend API key
   - RESEND_FROM      = the verified "from" address, e.g. "Salavai Bot <notifications@yourdomain.com>"
   - NOTIFY_EMAIL     = comma-separated recipient email addresses
"""

import os
import json
import urllib.request
import urllib.error
import threading


def send_lead_notification(lead_data, notify_emails=None):
    """
    Sends an email notification when a new franchise lead is captured.
    Runs in a background thread so it never blocks or crashes the request
    that captures the lead.
    """
    thread = threading.Thread(target=_send_email_sync, args=(lead_data, notify_emails), daemon=True)
    thread.start()


def _send_email_sync(lead_data, notify_emails=None):
    api_key = os.environ.get("RESEND_API_KEY")
    from_address = os.environ.get("RESEND_FROM", "onboarding@resend.dev")

    if notify_emails is None:
        notify_email_raw = os.environ.get("NOTIFY_EMAIL", "")
        notify_emails = [e.strip() for e in notify_email_raw.split(",") if e.strip()]

    if not api_key or not notify_emails:
        print("[email_utils] Resend not configured -- skipping email notification. "
              "Set RESEND_API_KEY env var and add recipient emails to enable.")
        return False

    subject = f"🧺 New Franchise Lead: {lead_data.get('name', 'Unknown')}"
    html_body = f"""
    <h2>New Franchise/Business Partner Lead</h2>
    <p>A new lead came in via the chatbot:</p>
    <ul>
      <li><strong>Name:</strong> {lead_data.get('name', '-')}</li>
      <li><strong>City/Area:</strong> {lead_data.get('city', '-')}</li>
      <li><strong>Prior Experience:</strong> {lead_data.get('experience', '-')}</li>
      <li><strong>Budget:</strong> {lead_data.get('budget', '-')}</li>
    </ul>
    <p>Follow up with them soon!</p>
    """

    payload = {
        "from": from_address,
        "to": notify_emails,
        "subject": subject,
        "html": html_body,
    }

    try:
        req = urllib.request.Request(
            url="https://api.resend.com/emails",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = response.read().decode("utf-8")
            print(f"[email_utils] Lead notification sent via Resend to {notify_emails}: {result}")
            return True
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[email_utils] Resend API error ({e.code}): {error_body}")
        return False
    except Exception as e:
        print(f"[email_utils] Failed to send email notification via Resend: {e}")
        return False
