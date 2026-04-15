"""
emailer.py — Format and send the daily AyaLocums physician job report.

Reads jobs_enriched.json, builds HTML email, sends via Gmail or SendGrid.

Each row shows facility + decision-maker contact info (name, title, email,
phone) sourced from the identifier.research_contact step. All AI-generated
fields pass through html.escape() before being embedded in the email.

Usage:
  python emailer.py              — send success email
  python emailer.py --failure    — send failure notification (reads FAILURE_REASON env var)
"""
import html
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

JOBS_ENRICHED_FILE = Path(__file__).parent / "jobs_enriched.json"

CONFIDENCE_COLORS = {
    "high":   "#d4edda",
    "medium": "#fff3cd",
    "low":    "#f8d7da",
    "none":   "#e2e3e5",
}

_TD = 'style="padding: 8px; border: 1px solid #ddd; vertical-align: top;"'
_TH = 'style="padding: 8px; border: 1px solid #ddd; text-align: left; background-color: #f2f2f2;"'


def _esc(value) -> str:
    """HTML-escape any value coerced to string. Empty/None -> empty string."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _build_contact_cell(contact: dict) -> str:
    """
    Render the contact cell: email as mailto link, phone on a new line.
    Shows an em dash if there is no valid contact info.
    """
    email = (contact.get("contact_email") or "").strip()
    phone = _esc(contact.get("contact_phone") or "")

    # Permissive email check — if it looks like an email, render as mailto
    if "@" in email and "." in email.split("@")[-1] and " " not in email:
        email_html = f'<a href="mailto:{_esc(email)}">{_esc(email)}</a>'
    elif email:
        email_html = "&mdash;"
    else:
        email_html = "&mdash;"

    parts = [email_html]
    if phone:
        parts.append(phone)
    return "<br>".join(parts)


def _build_decision_maker_cell(contact: dict) -> str:
    """Render the decision-maker cell: name + title on two lines, or em dash."""
    name = _esc(contact.get("contact_name") or "")
    title = _esc(contact.get("contact_title") or "")
    if not name and not title:
        return "&mdash;"
    if name and title:
        return f"<strong>{name}</strong><br><small>{title}</small>"
    return f"<strong>{name or title}</strong>"


def build_table(jobs: list) -> str:
    if not jobs:
        return "<p>No new job postings today.</p>"

    rows = []
    for job in jobs:
        ident = job.get("identification", {})
        contact = job.get("contact", {})
        confidence = ident.get("confidence", "none")
        bg = CONFIDENCE_COLORS.get(confidence, CONFIDENCE_COLORS["none"])

        facility_name = _esc(ident.get("facility_name", "Unable to identify"))
        facility_cell = f"<strong>{facility_name}</strong>"
        if confidence not in ("none", ""):
            facility_cell += f" ({_esc(confidence).capitalize()})"
        alt = ident.get("alternative_facility")
        if alt:
            facility_cell += f"<br><small><em>Also possible: {_esc(alt)}</em></small>"

        location = f"{_esc(job.get('city', ''))}, {_esc(job.get('state', ''))}"
        pay = _esc(job.get("pay_display") or "")
        shift = _esc(job.get("shift") or "")
        url = _esc(job.get("url", "#"))
        specialty = _esc(job.get("specialty", ""))

        decision_maker_cell = _build_decision_maker_cell(contact)
        contact_cell = _build_contact_cell(contact)

        rows.append(
            f'<tr style="background-color: {bg};">'
            f"<td {_TD}>{specialty}</td>"
            f"<td {_TD}>{location}</td>"
            f"<td {_TD}>{pay}</td>"
            f"<td {_TD}>{shift}</td>"
            f"<td {_TD}>{facility_cell}</td>"
            f"<td {_TD}>{decision_maker_cell}</td>"
            f"<td {_TD}>{contact_cell}</td>"
            f'<td {_TD}><a href="{url}">View</a></td>'
            "</tr>"
        )

    header = (
        '<table style="border-collapse: collapse; width: 100%; '
        'font-family: Arial, sans-serif; font-size: 13px;">'
        "<thead><tr>"
        f"<th {_TH}>Specialty</th>"
        f"<th {_TH}>Location</th>"
        f"<th {_TH}>Pay Rate</th>"
        f"<th {_TH}>Shift</th>"
        f"<th {_TH}>Likely Facility</th>"
        f"<th {_TH}>Decision-Maker</th>"
        f"<th {_TH}>Contact</th>"
        f"<th {_TH}>Link</th>"
        "</tr></thead><tbody>"
    )
    return header + "".join(rows) + "</tbody></table>"


def build_success_email(jobs: list) -> tuple:
    date = datetime.today().strftime("%Y-%m-%d")
    count = len(jobs)
    if count == 0:
        subject = f"AyaLocums New Jobs \u2014 {date} (No new postings)"
        body = f'<p style="font-family:Arial,sans-serif;">No new job postings found on {date}.</p>'
    else:
        noun = "posting" if count == 1 else "postings"
        subject = f"AyaLocums New Jobs \u2014 {date} ({count} new {noun})"
        body = (
            '<div style="font-family:Arial,sans-serif;">'
            f"<h2>AyaLocums New Physician &amp; CRNA Jobs \u2014 {date}</h2>"
            f"<p>{count} new {noun} since last run</p>"
            f"{build_table(jobs)}"
            "</div>"
        )
    return subject, body


def build_failure_email(reason: str) -> tuple:
    date = datetime.today().strftime("%Y-%m-%d")
    subject = f"\u26a0\ufe0f AyaLocums Scraper Failed \u2014 {date}"
    body = (
        f"AyaLocums scraper failed on {date}.\n\n"
        f"Error: {reason}\n\n"
        "Check GitHub Actions logs for full details.\n"
        "The scraper will retry on the next scheduled run."
    )
    return subject, body


def send_email(subject: str, body: str, is_html: bool = True) -> None:
    transport = os.environ.get("EMAIL_TRANSPORT", "gmail").lower()
    recipient = os.environ["RECIPIENT_EMAIL"]

    if transport == "sendgrid":
        _send_sendgrid(subject, body, recipient, is_html)
    else:
        _send_gmail(subject, body, recipient, is_html)


def _send_gmail(subject: str, body: str, recipient: str, is_html: bool) -> None:
    sender = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(body, "html" if is_html else "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())


def _send_sendgrid(subject: str, body: str, recipient: str, is_html: bool) -> None:
    import sendgrid
    from sendgrid.helpers.mail import Mail

    sender = os.environ.get("GMAIL_USER", "noreply@connecthealthstaff.com")
    message = Mail(
        from_email=sender,
        to_emails=recipient,
        subject=subject,
        html_content=body if is_html else None,
        plain_text_content=None if is_html else body,
    )
    sg = sendgrid.SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
    sg.send(message)


def main() -> None:
    failure_mode = "--failure" in sys.argv

    if failure_mode:
        reason = os.environ.get("FAILURE_REASON", "Unknown error \u2014 check GitHub Actions logs")
        subject, body = build_failure_email(reason)
        send_email(subject, body, is_html=False)
        print(f"Failure notification sent: {subject}")
        return

    with open(JOBS_ENRICHED_FILE) as f:
        jobs = json.load(f)

    subject, body = build_success_email(jobs)
    send_email(subject, body, is_html=True)
    print(f"Email sent: {subject}")


if __name__ == "__main__":
    main()
