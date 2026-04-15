"""tests/test_emailer.py"""
import os
import pytest
from unittest.mock import MagicMock, patch


def make_enriched_job(**kwargs) -> dict:
    defaults = {
        "id": "3236627",
        "specialty": "Obstetrics/Gynecology",
        "city": "Christiansburg",
        "state": "VA",
        "pay_display": "$4,500\u2013$5,000/day",
        "shift": "3x12-Hour 07:00 - 07:00",
        "url": "https://www.ayalocums.com/job/locum-physician/3236627/",
        "identification": {
            "facility_name": "Carilion New River Valley Medical Center",
            "facility_type": "Community Hospital",
            "confidence": "high",
            "reasoning": "Only hospital in Christiansburg at that address",
            "alternative_facility": None,
        },
    }
    return {**defaults, **kwargs}


class TestBuildTable:
    def test_returns_no_postings_message_for_empty_list(self):
        import emailer
        html = emailer.build_table([])
        assert "No new job postings" in html

    def test_includes_specialty(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "Obstetrics/Gynecology" in html

    def test_includes_pay_display(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "$4,500" in html
        assert "$5,000" in html

    def test_includes_shift(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "3x12-Hour" in html

    def test_includes_location(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "Christiansburg" in html
        assert "VA" in html

    def test_includes_facility_name(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "Carilion New River Valley Medical Center" in html

    def test_omits_also_possible_when_alternative_is_none(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "Also possible" not in html

    def test_shows_also_possible_when_alternative_present(self):
        import emailer
        job = make_enriched_job()
        job["identification"]["alternative_facility"] = "Montgomery Regional Hospital"
        html = emailer.build_table([job])
        assert "Also possible" in html
        assert "Montgomery Regional Hospital" in html

    def test_high_confidence_uses_green_background(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "#d4edda" in html

    def test_none_confidence_uses_gray_background(self):
        import emailer
        job = make_enriched_job()
        job["identification"]["confidence"] = "none"
        html = emailer.build_table([job])
        assert "#e2e3e5" in html

    def test_includes_link_to_job(self):
        import emailer
        html = emailer.build_table([make_enriched_job()])
        assert "ayalocums.com/job/locum-physician/3236627" in html


def make_contact(**kwargs) -> dict:
    defaults = {
        "contact_name": "Dr. Jane Smith, MD",
        "contact_title": "Chief Medical Officer",
        "contact_email": "jane.smith@carilionclinic.org",
        "contact_email_basis": "pattern-guessed from domain",
        "contact_phone": "(540) 731-2000",
        "contact_linkedin": None,
        "contact_confidence": "medium",
        "contact_reasoning": "CMO is standard for locum coverage",
        "research_summary": "146-bed community hospital in Christiansburg",
    }
    return {**defaults, **kwargs}


class TestBuildTableContactColumns:
    """Verify Decision-Maker and Contact columns render correctly."""

    def test_contact_name_appears_in_html(self):
        import emailer
        job = make_enriched_job(contact=make_contact())
        html = emailer.build_table([job])
        assert "Dr. Jane Smith, MD" in html

    def test_contact_title_appears_in_html(self):
        import emailer
        job = make_enriched_job(contact=make_contact())
        html = emailer.build_table([job])
        assert "Chief Medical Officer" in html

    def test_contact_email_renders_as_mailto_link(self):
        import emailer
        job = make_enriched_job(contact=make_contact())
        html = emailer.build_table([job])
        assert 'href="mailto:jane.smith@carilionclinic.org"' in html

    def test_contact_phone_appears_in_html(self):
        import emailer
        job = make_enriched_job(contact=make_contact())
        html = emailer.build_table([job])
        assert "(540) 731-2000" in html

    def test_missing_contact_shows_em_dash(self):
        import emailer
        # Job has no contact dict at all
        job = make_enriched_job()
        html = emailer.build_table([job])
        # Should render without crashing and show em dash placeholders
        assert "&mdash;" in html

    def test_fallback_contact_all_empty_shows_em_dash(self):
        import emailer
        empty_contact = {
            "contact_name": "",
            "contact_title": "",
            "contact_email": "",
            "contact_phone": None,
            "contact_confidence": "none",
        }
        job = make_enriched_job(contact=empty_contact)
        html = emailer.build_table([job])
        assert "&mdash;" in html

    def test_invalid_email_falls_back_to_em_dash(self):
        import emailer
        bad_contact = make_contact(contact_email="unable to guess")
        job = make_enriched_job(contact=bad_contact)
        html = emailer.build_table([job])
        assert "unable to guess" not in html

    def test_html_escapes_malicious_contact_name(self):
        import emailer
        bad_contact = make_contact(contact_name="<script>alert(1)</script>")
        job = make_enriched_job(contact=bad_contact)
        html = emailer.build_table([job])
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_table_has_decision_maker_and_contact_headers(self):
        import emailer
        html = emailer.build_table([make_enriched_job(contact=make_contact())])
        assert "Decision-Maker" in html
        assert "<th" in html and "Contact</th>" in html

    def test_table_still_has_link_column_last(self):
        """New columns inserted before Link, not after."""
        import emailer
        html = emailer.build_table([make_enriched_job(contact=make_contact())])
        # Link column should still be present and last in headers
        link_idx = html.find(">Link</th>")
        contact_idx = html.find(">Contact</th>")
        assert link_idx > contact_idx


class TestBuildSuccessEmail:
    def test_subject_includes_count(self):
        import emailer
        subject, _ = emailer.build_success_email([make_enriched_job()])
        assert "1 new posting" in subject

    def test_subject_no_postings_when_empty(self):
        import emailer
        subject, _ = emailer.build_success_email([])
        assert "No new postings" in subject

    def test_subject_includes_ayalocums_brand(self):
        import emailer
        subject, _ = emailer.build_success_email([make_enriched_job()])
        assert "AyaLocums" in subject

    def test_plural_postings_for_multiple_jobs(self):
        import emailer
        subject, _ = emailer.build_success_email([make_enriched_job(), make_enriched_job()])
        assert "2 new postings" in subject


class TestBuildFailureEmail:
    def test_subject_includes_failed(self):
        import emailer
        subject, _ = emailer.build_failure_email("Something broke")
        assert "Failed" in subject

    def test_subject_includes_ayalocums_brand(self):
        import emailer
        subject, _ = emailer.build_failure_email("Something broke")
        assert "AyaLocums" in subject

    def test_body_includes_reason(self):
        import emailer
        _, body = emailer.build_failure_email("Disk full error")
        assert "Disk full error" in body


class TestSendEmailGmail:
    def test_sends_via_gmail_by_default(self):
        import emailer
        env = {
            "EMAIL_TRANSPORT": "gmail",
            "RECIPIENT_EMAIL": "test@example.com",
            "GMAIL_USER": "sender@gmail.com",
            "GMAIL_APP_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env):
            with patch("emailer.smtplib.SMTP") as mock_smtp:
                server = MagicMock()
                mock_smtp.return_value.__enter__ = MagicMock(return_value=server)
                mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
                emailer.send_email("Subject", "<p>body</p>", is_html=True)
                server.sendmail.assert_called_once()


class TestSendEmailSendGrid:
    def test_sends_via_sendgrid_when_transport_set(self):
        import emailer
        env = {
            "EMAIL_TRANSPORT": "sendgrid",
            "RECIPIENT_EMAIL": "test@example.com",
            "GMAIL_USER": "sender@gmail.com",
            "SENDGRID_API_KEY": "SG.fake",
        }
        with patch.dict(os.environ, env):
            mock_sg = MagicMock()
            with patch("sendgrid.SendGridAPIClient", return_value=mock_sg):
                emailer.send_email("Subject", "<p>body</p>", is_html=True)
                mock_sg.send.assert_called_once()
