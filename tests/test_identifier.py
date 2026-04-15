"""tests/test_identifier.py"""
import json
import pytest
from unittest.mock import MagicMock, patch


def make_job(**kwargs) -> dict:
    defaults = {
        "id": "3236627",
        "specialty": "Obstetrics/Gynecology",
        "city": "Christiansburg",
        "state": "VA",
        "state_full": "Virginia",
        "zip_code": "24073",
        "street_address": "2875 Barn Rd",
        "lat": 37.09329,
        "lng": -80.50438,
        "pay_display": "$4,500\u2013$5,000/day",
        "shift": "3x12-Hour 07:00 - 07:00",
        "duration_weeks": 13,
        "start_date": "2026-07-01T04:00:00",
        "positions": 3,
        "url": "https://www.ayalocums.com/job/locum-physician/3236627/",
    }
    return {**defaults, **kwargs}


GOOD_RESPONSE = json.dumps({
    "facility_name": "Carilion New River Valley Medical Center",
    "facility_type": "Community Hospital",
    "confidence": "high",
    "reasoning": "Only hospital in Christiansburg at that address",
    "alternative_facility": None,
})


def make_mock_client(text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client.messages.create.return_value = msg
    return client


class TestIdentifyFacility:
    def test_returns_parsed_response_on_success(self):
        import identifier
        client = make_mock_client(GOOD_RESPONSE)
        result = identifier.identify_facility(client, make_job())
        assert result["facility_name"] == "Carilion New River Valley Medical Center"
        assert result["confidence"] == "high"
        assert result["reasoning"] != ""

    def test_strips_markdown_code_fences(self):
        import identifier
        wrapped = f"```json\n{GOOD_RESPONSE}\n```"
        client = make_mock_client(wrapped)
        result = identifier.identify_facility(client, make_job())
        assert result["facility_name"] == "Carilion New River Valley Medical Center"

    def test_returns_fallback_after_two_failures(self):
        import identifier
        client = make_mock_client("not valid json at all")
        with patch("identifier.time.sleep"):
            result = identifier.identify_facility(client, make_job())
        assert result == identifier.FALLBACK

    def test_retries_once_on_parse_failure(self):
        import identifier
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="bad json")]
        good_msg = MagicMock()
        good_msg.content = [MagicMock(text=GOOD_RESPONSE)]
        client = MagicMock()
        client.messages.create.side_effect = [bad_msg, good_msg]
        with patch("identifier.time.sleep"):
            result = identifier.identify_facility(client, make_job())
        assert result["facility_name"] == "Carilion New River Valley Medical Center"
        assert client.messages.create.call_count == 2

    def test_returns_fallback_when_required_keys_missing(self):
        import identifier
        incomplete = json.dumps({"facility_name": "Something", "confidence": "low"})
        client = make_mock_client(incomplete)
        with patch("identifier.time.sleep"):
            result = identifier.identify_facility(client, make_job())
        assert result == identifier.FALLBACK


# ---------------------------------------------------------------------------
# Contact research tests (second-stage Claude call with web_search tool)
# ---------------------------------------------------------------------------

GOOD_CONTACT_RESPONSE = json.dumps({
    "contact_name": "Dr. Jane Smith, MD",
    "contact_title": "Chief Medical Officer",
    "contact_email": "jane.smith@carilionclinic.org",
    "contact_email_basis": "pattern-guessed from facility domain; verified by other staff emails on leadership page",
    "contact_phone": "(540) 731-2000",
    "contact_linkedin": "https://linkedin.com/in/jane-smith-md",
    "contact_confidence": "medium",
    "contact_reasoning": "CMO is the standard decision-maker for locum coverage at community hospitals",
    "research_summary": "Carilion NRV is a 146-bed community hospital in Christiansburg VA. Dr. Jane Smith is the listed CMO on the leadership page.",
})


def make_identification(confidence="high", facility_name="Carilion New River Valley Medical Center") -> dict:
    return {
        "facility_name": facility_name,
        "facility_type": "Community Hospital",
        "confidence": confidence,
        "reasoning": "test",
        "alternative_facility": None,
    }


def make_text_block(text: str):
    """Mock an Anthropic content block with type='text'."""
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def make_tool_block():
    """Mock a non-text content block (e.g. server_tool_use / web_search_tool_result)."""
    b = MagicMock()
    b.type = "server_tool_use"
    b.text = None
    return b


def make_ws_mock_client(blocks) -> MagicMock:
    """Mock client whose response has the given content blocks."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = blocks
    client.messages.create.return_value = msg
    return client


class TestResearchContact:
    def test_skips_api_call_when_facility_confidence_is_none(self):
        import identifier
        client = MagicMock()
        result = identifier.research_contact(client, make_job(), make_identification(confidence="none"))
        assert client.messages.create.call_count == 0
        assert result == identifier.CONTACT_FALLBACK

    def test_happy_path_returns_parsed_contact(self):
        import identifier
        client = make_ws_mock_client([make_text_block(GOOD_CONTACT_RESPONSE)])
        result = identifier.research_contact(client, make_job(), make_identification())
        assert result["contact_name"] == "Dr. Jane Smith, MD"
        assert result["contact_email"] == "jane.smith@carilionclinic.org"
        assert result["contact_confidence"] == "medium"

    def test_uses_last_text_block_when_web_search_produced_tool_blocks(self):
        """web_search responses interleave tool_use blocks; final text block wins."""
        import identifier
        blocks = [
            make_text_block("scratchpad: let me search..."),
            make_tool_block(),
            make_tool_block(),
            make_text_block(GOOD_CONTACT_RESPONSE),
        ]
        client = make_ws_mock_client(blocks)
        result = identifier.research_contact(client, make_job(), make_identification())
        assert result["contact_name"] == "Dr. Jane Smith, MD"

    def test_retries_once_on_parse_failure(self):
        import identifier
        bad = MagicMock()
        bad.content = [make_text_block("not json")]
        good = MagicMock()
        good.content = [make_text_block(GOOD_CONTACT_RESPONSE)]
        client = MagicMock()
        client.messages.create.side_effect = [bad, good]
        with patch("identifier.time.sleep"):
            result = identifier.research_contact(client, make_job(), make_identification())
        assert result["contact_name"] == "Dr. Jane Smith, MD"
        assert client.messages.create.call_count == 2

    def test_returns_fallback_after_two_failures(self):
        import identifier
        client = make_ws_mock_client([make_text_block("garbage")])
        with patch("identifier.time.sleep"):
            result = identifier.research_contact(client, make_job(), make_identification())
        assert result["contact_confidence"] == "none"
        assert result["contact_name"] == ""

    def test_returns_fallback_when_required_keys_missing(self):
        import identifier
        incomplete = json.dumps({"contact_name": "Dr. Smith", "contact_confidence": "low"})
        client = make_ws_mock_client([make_text_block(incomplete)])
        with patch("identifier.time.sleep"):
            result = identifier.research_contact(client, make_job(), make_identification())
        assert result["contact_confidence"] == "none"

    def test_passes_web_search_tool_in_kwargs(self):
        import identifier
        client = make_ws_mock_client([make_text_block(GOOD_CONTACT_RESPONSE)])
        identifier.research_contact(client, make_job(), make_identification())
        kwargs = client.messages.create.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tools"][0]["type"] == "web_search_20250305"
        assert kwargs["tools"][0]["max_uses"] == 5
        # Sonnet model, not Haiku
        assert "sonnet" in kwargs["model"].lower()

    def test_prompt_uses_already_identified_facility_name(self):
        """The contact prompt should lead with the facility name from identification, not guess again."""
        import identifier
        client = make_ws_mock_client([make_text_block(GOOD_CONTACT_RESPONSE)])
        ident = make_identification(facility_name="Acme Regional Medical Center")
        identifier.research_contact(client, make_job(), ident)
        sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Acme Regional Medical Center" in sent_prompt

    def test_prompt_includes_injection_defense(self):
        """Prompt should explicitly label job content as untrusted data."""
        import identifier
        client = make_ws_mock_client([make_text_block(GOOD_CONTACT_RESPONSE)])
        identifier.research_contact(client, make_job(), make_identification())
        sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "untrusted" in sent_prompt.lower() or "not instructions" in sent_prompt.lower()
