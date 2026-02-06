import json
import logging

from backend.app.logging_config import StructuredFormatter


def test_structured_formatter_redacts_sensitive_fields_and_message_phi():
    formatter = StructuredFormatter()
    logger = logging.getLogger("test.logging.redaction")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        "Patient SSN 123-45-6789 email jane.doe@example.com phone +1 (555) 111-2222",
        (),
        None,
    )
    record.extra_data = {
        "ssn": "123-45-6789",
        "profile": {"email": "jane.doe@example.com"},
    }

    payload = json.loads(formatter.format(record))
    assert payload["data"]["ssn"] == "[REDACTED]"
    assert payload["data"]["profile"]["email"] == "[REDACTED]"
    assert "123-45-6789" not in payload["message"]
    assert "jane.doe@example.com" not in payload["message"]


def test_structured_formatter_truncates_large_text_fields():
    formatter = StructuredFormatter()
    logger = logging.getLogger("test.logging.truncation")
    record = logger.makeRecord(logger.name, logging.INFO, "", 0, "ok", (), None)
    record.extra_data = {"note": "x" * 1200}

    payload = json.loads(formatter.format(record))
    assert payload["data"]["note"].endswith("[TRUNCATED]")
