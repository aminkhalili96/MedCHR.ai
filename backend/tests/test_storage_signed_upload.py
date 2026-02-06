from unittest.mock import MagicMock, patch

from backend.app.storage import upload_bytes_via_signed_url


def test_upload_bytes_via_signed_url_uses_signed_endpoint():
    with patch("backend.app.storage.create_signed_upload_url") as mock_signed, patch(
        "backend.app.storage.httpx.Client"
    ) as mock_client_cls:
        mock_signed.return_value = {
            "upload_url": "https://storage.example/upload/sign/path?token=abc",
            "token": "abc",
            "path": "patient-1/document.pdf",
        }
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Key": "patient-1/document.pdf"}
        mock_client.put.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        result = upload_bytes_via_signed_url("uploads", "patient-1/document.pdf", b"abc", "application/pdf")

    assert result["path"] == "patient-1/document.pdf"
    assert result["Key"] == "patient-1/document.pdf"
    mock_client.put.assert_called_once()
