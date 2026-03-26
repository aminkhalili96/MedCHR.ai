"""Tests for config validation — production security guards."""

import pytest
from unittest.mock import MagicMock

from backend.app.config import validate_production_config


class TestProductionConfigValidation:
    def _make_settings(self, **overrides):
        s = MagicMock()
        s.app_secret_key = "a-real-secret-key-not-default"
        s.csrf_enabled = True
        s.app_env = "prod"
        s.hipaa_mode = False
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_valid_config_passes(self):
        s = self._make_settings()
        validate_production_config(s)  # Should not raise

    def test_default_secret_key_rejected(self):
        s = self._make_settings(app_secret_key="dev-secret")
        with pytest.raises(RuntimeError, match="APP_SECRET_KEY"):
            validate_production_config(s)

    def test_csrf_disabled_rejected(self):
        s = self._make_settings(csrf_enabled=False)
        with pytest.raises(RuntimeError, match="CSRF_ENABLED"):
            validate_production_config(s)

    def test_both_issues_reported(self):
        s = self._make_settings(app_secret_key="dev-secret", csrf_enabled=False)
        with pytest.raises(RuntimeError) as exc_info:
            validate_production_config(s)
        msg = str(exc_info.value)
        assert "APP_SECRET_KEY" in msg
        assert "CSRF_ENABLED" in msg
