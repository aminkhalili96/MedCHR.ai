from types import SimpleNamespace

from backend.app.authz import has_permission, is_step_up_verified, mark_step_up_verified, require_permission


def test_require_permission_accepts_allowed_role():
    require_permission("admin", "user.manage")
    require_permission("clinician", "report.finalize")


def test_has_permission_rejects_unknown_permission():
    assert has_permission("clinician", "user.manage") is False


def test_mark_step_up_verified_sets_timestamp():
    request = SimpleNamespace(session={})
    mark_step_up_verified(request)
    assert is_step_up_verified(request, max_age_minutes=15) is True
