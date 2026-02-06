import ast
import inspect

from backend.app.main import app


_PHI_PREFIXES = ("/patients", "/documents", "/chr", "/fhir", "/api/gap", "/jobs")
_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_AUDIT_CALLS = {
    "_log_action",
    "append_audit_event",
    "_audit_clinical_event",
    "_audit_gap_event",
    "_upload_document",
    "_extract_document",
    "_embed_document",
    "_draft_chr",
    "_register_signed_upload",
    "_issue_signed_upload",
}


def _collect_called_functions(source: str) -> set[str]:
    calls: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            calls.add(func.id)
        elif isinstance(func, ast.Attribute):
            calls.add(func.attr)
    return calls


def test_phi_routes_have_explicit_audit_policy_marker():
    missing: list[str] = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", "")
        if not methods or not path.startswith(_PHI_PREFIXES):
            continue
        if not (methods & _METHODS):
            continue
        source = inspect.getsource(route.endpoint)
        calls = _collect_called_functions(source)
        if calls.isdisjoint(_AUDIT_CALLS):
            listed_methods = ",".join(sorted(method for method in methods if method in _METHODS))
            missing.append(f"{listed_methods} {path}")

    assert not missing, "Missing audit marker(s): " + "; ".join(missing)
