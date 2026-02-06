from backend.app.uploads import sanitize_filename


def test_sanitize_filename_strips_paths():
    assert sanitize_filename("../secret/../../evil.txt") == "evil.txt"


def test_sanitize_filename_fallback():
    assert sanitize_filename("////") == "upload.bin"
