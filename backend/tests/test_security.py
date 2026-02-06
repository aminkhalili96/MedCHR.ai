from backend.app.security import render_markdown


def test_render_markdown_sanitizes_script():
    html = render_markdown("Hello <script>alert(1)</script>")
    assert "<script>" not in html


def test_render_markdown_allows_links():
    html = render_markdown("[link](https://example.com)")
    assert 'href="https://example.com"' in html
