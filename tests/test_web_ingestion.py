
import socket
from unittest.mock import Mock

import pytest

from modules import web_ingestion as web


def public_dns(*args, **kwargs):
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("93.184.216.34", 80),
        )
    ]


def test_validate_public_url_accepts_public_http(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", public_dns)

    result = web.validate_public_url("https://example.com")

    assert result == "https://example.com"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "file:///etc/passwd",
        "http:///missing-host",
        "http://localhost",
        "http://server.local",
        "http://example",
        "http://user:password@example.com",
        "http://example.com:8080",
        "http://example.com:invalid",
    ],
)
def test_validate_public_url_rejects_invalid_urls(url):
    with pytest.raises(ValueError):
        web.validate_public_url(url)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "::1",
    ],
)
def test_validate_public_url_rejects_private_ip(monkeypatch, address):
    with pytest.raises(ValueError):
        web.validate_public_url(f"http://{address}")


def test_validate_public_url_rejects_private_dns(monkeypatch):
    def private_dns(*args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.10", 80),
            )
        ]

    monkeypatch.setattr(web.socket, "getaddrinfo", private_dns)

    with pytest.raises(ValueError, match="non-public"):
        web.validate_public_url("https://example.com")


def test_validate_public_url_rejects_mixed_public_private_dns(
    monkeypatch,
):
    def mixed_dns(*args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 80),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.10", 80),
            ),
        ]

    monkeypatch.setattr(web.socket, "getaddrinfo", mixed_dns)

    with pytest.raises(ValueError, match="non-public"):
        web.validate_public_url("https://example.com")


def test_validate_public_url_handles_dns_failure(monkeypatch):
    def failed_dns(*args, **kwargs):
        raise socket.gaierror("DNS failure")

    monkeypatch.setattr(web.socket, "getaddrinfo", failed_dns)

    with pytest.raises(ValueError, match="resolve"):
        web.validate_public_url("https://example.com")


def test_validate_public_url_rejects_invalid_port():
    with pytest.raises(ValueError, match="port"):
        web.validate_public_url("https://example.com:99999")


def test_extract_page_content_removes_non_content_elements():
    html = """
    <html>
      <head><title>Example Article</title></head>
      <body>
        <nav>Navigation text</nav>
        <script>secret_script()</script>
        <main>
          <h1>Research topic</h1>
          <p>This is readable article content with useful information.</p>
          <p>This paragraph contains additional information for testing.</p>
          <footer>Footer text</footer>
        </main>
      </body>
    </html>
    """

    title, content = web.extract_page_content(html)

    assert title == "Example Article"
    assert "Research topic" in content
    assert "readable article content" in content
    assert "secret_script" not in content
    assert "Navigation text" not in content
    assert "Footer text" not in content


def test_extract_page_content_uses_untitled_default():
    html = (
        "<html><body><main>"
        + ("Useful content " * 20)
        + "</main></body></html>"
    )

    title, content = web.extract_page_content(html)

    assert title == "Untitled webpage"
    assert len(content) >= 100


def test_extract_page_content_rejects_insufficient_content():
    with pytest.raises(ValueError, match="Insufficient"):
        web.extract_page_content(
            "<html><body><main>Too short</main></body></html>"
        )


def test_extract_page_content_truncates_long_title():
    html = (
        "<html><head><title>"
        + ("A" * 400)
        + "</title></head><body><main>"
        + ("Useful article content " * 10)
        + "</main></body></html>"
    )

    title, _ = web.extract_page_content(html)

    assert len(title) == 300


def test_fetch_web_page_returns_html(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", public_dns)

    response = Mock()
    response.is_redirect = False
    response.is_permanent_redirect = False
    response.headers = {"Content-Type": "text/html"}
    response.encoding = "utf-8"
    response.url = "https://example.com"
    response.iter_content.return_value = [b"<html>Example</html>"]

    session = Mock()
    session.get.return_value = response
    session_context = Mock()
    session_context.__enter__ = Mock(return_value=session)
    session_context.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(
        web.requests, "Session", lambda: session_context
    )

    url, html = web.fetch_web_page("https://example.com")

    assert url == "https://example.com"
    assert "Example" in html
    response.close.assert_called_once()
    session.get.assert_called_once()


def test_fetch_web_page_rejects_non_html(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", public_dns)

    response = Mock()
    response.is_redirect = False
    response.is_permanent_redirect = False
    response.headers = {"Content-Type": "application/pdf"}

    session = Mock()
    session.get.return_value = response
    context = Mock()
    context.__enter__ = Mock(return_value=session)
    context.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(
        web.requests, "Session", lambda: context
    )

    with pytest.raises(ValueError, match="HTML"):
        web.fetch_web_page("https://example.com")

    response.close.assert_called_once()


def test_fetch_web_page_rejects_oversized_content(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", public_dns)

    response = Mock()
    response.is_redirect = False
    response.is_permanent_redirect = False
    response.headers = {
        "Content-Type": "text/html",
        "Content-Length": str(web.MAX_PAGE_SIZE + 1),
    }

    session = Mock()
    session.get.return_value = response
    context = Mock()
    context.__enter__ = Mock(return_value=session)
    context.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(
        web.requests, "Session", lambda: context
    )

    with pytest.raises(ValueError, match="2 MB"):
        web.fetch_web_page("https://example.com")

    response.close.assert_called_once()


def test_process_web_page_generates_chunks_and_ids(monkeypatch):
    html = (
        "This is a useful research article containing information. "
        * 150
    )

    monkeypatch.setattr(
        web,
        "fetch_web_page",
        lambda url: ("https://example.com/article", html),
    )
    monkeypatch.setattr(
        web,
        "extract_page_content",
        lambda html: ("Example Article", html),
    )

    chunks, ids, metadata = web.process_web_page(
        "https://example.com/article"
    )

    assert len(chunks) > 0
    assert len(chunks) == len(ids)
    assert metadata["title"] == "Example Article"
    assert metadata["source_type"] == "web_page"
    assert metadata["file_type"] == "html"
    assert metadata["url"] == "https://example.com/article"
    assert metadata["file_id"]
    assert chunks[0].metadata["chunk_index"] == 0
    assert ids[0].startswith("web-")


def test_process_web_page_rejects_empty_chunks(monkeypatch):
    monkeypatch.setattr(
        web,
        "fetch_web_page",
        lambda url: ("https://example.com", "content"),
    )
    monkeypatch.setattr(
        web,
        "extract_page_content",
        lambda html: ("Example", "content"),
    )

    monkeypatch.setattr(
        web.RecursiveCharacterTextSplitter,
        "split_documents",
        lambda self, documents: [],
    )

    with pytest.raises(ValueError, match="No indexable"):
        web.process_web_page("https://example.com")
