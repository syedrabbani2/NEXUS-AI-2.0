
import hashlib
import ipaddress
import socket
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# WEB INGESTION CONFIGURATION
# --------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (compatible; NEXUS-AI/2.0; "
    "+https://www.python.org/)"
)

REQUEST_TIMEOUT = (5, 15)
MAX_PAGE_SIZE = 2 * 1024 * 1024
MAX_REDIRECTS = 4


# --------------------------------------------------
# PUBLIC URL VALIDATION
# --------------------------------------------------

def validate_public_url(url: str) -> str:
    """Validate that a URL points to a public HTTP/HTTPS destination."""

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed.hostname:
        raise ValueError("The URL does not contain a valid hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")

    hostname = parsed.hostname.lower().rstrip(".")

    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or "." not in hostname
    ):
        raise ValueError("Private or local hostnames are not allowed.")

    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid URL port.")

    if port is not None and port not in (80, 443):
        raise ValueError("Only ports 80 and 443 are supported.")

    # Check literal IP addresses directly.
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        # Resolve DNS and inspect every returned address.
        try:
            results = socket.getaddrinfo(
                hostname,
                port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
            addresses = list({
                ipaddress.ip_address(result[4][0])
                for result in results
            })
        except socket.gaierror:
            raise ValueError("Unable to resolve the website hostname.")

    if not addresses:
        raise ValueError("The website hostname has no valid IP address.")

    for address in addresses:
        if not address.is_global:
            raise ValueError(
                "Private, reserved and non-public IP destinations are blocked."
            )

    return url.strip()


# --------------------------------------------------
# FETCH WEBPAGE
# --------------------------------------------------

def fetch_web_page(url: str):
    """Fetch a public webpage while validating redirect destinations."""

    current_url = validate_public_url(url)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }

    with requests.Session() as session:

        for redirect_number in range(MAX_REDIRECTS + 1):

            # Revalidate immediately before each request.
            current_url = validate_public_url(current_url)

            response = session.get(
                current_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )

            try:
                if response.is_redirect or response.is_permanent_redirect:

                    location = response.headers.get("Location")

                    if not location:
                        raise ValueError("Redirect has no destination.")

                    if redirect_number >= MAX_REDIRECTS:
                        raise ValueError("Maximum redirect limit exceeded.")

                    current_url = urljoin(current_url, location)
                    current_url = validate_public_url(current_url)
                    continue

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError(
                        "The URL does not return an HTML webpage."
                    )

                content_length = response.headers.get("Content-Length")

                if content_length:
                    try:
                        if int(content_length) > MAX_PAGE_SIZE:
                            raise ValueError(
                                "Webpage exceeds the 2 MB size limit."
                            )
                    except ValueError as exc:
                        if "2 MB" in str(exc):
                            raise
                        raise ValueError("Invalid Content-Length header.")

                content = bytearray()

                for part in response.iter_content(chunk_size=16384):

                    if not part:
                        continue

                    content.extend(part)

                    if len(content) > MAX_PAGE_SIZE:
                        raise ValueError(
                            "Webpage exceeds the 2 MB size limit."
                        )

                response.encoding = response.encoding or "utf-8"

                html = bytes(content).decode(
                    response.encoding,
                    errors="replace",
                )

                final_url = validate_public_url(response.url)

                return final_url, html

            finally:
                response.close()

    raise ValueError("Unable to fetch webpage.")


# --------------------------------------------------
# EXTRACT CLEAN WEBPAGE CONTENT
# --------------------------------------------------

def extract_page_content(html: str):
    """Extract readable webpage text and its title."""

    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled webpage"
    )

    # Remove non-content elements.
    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "nav",
        "footer",
        "header",
        "form",
        "button",
    ]):
        element.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    text = main_content.get_text(separator=" ", strip=True)

    # Normalize whitespace.
    text = " ".join(text.split())

    if len(text) < 100:
        raise ValueError(
            "Insufficient readable content found on this webpage."
        )

    return title[:300], text


# --------------------------------------------------
# PROCESS WEBPAGE INTO LANGCHAIN DOCUMENTS
# --------------------------------------------------

def process_web_page(url: str):
    """
    Fetch, extract and split a webpage.

    Returns:
        chunks: LangChain Document objects
        ids: Unique ChromaDB IDs
        page_metadata: Webpage metadata
    """

    final_url, html = fetch_web_page(url)

    title, page_text = extract_page_content(html)

    # Hash the webpage content.
    content_hash = hashlib.sha256(
        page_text.encode("utf-8")
    ).hexdigest()[:16]

    # Hash the URL so different URLs have distinct IDs.
    url_hash = hashlib.sha256(
        final_url.encode("utf-8")
    ).hexdigest()[:10]

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    page_metadata = {
        "source": final_url,
        "source_type": "web_page",
        "file_type": "html",
        "title": title,
        "url": final_url,
        "retrieved_at": retrieved_at,
        "file_id": content_hash,
    }

    document = Document(
        page_content=page_text,
        metadata=page_metadata.copy(),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise ValueError("No indexable webpage content was generated.")

    ids = []

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = index

        ids.append(
            f"web-{url_hash}-{content_hash}-{index}"
        )

    return chunks, ids, page_metadata
