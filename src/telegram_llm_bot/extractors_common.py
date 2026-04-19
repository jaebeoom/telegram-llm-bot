import os
import re
from html import unescape
from urllib.parse import urlparse

MAX_PDF_CHARS = 120000
MAX_TRANSCRIPT_CHARS = 120000
MAX_WEB_CHARS = 120000
MAX_FETCH_BYTES = 15 * 1024 * 1024
MAX_REDIRECTS = 5

TWEET_URL_PATTERN = re.compile(
    r"https?://(?:twitter\.com|x\.com)/\w+/status/(\d+)(?:\?\S*)?"
)
YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?(?:[^\s#]*?&)?v=|youtu\.be/)([\w-]{11})(?:[?&][^\s#]*)?"
)
GENERAL_URL_PATTERN = re.compile(r"https?://\S+")

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}
_META_DESCRIPTION_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:og:description|twitter:description|description)["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_X_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com"}
_NOISE_SNIPPETS = (
    "you can see a list of supported browsers in our help center",
    "we’ve detected that javascript is disabled in this browser",
    "we've detected that javascript is disabled in this browser",
    "browser is no longer supported",
    "this browser is no longer supported",
    "checking if the site connection is secure",
    "verify you are human",
    "enable javascript and cookies to continue",
    "attention required!",
    "enable javascript",
    "sign in to x",
    "join x today",
)
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY_VALUES


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default

    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _is_x_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").removeprefix("www.").casefold()
    return host in _X_HOSTS


def _clean_extracted_text(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    clipped = text.strip()[:limit]
    lowered = " ".join(clipped.split()).casefold()
    if any(snippet in lowered for snippet in _NOISE_SNIPPETS):
        return None
    return clipped if clipped else None


def _extract_meta_text(html: str, limit: int) -> str | None:
    for pattern in (_META_DESCRIPTION_RE, _TITLE_RE):
        match = pattern.search(html[:20000])
        if not match:
            continue
        text = re.sub(r"\s+", " ", unescape(match.group(1))).strip()
        cleaned = _clean_extracted_text(text, limit)
        if cleaned:
            return cleaned
    return None
