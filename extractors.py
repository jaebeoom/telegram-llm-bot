"""
콘텐츠 추출 모듈.
X(Twitter), PDF, YouTube, 웹페이지에서 텍스트를 뽑는 순수 함수 모음.
"""

import ipaddress
import re
import logging
import socket
from urllib.parse import urljoin, urlparse

import requests
import fitz
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
MAX_PDF_CHARS = 20000
MAX_TRANSCRIPT_CHARS = 20000
MAX_WEB_CHARS = 20000
MAX_FETCH_BYTES = 15 * 1024 * 1024
MAX_REDIRECTS = 5

TWEET_URL_PATTERN = re.compile(
    r"https?://(?:twitter\.com|x\.com)/\w+/status/(\d+)(?:\?\S*)?"
)
YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
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
    "enable javascript",
    "sign in to x",
    "join x today",
)


def _is_private_hostname(hostname: str) -> bool:
    lowered = hostname.lower()
    if lowered in {"localhost"} or lowered.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(lowered)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return True

    for info in infos:
        resolved_ip = ipaddress.ip_address(info[4][0])
        if (
            resolved_ip.is_private
            or resolved_ip.is_loopback
            or resolved_ip.is_link_local
            or resolved_ip.is_multicast
            or resolved_ip.is_reserved
            or resolved_ip.is_unspecified
        ):
            return True

    return False


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("http/https URL만 허용됩니다.")
    if not parsed.hostname:
        raise ValueError("유효한 호스트가 없습니다.")
    if _is_private_hostname(parsed.hostname):
        raise ValueError("로컬/사설 네트워크 주소에는 접근할 수 없습니다.")
    return url


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
        text = re.sub(r"\s+", " ", match.group(1)).strip()
        cleaned = _clean_extracted_text(text, limit)
        if cleaned:
            return cleaned
    return None


def _download_public_url(url: str) -> bytes:
    current_url = _validate_public_url(url)

    with requests.Session() as session:
        for _ in range(MAX_REDIRECTS + 1):
            response = session.get(
                current_url,
                timeout=30,
                headers=REQUEST_HEADERS,
                allow_redirects=False,
                stream=True,
            )

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("리다이렉트 Location 헤더가 없습니다.")
                current_url = _validate_public_url(urljoin(current_url, location))
                response.close()
                continue

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_FETCH_BYTES:
                raise ValueError("다운로드 가능한 최대 크기를 초과했습니다.")

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_FETCH_BYTES:
                    raise ValueError("다운로드 가능한 최대 크기를 초과했습니다.")
                chunks.append(chunk)

            return b"".join(chunks)

    raise ValueError("리다이렉트가 너무 많습니다.")


def _extract_tweet_from_html(url: str) -> str | None:
    parsed = urlparse(url)
    candidates = [url]
    if _is_x_url(url):
        for host in ("fxtwitter.com", "vxtwitter.com"):
            candidates.append(parsed._replace(netloc=host).geturl())

    for candidate in dict.fromkeys(candidates):
        try:
            html = _download_public_url(candidate).decode("utf-8", errors="ignore")
            text = _extract_meta_text(html, MAX_PDF_CHARS)
            if text:
                return f"[X Post]\n{text}"
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────
# X(Twitter)
# ──────────────────────────────────────────────
def extract_tweet(tweet_id: str) -> str | None:
    """fxtwitter API로 X 피드 텍스트 추출"""
    try:
        r = requests.get(f"https://api.fxtwitter.com/status/{tweet_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        tweet = data.get("tweet", {})
        name = tweet.get("author", {}).get("name", "Unknown")
        text = tweet.get("text", "") or tweet.get("raw_text", {}).get("text", "")
        created = tweet.get("created_at", "")

        # 아티클(장문 포스트) 감지 → content.blocks에서 본문 추출
        article = tweet.get("article")
        if article:
            title = article.get("title", "")
            blocks = article.get("content", {}).get("blocks", [])
            body = "\n".join(b.get("text", "") for b in blocks if b.get("text"))
            if body:
                body = _clean_extracted_text(body, MAX_PDF_CHARS)
                if not body:
                    return None
                return f"[X Article]\n작성자: {name}\n제목: {title}\nDate: {created}\n\n{body}"

        # 일반 피드
        media = tweet.get("media", {})
        if not text or text.startswith("https://t.co/"):
            media_types = [m.get("type", "unknown") for m in media.get("all", [])]
            desc = f"[미디어: {', '.join(media_types)}]" if media_types else "[텍스트 없음]"
            return f"[X Post]\n{name}: {desc}\nDate: {created}"
        cleaned = _clean_extracted_text(text, MAX_PDF_CHARS)
        if cleaned:
            return f"[X Post]\n{name}: {cleaned}\nDate: {created}"
    except Exception as e:
        logger.error(f"X post extraction error: {e}")
    return None


def extract_tweet_from_url(url: str) -> str | None:
    match = TWEET_URL_PATTERN.search(url)
    if not match:
        return None
    extracted = extract_tweet(match.group(1))
    if extracted:
        return extracted
    return _extract_tweet_from_html(url)


# ──────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str | None:
    """pymupdf로 PDF 바이트에서 텍스트 추출"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        total = 0
        for page in doc:
            text = page.get_text()
            if total + len(text) > MAX_PDF_CHARS:
                pages.append(text[: MAX_PDF_CHARS - total])
                break
            pages.append(text)
            total += len(text)
        doc.close()
        result = "\n".join(pages).strip()
        if not result:
            return None
        return f"[PDF Document]\n{result}"
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None


def extract_pdf_from_url(url: str) -> str | None:
    """URL에서 PDF 다운로드 후 텍스트 추출"""
    try:
        return extract_pdf_text(_download_public_url(url))
    except Exception as e:
        logger.error(f"PDF URL extraction error: {e}")
        return None


# ──────────────────────────────────────────────
# YouTube
# ──────────────────────────────────────────────
def extract_youtube_transcript(video_id: str) -> str | None:
    """youtube-transcript-api로 스크립트 추출"""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["ko", "en"])
        text = " ".join(s.text for s in transcript.snippets)
        if not text:
            return None
        return f"[YouTube Transcript]\n{text[:MAX_TRANSCRIPT_CHARS]}"
    except Exception as e:
        logger.error(f"YouTube transcript error: {e}")
        return None


# ──────────────────────────────────────────────
# 웹페이지
# ──────────────────────────────────────────────
def extract_web_text(url: str) -> str | None:
    """trafilatura로 웹페이지 본문 추출"""
    if _is_x_url(url):
        return None
    try:
        downloaded = _download_public_url(url).decode("utf-8", errors="ignore")
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        if not text:
            return None
        cleaned = _clean_extracted_text(text, MAX_WEB_CHARS)
        if not cleaned:
            return None
        return f"[Web Article]\nURL: {url}\n\n{cleaned}"
    except Exception as e:
        logger.error(f"Web extraction error: {e}")
        return None
