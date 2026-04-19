import logging
from urllib.parse import urlparse

import requests

from .extractors_common import (
    MAX_PDF_CHARS,
    MAX_TRANSCRIPT_CHARS,
    TWEET_URL_PATTERN,
    _clean_extracted_text,
    _extract_meta_text,
    _is_x_url,
)
from .extractors_network import _download_public_url

logger = logging.getLogger(__name__)


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


def extract_pdf_text(file_bytes: bytes) -> str | None:
    """pymupdf로 PDF 바이트에서 텍스트 추출"""
    try:
        import fitz

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


def extract_youtube_transcript(video_id: str) -> str | None:
    """youtube-transcript-api로 스크립트 추출"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["ko", "en"])
        text = " ".join(s.text for s in transcript.snippets)
        if not text:
            return None
        return f"[YouTube Transcript]\n{text[:MAX_TRANSCRIPT_CHARS]}"
    except Exception as e:
        logger.error(f"YouTube transcript error: {e}")
        return None
