"""
콘텐츠 추출 모듈.
X(Twitter), PDF, YouTube, 웹페이지에서 텍스트를 뽑는 순수 함수 모음.
"""

import re
import logging

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

TWEET_URL_PATTERN = re.compile(
    r"https?://(?:twitter\.com|x\.com)/\w+/status/(\d+)(?:\?\S*)?"
)
YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)
GENERAL_URL_PATTERN = re.compile(r"https?://\S+")


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
                body = body[:MAX_PDF_CHARS]
                return f"[X Article]\n작성자: {name}\n제목: {title}\nDate: {created}\n\n{body}"

        # 일반 피드
        media = tweet.get("media", {})
        if not text or text.startswith("https://t.co/"):
            media_types = [m.get("type", "unknown") for m in media.get("all", [])]
            desc = f"[미디어: {', '.join(media_types)}]" if media_types else "[텍스트 없음]"
            return f"[X Post]\n{name}: {desc}\nDate: {created}"
        return f"[X Post]\n{name}: {text}\nDate: {created}"
    except Exception as e:
        logger.error(f"X post extraction error: {e}")
        return None


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
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        r = requests.get(url, timeout=30, headers=headers)
        r.raise_for_status()
        return extract_pdf_text(r.content)
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
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        if not text:
            return None
        return f"[Web Article]\nURL: {url}\n\n{text[:MAX_WEB_CHARS]}"
    except Exception as e:
        logger.error(f"Web extraction error: {e}")
        return None
