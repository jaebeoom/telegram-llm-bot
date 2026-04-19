import logging
from dataclasses import dataclass
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

YOUTUBE_TRANSCRIPT_PREFERRED_LANGUAGES = ("ko", "en")
YOUTUBE_TRANSCRIPT_FALLBACK_TRANSLATION_LANGUAGE = "ko"


@dataclass(frozen=True)
class YouTubeTranscriptExtractionResult:
    content: str | None
    status: str
    message: str
    language_code: str | None = None
    language: str | None = None
    is_generated: bool | None = None
    selection: str = ""


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


def _format_youtube_transcript_error(exc: Exception) -> tuple[str, str]:
    from youtube_transcript_api._errors import (
        AgeRestricted,
        InvalidVideoId,
        IpBlocked,
        NoTranscriptFound,
        PoTokenRequired,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
        VideoUnplayable,
    )

    if isinstance(exc, TranscriptsDisabled):
        return (
            "transcripts_disabled",
            "이 영상은 YouTube에서 공개 스크립트/자막이 비활성화되어 있습니다.",
        )
    if isinstance(exc, NoTranscriptFound):
        return (
            "no_transcript_found",
            "이 영상에서 가져올 수 있는 공개 스크립트/자막 트랙을 찾지 못했습니다.",
        )
    if isinstance(exc, (RequestBlocked, IpBlocked)):
        return (
            "request_blocked",
            "YouTube가 현재 스크립트 요청을 차단했습니다. 나중에 다시 시도하거나 다른 네트워크가 필요할 수 있습니다.",
        )
    if isinstance(exc, PoTokenRequired):
        return (
            "po_token_required",
            "YouTube가 추가 재생 토큰을 요구해서 스크립트를 가져올 수 없습니다.",
        )
    if isinstance(exc, AgeRestricted):
        return (
            "age_restricted",
            "연령 제한 영상이라 공개 스크립트를 가져올 수 없습니다.",
        )
    if isinstance(exc, (VideoUnavailable, VideoUnplayable)):
        return (
            "video_unavailable",
            "영상을 재생할 수 없어서 스크립트를 가져올 수 없습니다.",
        )
    if isinstance(exc, InvalidVideoId):
        return ("invalid_video_id", "YouTube 영상 ID가 올바르지 않습니다.")
    return ("transcript_error", "YouTube 스크립트를 가져오는 중 오류가 발생했습니다.")


def _select_youtube_transcript(transcript_list, preferred_languages: tuple[str, ...]):
    from youtube_transcript_api._errors import NoTranscriptFound

    selection_attempts = (
        ("preferred_manual", transcript_list.find_manually_created_transcript),
        ("preferred_generated", transcript_list.find_generated_transcript),
        ("preferred_any", transcript_list.find_transcript),
    )
    for selection, finder in selection_attempts:
        try:
            return finder(preferred_languages), selection
        except NoTranscriptFound:
            continue

    available = list(transcript_list)
    for transcript in available:
        if not transcript.is_generated:
            return transcript, "fallback_manual"
    if available:
        return available[0], "fallback_generated"
    raise NoTranscriptFound(
        video_id="",
        requested_language_codes=preferred_languages,
        transcript_data=transcript_list,
    )


def _translate_youtube_transcript_if_useful(transcript, preferred_languages: tuple[str, ...]):
    if transcript.language_code in preferred_languages:
        return transcript, False
    if not transcript.is_translatable:
        return transcript, False
    try:
        return transcript.translate(YOUTUBE_TRANSCRIPT_FALLBACK_TRANSLATION_LANGUAGE), True
    except Exception as exc:
        logger.info("YouTube transcript translation unavailable: %s", exc)
        return transcript, False


def extract_youtube_transcript_result(video_id: str) -> YouTubeTranscriptExtractionResult:
    """youtube-transcript-api로 스크립트를 추출하고 실패 사유를 보존한다."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        transcript, selection = _select_youtube_transcript(
            transcript_list,
            YOUTUBE_TRANSCRIPT_PREFERRED_LANGUAGES,
        )
        transcript, translated = _translate_youtube_transcript_if_useful(
            transcript,
            YOUTUBE_TRANSCRIPT_PREFERRED_LANGUAGES,
        )
        if translated:
            selection = f"{selection}_translated"

        fetched_transcript = transcript.fetch()
        text = " ".join(s.text for s in fetched_transcript.snippets).strip()
        if not text:
            return YouTubeTranscriptExtractionResult(
                content=None,
                status="empty_transcript",
                message="스크립트 트랙은 있지만 텍스트가 비어 있습니다.",
                language_code=transcript.language_code,
                language=transcript.language,
                is_generated=transcript.is_generated,
                selection=selection,
            )
        return YouTubeTranscriptExtractionResult(
            content=f"[YouTube Transcript]\n{text[:MAX_TRANSCRIPT_CHARS]}",
            status="ok",
            message="",
            language_code=transcript.language_code,
            language=transcript.language,
            is_generated=transcript.is_generated,
            selection=selection,
        )
    except Exception as e:
        status, message = _format_youtube_transcript_error(e)
        logger.error("YouTube transcript error status=%s: %s", status, e)
        return YouTubeTranscriptExtractionResult(content=None, status=status, message=message)


def extract_youtube_transcript(video_id: str) -> str | None:
    """기존 호출부와 테스트를 위한 호환 wrapper."""
    return extract_youtube_transcript_result(video_id).content
