import logging
import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urljoin, urlparse

import trafilatura

from .extractors_common import MAX_WEB_CHARS, _clean_extracted_text, _extract_meta_text, _is_x_url
from .extractors_network import _download_public_url, _validate_public_url
from .extractors_rendering import _render_documents_with_playwright

try:
    from lxml.html import fromstring
    from readability import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None
    fromstring = None

logger = logging.getLogger(__name__)

_IFRAME_TAG_RE = re.compile(r"<iframe\b[^>]*>", re.IGNORECASE)
_IFRAME_SRC_RE = re.compile(r'\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class WebExtractionResult:
    content: str
    method: str
    document_url: str


def _extract_iframe_urls(url: str, html: str) -> list[str]:
    parsed_base = urlparse(url)
    base_host = (parsed_base.hostname or "").casefold()
    candidates: list[tuple[int, str]] = []

    for tag in _IFRAME_TAG_RE.findall(html):
        src_match = _IFRAME_SRC_RE.search(tag)
        if not src_match:
            continue
        raw_src = unescape(src_match.group(1)).strip()
        if not raw_src or raw_src.casefold().startswith(("javascript:", "data:", "about:")):
            continue
        resolved = urljoin(url, raw_src)
        resolved_host = (urlparse(resolved).hostname or "").casefold()
        tag_lower = tag.casefold()
        resolved_lower = resolved.casefold()
        score = 0

        if resolved_host and resolved_host == base_host:
            score += 100
        if "mainframe" in tag_lower:
            score += 150
        if any(token in resolved_lower for token in ("postview", "article", "content", "viewer", "entry")):
            score += 50
        if any(token in tag_lower for token in ("content", "article", "viewer")):
            score += 20
        if any(token in resolved_lower for token in ("ad", "ads", "banner", "comment", "reply", "share", "video", "player")):
            score -= 80

        candidates.append((score, resolved))

    ordered_urls: list[str] = []
    seen: set[str] = set()
    for _, iframe_url in sorted(candidates, key=lambda item: item[0], reverse=True):
        if iframe_url in seen:
            continue
        ordered_urls.append(iframe_url)
        seen.add(iframe_url)
    return ordered_urls


def _extract_with_trafilatura(html: str) -> str | None:
    try:
        text = trafilatura.extract(html)
    except Exception:
        text = None
    return _clean_extracted_text(text, MAX_WEB_CHARS)


def _extract_with_readability(html: str) -> str | None:
    if Document is None:
        return None

    try:
        summary_html = Document(html).summary(html_partial=True)
    except Exception:
        return None

    text = _extract_with_trafilatura(summary_html)
    if text:
        return text

    if fromstring is None:
        return None

    try:
        text = fromstring(summary_html).text_content()
    except Exception:
        return None
    return _clean_extracted_text(text, MAX_WEB_CHARS)


def _extract_web_result_from_html(
    url: str,
    html: str,
    visited: set[str],
    *,
    stage: str,
) -> tuple[str, str, str] | None:
    if url in visited:
        return None
    visited.add(url)

    cleaned = _extract_with_trafilatura(html)
    if cleaned:
        return cleaned, f"{stage}_trafilatura", url

    readability_cleaned = _extract_with_readability(html)
    if readability_cleaned:
        return readability_cleaned, f"{stage}_readability", url

    for iframe_url in _extract_iframe_urls(url, html):
        if iframe_url in visited:
            continue
        try:
            iframe_html = _download_public_url(iframe_url).decode("utf-8", errors="ignore")
        except Exception:
            continue
        nested = _extract_web_result_from_html(iframe_url, iframe_html, visited, stage=stage)
        if nested:
            return nested

    meta_text = _extract_meta_text(html, MAX_WEB_CHARS)
    if meta_text:
        return meta_text, f"{stage}_meta", url
    return None


def _extract_web_result_from_documents(
    base_url: str,
    documents: list[tuple[str, str]],
    *,
    stage: str,
) -> tuple[str, str, str] | None:
    visited: set[str] = set()
    for document_url, html in documents:
        result = _extract_web_result_from_html(document_url or base_url, html, visited, stage=stage)
        if result:
            return result
    return None


def _extract_rendered_web_result(url: str) -> tuple[str, str, str] | None:
    rendered_documents = _render_documents_with_playwright(url)
    if not rendered_documents:
        return None
    return _extract_web_result_from_documents(url, rendered_documents, stage="rendered")


def _extract_rendered_web_text(url: str) -> str | None:
    result = _extract_rendered_web_result(url)
    return result[0] if result else None


def extract_web_result(url: str) -> WebExtractionResult | None:
    """웹페이지 본문과 추출 방식을 함께 반환한다."""
    if _is_x_url(url):
        return None
    try:
        validated_url = _validate_public_url(url)
        result = None
        try:
            downloaded = _download_public_url(validated_url).decode("utf-8", errors="ignore")
        except Exception as exc:
            downloaded = ""
            logger.warning("Static web extraction fetch failed for %s: %s", validated_url, exc)

        if downloaded:
            result = _extract_web_result_from_html(validated_url, downloaded, visited=set(), stage="static")

        if not result:
            result = _extract_rendered_web_result(validated_url)

        if not result:
            return None

        text, method, document_url = result
        logger.info(
            "Web extraction success url=%s method=%s document_url=%s chars=%s",
            validated_url,
            method,
            document_url,
            len(text),
        )
        return WebExtractionResult(
            content=f"[Web Article]\nURL: {validated_url}\n\n{text}",
            method=method,
            document_url=document_url,
        )
    except Exception as e:
        logger.error(f"Web extraction error: {e}")
        return None


def extract_web_text(url: str) -> str | None:
    """trafilatura로 웹페이지 본문 추출"""
    result = extract_web_result(url)
    return result.content if result else None
