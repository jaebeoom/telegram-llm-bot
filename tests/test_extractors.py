from types import SimpleNamespace
from concurrent.futures import Future
import subprocess
import sys

from extractors import (
    _PlaywrightRenderer,
    extract_web_result,
    _clean_extracted_text,
    extract_tweet_from_url,
    extract_web_text,
)


def test_clean_extracted_text_filters_browser_gate_noise():
    text = "You can see a list of supported browsers in our Help Center"

    assert _clean_extracted_text(text, 200) is None


def test_extract_web_text_skips_x_urls():
    assert extract_web_text("https://x.com/example/status/123") is None


def test_importing_extractors_does_not_import_fitz():
    result = subprocess.run(
        [sys.executable, "-c", "import extractors, sys; print('fitz' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_extract_web_text_follows_iframe_when_outer_page_has_no_article(monkeypatch):
    downloaded_urls: list[str] = []
    outer_url = "https://blog.naver.com/energyinfo/224246917495"
    iframe_url = "https://blog.naver.com/PostView.naver?blogId=energyinfo&logNo=224246917495"

    def fake_download(url: str) -> bytes:
        downloaded_urls.append(url)
        if url == outer_url:
            return '<iframe id="mainFrame" src="/PostView.naver?blogId=energyinfo&logNo=224246917495"></iframe>'.encode(
                "utf-8"
            )
        if url == iframe_url:
            return '<div class="se-main-container">본문</div>'.encode("utf-8")
        raise AssertionError(f"unexpected url: {url}")

    def fake_extract(html: str) -> str | None:
        return "네이버 블로그 본문" if "se-main-container" in html else None

    monkeypatch.setattr("extractors_web._download_public_url", fake_download)
    monkeypatch.setattr("extractors_web.trafilatura.extract", fake_extract)
    monkeypatch.setattr("extractors_web._validate_public_url", lambda url: url)

    text = extract_web_text(outer_url)

    assert downloaded_urls == [outer_url, iframe_url]
    assert text == f"[Web Article]\nURL: {outer_url}\n\n네이버 블로그 본문"


def test_extract_web_text_falls_back_to_iframe_meta_description(monkeypatch):
    outer_url = "https://blog.naver.com/energyinfo/224246917495"

    def fake_download(url: str) -> bytes:
        if url == outer_url:
            return '<iframe id="mainFrame" src="/PostView.naver?blogId=energyinfo&logNo=224246917495"></iframe>'.encode(
                "utf-8"
            )
        return '<meta property="og:description" content="본문 요약">'.encode("utf-8")

    monkeypatch.setattr("extractors_web._download_public_url", fake_download)
    monkeypatch.setattr("extractors_web.trafilatura.extract", lambda html: None)
    monkeypatch.setattr("extractors_web._validate_public_url", lambda url: url)

    text = extract_web_text(outer_url)

    assert text == f"[Web Article]\nURL: {outer_url}\n\n본문 요약"


def test_extract_web_text_uses_rendered_dom_fallback_when_static_html_is_empty_shell(monkeypatch):
    url = "https://example.com/article"

    monkeypatch.setattr("extractors_web._download_public_url", lambda fetched_url: b'<div id="app"></div>')
    monkeypatch.setattr("extractors_web.trafilatura.extract", lambda html: None)
    monkeypatch.setattr(
        "extractors_web._extract_rendered_web_result",
        lambda rendered_url: ("렌더된 본문", "rendered_trafilatura", rendered_url),
    )
    monkeypatch.setattr("extractors_web._validate_public_url", lambda validated_url: validated_url)

    text = extract_web_text(url)

    assert text == f"[Web Article]\nURL: {url}\n\n렌더된 본문"


def test_extract_web_text_does_not_render_private_url(monkeypatch):
    monkeypatch.setattr("extractors_web._extract_rendered_web_result", lambda url: (_ for _ in ()).throw(AssertionError))

    assert extract_web_text("http://127.0.0.1/internal") is None


def test_extract_web_result_reports_static_readability_method(monkeypatch):
    url = "https://example.com/article"

    class FakeDocument:
        def __init__(self, html: str):
            self.html = html

        def summary(self, html_partial: bool = True) -> str:
            return "<article>정리된 본문</article>"

    def fake_extract(html: str) -> str | None:
        if "정리된 본문" in html:
            return "정리된 본문"
        return None

    monkeypatch.setattr("extractors_web.Document", FakeDocument)
    monkeypatch.setattr("extractors_web._download_public_url", lambda fetched_url: b"<html><body><div>shell</div></body></html>")
    monkeypatch.setattr("extractors_web.trafilatura.extract", fake_extract)
    monkeypatch.setattr("extractors_web._validate_public_url", lambda validated_url: validated_url)

    result = extract_web_result(url)

    assert result is not None
    assert result.method == "static_readability"
    assert result.document_url == url
    assert result.content == f"[Web Article]\nURL: {url}\n\n정리된 본문"


def test_playwright_route_aborts_private_requests(monkeypatch):
    monkeypatch.setattr(
        "extractors_rendering._validate_public_url",
        lambda url: (_ for _ in ()).throw(ValueError("blocked")),
    )

    route = SimpleNamespace(
        request=SimpleNamespace(resource_type="xhr", url="http://127.0.0.1/internal"),
        aborted=False,
        continued=False,
    )
    route.abort = lambda: setattr(route, "aborted", True)
    route.continue_ = lambda: setattr(route, "continued", True)

    _PlaywrightRenderer._handle_route(route)

    assert route.aborted is True
    assert route.continued is False


def test_playwright_route_continues_public_document_requests(monkeypatch):
    monkeypatch.setattr("extractors_rendering._validate_public_url", lambda url: url)

    route = SimpleNamespace(
        request=SimpleNamespace(resource_type="document", url="https://example.com/article"),
        aborted=False,
        continued=False,
    )
    route.abort = lambda: setattr(route, "aborted", True)
    route.continue_ = lambda: setattr(route, "continued", True)

    _PlaywrightRenderer._handle_route(route)

    assert route.aborted is False
    assert route.continued is True


def test_playwright_renderer_returns_none_when_queue_is_full(monkeypatch):
    monkeypatch.setattr("extractors_rendering.sync_playwright", object())
    monkeypatch.setattr("extractors_rendering._playwright_max_queue_size", lambda: 1)
    monkeypatch.setattr("extractors_rendering._playwright_queue_wait_timeout_ms", lambda: 0)

    renderer = _PlaywrightRenderer()
    monkeypatch.setattr(renderer, "_ensure_thread", lambda: None)
    renderer._tasks.put(("busy", Future()), block=False)

    assert renderer.render("https://example.com/article") is None


def test_extract_tweet_from_url_falls_back_to_html(monkeypatch):
    monkeypatch.setattr("extractors_content.extract_tweet", lambda tweet_id: None)
    monkeypatch.setattr(
        "extractors_content._download_public_url",
        lambda url: '<meta property="og:description" content="실제 X 게시물 본문">'.encode("utf-8"),
    )

    text = extract_tweet_from_url("https://x.com/example/status/123")

    assert text == "[X Post]\n실제 X 게시물 본문"
