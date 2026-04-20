from types import SimpleNamespace
from concurrent.futures import Future
from pathlib import Path
import subprocess
import sys
from xml.etree.ElementTree import ParseError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractors import (
    _PlaywrightRenderer,
    extract_web_result,
    _clean_extracted_text,
    extract_tweet_from_url,
    extract_web_text,
)
from extractors_content import _format_youtube_transcript_error
from extractors_content import (
    YOUTUBE_TRANSLATION_BLOCKED_FALLBACK_MESSAGE,
    _select_youtube_transcript,
    extract_youtube_transcript_result,
)


class FakeTranscript:
    def __init__(
        self,
        language_code: str,
        *,
        language: str | None = None,
        is_generated: bool = False,
        is_translatable: bool = False,
        text: str = "transcript text",
        translated=None,
        fetch_error: Exception | None = None,
    ):
        self.language_code = language_code
        self.language = language or language_code
        self.is_generated = is_generated
        self.is_translatable = is_translatable
        self.text = text
        self.translated = translated
        self.fetch_error = fetch_error

    def fetch(self):
        if self.fetch_error is not None:
            raise self.fetch_error
        return SimpleNamespace(snippets=[SimpleNamespace(text=self.text)])

    def translate(self, language_code: str):
        assert language_code == "ko"
        if self.translated is None:
            raise AssertionError("unexpected translation request")
        return self.translated


class FakeTranscriptList:
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def __iter__(self):
        return iter(self.transcripts)

    def _find(self, language_codes, *, is_generated: bool | None = None):
        from youtube_transcript_api._errors import NoTranscriptFound

        for transcript in self.transcripts:
            if is_generated is not None and transcript.is_generated != is_generated:
                continue
            if transcript.language_code in language_codes:
                return transcript
        raise NoTranscriptFound(
            video_id="video-id",
            requested_language_codes=language_codes,
            transcript_data=self,
        )

    def find_manually_created_transcript(self, language_codes):
        return self._find(language_codes, is_generated=False)

    def find_generated_transcript(self, language_codes):
        return self._find(language_codes, is_generated=True)

    def find_transcript(self, language_codes):
        return self._find(language_codes)


def test_clean_extracted_text_filters_browser_gate_noise():
    text = "You can see a list of supported browsers in our Help Center"

    assert _clean_extracted_text(text, 200) is None


def test_clean_extracted_text_filters_substack_js_shell_noise():
    text = "Home Subscriptions Profile This site requires JavaScript to run correctly. Please turn on JavaScript or unblock scripts"

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


def test_youtube_transcript_parse_error_reports_specific_status():
    status, message = _format_youtube_transcript_error(ParseError("not well-formed"))

    assert status == "transcript_parse_error"
    assert message == "공개 자막 데이터가 깨져서 읽을 수 없습니다."


def test_youtube_transcript_selection_accepts_locale_variant_for_base_language():
    transcript = FakeTranscript("en-US", language="English (United States)")
    transcript_list = FakeTranscriptList([transcript])

    selected, selection = _select_youtube_transcript(transcript_list, ("ko", "en"))

    assert selected is transcript
    assert selection == "preferred_locale_manual"


def test_youtube_transcript_uses_original_when_translation_fetch_is_blocked(monkeypatch):
    import youtube_transcript_api
    from youtube_transcript_api._errors import IpBlocked

    translated = FakeTranscript("ko", language="Korean", fetch_error=IpBlocked("video-id"))
    original = FakeTranscript(
        "es",
        language="Spanish",
        is_translatable=True,
        text="original transcript",
        translated=translated,
    )

    class FakeApi:
        def list(self, video_id):
            assert video_id == "video-id"
            return FakeTranscriptList([original])

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", FakeApi)

    result = extract_youtube_transcript_result("video-id")

    assert result.status == "ok"
    assert result.message == YOUTUBE_TRANSLATION_BLOCKED_FALLBACK_MESSAGE
    assert result.language_code == "es"
    assert result.selection == "fallback_manual"
    assert result.content == "[YouTube Transcript]\noriginal transcript"


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


def test_extract_web_result_follows_substack_app_route_canonical_url(monkeypatch):
    app_url = "https://substack.com/@writer/p-123"
    canonical_url = "https://writer.substack.com/p/post-title"
    downloaded_urls: list[str] = []

    def fake_download(url: str) -> bytes:
        downloaded_urls.append(url)
        if url == app_url:
            return (
                '<div>Home Subscriptions Profile This site requires JavaScript to run correctly. '
                'Please turn on JavaScript or unblock scripts</div>'
                '<script>{\\"canonical_url\\":\\"https://writer.substack.com/p/post-title\\"}</script>'
            ).encode("utf-8")
        if url == canonical_url:
            return "<article><h1>Actual article</h1><p>실제 글 본문</p></article>".encode("utf-8")
        raise AssertionError(f"unexpected url: {url}")

    def fake_extract(html: str) -> str | None:
        if "Actual article" in html:
            return "Actual article\n실제 글 본문"
        if "This site requires JavaScript" in html:
            return "Home Subscriptions Profile This site requires JavaScript to run correctly"
        return None

    monkeypatch.setattr("extractors_web._download_public_url", fake_download)
    monkeypatch.setattr("extractors_web._validate_public_url", lambda url: url)
    monkeypatch.setattr("extractors_web._extract_with_trafilatura", fake_extract)
    monkeypatch.setattr("extractors_web._extract_rendered_web_result", lambda url: None)

    result = extract_web_result(app_url)

    assert downloaded_urls == [app_url, canonical_url]
    assert result is not None
    assert result.document_url == canonical_url
    assert result.method == "static_trafilatura"
    assert result.content == f"[Web Article]\nURL: {app_url}\n\nActual article\n실제 글 본문"


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
