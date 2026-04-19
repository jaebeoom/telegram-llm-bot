import atexit
import logging
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

from .extractors_common import REQUEST_HEADERS, _env_flag, _env_int
from .extractors_network import _validate_public_url

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional dependency
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None

logger = logging.getLogger(__name__)

_PLAYWRIGHT_TASK_POLL_SECONDS = 1.0
_PLAYWRIGHT_FALLBACK_DISABLE_SECONDS = 300
_PLAYWRIGHT_BLOCKED_RESOURCE_TYPES = {
    "eventsource",
    "font",
    "image",
    "manifest",
    "media",
    "texttrack",
    "websocket",
}


def _playwright_fallback_enabled() -> bool:
    return _env_flag("ENABLE_PLAYWRIGHT_FALLBACK", True)


def _playwright_render_timeout_ms() -> int:
    return _env_int("PLAYWRIGHT_RENDER_TIMEOUT_MS", 12000, minimum=1000, maximum=60000)


def _playwright_idle_timeout_seconds() -> int:
    return _env_int("PLAYWRIGHT_IDLE_TIMEOUT_SECONDS", 600, minimum=30, maximum=3600)


def _playwright_settle_delay_ms() -> int:
    return _env_int("PLAYWRIGHT_SETTLE_DELAY_MS", 750, minimum=0, maximum=5000)


def _playwright_max_queue_size() -> int:
    return _env_int("PLAYWRIGHT_MAX_QUEUE_SIZE", 4, minimum=1, maximum=32)


def _playwright_queue_wait_timeout_ms() -> int:
    return _env_int("PLAYWRIGHT_QUEUE_WAIT_TIMEOUT_MS", 1000, minimum=0, maximum=10000)


class _PlaywrightRenderer:
    def __init__(self) -> None:
        self._tasks: queue.Queue[tuple[str | None, Future | None]] = queue.Queue(
            maxsize=_playwright_max_queue_size()
        )
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cooldown_until = 0.0

    def render(self, url: str) -> list[tuple[str, str]] | None:
        if sync_playwright is None:
            return None

        now = time.monotonic()
        if now < self._cooldown_until:
            return None

        task = Future()
        self._ensure_thread()
        wait_timeout_seconds = _playwright_queue_wait_timeout_ms() / 1000
        try:
            self._tasks.put((url, task), timeout=wait_timeout_seconds)
        except queue.Full:
            logger.warning(
                "Playwright render queue full for %s: size=%s timeout_ms=%s",
                url,
                self._tasks.qsize(),
                _playwright_queue_wait_timeout_ms(),
            )
            return None
        try:
            return task.result(timeout=(_playwright_render_timeout_ms() / 1000) + 5)
        except FutureTimeoutError:
            logger.warning("Playwright rendered fallback timed out while waiting for worker: %s", url)
            return None
        except Exception as exc:
            logger.warning("Playwright rendered fallback error for %s: %s", url, exc)
            return None

    def close(self) -> None:
        with self._thread_lock:
            thread = self._thread
            self._thread = None
            self._stop_event.set()
        if thread and thread.is_alive():
            self._tasks.put((None, None))
            thread.join(timeout=2)
        self._stop_event.clear()

    def _ensure_thread(self) -> None:
        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="playwright-renderer",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        playwright = None
        browser = None
        last_used_at = 0.0

        while not self._stop_event.is_set():
            try:
                url, task = self._tasks.get(timeout=_PLAYWRIGHT_TASK_POLL_SECONDS)
            except queue.Empty:
                if browser and time.monotonic() - last_used_at >= _playwright_idle_timeout_seconds():
                    browser, playwright = self._close_browser(browser, playwright)
                    logger.info("Closed idle Playwright browser after %ss", _playwright_idle_timeout_seconds())
                continue

            if url is None or task is None:
                break

            launch_failed = browser is None
            try:
                playwright, browser = self._ensure_browser(playwright, browser)
                documents = self._render_documents(browser, url)
                last_used_at = time.monotonic()
                task.set_result(documents)
            except Exception as exc:
                if launch_failed:
                    self._cooldown_until = time.monotonic() + _PLAYWRIGHT_FALLBACK_DISABLE_SECONDS
                browser, playwright = self._close_browser(browser, playwright)
                task.set_exception(exc)

        self._close_browser(browser, playwright)

    def _ensure_browser(self, playwright, browser):
        if playwright is not None and browser is not None:
            return playwright, browser
        if sync_playwright is None:
            raise RuntimeError("Playwright is not installed.")

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        logger.info("Started Playwright fallback browser")
        return playwright, browser

    def _render_documents(self, browser, url: str) -> list[tuple[str, str]]:
        context = browser.new_context(
            user_agent=REQUEST_HEADERS["User-Agent"],
            locale="ko-KR",
        )
        try:
            page = context.new_page()
            page.route("**/*", self._handle_route)
            page.goto(url, wait_until="domcontentloaded", timeout=_playwright_render_timeout_ms())
            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=min(2500, _playwright_render_timeout_ms()),
                )
            except PlaywrightTimeoutError:
                pass

            settle_delay = _playwright_settle_delay_ms()
            if settle_delay:
                page.wait_for_timeout(settle_delay)

            documents: list[tuple[str, str]] = []
            self._append_document(documents, page.url, page.content())
            for frame in page.frames:
                if frame == page.main_frame:
                    continue
                try:
                    self._append_document(documents, frame.url, frame.content())
                except Exception:
                    continue
            return documents
        finally:
            context.close()

    @staticmethod
    def _handle_route(route) -> None:
        request = route.request
        if request.resource_type in _PLAYWRIGHT_BLOCKED_RESOURCE_TYPES:
            route.abort()
            return
        try:
            _validate_public_url(request.url)
        except ValueError:
            route.abort()
            return
        route.continue_()

    @staticmethod
    def _append_document(documents: list[tuple[str, str]], document_url: str, html: str) -> None:
        cleaned_html = html.strip()
        if not cleaned_html:
            return
        if any(existing_html == cleaned_html for _, existing_html in documents):
            return
        documents.append((document_url, cleaned_html))

    @staticmethod
    def _close_browser(browser, playwright):
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass
        return None, None


_PLAYWRIGHT_RENDERER = _PlaywrightRenderer()
atexit.register(_PLAYWRIGHT_RENDERER.close)


def _render_documents_with_playwright(url: str) -> list[tuple[str, str]] | None:
    if not _playwright_fallback_enabled():
        return None
    return _PLAYWRIGHT_RENDERER.render(url)
