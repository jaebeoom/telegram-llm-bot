import asyncio
import re
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from extractors import WebExtractionResult
from prompt_profiles import load_prompt_profile, normalize_model_name, render_prompt_profile


def reset_bot_runtime_state() -> None:
    bot.conversations.clear()
    bot.session_histories.clear()
    bot.source_memories.clear()
    bot.active_source_sessions.clear()
    bot.inbox_context_prefetch_cache.clear()
    bot.inbox_context_prefetch_inflight.clear()
    bot.inbox_context_prefetch_tasks.clear()
    bot.inbox_context_prefetch_task = None
    bot.inbox_context_ready_list_api_available = None
    bot.session_identifiers.clear()
    bot.last_activity_at_by_session.clear()
    bot.youtube_audio_transcription_semaphore = None
    bot.clear_persistent_inbox_context_prefetch_cache()


@pytest.fixture(autouse=True)
def clear_histories(monkeypatch, tmp_path):
    monkeypatch.setattr(bot, "ENABLE_INBOX_CONTEXT_PREFETCH", False)
    monkeypatch.setattr(
        bot,
        "INBOX_CONTEXT_PREFETCH_PERSISTENT_CACHE_PATH",
        tmp_path / "inbox-context-prefetch.sqlite3",
    )
    reset_bot_runtime_state()
    yield
    reset_bot_runtime_state()


class DummyFile:
    async def download_as_bytearray(self):
        return bytearray(b"%PDF-1.4")


class DummyDocument:
    mime_type = "application/pdf"

    async def get_file(self):
        return DummyFile()


class DummyBot:
    def __init__(self, fail_draft=False, fail_draft_calls=None):
        self.fail_draft = fail_draft
        self.fail_draft_calls = set(fail_draft_calls or [])
        self.drafts = []
        self.chat_actions = []
        self._draft_call_count = 0

    async def send_message_draft(self, **kwargs):
        self._draft_call_count += 1
        if self.fail_draft or self._draft_call_count in self.fail_draft_calls:
            raise bot.TelegramError("draft unsupported")
        self.drafts.append(kwargs)
        return True

    async def send_chat_action(self, **kwargs):
        self.chat_actions.append(kwargs)
        return True


class DummyReply:
    def __init__(self, text):
        self.text = text
        self.edits = []
        self.deleted = False

    async def edit_text(self, text):
        self.text = text
        self.edits.append(text)
        return self

    async def delete(self):
        self.deleted = True
        return None


class DummyMessage:
    def __init__(
        self,
        text=None,
        caption=None,
        document=None,
        bot_instance=None,
        message_id=1,
        chat_id=999,
        message_thread_id=None,
    ):
        self.text = text
        self.caption = caption
        self.document = document
        self.replies = []
        self.reply_messages = []
        self.chat_id = chat_id
        self.message_id = message_id
        self.message_thread_id = message_thread_id
        self._bot = bot_instance or DummyBot()

    async def reply_text(self, text):
        self.replies.append(text)
        reply = DummyReply(text)
        self.reply_messages.append(reply)
        return reply

    def get_bot(self):
        return self._bot


async def _async_noop():
    return None


def session_key(user_id: int, chat_id: int = 999, message_thread_id: int | None = None):
    return (user_id, chat_id, message_thread_id or 0)


def use_delivery(monkeypatch, mode: str):
    monkeypatch.setattr(bot, "TELEGRAM_RESPONSE_DELIVERY", mode)


def test_build_context_prompt_uses_default_for_blank_text():
    assert bot.build_context_prompt("   ") == "이 내용을 한국어로 간단히 요약해줘."


def test_build_context_prompt_keeps_user_text():
    assert bot.build_context_prompt("요약 말고 핵심만") == "요약 말고 핵심만"


def test_direct_long_source_context_uses_legacy_full_context():
    key = session_key(301)
    long_context = "[Web Article]\n" + "첫 구간 성장 서술. " * 900 + "중간 구간 비용 압박. " * 900

    effective_context, source_kind, source_url, store_context = bot.resolve_source_context_for_request(
        key,
        bot.DEFAULT_CONTEXT_PROMPT,
        long_context,
        source_kind="web",
        source_url="https://example.com/article",
    )

    assert effective_context == long_context
    assert source_kind == "web"
    assert source_url == "https://example.com/article"
    assert store_context is True
    assert bot.source_memories.get(key) is None


def test_direct_context_replaces_active_inbox_context_mode():
    key = session_key(301)
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="web",
        source_url="https://example.com/inbox",
        title="Inbox Article",
        text="[Web Article]\nInbox 본문",
        remaining_ready_count=0,
    )
    bot.apply_inbox_context_source_to_session(key, source)

    effective_context, source_kind, source_url, store_context = bot.resolve_source_context_for_request(
        key,
        bot.DEFAULT_CONTEXT_PROMPT,
        "[Web Article]\n직접 보낸 본문",
        source_kind="web",
        source_url="https://example.com/direct",
    )

    assert effective_context == "[Web Article]\n직접 보낸 본문"
    assert source_kind == "web"
    assert source_url == "https://example.com/direct"
    assert store_context is True
    assert key not in bot.active_source_sessions


def test_search_context_can_preserve_active_inbox_context():
    key = session_key(301)
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="web",
        source_url="https://example.com/inbox",
        title="Inbox Article",
        text="[Web Article]\nInbox 본문",
        remaining_ready_count=0,
    )
    bot.apply_inbox_context_source_to_session(key, source)

    effective_context, source_kind, source_url, store_context = bot.resolve_source_context_for_request(
        key,
        "이 글의 현재 소비자 반응은?",
        "[Web Search Results]\n검색 결과",
        preserve_active_source_context=True,
    )

    assert effective_context.startswith("[Combined Source And Web Search Context]")
    assert "[Prior Source Context]" in effective_context
    assert "[Current Web Search Context]" in effective_context
    assert "Inbox 본문" in effective_context
    assert "검색 결과" in effective_context
    assert source_kind == "web"
    assert source_url == "https://example.com/inbox"
    assert store_context is False
    assert key in bot.active_source_sessions


def test_inbox_initial_source_context_uses_full_context_while_registering_memory():
    key = session_key(301)
    long_context = "[Web Article]\n" + "첫 구간 성장 서술. " * 900 + "중간 구간 비용 압박. " * 900

    effective_context, source_kind, source_url, store_context = bot.resolve_source_context_for_request(
        key,
        bot.DEFAULT_CONTEXT_PROMPT,
        long_context,
        source_kind="web",
        source_url="https://example.com/article",
        use_source_memory_for_context=True,
    )

    assert effective_context == long_context
    assert source_kind == "web"
    assert source_url == "https://example.com/article"
    assert store_context is True
    assert bot.source_memories[key][-1].content == long_context.strip()
    assert key in bot.active_source_sessions


def test_inbox_source_context_is_chunked_and_retrieved_representatively():
    key = session_key(301)
    long_context = "[Web Article]\n" + "첫 구간 성장 서술. " * 900 + "중간 구간 비용 압박. " * 900
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="web",
        source_url="https://example.com/article",
        title="Article",
        text=long_context,
        remaining_ready_count=0,
    )

    bot.apply_inbox_context_source_to_session(key, source)
    effective_context, source_kind, source_url, store_context = bot.resolve_source_context_for_request(
        key,
        bot.DEFAULT_CONTEXT_PROMPT,
        "",
    )

    assert effective_context.startswith("[Retrieved Source Context]")
    assert "Selection mode: representative" in effective_context
    assert "Chunks selected:" in effective_context
    assert len(effective_context) < len(long_context)
    assert source_kind == "web"
    assert source_url == "https://example.com/article"
    assert store_context is False
    assert bot.source_memories[key][-1].content == long_context.strip()


def test_source_followup_retrieves_relevant_chunk_without_storing_retrieval_prompt(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(302)
    long_context = (
        "[Web Article]\n"
        + "초반부 매출 성장과 제품 전략. " * 800
        + "수요 신호는 재고 회전, 예약 주문, 기업 고객 전환율에서 확인된다. " * 80
        + "후반부 비용 구조와 리스크. " * 800
    )
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="web",
        source_url="https://example.com/article",
        title="Article",
        text=long_context,
        remaining_ready_count=0,
    )
    bot.apply_inbox_context_source_to_session(key, source)

    effective_context, source_kind, source_url, store_context = bot.resolve_source_context_for_request(
        key,
        "이 글에서 수요 신호만",
        "",
    )
    messages = bot.prepare_messages(
        key,
        "이 글에서 수요 신호만",
        effective_context,
        source_kind=source_kind,
        source_url=source_url,
        store_context_in_history=store_context,
    )

    assert "Selection mode: relevant" in messages[-1]["content"]
    assert "재고 회전" in messages[-1]["content"]
    assert bot.conversations[key][-1] == {"role": "user", "content": "이 글에서 수요 신호만"}
    assert bot.session_histories[key][-1]["source_kind"] == "web"
    assert bot.session_histories[key][-1]["source_url"] == "https://example.com/article"


def test_legacy_source_followup_detection_uses_recent_window():
    key = session_key(303)
    bot.append_history_message(
        key,
        "user",
        "[Web Article]\n본문\n\n[Response Rules]\n...\n\n[User Question]\n요약",
    )
    bot.append_history_message(key, "assistant", "첫 요약")
    for turn in range(5):
        bot.append_history_message(key, "user", f"일반 질문 {turn}")
        bot.append_history_message(key, "assistant", f"일반 답변 {turn}")

    assert bot.should_apply_source_followup_rules(key, "이건 어때?") is False


def test_active_inbox_context_applies_without_explicit_reference():
    key = session_key(304)
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="영상 제목",
        text="[YouTube Transcript]\n본문",
        remaining_ready_count=0,
    )

    memory = bot.apply_inbox_context_source_to_session(key, source)

    assert memory.source_kind == "youtube"
    assert key in bot.active_source_sessions
    assert bot.should_apply_source_followup_rules(key, "요약해줘") is True


def test_parse_inbox_context_source_payload():
    parsed = bot.parse_inbox_context_source_payload(
        {
            "source": {
                "id": 7,
                "kind": "web",
                "title": "Article",
                "url": "https://example.com/article",
                "text": "[Web Article]\n본문",
            },
            "remaining_ready_count": 3,
        }
    )

    assert parsed == bot.InboxContextSource(
        source_id=7,
        source_kind="web",
        source_url="https://example.com/article",
        title="Article",
        text="[Web Article]\n본문",
        remaining_ready_count=3,
    )


def test_parse_inbox_context_sources_payload():
    parsed = bot.parse_inbox_context_sources_payload(
        {
            "sources": [
                {
                    "id": 7,
                    "kind": "youtube",
                    "title": "Video",
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                    "text": "[YouTube Transcript]\n본문",
                },
                {
                    "id": 8,
                    "kind": "web",
                    "title": "Article",
                    "url": "https://example.com/article",
                    "text": "[Web Article]\n본문",
                },
            ],
            "ready_count": 4,
        }
    )

    assert parsed == [
        bot.InboxContextSource(
            source_id=7,
            source_kind="youtube",
            source_url="https://www.youtube.com/watch?v=abcdefghijk",
            title="Video",
            text="[YouTube Transcript]\n본문",
            remaining_ready_count=3,
        ),
        bot.InboxContextSource(
            source_id=8,
            source_kind="web",
            source_url="https://example.com/article",
            title="Article",
            text="[Web Article]\n본문",
            remaining_ready_count=2,
        ),
    ]


def test_build_inbox_context_applied_reply_includes_summary_and_url():
    source = bot.InboxContextSource(
        source_id=7,
        source_kind="web",
        source_url="https://example.com/article",
        title="Article",
        text="[Web Article]\n본문",
        remaining_ready_count=3,
    )

    reply = bot.build_inbox_context_applied_reply(source, "핵심 요약입니다.")

    assert "컨텍스트 적용됨: #7 Article" in reply
    assert "출처: https://example.com/article" in reply
    assert "남은 준비된 컨텍스트 큐: 3개" in reply
    assert "요약\n핵심 요약입니다." in reply


def test_build_inbox_context_processing_reply_includes_title_and_url():
    source = bot.InboxContextSource(
        source_id=8,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video Title",
        text="[YouTube Transcript]\n본문",
        remaining_ready_count=0,
    )

    reply = bot.build_inbox_context_processing_reply(source)

    assert "컨텍스트 준비 중: #8 Video Title" in reply
    assert "종류: YouTube" in reply
    assert "출처: https://www.youtube.com/watch?v=abcdefghijk" in reply
    assert "먼저 읽은 본문:" in reply
    assert "YouTube 자막을 확인해 필요하면 오디오 전사로 보강합니다." in reply


def test_build_inbox_context_status_reply_omits_repeated_url():
    source = bot.InboxContextSource(
        source_id=8,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video Title",
        text="[YouTube Transcript]\n본문",
        remaining_ready_count=0,
    )

    reply = bot.build_inbox_context_status_reply(source, enhanced=True)

    assert "컨텍스트 적용됨: #8 Video Title" in reply
    assert "출처:" not in reply
    assert "본문:" in reply
    assert "YouTube 원문으로 본문을 보강했습니다." in reply
    assert "남은 준비된 컨텍스트 큐: 0개" in reply


def test_resolve_auto_search_decision_skips_source_local_active_context(monkeypatch):
    key = session_key(305)
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="영상 제목",
        text="[YouTube Transcript]\n본문",
        remaining_ready_count=0,
    )
    bot.apply_inbox_context_source_to_session(key, source)

    monkeypatch.setattr(bot, "tavily", object())
    monkeypatch.setattr(
        bot,
        "classify_recency_need",
        lambda *_args, **_kwargs: pytest.fail("source-local summary should not classify"),
    )

    decision = bot.resolve_auto_search_decision("방금 넣은 것 무슨 내용인지 요약해줘", key)

    assert decision == bot.AutoSearchDecision(
        False,
        reason="active source local follow-up",
        source="source_context",
    )


def test_resolve_auto_search_decision_uses_classifier_for_active_source_context(monkeypatch):
    key = session_key(305)
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="영상 제목",
        text="[YouTube Transcript]\n본문",
        remaining_ready_count=0,
    )
    bot.apply_inbox_context_source_to_session(key, source)

    monkeypatch.setattr(bot, "tavily", object())
    monkeypatch.setattr(
        bot,
        "classify_recency_need",
        lambda user_message, session_key=None: bot.AutoSearchDecision(
            False,
            reason="classifier checked active source",
            source="classifier",
        ),
    )

    decision = bot.resolve_auto_search_decision("이 주장과 내 논리가 충돌하는지 봐줘", key)

    assert decision == bot.AutoSearchDecision(
        False,
        reason="classifier checked active source",
        source="classifier",
    )


def test_resolve_auto_search_decision_uses_guardrail_inside_active_source_context(monkeypatch):
    key = session_key(305)
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="영상 제목",
        text="[YouTube Transcript]\n본문",
        remaining_ready_count=0,
    )
    bot.apply_inbox_context_source_to_session(key, source)

    monkeypatch.setattr(bot, "tavily", object())
    monkeypatch.setattr(
        bot,
        "classify_recency_need",
        lambda *_args, **_kwargs: pytest.fail("guardrail should run before classifier"),
    )

    decision = bot.resolve_auto_search_decision("이 주제의 현재 시장 반응은?", key)

    assert decision.needs_search is True
    assert decision.source == "guardrail"


def test_resolve_auto_search_decision_uses_classifier_for_regular_context_followup(monkeypatch):
    key = session_key(306)
    bot.conversations[key] = [
        {
            "role": "user",
            "content": "[Web Article]\n원문\n\n[Response Rules]\n...\n\n[User Question]\n요약",
        },
        {"role": "assistant", "content": "첫 요약"},
    ]

    monkeypatch.setattr(bot, "tavily", object())
    monkeypatch.setattr(
        bot,
        "classify_recency_need",
        lambda user_message, session_key=None: bot.AutoSearchDecision(
            False,
            reason="classifier checked",
            source="classifier",
        ),
    )

    decision = bot.resolve_auto_search_decision("수요 신호만", key)

    assert decision == bot.AutoSearchDecision(False, reason="classifier checked", source="classifier")


def test_normalize_source_url_canonicalizes_x_and_youtube():
    assert bot.normalize_source_url("https://twitter.com/test/status/12345?ref=share", "x") == (
        "https://x.com/test/status/12345"
    )
    assert bot.normalize_source_url("https://youtu.be/abcdefghijk?si=share", "youtube") == (
        "https://www.youtube.com/watch?v=abcdefghijk"
    )
    assert bot.normalize_source_url("https://example.com/doc.pdf?token=secret#page=2", "pdf") == (
        "https://example.com/doc.pdf"
    )


def test_handle_document_without_caption_uses_default_summary_prompt(monkeypatch):
    calls = []

    async def fake_stream_reply(
        update,
        user_id,
        user_message,
        search_context="",
        source="context",
        source_kind=None,
        source_url=None,
    ):
        calls.append((user_id, user_message, search_context, source_kind, source_url))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "extract_pdf_text", lambda file_bytes: "[PDF Document]\n본문")
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=123),
        message=DummyMessage(caption=None, document=DummyDocument()),
    )

    asyncio.run(bot.handle_document(update, None))

    assert calls == [(123, bot.DEFAULT_CONTEXT_PROMPT, "[PDF Document]\n본문", None, None)]


def test_handle_document_with_extract_caption_returns_pdf_text_without_llm(monkeypatch):
    async def fail_stream_reply(*_args, **_kwargs):
        pytest.fail("extract-only PDF uploads should not call the LLM")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "extract_pdf_text", lambda file_bytes: "[PDF Document]\n본문")
    monkeypatch.setattr(bot, "stream_reply", fail_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=124),
        message=DummyMessage(caption="/extract", document=DummyDocument()),
    )

    asyncio.run(bot.handle_document(update, None))

    assert update.message.replies == [
        "📄 PDF 읽는 중...",
        "PDF 텍스트\n\n본문",
    ]
    assert bot.conversations.get(session_key(124)) is None


def test_handle_message_with_x_url_uses_context_reply(monkeypatch):
    calls = []

    async def fake_stream_reply(
        update,
        user_id,
        user_message,
        search_context="",
        source="context",
        source_kind=None,
        source_url=None,
    ):
        calls.append((user_id, user_message, search_context, source_kind, source_url))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "extract_tweet_from_url", lambda url: "[X Post]\n본문")
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=7),
        message=DummyMessage(text="https://x.com/test/status/123"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert calls == [
        (7, bot.DEFAULT_CONTEXT_PROMPT, "[X Post]\n본문", "x", "https://x.com/test/status/123")
    ]
    assert update.message.replies[0] == "📰 X 피드 읽는 중..."


def test_handle_message_with_youtube_share_url_uses_canonical_source(monkeypatch):
    calls = []

    async def fake_stream_reply(
        update,
        user_id,
        user_message,
        search_context="",
        source="context",
        source_kind=None,
        source_url=None,
    ):
        calls.append((user_id, user_message, search_context, source, source_kind, source_url))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content="[YouTube Transcript]\n본문",
            status="ok",
            message="",
        ),
    )
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=8),
        message=DummyMessage(text="https://youtu.be/abcdefghijk?si=share 이거 요약"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert calls == [
        (
            8,
            "이거 요약",
            "[YouTube Transcript]\n본문",
            "youtube",
            "youtube",
            "https://www.youtube.com/watch?v=abcdefghijk",
        )
    ]
    assert update.message.replies[0] == "🎬 YouTube 스크립트 확인 중..."
    assert update.message.reply_messages[0].text == (
        "컨텍스트 준비 완료: YouTube\n"
        "본문: 23자\n"
        "답변 생성 중..."
    )


def test_handle_message_with_extract_suffix_returns_youtube_transcript_without_llm(monkeypatch):
    async def fail_stream_reply(*_args, **_kwargs):
        pytest.fail("extract-only requests should not call the LLM")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content="[YouTube Transcript]\n원문 본문",
            status="ok",
            message="",
        ),
    )
    monkeypatch.setattr(bot, "stream_reply", fail_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=18),
        message=DummyMessage(text="https://youtu.be/abcdefghijk /extract"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert update.message.replies == [
        "🎬 YouTube 스크립트 확인 중...",
        "YouTube 스크립트\n\n원문 본문",
    ]
    assert update.message.reply_messages[0].text == (
        "컨텍스트 준비 완료: YouTube\n"
        "본문: 26자\n"
        "원문 전송 중..."
    )
    assert bot.conversations.get(session_key(18)) is None
    assert bot.session_histories[session_key(18)][-1] == {
        "role": "assistant",
        "content": "YouTube 스크립트\n\n원문 본문",
    }


def test_handle_message_with_e_suffix_returns_youtube_transcript_without_llm(monkeypatch):
    async def fail_stream_reply(*_args, **_kwargs):
        pytest.fail("extract-only requests should not call the LLM")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content="[YouTube Transcript]\n원문 본문",
            status="ok",
            message="",
        ),
    )
    monkeypatch.setattr(bot, "stream_reply", fail_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=20),
        message=DummyMessage(text="https://youtu.be/abcdefghijk /e"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert update.message.replies == [
        "🎬 YouTube 스크립트 확인 중...",
        "YouTube 스크립트\n\n원문 본문",
    ]


def test_handle_message_reports_youtube_transcript_success_notice(monkeypatch):
    async def fail_stream_reply(*_args, **_kwargs):
        pytest.fail("extract-only requests should not call the LLM")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content="[YouTube Transcript]\n원본 영어 자막",
            status="ok",
            message="YouTube가 한국어 번역 자막 요청을 차단해 원본 자막을 사용했습니다.",
        ),
    )
    monkeypatch.setattr(bot, "stream_reply", fail_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=25),
        message=DummyMessage(text="https://youtu.be/abcdefghijk /e"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert update.message.replies == [
        "🎬 YouTube 스크립트 확인 중...",
        "YouTube 스크립트\n\n원본 영어 자막",
    ]
    assert update.message.reply_messages[0].text == (
        "컨텍스트 준비 완료: YouTube\n"
        "본문: 29자\n"
        "참고: YouTube가 한국어 번역 자막 요청을 차단해 원본 자막을 사용했습니다.\n"
        "원문 전송 중..."
    )


def test_handle_message_reports_specific_youtube_transcript_failure(monkeypatch):
    async def fail_stream_reply(*_args, **_kwargs):
        pytest.fail("missing YouTube transcript should not call the LLM")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION", False)
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content=None,
            status="transcripts_disabled",
            message="이 영상은 YouTube에서 공개 스크립트/자막이 비활성화되어 있습니다.",
        ),
    )
    monkeypatch.setattr(bot, "stream_reply", fail_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=21),
        message=DummyMessage(text="https://youtu.be/abcdefghijk 요약"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert update.message.replies == ["🎬 YouTube 스크립트 확인 중..."]
    assert update.message.reply_messages[0].text == "⚠️ 이 영상은 YouTube에서 공개 스크립트/자막이 비활성화되어 있습니다."


def test_handle_message_auto_starts_youtube_audio_transcription_when_enabled(monkeypatch):
    calls = []

    async def fake_execute_youtube_audio_transcription(update, pending, status_message=None):
        assert pending.video_id == "abcdefghijk"
        assert pending.user_message == "요약"
        assert pending.duration == 5400
        assert status_message is not None
        return (
            {
                "ok": True,
                "status": "ok",
                "message": "transcribed",
                "content": "[YouTube Transcript]\n전사 본문",
            },
            1234,
        )

    async def fake_stream_context_reply(
        update,
        user_id,
        user_message,
        search_context,
        source="context",
        source_kind=None,
        source_url=None,
    ):
        calls.append((user_id, user_message, search_context, source, source_kind, source_url))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION", True)
    monkeypatch.setattr(bot.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        bot,
        "fetch_youtube_audio_metadata_for_prompt",
        lambda video_id: (
            {
                "title": "Long interview",
                "channel": "Test Channel",
                "duration": 5400,
            },
            True,
            "",
        ),
    )
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content=None,
            status="transcripts_disabled",
            message="이 영상은 YouTube에서 공개 스크립트/자막이 비활성화되어 있습니다.",
        ),
    )
    monkeypatch.setattr(bot, "execute_youtube_audio_transcription", fake_execute_youtube_audio_transcription)
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=22),
        message=DummyMessage(text="https://youtu.be/abcdefghijk 요약"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert update.message.replies == ["🎬 YouTube 스크립트 확인 중..."]
    assert update.message.reply_messages[0].text == (
        "컨텍스트 준비 완료: YouTube\n"
        "본문: 26자\n"
        "답변 생성 중..."
    )
    assert calls == [
        (
            22,
            "요약",
            "[YouTube Transcript]\n전사 본문",
            "youtube_audio",
            "youtube",
            "https://www.youtube.com/watch?v=abcdefghijk",
        )
    ]


def test_youtube_transcript_parse_error_is_audio_fallback_eligible():
    assert "transcript_parse_error" in bot.YOUTUBE_TRANSCRIPTION_FALLBACK_STATUSES


def test_youtube_transcript_request_blocked_is_audio_fallback_eligible():
    assert "request_blocked" in bot.YOUTUBE_TRANSCRIPTION_FALLBACK_STATUSES


def test_run_youtube_audio_transcription_worker_reads_large_final_payload(monkeypatch):
    script = (
        "import json\n"
        "print(json.dumps({'ok': True, 'status': 'ok', 'message': 'transcribed', 'content': '가' * 120000}), flush=True)\n"
    )
    pending = bot.PendingYouTubeTranscription(
        video_id="abcdefghijk",
        youtube_url="https://youtu.be/abcdefghijk",
        canonical_youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        user_message="요약",
        extract_only_requested=False,
        requested_at=bot.time.time(),
        failure_status="transcripts_disabled",
        failure_message="자막 비활성화",
    )
    monkeypatch.setattr(bot, "youtube_audio_worker_command", lambda *_args: [sys.executable, "-c", script])
    monkeypatch.setattr(bot, "YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS", 2)

    update = SimpleNamespace(message=DummyMessage(text="1"))

    result = asyncio.run(bot.run_youtube_audio_transcription_worker(update, pending))

    assert result["status"] == "ok"
    assert len(result["content"]) == 120000


def test_run_youtube_audio_transcription_worker_drains_stderr(monkeypatch):
    script = (
        "import json, sys\n"
        "sys.stderr.write('x' * 200000)\n"
        "sys.stderr.flush()\n"
        "print(json.dumps({'ok': False, 'status': 'error', 'message': 'worker failed'}), flush=True)\n"
    )
    pending = bot.PendingYouTubeTranscription(
        video_id="abcdefghijk",
        youtube_url="https://youtu.be/abcdefghijk",
        canonical_youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        user_message="요약",
        extract_only_requested=False,
        requested_at=bot.time.time(),
        failure_status="transcripts_disabled",
        failure_message="자막 비활성화",
    )
    monkeypatch.setattr(bot, "youtube_audio_worker_command", lambda *_args: [sys.executable, "-c", script])
    monkeypatch.setattr(bot, "YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS", 2)

    update = SimpleNamespace(message=DummyMessage(text="1"))

    result = asyncio.run(bot.run_youtube_audio_transcription_worker(update, pending))

    assert result["status"] == "error"
    assert result["message"] == "worker failed"


def test_run_youtube_audio_transcription_worker_reports_stall(monkeypatch):
    script = (
        "import json, time\n"
        "print(json.dumps({'event': 'chunk_done', 'video_id': 'abcdefghijk', 'index': 6, 'total': 8}), flush=True)\n"
        "time.sleep(60)\n"
    )
    pending = bot.PendingYouTubeTranscription(
        video_id="abcdefghijk",
        youtube_url="https://youtu.be/abcdefghijk",
        canonical_youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        user_message="요약",
        extract_only_requested=False,
        requested_at=bot.time.time(),
        failure_status="transcripts_disabled",
        failure_message="자막 비활성화",
    )
    monkeypatch.setattr(bot, "youtube_audio_worker_command", lambda *_args: [sys.executable, "-c", script])
    monkeypatch.setattr(bot, "YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS", 0.05)

    update = SimpleNamespace(message=DummyMessage(text="1"))

    result = asyncio.run(bot.run_youtube_audio_transcription_worker(update, pending))

    assert result["status"] == "stalled"
    assert "마지막 진행: 6/8" in result["message"]
    assert update.message.replies == ["🎙️ 오디오 전사 진행 중... 6/8"]


def test_handle_extract_command_returns_web_text_without_llm(monkeypatch):
    async def fail_stream_reply(*_args, **_kwargs):
        pytest.fail("extract-only commands should not call the LLM")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_web_result",
        lambda url: WebExtractionResult(
            content="[Web Article]\n본문",
            method="static_trafilatura",
            document_url="https://example.com/article",
        ),
    )
    monkeypatch.setattr(bot, "stream_reply", fail_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=19),
        message=DummyMessage(text="/extract https://example.com/article"),
    )
    context = SimpleNamespace(args=["https://example.com/article"])

    asyncio.run(bot.handle_extract(update, context))

    assert update.message.replies == [
        "📖 웹페이지 읽는 중...",
        "웹페이지 본문\n\n본문",
    ]


def test_handle_inbox_context_applies_source_and_consumes(monkeypatch):
    consumed = []
    stream_calls = []
    source = bot.InboxContextSource(
        source_id=12,
        source_kind="web",
        source_url="https://example.com/article",
        title="Article",
        text="[Web Article]\n본문",
        remaining_ready_count=2,
    )

    async def fake_stream_context_reply(update, user_id, user_message, search_context, **kwargs):
        stream_calls.append((user_id, user_message, search_context, kwargs))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", lambda: source)
    monkeypatch.setattr(bot, "mark_inbox_context_source_consumed", lambda source_id: consumed.append(source_id))
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=31),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    key = session_key(31)
    assert consumed == [12]
    assert bot.source_memories[key][-1].source_url == "https://example.com/article"
    assert key in bot.active_source_sessions
    assert update.message.replies == [
        bot.build_inbox_context_processing_reply(source),
        bot.build_inbox_context_status_reply(source),
    ]
    assert stream_calls == [
        (
            31,
            bot.DEFAULT_CONTEXT_PROMPT,
            "[Web Article]\n본문",
            {
                "source": "inbox_context",
                "source_kind": "web",
                "source_url": "https://example.com/article",
            },
        )
    ]


def test_handle_inbox_context_uses_prefetched_source_and_summary(monkeypatch):
    consumed = []
    stream_calls = []
    source = bot.InboxContextSource(
        source_id=15,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n짧은 inbox 본문",
        remaining_ready_count=1,
    )
    prefetched_source = bot.InboxContextSource(
        source_id=15,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n긴 오디오 전사 본문",
        remaining_ready_count=1,
    )
    bot.inbox_context_prefetch_cache[15] = bot.PrefetchedInboxContextSource(
        source=prefetched_source,
        enhanced=True,
        cached_at=bot.time.time(),
        initial_reply="미리 만든 요약",
        original_source=source,
    )

    async def fake_stream_context_reply(*args, **kwargs):
        stream_calls.append((args, kwargs))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", lambda: source)
    monkeypatch.setattr(bot, "mark_inbox_context_source_consumed", lambda source_id: consumed.append(source_id))
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)
    monkeypatch.setattr(
        bot,
        "enhance_inbox_youtube_context_source",
        lambda *_args, **_kwargs: pytest.fail("prefetched source should skip hydration"),
    )

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=35),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    key = session_key(35)
    assert consumed == [15]
    assert stream_calls == []
    assert bot.source_memories[key][-1].content == "[YouTube Transcript]\n긴 오디오 전사 본문"
    assert bot.session_histories[key][-1] == {"role": "assistant", "content": "미리 만든 요약"}
    assert update.message.replies == [
        bot.build_inbox_context_processing_reply(source),
        bot.build_inbox_context_status_reply(prefetched_source, enhanced=True),
        "미리 만든 요약",
    ]


def test_handle_inbox_context_uses_persistent_prefetched_source_and_summary(monkeypatch):
    consumed = []
    stream_calls = []
    source = bot.InboxContextSource(
        source_id=16,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n짧은 inbox 본문",
        remaining_ready_count=0,
    )
    prefetched_source = bot.InboxContextSource(
        source_id=16,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n긴 오디오 전사 본문",
        remaining_ready_count=0,
    )
    bot.persist_prefetched_inbox_context_source(
        bot.PrefetchedInboxContextSource(
            source=prefetched_source,
            enhanced=True,
            cached_at=bot.time.time(),
            initial_reply="디스크 캐시 요약",
            original_source=source,
        )
    )

    async def fake_stream_context_reply(*args, **kwargs):
        stream_calls.append((args, kwargs))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", lambda: source)
    monkeypatch.setattr(bot, "mark_inbox_context_source_consumed", lambda source_id: consumed.append(source_id))
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)
    monkeypatch.setattr(
        bot,
        "enhance_inbox_youtube_context_source",
        lambda *_args, **_kwargs: pytest.fail("persistent prefetched source should skip hydration"),
    )

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=36),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    key = session_key(36)
    assert consumed == [16]
    assert stream_calls == []
    assert bot.source_memories[key][-1].content == "[YouTube Transcript]\n긴 오디오 전사 본문"
    assert bot.session_histories[key][-1] == {"role": "assistant", "content": "디스크 캐시 요약"}
    assert update.message.replies == [
        bot.build_inbox_context_processing_reply(source),
        bot.build_inbox_context_status_reply(prefetched_source, enhanced=True),
        "디스크 캐시 요약",
    ]
    assert bot.load_persistent_prefetched_inbox_context_source(16) is None


def test_handle_inbox_context_revalidates_persistent_prefetched_source(monkeypatch):
    consumed = []
    enhanced_calls = []
    source = bot.InboxContextSource(
        source_id=17,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n현재 inbox 본문",
        remaining_ready_count=0,
    )
    stale_source = bot.InboxContextSource(
        source_id=17,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n오래된 inbox 본문",
        remaining_ready_count=0,
    )
    hydrated_source = bot.InboxContextSource(
        source_id=17,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n실시간 재보강 본문",
        remaining_ready_count=0,
    )
    bot.persist_prefetched_inbox_context_source(
        bot.PrefetchedInboxContextSource(
            source=stale_source,
            enhanced=True,
            cached_at=bot.time.time(),
            initial_reply="stale summary",
            original_source=stale_source,
        )
    )

    async def fake_enhance(*_args, **_kwargs):
        enhanced_calls.append(True)
        return hydrated_source, True

    async def fake_stream_context_reply(update, user_id, prompt, search_context, **kwargs):
        await update.message.reply_text("실시간 요약")

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", lambda: source)
    monkeypatch.setattr(bot, "mark_inbox_context_source_consumed", lambda source_id: consumed.append(source_id))
    monkeypatch.setattr(bot, "enhance_inbox_youtube_context_source", fake_enhance)
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=37),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    key = session_key(37)
    assert consumed == [17]
    assert enhanced_calls == [True]
    assert bot.source_memories[key][-1].content == "[YouTube Transcript]\n실시간 재보강 본문"
    assert update.message.replies == [
        bot.build_inbox_context_processing_reply(source),
        bot.build_inbox_context_status_reply(hydrated_source, enhanced=True),
        "실시간 요약",
    ]
    assert bot.load_persistent_prefetched_inbox_context_source(17) is None


def test_handle_inbox_context_reports_empty_queue(monkeypatch):
    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", lambda: None)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=32),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    assert update.message.replies == [
        "가져올 준비된 컨텍스트가 없어요. Inbox bot에 URL /ctx 로 먼저 넣어두면 됩니다."
    ]


def test_handle_inbox_context_keeps_typing_visible_while_waiting(monkeypatch):
    events = []
    consumed = []
    stream_calls = []
    source = bot.InboxContextSource(
        source_id=13,
        source_kind="web",
        source_url="https://example.com/article",
        title="Article",
        text="[Web Article]\n본문",
        remaining_ready_count=0,
    )

    async def fake_keep_typing(update, stop_event):
        events.append("started")
        await stop_event.wait()
        events.append("stopped")

    def slow_fetch():
        time.sleep(0.01)
        return source

    async def fake_stream_context_reply(update, user_id, user_message, search_context, **kwargs):
        stream_calls.append((user_id, user_message, search_context, kwargs))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "keep_typing_until_visible", fake_keep_typing)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", slow_fetch)
    monkeypatch.setattr(bot, "mark_inbox_context_source_consumed", lambda source_id: consumed.append(source_id))
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=33),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    assert events == ["started", "stopped"]
    assert consumed == [13]
    assert update.message.replies == [
        bot.build_inbox_context_processing_reply(source),
        bot.build_inbox_context_status_reply(source),
    ]
    assert stream_calls


def test_handle_inbox_context_enhances_youtube_with_audio_fallback(monkeypatch):
    consumed = []
    stream_calls = []
    source = bot.InboxContextSource(
        source_id=14,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n짧은 inbox 본문",
        remaining_ready_count=1,
    )

    def fake_build_pending(video_id, youtube_url, canonical_url, user_message, extract_only, yt_result):
        assert video_id == "abcdefghijk"
        assert canonical_url == "https://www.youtube.com/watch?v=abcdefghijk"
        assert user_message == bot.DEFAULT_CONTEXT_PROMPT
        assert extract_only is False
        assert yt_result.status == "request_blocked"
        return (
            bot.PendingYouTubeTranscription(
                video_id=video_id,
                youtube_url=youtube_url,
                canonical_youtube_url=canonical_url,
                user_message=user_message,
                extract_only_requested=False,
                requested_at=bot.time.time(),
                failure_status=yt_result.status,
                failure_message=yt_result.message,
            ),
            None,
        )

    async def fake_execute_youtube_audio_transcription(update, pending):
        return (
            {
                "ok": True,
                "status": "ok",
                "message": "transcribed",
                "content": "[YouTube Transcript]\n긴 오디오 전사 본문",
            },
            1234,
        )

    async def fake_stream_context_reply(update, user_id, user_message, search_context, **kwargs):
        stream_calls.append((user_id, user_message, search_context, kwargs))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "fetch_next_inbox_context_source", lambda: source)
    monkeypatch.setattr(bot, "mark_inbox_context_source_consumed", lambda source_id: consumed.append(source_id))
    monkeypatch.setattr(
        bot,
        "extract_youtube_transcript_result",
        lambda video_id: bot.YouTubeTranscriptExtractionResult(
            content=None,
            status="request_blocked",
            message="YouTube가 현재 스크립트 요청을 차단했습니다.",
        ),
    )
    monkeypatch.setattr(bot, "build_pending_youtube_transcription", fake_build_pending)
    monkeypatch.setattr(bot, "execute_youtube_audio_transcription", fake_execute_youtube_audio_transcription)
    monkeypatch.setattr(bot, "stream_context_reply", fake_stream_context_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=34),
        message=DummyMessage(text="/ctx"),
    )
    context = SimpleNamespace(args=[])

    asyncio.run(bot.handle_inbox_context(update, context))

    enhanced_source = bot.InboxContextSource(
        source_id=14,
        source_kind="youtube",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Video",
        text="[YouTube Transcript]\n긴 오디오 전사 본문",
        remaining_ready_count=1,
    )
    assert consumed == [14]
    assert update.message.replies == [
        bot.build_inbox_context_processing_reply(source),
        "🎙️ 공개 자막을 가져오지 못해 오디오 전사를 시작합니다.",
        bot.build_inbox_context_status_reply(enhanced_source, enhanced=True),
    ]
    assert stream_calls == [
        (
            34,
            bot.DEFAULT_CONTEXT_PROMPT,
            "[YouTube Transcript]\n긴 오디오 전사 본문",
            {
                "source": "inbox_context",
                "source_kind": "youtube",
                "source_url": "https://www.youtube.com/watch?v=abcdefghijk",
            },
        )
    ]
    assert bot.source_memories[session_key(34)][-1].content == "[YouTube Transcript]\n긴 오디오 전사 본문"


def test_handle_e_command_returns_web_text_without_llm(monkeypatch):
    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_web_result",
        lambda url: WebExtractionResult(
            content="[Web Article]\n본문",
            method="static_trafilatura",
            document_url="https://example.com/article",
        ),
    )

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=21),
        message=DummyMessage(text="/e https://example.com/article"),
    )
    context = SimpleNamespace(args=["https://example.com/article"])

    asyncio.run(bot.handle_extract(update, context))

    assert update.message.replies == [
        "📖 웹페이지 읽는 중...",
        "웹페이지 본문\n\n본문",
    ]


def test_handle_message_with_web_url_logs_extraction_method(monkeypatch):
    calls = []
    metrics = []

    async def fake_stream_reply(
        update,
        user_id,
        user_message,
        search_context="",
        source="context",
        source_kind=None,
        source_url=None,
    ):
        calls.append((user_id, user_message, search_context, source, source_kind, source_url))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "extract_web_result",
        lambda url: WebExtractionResult(
            content="[Web Article]\nURL: https://example.com\n\n본문",
            method="rendered_readability",
            document_url="https://example.com/article",
        ),
    )
    monkeypatch.setattr(
        bot,
        "log_stage_metrics",
        lambda stage, user_id, elapsed_ms, **kwargs: metrics.append((stage, user_id, elapsed_ms, kwargs)),
    )
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=9),
        message=DummyMessage(text="https://example.com 이거 요약"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert calls == [
        (
            9,
            "이거 요약",
            "[Web Article]\nURL: https://example.com\n\n본문",
            "web",
            "web",
            "https://example.com/article",
        )
    ]
    assert update.message.replies[0] == "📖 웹페이지 읽는 중..."
    assert metrics[0][0] == "extract"
    assert metrics[0][3]["source"] == "web"
    assert metrics[0][3]["method"] == "rendered_readability"
    assert metrics[0][3]["detail"] == "https://example.com/article"


def test_load_environment_loads_project_then_parent_shared(monkeypatch):
    calls = []

    def fake_exists(self):
        return self.name in {".env", ".shared-ai.env"}

    def fake_load_dotenv(path, override=False):
        calls.append((Path(path).name, override))

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(bot, "load_dotenv", fake_load_dotenv)

    loaded = bot.load_environment()

    assert loaded == [
        str(Path(bot.__file__).resolve().parent / ".env"),
        str(Path(bot.__file__).resolve().parent.parent / ".shared-ai.env"),
    ]
    assert calls == [
        (".env", True),
        (".shared-ai.env", True),
    ]


def test_sanitize_replacement_chars_removes_broken_unicode_marker():
    assert bot.sanitize_replacement_chars("제임스 �(James Hespel)") == "제임스 (James Hespel)"


def test_normalize_response_text_converts_common_latex_arrows():
    assert bot.normalize_response_text(r"흐름은 $\rightarrow$ 이렇게 갑니다.") == "흐름은 → 이렇게 갑니다."
    assert bot.normalize_response_text(r"상태 변화는 $A \rightarrow B$ 입니다.") == "상태 변화는 A → B 입니다."


def test_normalize_response_text_strips_markdown_syntax_for_plain_telegram_display():
    source = "**결론**\n\n- `Q4`가 낫습니다.\n\n## 이유\n짧게 봐야 합니다."

    assert bot.normalize_response_text(source) == "결론\n\n- Q4가 낫습니다.\n\n이유\n짧게 봐야 합니다."


def test_normalize_response_text_converts_markdown_table_to_comparison_list():
    source = """| 구분 | 블록 (Block) | 소파이 (SoFi) |
| :--- | :--- | :--- |
| 정체성 | 결제 및 생태계 중심 (Ecosystem/Payment) | 은행 및 신용 중심 (Banking/Credit) |
| 핵심 엔진 | 결제(Square) + 암호화폐(Cash App/TBD) | 은행 라이선스(Bank Charter) + 대출(Lending) |"""

    assert bot.normalize_response_text(source) == (
        "블록 (Block) vs 소파이 (SoFi)\n\n"
        "정체성\n"
        "- 블록 (Block): 결제 및 생태계 중심 (Ecosystem/Payment)\n"
        "- 소파이 (SoFi): 은행 및 신용 중심 (Banking/Credit)\n\n"
        "핵심 엔진\n"
        "- 블록 (Block): 결제(Square) + 암호화폐(Cash App/TBD)\n"
        "- 소파이 (SoFi): 은행 라이선스(Bank Charter) + 대출(Lending)"
    )


def test_response_validation_issue_detects_chinese_and_japanese_characters():
    assert bot.response_validation_issue("这个视频主要讲的是 AI") == "contains_chinese_or_japanese_characters"
    assert bot.response_validation_issue("これはテストです") == "contains_chinese_or_japanese_characters"
    assert bot.response_validation_issue("이 영상은 AI를 설명합니다.") is None


def test_build_validation_safe_fallback_omits_untranslated_cjk():
    fallback = bot.build_validation_safe_fallback("핵심은 这个视频 입니다. これは테스트.")

    assert fallback is not None
    assert "중국어/일본어 원문 표기 생략" in fallback
    assert bot.response_validation_issue(fallback) is None


def test_validate_and_rewrite_response_retries_when_translation_still_violates(monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_RESPONSE_VALIDATION", True)
    monkeypatch.setattr(bot, "RESPONSE_REWRITE_MAX_ATTEMPTS", 3)
    calls = []

    def fake_rewrite(original_text, user_message, search_context, issue, attempt=1):
        calls.append((original_text, user_message, search_context, issue, attempt))
        if attempt == 1:
            return "仍然是中文"
        return "이 영상은 AI를 설명합니다."

    monkeypatch.setattr(bot, "rewrite_invalid_response", fake_rewrite)

    result = asyncio.run(bot.validate_and_rewrite_response("这是中文", "요약", "[YouTube Transcript]\n본문"))

    assert result == "이 영상은 AI를 설명합니다."
    assert calls == [
        ("这是中文", "요약", "[YouTube Transcript]\n본문", "contains_chinese_or_japanese_characters", 1),
        ("仍然是中文", "요약", "[YouTube Transcript]\n본문", "contains_chinese_or_japanese_characters", 2),
    ]


def test_validate_and_rewrite_response_uses_safe_omission_after_max_translation_attempts(monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_RESPONSE_VALIDATION", True)
    monkeypatch.setattr(bot, "RESPONSE_REWRITE_MAX_ATTEMPTS", 2)
    calls = []

    def fake_rewrite(_original_text, _user_message, _search_context, _issue, attempt=1):
        calls.append(attempt)
        return "仍然是中文"

    monkeypatch.setattr(bot, "rewrite_invalid_response", fake_rewrite)

    result = asyncio.run(bot.validate_and_rewrite_response("这是中文", "요약", "[YouTube Transcript]\n본문"))

    assert result == (
        "응답 일부에 번역되지 않은 중국어/일본어 표기가 있어 해당 표기를 생략했습니다.\n\n"
        f"{bot.CJK_OMISSION_PLACEHOLDER}"
    )
    assert calls == [1, 2]
    assert bot.response_validation_issue(result) is None


def test_normalize_model_name_slugifies_consistently():
    assert normalize_model_name("Gemma 4 26B A4B 6 bit MLX") == "gemma-4-26b-a4b-6-bit-mlx"


def test_load_prompt_profile_includes_base_and_matching_model_rules():
    profile = load_prompt_profile("Gemma 4 26B A4B 6 bit MLX")

    assert "Answer in Korean." in profile
    assert "Do not agree by default. Test the premise first." in profile


def test_build_system_prompt_includes_runtime_date_and_profile_rules():
    prompt = bot.build_system_prompt("Gemma 4 26B A4B 6 bit MLX", today=bot.datetime(2026, 4, 5))

    assert "Today's date is 2026-04-05." in prompt
    assert "Be a thoughtful collaborator, not a cheerleader." in prompt


def test_load_prompt_profile_matches_qwen36_model_rules():
    profile = load_prompt_profile("spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit")

    assert "Answer in Korean." in profile
    assert "Be a rigorous thought partner for investment research" in profile
    assert "For current or date-sensitive facts, do not rely on internal memory alone." in profile


def test_render_prompt_profile_replaces_template_variables():
    prompt = render_prompt_profile("Gemma 4 26B A4B 6 bit MLX", variables={"today": "2026-04-05"})

    assert "{today}" not in prompt
    assert "Today's date is 2026-04-05." in prompt


def test_render_prompt_profile_keeps_literal_braces():
    prompts_dir = Path(__file__).resolve().parent / "fixtures" / "prompts-with-braces"
    prompt = render_prompt_profile("test-model", variables={"today": "2026-04-05"}, prompts_dir=prompts_dir)

    assert 'Return JSON like {"ok": true}.' in prompt
    assert "Today's date is 2026-04-05." in prompt


def test_validate_runtime_config_reports_missing_required_values(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_TOKEN", "")
    monkeypatch.setattr(bot, "MODEL_NAME", "")
    monkeypatch.setattr(bot, "SESSION_INACTIVE_TTL_CONFIG_ERROR", None)

    assert bot.validate_runtime_config() == ["TELEGRAM_TOKEN", "OMLX_MODEL or MODEL_NAME"]


def test_validate_runtime_config_accepts_present_required_values(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_TOKEN", "telegram-token")
    monkeypatch.setattr(bot, "MODEL_NAME", "current-model")
    monkeypatch.setattr(bot, "SESSION_INACTIVE_TTL_CONFIG_ERROR", None)

    assert bot.validate_runtime_config() == []


def test_validate_runtime_config_reports_invalid_session_inactive_ttl(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_TOKEN", "telegram-token")
    monkeypatch.setattr(bot, "MODEL_NAME", "current-model")
    monkeypatch.setattr(
        bot,
        "SESSION_INACTIVE_TTL_CONFIG_ERROR",
        "SESSION_INACTIVE_TTL_SECONDS must be an integer number of seconds",
    )

    assert bot.validate_runtime_config() == ["SESSION_INACTIVE_TTL_SECONDS must be an integer number of seconds"]


def test_parse_session_inactive_ttl_seconds_supports_default_disable_and_invalid():
    assert bot.parse_session_inactive_ttl_seconds(None) == (86400, None)
    assert bot.parse_session_inactive_ttl_seconds("") == (None, None)
    assert bot.parse_session_inactive_ttl_seconds("0") == (None, None)
    assert bot.parse_session_inactive_ttl_seconds("3600") == (3600, None)
    assert bot.parse_session_inactive_ttl_seconds("-1") == (
        None,
        "SESSION_INACTIVE_TTL_SECONDS must be greater than or equal to 0",
    )
    assert bot.parse_session_inactive_ttl_seconds("oops") == (
        None,
        "SESSION_INACTIVE_TTL_SECONDS must be an integer number of seconds",
    )


def test_search_web_fails_fast_without_tavily_key(monkeypatch):
    monkeypatch.setattr(bot, "tavily", None)

    assert bot.search_web("테스트") == "Search failed: missing TAVILY_API_KEY"


def test_extract_json_object_handles_thinking_and_fenced_json():
    text = '<think>검토</think>\n```json\n{"needs_search": true, "query": "GPT-5 reaction"}\n```'

    assert bot.extract_json_object(text) == {"needs_search": True, "query": "GPT-5 reaction"}


def test_resolve_auto_search_decision_uses_hard_guardrail(monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_AUTO_SEARCH", True)
    monkeypatch.setattr(bot, "tavily", object())
    monkeypatch.setattr(
        bot,
        "classify_recency_need",
        lambda *_args, **_kwargs: pytest.fail("classifier should not run for hard guardrail matches"),
    )

    decision = bot.resolve_auto_search_decision("GPT 5.0 현재 소비자 반응은 어때?", session_key(42))

    assert decision.needs_search is True
    assert decision.source == "guardrail"
    assert decision.query == "GPT 5.0 현재 소비자 반응은 어때?"


def test_classify_recency_need_parses_json_response(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"needs_search": true, "query": "GPT-5 consumer reaction", "reason": "recent product reception"}'
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return FakeResponse()

    monkeypatch.setattr(bot.requests, "post", fake_post)
    monkeypatch.setattr(bot, "MODEL_NAME", "test-model")

    decision = bot.classify_recency_need("GPT-5가 소비자에게 외면받았다는 말이 맞아?", session_key(42))

    assert decision.needs_search is True
    assert decision.query == "GPT-5 consumer reaction"
    assert decision.source == "classifier"
    assert calls[0][2]["stream"] is False
    assert calls[0][2]["temperature"] == 0
    assert calls[0][2]["chat_template_kwargs"] == {"enable_thinking": False}


def test_handle_message_auto_searches_when_classifier_requests_it(monkeypatch):
    calls = []

    async def fake_stream_reply(
        update,
        user_id,
        user_message,
        search_context="",
        source="context",
        source_kind=None,
        source_url=None,
        status_message=None,
    ):
        calls.append((user_id, user_message, search_context, source, source_kind, source_url, status_message))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(
        bot,
        "resolve_auto_search_decision",
        lambda user_message, session_key=None: bot.AutoSearchDecision(
            True,
            query="GPT-5 consumer reaction",
            reason="current public reaction",
            source="classifier",
        ),
    )
    monkeypatch.setattr(bot, "search_web", lambda query: "[Web Search Results]\n검색 결과")
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=66),
        message=DummyMessage(text="GPT-5가 소비자에게 외면받았다는 말이 맞아?"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert update.message.replies == ["🔍 최신 정보 확인 중..."]
    status_message = update.message.reply_messages[0]
    assert calls == [
        (
            66,
            "GPT-5가 소비자에게 외면받았다는 말이 맞아?",
            "[Web Search Results]\n검색 결과",
            "auto_search",
            None,
            None,
            status_message,
        )
    ]


def test_cleanup_inactive_sessions_removes_expired_state_from_memory_only(monkeypatch, caplog):
    expired_key = session_key(77, chat_id=999)
    fresh_key = session_key(77, chat_id=111)

    monkeypatch.setattr(bot, "SESSION_INACTIVE_TTL_SECONDS", 10)

    bot.conversations[expired_key] = [{"role": "user", "content": "오래된 채팅"}]
    bot.session_histories[expired_key] = [{"role": "assistant", "content": "오래된 응답"}]
    bot.session_identifiers[expired_key] = "tg:999:101"
    bot.last_activity_at_by_session[expired_key] = 100

    bot.conversations[fresh_key] = [{"role": "user", "content": "최근 채팅"}]
    bot.session_histories[fresh_key] = [{"role": "assistant", "content": "최근 응답"}]
    bot.session_identifiers[fresh_key] = "tg:111:202"
    bot.last_activity_at_by_session[fresh_key] = 995

    with caplog.at_level("INFO"):
        cleaned_up_keys = bot.cleanup_inactive_sessions(now=1000)

    assert cleaned_up_keys == [expired_key]
    assert expired_key not in bot.conversations
    assert expired_key not in bot.session_histories
    assert expired_key not in bot.session_identifiers
    assert expired_key not in bot.last_activity_at_by_session
    assert bot.conversations[fresh_key] == [{"role": "user", "content": "최근 채팅"}]
    assert bot.session_identifiers[fresh_key] == "tg:111:202"
    assert "Cleaned up inactive session from memory" in caplog.text


def test_prepare_messages_builds_fresh_system_prompt(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: f"prompt-for-{model_name}")

    messages = bot.prepare_messages(session_key(42), "안녕")

    assert messages[0] == {"role": "system", "content": f"prompt-for-{bot.MODEL_NAME}"}


def test_prepare_messages_keeps_source_metadata_out_of_llm_payload(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(420)

    messages = bot.prepare_messages(
        key,
        "핵심만",
        "[Web Article]\n본문",
        source_kind="web",
        source_url="https://example.com/article",
    )

    assert messages[1] == {
        "role": "user",
        "content": (
            "[Web Article]\n본문\n\n"
            "[Response Rules]\n"
            "- 반드시 한국어로만 답하세요.\n"
            "- 원문 언어가 영어, 중국어, 일본어 또는 다른 언어여도 한국어로 번역, 요약, 설명하세요.\n"
            "- 사용자가 명시적으로 원문 인용을 요청한 경우가 아니라면 중국어/일본어 문자를 출력하지 마세요.\n"
            "- 사용자가 영어 답변을 명시적으로 요청한 경우에만 영어로 답하세요.\n"
            "- Treat the provided context/search results as authoritative for current facts.\n"
            "- If the provided context conflicts with your internal memory, prefer the provided context.\n"
            "- Do not assert current facts that are not supported by the provided context; say they are unverified.\n"
            "- For current-event, product, company, market, or investment answers, separate verified facts from analysis and unresolved uncertainties.\n\n"
            "[User Question]\n핵심만"
        ),
    }
    assert bot.conversations[key][-1] == messages[1]
    assert bot.session_histories[key][-1]["source_kind"] == "web"
    assert bot.session_histories[key][-1]["source_url"] == "https://example.com/article"
    assert "source_kind" not in messages[1]
    assert "source_url" not in messages[1]


def test_prepare_messages_applies_source_followup_rules_without_storing_them(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(421)
    bot.conversations[key] = [
        {
            "role": "user",
            "content": "[YouTube Transcript]\n원문\n\n[Response Rules]\n...\n\n[User Question]\n요약",
        },
        {"role": "assistant", "content": "첫 요약"},
    ]
    bot.session_histories[key] = list(bot.conversations[key])

    messages = bot.prepare_messages(key, "이 자료에서 수요 신호만")

    assert messages[-1]["role"] == "user"
    assert "[Follow-up Source Rules]" in messages[-1]["content"]
    assert "Treat the user's latest message as the priority lens" in messages[-1]["content"]
    assert "[User Question]\n이 자료에서 수요 신호만" in messages[-1]["content"]
    assert bot.conversations[key][-1] == {"role": "user", "content": "이 자료에서 수요 신호만"}
    assert bot.session_histories[key][-1] == {"role": "user", "content": "이 자료에서 수요 신호만"}


def test_prepare_messages_applies_legacy_source_followup_without_explicit_source_reference(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(424)
    bot.conversations[key] = [
        {
            "role": "user",
            "content": "[Web Article]\n원문\n\n[Response Rules]\n...\n\n[User Question]\n요약",
        },
        {"role": "assistant", "content": "첫 요약"},
    ]
    bot.session_histories[key] = list(bot.conversations[key])

    messages = bot.prepare_messages(key, "수요 신호만")

    assert messages[-1]["role"] == "user"
    assert "[Follow-up Source Rules]" in messages[-1]["content"]
    assert "[User Question]\n수요 신호만" in messages[-1]["content"]
    assert bot.conversations[key][-1] == {"role": "user", "content": "수요 신호만"}


def test_prepare_messages_skips_source_followup_rules_without_recent_context(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(422)
    bot.append_history_message(key, "user", "이전 일반 질문")
    bot.append_history_message(key, "assistant", "이전 일반 답변")

    messages = bot.prepare_messages(key, "수요 신호만")

    assert messages[-1] == {"role": "user", "content": "수요 신호만"}
    assert "[Follow-up Source Rules]" not in messages[-1]["content"]


def test_prepare_messages_skips_source_followup_rules_when_user_asks_to_ignore_context(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(423)
    bot.conversations[key] = [
        {
            "role": "user",
            "content": "[Web Article]\n원문\n\n[Response Rules]\n...\n\n[User Question]\n요약",
        },
        {"role": "assistant", "content": "첫 요약"},
    ]
    bot.session_histories[key] = list(bot.conversations[key])

    messages = bot.prepare_messages(key, "다른 얘기인데 qwen 설정은?")

    assert messages[-1] == {"role": "user", "content": "다른 얘기인데 qwen 설정은?"}
    assert "[Follow-up Source Rules]" not in messages[-1]["content"]


def test_prepare_messages_keeps_ten_pairs_plus_current_user(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(7)

    for turn in range(1, 11):
        bot.append_history_message(key, "user", f"u{turn}")
        bot.append_history_message(key, "assistant", f"a{turn}")

    messages = bot.prepare_messages(key, "u11")

    assert messages[0] == {"role": "system", "content": "prompt"}
    assert len(bot.conversations[key]) == 21
    assert bot.conversations[key][0] == {"role": "user", "content": "u1"}
    assert bot.conversations[key][-1] == {"role": "user", "content": "u11"}

    bot.append_history_message(key, "assistant", "a11")

    assert len(bot.conversations[key]) == 20
    assert bot.conversations[key][0] == {"role": "user", "content": "u2"}
    assert bot.conversations[key][-2:] == [
        {"role": "user", "content": "u11"},
        {"role": "assistant", "content": "a11"},
    ]
    assert len(bot.session_histories[key]) == 22


def test_prepare_messages_compacts_assistant_history_in_llm_payload(monkeypatch):
    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: "prompt")
    key = session_key(8)
    old_answer = "오래된 답변 " * 200
    recent_answer = "최근 답변 " * 300

    bot.append_history_message(key, "user", "u1")
    bot.append_history_message(key, "assistant", old_answer)
    bot.append_history_message(key, "user", "u2")
    bot.append_history_message(key, "assistant", recent_answer)

    messages = bot.prepare_messages(key, "새 질문")
    assistant_messages = [message for message in messages if message["role"] == "assistant"]

    assert old_answer in [message["content"] for message in bot.session_histories[key]]
    assert recent_answer in [message["content"] for message in bot.session_histories[key]]
    assert assistant_messages[0]["content"].startswith("오래된 답변")
    assert assistant_messages[1]["content"].startswith("최근 답변")
    assert len(assistant_messages[0]["content"]) < len(old_answer)
    assert len(assistant_messages[1]["content"]) < len(recent_answer)
    assert "이전 AI 답변 일부 생략" in assistant_messages[0]["content"]
    assert "이전 AI 답변 일부 생략" in assistant_messages[1]["content"]


def test_build_draft_id_is_positive_non_zero():
    assert bot.build_draft_id() > 0


def test_parse_telegram_response_delivery_defaults_to_final(monkeypatch):
    monkeypatch.delenv("TELEGRAM_RESPONSE_DELIVERY", raising=False)
    monkeypatch.delenv("ENABLE_TELEGRAM_DRAFT_STREAMING", raising=False)

    assert bot.parse_telegram_response_delivery() == "final"


def test_parse_telegram_response_delivery_supports_legacy_draft_flag(monkeypatch):
    monkeypatch.delenv("TELEGRAM_RESPONSE_DELIVERY", raising=False)
    monkeypatch.setenv("ENABLE_TELEGRAM_DRAFT_STREAMING", "false")

    assert bot.parse_telegram_response_delivery() == "edit"

    monkeypatch.setenv("ENABLE_TELEGRAM_DRAFT_STREAMING", "true")
    assert bot.parse_telegram_response_delivery() == "draft"


def test_parse_enable_thinking_for_context_prefers_positive_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_THINKING_FOR_CONTEXT", "true")
    monkeypatch.setenv("DISABLE_THINKING_FOR_CONTEXT", "true")

    assert bot.parse_enable_thinking_for_context() is True


def test_parse_enable_thinking_for_context_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_THINKING_FOR_CONTEXT", raising=False)
    monkeypatch.delenv("DISABLE_THINKING_FOR_CONTEXT", raising=False)

    assert bot.parse_enable_thinking_for_context() is True


def test_parse_enable_thinking_for_context_honors_legacy_disable(monkeypatch):
    monkeypatch.delenv("ENABLE_THINKING_FOR_CONTEXT", raising=False)
    monkeypatch.setenv("DISABLE_THINKING_FOR_CONTEXT", "true")

    assert bot.parse_enable_thinking_for_context() is False


def test_build_chat_completion_payload_disables_thinking_for_context_when_configured(monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_THINKING_FOR_CONTEXT", False)

    payload = bot.build_chat_completion_payload(
        [{"role": "user", "content": "요약"}],
        search_context="[Web Article]\n본문",
    )

    assert payload["stream_options"] == {"include_usage": True}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_chat_completion_payload_keeps_thinking_for_context_when_configured(monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_THINKING_FOR_CONTEXT", True)

    payload = bot.build_chat_completion_payload(
        [{"role": "user", "content": "요약"}],
        search_context="[Web Article]\n본문",
    )

    assert payload["stream_options"] == {"include_usage": True}
    assert "chat_template_kwargs" not in payload


def test_build_chat_completion_payload_keeps_server_default_for_plain_chat(monkeypatch):
    monkeypatch.setattr(bot, "ENABLE_THINKING_FOR_CONTEXT", False)

    payload = bot.build_chat_completion_payload(
        [{"role": "user", "content": "안녕"}],
        search_context="",
    )

    assert payload["stream_options"] == {"include_usage": True}
    assert "chat_template_kwargs" not in payload


def test_redact_log_secrets_removes_telegram_bot_token():
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghi"
    url = f"https://api.telegram.org/bot{token}/getUpdates"

    redacted = bot.redact_log_secrets(f"HTTP Request: POST {url}")

    assert token not in redacted
    assert "api.telegram.org/" in redacted
    assert bot.REDACTED_TELEGRAM_BOT_TOKEN in redacted
    assert "/getUpdates" in redacted


def test_secret_redaction_filter_redacts_log_record_args():
    token = "987654321:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghi"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    record = bot.logging.LogRecord(
        "httpx",
        bot.logging.INFO,
        __file__,
        1,
        "HTTP Request: POST %s %s",
        (url, {"token": token}),
        None,
    )

    assert bot.SecretRedactionFilter().filter(record) is True
    rendered = record.getMessage()

    assert token not in rendered
    assert bot.REDACTED_TELEGRAM_BOT_TOKEN in rendered


def test_save_session_to_vault_uses_full_session_history(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "VAULT_CAPTURE_PATH", str(tmp_path))
    monkeypatch.setattr(bot, "generate_tags", lambda history: "#test-tag")
    key = session_key(55)

    bot.session_histories[key] = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
        {"role": "user", "content": "둘째 질문"},
    ]
    bot.conversations[key] = [{"role": "user", "content": "둘째 질문"}]

    assert bot.save_session_to_vault(key) is True

    saved_files = list(tmp_path.glob("*.md"))
    assert len(saved_files) == 1
    saved_text = saved_files[0].read_text(encoding="utf-8")
    assert "첫 질문" in saved_text
    assert "첫 답변" in saved_text
    assert "둘째 질문" in saved_text
    assert "#test-tag" in saved_text


def test_save_session_to_vault_writes_session_marker_below_header(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "VAULT_CAPTURE_PATH", str(tmp_path))
    monkeypatch.setattr(bot, "generate_tags", lambda history: "#test-tag")
    key = session_key(56)

    bot.session_histories[key] = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
    ]
    bot.session_identifiers[key] = "tg:999:123456"

    assert bot.save_session_to_vault(key) is True

    saved_text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert re.search(
        r"## AI 세션 \(\d{2}:\d{2}, .+?\)\n<!-- capture:session-id=tg:999:123456 -->\n\n\*\*나\*\*: 첫 질문",
        saved_text,
    )


def test_save_session_to_vault_writes_source_comment_below_user_block(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "VAULT_CAPTURE_PATH", str(tmp_path))
    monkeypatch.setattr(bot, "generate_tags", lambda history: "#test-tag")
    key = session_key(58)

    bot.session_histories[key] = [
        {
            "role": "user",
            "content": "[YouTube Transcript]\n본문\n\n[User Question]\n핵심만",
            "source_kind": "youtube",
            "source_url": "https://www.youtube.com/watch?v=abcdefghijk",
        },
        {"role": "assistant", "content": "첫 답변"},
    ]

    assert bot.save_session_to_vault(key) is True

    saved_text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert re.search(
        r"\*\*나\*\* \(YouTube 첨부\): 핵심만\n<!-- source: https://www\.youtube\.com/watch\?v=abcdefghijk -->\n\n\*\*AI\*\*: 첫 답변",
        saved_text,
    )


def test_save_session_to_vault_keeps_legacy_format_when_identifier_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "VAULT_CAPTURE_PATH", str(tmp_path))
    monkeypatch.setattr(bot, "generate_tags", lambda history: "#test-tag")
    key = session_key(57)

    bot.session_histories[key] = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
    ]

    assert bot.save_session_to_vault(key) is True

    saved_text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "capture:session-id=" not in saved_text
    assert re.search(r"## AI 세션 \(\d{2}:\d{2}, .+?\)\n\n\*\*나\*\*: 첫 질문", saved_text)


def test_should_show_reasoning_status_skips_whitespace_and_single_char():
    assert bot.should_show_reasoning_status("\n") is False
    assert bot.should_show_reasoning_status("*") is False
    assert bot.should_show_reasoning_status("가") is False
    assert bot.should_show_reasoning_status("계산") is True


def test_handle_message_shares_early_typing_indicator_with_stream_reply(monkeypatch):
    events = []
    stream_saw_typing = []

    async def fake_keep_typing(update, stop_event):
        events.append("started")
        await stop_event.wait()
        events.append("stopped")

    async def fake_stream_reply(
        update,
        user_id,
        user_message,
        search_context="",
        source="context",
        source_kind=None,
        source_url=None,
    ):
        stream_saw_typing.append(bot._active_typing_indicator.get() is not None)

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "keep_typing_until_visible", fake_keep_typing)
    monkeypatch.setattr(bot, "resolve_auto_search_decision", lambda *_args, **_kwargs: bot.AutoSearchDecision(False))
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=320),
        message=DummyMessage(text="질문"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert stream_saw_typing == [True]
    assert events == ["started", "stopped"]


def test_keep_typing_retries_after_transient_chat_action_failure(monkeypatch):
    class FlakyBot(DummyBot):
        def __init__(self):
            super().__init__()
            self.chat_action_calls = 0

        async def send_chat_action(self, **kwargs):
            self.chat_action_calls += 1
            if self.chat_action_calls == 1:
                raise bot.TelegramError("temporary chat action failure")
            return await super().send_chat_action(**kwargs)

    async def run_once():
        flaky_bot = FlakyBot()
        update = SimpleNamespace(message=DummyMessage(text="질문", bot_instance=flaky_bot))
        stop_event = asyncio.Event()
        task = asyncio.create_task(bot.keep_typing_until_visible(update, stop_event))
        for _ in range(20):
            if flaky_bot.chat_actions:
                break
            await asyncio.sleep(0.01)
        stop_event.set()
        await task
        return flaky_bot

    monkeypatch.setattr(bot, "TYPING_ACTION_RETRY_INTERVAL", 0.01)
    monkeypatch.setattr(bot, "TYPING_ACTION_INTERVAL", 60)

    flaky_bot = asyncio.run(run_once())

    assert flaky_bot.chat_action_calls >= 2
    assert flaky_bot.chat_actions[-1]["action"] == bot.ChatAction.TYPING


def test_stream_reply_sends_final_message_only_by_default(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "안"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "녕"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "final")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=321),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 321, "질문"))

    assert dummy_bot.drafts == []
    assert message.replies == ["안녕"]
    assert message.reply_messages[0].edits == []
    assert bot.conversations[session_key(321)][-1] == {"role": "assistant", "content": "안녕"}


def test_stream_reply_keeps_first_session_identifier_for_user(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "응답"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    first_update = SimpleNamespace(
        effective_user=SimpleNamespace(id=900),
        message=DummyMessage(text="첫 질문", message_id=101),
    )
    second_update = SimpleNamespace(
        effective_user=SimpleNamespace(id=900),
        message=DummyMessage(text="둘째 질문", message_id=202),
    )

    asyncio.run(bot.stream_reply(first_update, 900, "첫 질문"))
    asyncio.run(bot.stream_reply(second_update, 900, "둘째 질문"))

    assert bot.session_identifiers[session_key(900)] == "tg:999:101"


def test_stream_reply_keeps_parallel_scoped_sessions_independent(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "응답"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    first_update = SimpleNamespace(
        effective_user=SimpleNamespace(id=901),
        message=DummyMessage(text="첫 질문", message_id=101, chat_id=999),
    )
    second_update = SimpleNamespace(
        effective_user=SimpleNamespace(id=901),
        message=DummyMessage(text="둘째 질문", message_id=202, chat_id=111),
    )

    asyncio.run(bot.stream_reply(first_update, 901, "첫 질문"))
    asyncio.run(bot.stream_reply(second_update, 901, "둘째 질문"))

    assert bot.conversations[session_key(901, chat_id=999)][-1] == {"role": "assistant", "content": "응답"}
    assert bot.conversations[session_key(901, chat_id=111)][-1] == {"role": "assistant", "content": "응답"}
    assert bot.session_identifiers[session_key(901, chat_id=999)] == "tg:999:101"
    assert bot.session_identifiers[session_key(901, chat_id=111)] == "tg:111:202"


def test_stream_reply_resumes_existing_scoped_session_after_other_scope(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "응답"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    first_scope = SimpleNamespace(
        effective_user=SimpleNamespace(id=902),
        message=DummyMessage(text="첫 질문", message_id=101, chat_id=999),
    )
    second_scope = SimpleNamespace(
        effective_user=SimpleNamespace(id=902),
        message=DummyMessage(text="다른 채팅", message_id=202, chat_id=111),
    )
    return_to_first_scope = SimpleNamespace(
        effective_user=SimpleNamespace(id=902),
        message=DummyMessage(text="다시 첫 채팅", message_id=303, chat_id=999),
    )

    asyncio.run(bot.stream_reply(first_scope, 902, "첫 질문"))
    asyncio.run(bot.stream_reply(second_scope, 902, "다른 채팅"))
    asyncio.run(bot.stream_reply(return_to_first_scope, 902, "다시 첫 채팅"))

    first_key = session_key(902, chat_id=999)
    second_key = session_key(902, chat_id=111)

    assert bot.session_identifiers[first_key] == "tg:999:101"
    assert bot.session_identifiers[second_key] == "tg:111:202"
    assert bot.conversations[first_key][-2:] == [
        {"role": "user", "content": "다시 첫 채팅"},
        {"role": "assistant", "content": "응답"},
    ]


def test_stream_reply_starts_new_in_memory_session_after_idle_prune(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "응답"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")
    monkeypatch.setattr(bot, "SESSION_INACTIVE_TTL_SECONDS", 10)
    monkeypatch.setattr(bot.time, "time", lambda: 1000)

    key = session_key(903, chat_id=999)
    bot.conversations[key] = [{"role": "user", "content": "오래된 질문"}]
    bot.session_histories[key] = [{"role": "assistant", "content": "오래된 응답"}]
    bot.session_identifiers[key] = "tg:999:101"
    bot.last_activity_at_by_session[key] = 0

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=903),
        message=DummyMessage(text="새 질문", message_id=303, chat_id=999),
    )

    asyncio.run(bot.stream_reply(update, 903, "새 질문"))

    assert bot.session_identifiers[key] == "tg:999:303"
    assert bot.conversations[key][-2:] == [
        {"role": "user", "content": "새 질문"},
        {"role": "assistant", "content": "응답"},
    ]


def test_stream_reply_normalizes_latex_arrow_before_sending(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", r"$A \rightarrow B$"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=322),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 322, "질문"))

    assert dummy_bot.drafts[-1]["text"] == "A → B"
    assert "A → B" in message.replies
    assert bot.conversations[session_key(322)][-1] == {"role": "assistant", "content": "A → B"}


def test_stream_reply_normalizes_markdown_table_before_sending(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            (
                "token",
                "| 구분 | 블록 | 소파이 |\n| :--- | :--- | :--- |\n| 정체성 | 결제 중심 | 은행 중심 |",
            ),
        )
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=323),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 323, "질문"))

    expected = "블록 vs 소파이\n\n정체성\n- 블록: 결제 중심\n- 소파이: 은행 중심"
    assert dummy_bot.drafts[-1]["text"] == expected
    assert expected in message.replies
    assert bot.conversations[session_key(323)][-1] == {"role": "assistant", "content": expected}


def test_stream_reply_rewrites_cjk_final_response_before_sending(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "这个视频主要讲的是 AI"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    def fake_rewrite(original_text, user_message, search_context, issue, attempt=1):
        assert original_text == "这个视频主要讲的是 AI"
        assert user_message == "요약"
        assert search_context == "[YouTube Transcript]\n본문"
        assert issue == "contains_chinese_or_japanese_characters"
        assert attempt == 1
        return "이 영상은 AI를 설명합니다."

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "rewrite_invalid_response", fake_rewrite)
    monkeypatch.setattr(bot, "ENABLE_RESPONSE_VALIDATION", True)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=324),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 324, "요약", "[YouTube Transcript]\n본문", source="youtube"))

    assert [draft["text"] for draft in dummy_bot.drafts] == ["이 영상은 AI를 설명합니다."]
    assert dummy_bot.drafts[-1]["text"] == "이 영상은 AI를 설명합니다."
    assert message.replies == ["이 영상은 AI를 설명합니다."]
    assert bot.conversations[session_key(324)][-1] == {
        "role": "assistant",
        "content": "이 영상은 AI를 설명합니다.",
    }


def test_stream_reply_shows_reasoning_status_before_content(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("reasoning", "곱셈을 계산한다"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "323"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=777),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 777, "질문"))

    draft_texts = [draft["text"] for draft in dummy_bot.drafts]
    assert "🧠 추론 중..." in draft_texts
    assert "323" in draft_texts
    assert "323" in message.replies
    assert bot.conversations[session_key(777)][-1] == {"role": "assistant", "content": "323"}


def test_stream_reply_final_delivery_sends_temporary_reasoning_status(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("reasoning", "곱셈을 계산한다"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "323"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    use_delivery(monkeypatch, "final")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=779),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 779, "질문"))

    assert dummy_bot.drafts == []
    assert message.replies == ["🧠 추론 중...", "323"]
    assert message.reply_messages[0].deleted is True
    assert message.reply_messages[0].edits == []
    assert message.reply_messages[1].deleted is False
    assert message.reply_messages[1].edits == []
    assert bot.conversations[session_key(779)][-1] == {"role": "assistant", "content": "323"}


def test_stream_reply_final_delivery_reuses_status_message(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("reasoning", "검색 결과를 검토한다"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "최종 답변"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    use_delivery(monkeypatch, "final")

    message = DummyMessage(text="질문")
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=780),
        message=message,
    )
    status_message = asyncio.run(message.reply_text("🔍 검색 중..."))

    asyncio.run(bot.stream_reply(update, 780, "질문", "[Web Search Results]\n검색 결과", source="auto_search", status_message=status_message))

    assert message.replies == ["🔍 검색 중..."]
    assert status_message.edits == ["🧠 추론 중...", "최종 답변"]
    assert status_message.deleted is False
    assert bot.conversations[session_key(780)][-1] == {"role": "assistant", "content": "최종 답변"}


def test_stream_reply_skips_reasoning_status_for_trivial_reasoning(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("reasoning", "\n"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "323"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=778),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 778, "질문"))

    draft_texts = [draft["text"] for draft in dummy_bot.drafts]
    assert "🧠 추론 중..." not in draft_texts
    assert "323" in draft_texts
    assert bot.conversations[session_key(778)][-1] == {"role": "assistant", "content": "323"}


def test_stream_reply_flushes_final_draft_before_sending_message(monkeypatch):
    bot.conversations.clear()

    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "테"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "스트"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 999)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=654),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 654, "질문"))

    assert len(dummy_bot.drafts) >= 1
    assert dummy_bot.drafts[0]["text"] == "테"
    assert dummy_bot.drafts[-1]["text"] == "테스트"
    assert "테스트" in message.replies
    assert message.reply_messages[0].edits == []
    assert bot.conversations[session_key(654)][-1] == {"role": "assistant", "content": "테스트"}


def test_stream_reply_keeps_draft_mode_after_visible_draft_failure(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "테"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "스"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "트"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot(fail_draft_calls={2})
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=655),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 655, "질문"))

    assert [draft["text"] for draft in dummy_bot.drafts] == ["테", "테스트"]
    assert message.replies == ["테스트"]
    assert message.reply_messages[0].edits == []
    assert bot.conversations[session_key(655)][-1] == {"role": "assistant", "content": "테스트"}


def test_stream_reply_uses_edit_delivery_when_configured(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "테"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "스트"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    use_delivery(monkeypatch, "edit")

    dummy_bot = DummyBot()
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=656),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 656, "질문"))

    assert dummy_bot.drafts == []
    assert message.replies[0] == "테"
    assert "테스트" in message.reply_messages[0].edits
    assert bot.conversations[session_key(656)][-1] == {"role": "assistant", "content": "테스트"}


def test_stream_reply_falls_back_to_edit_text_when_draft_fails(monkeypatch):
    def fake_stream(messages, loop, queue):
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "테"))
        loop.call_soon_threadsafe(queue.put_nowait, ("token", "스트"))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    monkeypatch.setattr(bot, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(bot, "STREAM_EDIT_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_START_INTERVAL", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_MIN_CHARS_DELTA", 0)
    monkeypatch.setattr(bot, "DRAFT_STREAM_FINAL_FLUSH_DELAY", 0)
    use_delivery(monkeypatch, "draft")

    dummy_bot = DummyBot(fail_draft=True)
    message = DummyMessage(text="질문", bot_instance=dummy_bot)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=654),
        message=message,
    )

    asyncio.run(bot.stream_reply(update, 654, "질문"))

    assert message.replies[0] == "테"
    assert "테스트" in message.reply_messages[0].edits
    assert bot.conversations[session_key(654)][-1] == {"role": "assistant", "content": "테스트"}


def test_clear_history_pops_context_and_full_session(monkeypatch):
    saved_session_keys = []
    key = session_key(88)

    monkeypatch.setattr(bot, "save_session_to_vault", lambda current_key: saved_session_keys.append(current_key) or True)

    bot.conversations[key] = [{"role": "user", "content": "최근"}]
    bot.session_histories[key] = [
        {"role": "user", "content": "예전"},
        {"role": "assistant", "content": "답변"},
        {"role": "user", "content": "최근"},
    ]
    bot.session_identifiers[key] = "tg:999:123"
    bot.last_activity_at_by_session[key] = 1000

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=88),
        message=DummyMessage(text="/c"),
    )

    asyncio.run(bot.clear_history(update, None))

    assert saved_session_keys == [key]
    assert key not in bot.conversations
    assert key not in bot.session_histories
    assert key not in bot.session_identifiers
    assert key not in bot.last_activity_at_by_session
    assert update.message.replies == ["🗑️ 대화 기록 초기화 완료."]


def test_clear_history_only_clears_current_scoped_session(monkeypatch):
    saved_session_keys = []
    current_key = session_key(89, chat_id=999)
    other_key = session_key(89, chat_id=111)

    monkeypatch.setattr(bot, "save_session_to_vault", lambda current_key_arg: saved_session_keys.append(current_key_arg) or True)
    monkeypatch.setattr(bot, "SESSION_INACTIVE_TTL_SECONDS", None)

    bot.conversations[current_key] = [{"role": "user", "content": "현재 채팅"}]
    bot.session_histories[current_key] = [{"role": "assistant", "content": "현재 응답"}]
    bot.session_identifiers[current_key] = "tg:999:123"
    bot.last_activity_at_by_session[current_key] = 1000

    bot.conversations[other_key] = [{"role": "user", "content": "다른 채팅"}]
    bot.session_histories[other_key] = [{"role": "assistant", "content": "다른 응답"}]
    bot.session_identifiers[other_key] = "tg:111:456"
    bot.last_activity_at_by_session[other_key] = 1000

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=89),
        message=DummyMessage(text="/c", chat_id=999),
    )

    asyncio.run(bot.clear_history(update, None))

    assert saved_session_keys == [current_key]
    assert current_key not in bot.conversations
    assert current_key not in bot.session_histories
    assert current_key not in bot.session_identifiers
    assert current_key not in bot.last_activity_at_by_session
    assert bot.conversations[other_key] == [{"role": "user", "content": "다른 채팅"}]
    assert bot.session_histories[other_key] == [{"role": "assistant", "content": "다른 응답"}]
    assert bot.session_identifiers[other_key] == "tg:111:456"
    assert bot.last_activity_at_by_session[other_key] == 1000


def test_clear_history_prunes_other_idle_scopes_after_current_clear(monkeypatch):
    saved_session_keys = []
    current_key = session_key(90, chat_id=999)
    idle_other_key = session_key(90, chat_id=111)

    monkeypatch.setattr(bot, "save_session_to_vault", lambda current_key_arg: saved_session_keys.append(current_key_arg) or True)
    monkeypatch.setattr(bot, "SESSION_INACTIVE_TTL_SECONDS", 10)
    monkeypatch.setattr(bot.time, "time", lambda: 1000)

    bot.conversations[current_key] = [{"role": "user", "content": "현재 채팅"}]
    bot.session_histories[current_key] = [{"role": "assistant", "content": "현재 응답"}]
    bot.session_identifiers[current_key] = "tg:999:123"
    bot.last_activity_at_by_session[current_key] = 995

    bot.conversations[idle_other_key] = [{"role": "user", "content": "오래된 채팅"}]
    bot.session_histories[idle_other_key] = [{"role": "assistant", "content": "오래된 응답"}]
    bot.session_identifiers[idle_other_key] = "tg:111:456"
    bot.last_activity_at_by_session[idle_other_key] = 100

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=90),
        message=DummyMessage(text="/c", chat_id=999),
    )

    asyncio.run(bot.clear_history(update, None))

    assert saved_session_keys == [current_key]
    assert current_key not in bot.conversations
    assert idle_other_key not in bot.conversations
    assert idle_other_key not in bot.session_histories
    assert idle_other_key not in bot.session_identifiers
    assert idle_other_key not in bot.last_activity_at_by_session


def test_show_model_marks_unset_model(monkeypatch):
    monkeypatch.setattr(bot, "MODEL_NAME", "")
    monkeypatch.setattr(bot, "ENV_FILES_LOADED", [])

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        message=DummyMessage(text="/m"),
    )

    asyncio.run(bot.show_model(update, None))

    assert update.message.replies == ["📦 현재 모델: (unset)\n🔧 설정 소스: process env"]
