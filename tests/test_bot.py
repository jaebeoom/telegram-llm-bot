import asyncio
from types import SimpleNamespace

import bot


class DummyFile:
    async def download_as_bytearray(self):
        return bytearray(b"%PDF-1.4")


class DummyDocument:
    mime_type = "application/pdf"

    async def get_file(self):
        return DummyFile()


class DummyMessage:
    def __init__(self, text=None, caption=None, document=None):
        self.text = text
        self.caption = caption
        self.document = document
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)
        return SimpleNamespace(delete=_async_noop)


async def _async_noop():
    return None


def test_build_context_prompt_uses_default_for_blank_text():
    assert bot.build_context_prompt("   ") == bot.DEFAULT_CONTEXT_PROMPT


def test_build_context_prompt_keeps_user_text():
    assert bot.build_context_prompt("요약 말고 핵심만") == "요약 말고 핵심만"


def test_handle_document_without_caption_uses_default_summary_prompt(monkeypatch):
    calls = []

    async def fake_stream_reply(update, user_id, user_message, search_context=""):
        calls.append((user_id, user_message, search_context))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "extract_pdf_text", lambda file_bytes: "[PDF Document]\n본문")
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=123),
        message=DummyMessage(caption=None, document=DummyDocument()),
    )

    asyncio.run(bot.handle_document(update, None))

    assert calls == [(123, bot.DEFAULT_CONTEXT_PROMPT, "[PDF Document]\n본문")]


def test_handle_message_with_x_url_uses_context_reply(monkeypatch):
    calls = []

    async def fake_stream_reply(update, user_id, user_message, search_context=""):
        calls.append((user_id, user_message, search_context))

    monkeypatch.setattr(bot, "is_allowed", lambda user_id: True)
    monkeypatch.setattr(bot, "extract_tweet_from_url", lambda url: "[X Post]\n본문")
    monkeypatch.setattr(bot, "stream_reply", fake_stream_reply)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=7),
        message=DummyMessage(text="https://x.com/test/status/123"),
    )

    asyncio.run(bot.handle_message(update, None))

    assert calls == [(7, bot.DEFAULT_CONTEXT_PROMPT, "[X Post]\n본문")]
