import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from prompt_profiles import load_prompt_profile, normalize_model_name, render_prompt_profile


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


def test_render_prompt_profile_replaces_template_variables():
    prompt = render_prompt_profile("Gemma 4 26B A4B 6 bit MLX", variables={"today": "2026-04-05"})

    assert "{today}" not in prompt
    assert "Today's date is 2026-04-05." in prompt


def test_render_prompt_profile_keeps_literal_braces():
    prompts_dir = Path(__file__).resolve().parent / "fixtures" / "prompts-with-braces"
    prompt = render_prompt_profile("test-model", variables={"today": "2026-04-05"}, prompts_dir=prompts_dir)

    assert 'Return JSON like {"ok": true}.' in prompt
    assert "Today's date is 2026-04-05." in prompt


def test_prepare_messages_builds_fresh_system_prompt(monkeypatch):
    bot.conversations.clear()
    bot.last_activity.clear()

    monkeypatch.setattr(bot, "build_system_prompt", lambda model_name: f"prompt-for-{model_name}")

    messages = bot.prepare_messages(42, "안녕")

    assert messages[0] == {"role": "system", "content": f"prompt-for-{bot.MODEL_NAME}"}
