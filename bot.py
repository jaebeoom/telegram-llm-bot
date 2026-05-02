from __future__ import annotations

import os
import re
import asyncio
import hashlib
import json
import logging
import sys
import time
import shutil
import subprocess
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from tagger import generate_tags
from prompt_profiles import render_prompt_profile
from inbox_prefetch_cache import (
    InboxPrefetchPersistentCache,
    PersistentInboxPrefetchRecord,
)
from telegram_llm_bot.youtube_audio_transcription import (
    load_config_from_env as load_youtube_audio_transcription_config,
    transcript_cache_path,
)
from extractors import (
    TWEET_URL_PATTERN,
    YOUTUBE_URL_PATTERN,
    GENERAL_URL_PATTERN,
    YouTubeTranscriptExtractionResult,
    extract_tweet_from_url,
    extract_pdf_text,
    extract_pdf_from_url,
    extract_web_result,
    extract_youtube_transcript_result,
)

ENV_FILES_LOADED: list[str] = []
ChatMessage = dict[str, str]
SessionKey = tuple[int, int, int]
CAPTURE_SESSION_ID_MARKER_PREFIX = "<!-- capture:session-id="
SOURCE_KIND_LABELS = {
    "x": "X 포스트",
    "youtube": "YouTube",
    "pdf": "PDF",
    "web": "웹 아티클",
}
TELEGRAM_BOT_TOKEN_RE = re.compile(
    r"bot\d{6,}:[A-Za-z0-9_-]{20,}|(?<![A-Za-z0-9_-])\d{6,}:[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
)
REDACTED_TELEGRAM_BOT_TOKEN = "[REDACTED_TELEGRAM_BOT_TOKEN]"


def redact_log_secrets(text: str) -> str:
    return TELEGRAM_BOT_TOKEN_RE.sub(REDACTED_TELEGRAM_BOT_TOKEN, text)


def _redact_log_value(value):
    if isinstance(value, str):
        return redact_log_secrets(value)
    if isinstance(value, tuple):
        return tuple(_redact_log_value(item) for item in value)
    if isinstance(value, list):
        return [_redact_log_value(item) for item in value]
    if isinstance(value, dict):
        return {_redact_log_value(key): _redact_log_value(item) for key, item in value.items()}
    return value


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_log_value(record.msg)
        record.args = _redact_log_value(record.args)
        return True


def install_log_redaction_filter() -> None:
    root_logger = logging.getLogger()
    secret_filter = SecretRedactionFilter()
    for handler in root_logger.handlers:
        handler.addFilter(secret_filter)


def load_environment() -> list[str]:
    """프로젝트 .env와 상위 공용 env 파일을 순서대로 로드한다.

    상위 공용 파일은 프로젝트 .env보다 우선하도록 마지막에 override=True로 덮어쓴다.
    """

    root_dir = Path(__file__).resolve().parent
    candidate_paths = [
        root_dir / ".env",
        root_dir.parent / "shared_ai.env",
        root_dir.parent / ".shared_ai.env",
        root_dir.parent / "shared-ai.env",
        root_dir.parent / ".shared-ai.env",
    ]

    loaded_files: list[str] = []
    for env_path in candidate_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            loaded_files.append(str(env_path))

    return loaded_files


ENV_FILES_LOADED = load_environment()

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()
LLM_API_BASE_URL = (
    os.getenv("OMLX_BASE_URL")
    or os.getenv("LLM_API_BASE_URL")
    or os.getenv("LLM_DEFAULT_BASE_URL")
    or os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "") or os.getenv("LLM_DEFAULT_API_KEY", "")
if "openrouter.ai" in LLM_API_BASE_URL.lower():
    LLM_API_KEY = OPENROUTER_API_KEY or LLM_DEFAULT_API_KEY or os.getenv("OMLX_API_KEY", "")
else:
    LLM_API_KEY = os.getenv("OMLX_API_KEY") or LLM_DEFAULT_API_KEY or OPENROUTER_API_KEY
LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER_NAME", "OMLX")
ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "")
MODEL_NAME = ((os.getenv("OMLX_MODEL") or os.getenv("MODEL_NAME") or os.getenv("LLM_DEFAULT_MODEL")) or "").strip()
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()
LLM_PROVIDER_DATA_COLLECTION = os.getenv("LLM_PROVIDER_DATA_COLLECTION", "").strip()
LLM_REQUIRE_PARAMETERS = os.getenv("LLM_REQUIRE_PARAMETERS", "").strip().lower() in {"1", "true", "yes", "on"}
LLM_ZERO_DATA_RETENTION = os.getenv("LLM_ZERO_DATA_RETENTION", "").strip().lower() in {"1", "true", "yes", "on"}
LLM_ALLOW_FALLBACKS_RAW = os.getenv("LLM_ALLOW_FALLBACKS", "").strip().lower()
LLM_TASK_NAMES = (
    "chat",
    "context",
    "context_summary",
    "context_analysis",
    "router",
    "summary",
    "rewrite",
    "prefetch",
    "prefetch_summary",
)
LLM_TASK_FALLBACKS = {
    "context_summary": ("context",),
    "context_analysis": ("context",),
    "prefetch_summary": ("prefetch", "summary"),
}
VAULT_CAPTURE_PATH = (
    os.getenv("VAULT_CAPTURE_PATH") or os.getenv("VAULT_HAIKU_PATH") or ""
).strip()
INBOX_API_BASE_URL = os.getenv("INBOX_API_BASE_URL", "http://localhost:8000").strip().rstrip("/")
INBOX_API_ACCESS_TOKEN = os.getenv("INBOX_API_ACCESS_TOKEN", "").strip()


def parse_session_inactive_ttl_seconds(raw_value: str | None) -> tuple[int | None, str | None]:
    if raw_value is None:
        return 86400, None

    stripped = raw_value.strip()
    if not stripped:
        return None, None

    try:
        ttl_seconds = int(stripped)
    except ValueError:
        return None, "SESSION_INACTIVE_TTL_SECONDS must be an integer number of seconds"

    if ttl_seconds < 0:
        return None, "SESSION_INACTIVE_TTL_SECONDS must be greater than or equal to 0"
    if ttl_seconds == 0:
        return None, None
    return ttl_seconds, None


SESSION_INACTIVE_TTL_SECONDS, SESSION_INACTIVE_TTL_CONFIG_ERROR = parse_session_inactive_ttl_seconds(
    os.getenv("SESSION_INACTIVE_TTL_SECONDS") or os.getenv("SESSION_IDLE_TTL_SECONDS")
)

# Tavily 클라이언트
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# 대화 기록
# conversations: LLM 컨텍스트에 보낼 최근 대화
# session_histories: /c 저장용 전체 세션 누적 대화
conversations: dict[SessionKey, list[ChatMessage]] = {}
session_histories: dict[SessionKey, list[ChatMessage]] = {}
source_memories: dict[SessionKey, list[SourceMemory]] = {}
active_source_sessions: set[SessionKey] = set()
inbox_context_prefetch_cache: dict[int, PrefetchedInboxContextSource] = {}
inbox_context_prefetch_inflight: set[int] = set()
inbox_context_prefetch_tasks: dict[int, asyncio.Task] = {}
inbox_context_prefetch_task: asyncio.Task | None = None
inbox_context_ready_list_api_available: bool | None = None
session_identifiers: dict[SessionKey, str] = {}
last_activity_at_by_session: dict[SessionKey, float] = {}
youtube_audio_transcription_semaphore: asyncio.Semaphore | None = None
MAX_HISTORY_PAIRS = 10
MAX_ASSISTANT_CONTEXT_CHARS = 700
MAX_RECENT_ASSISTANT_CONTEXT_CHARS = 1200
ASSISTANT_CONTEXT_TRUNCATION_NOTICE = (
    "\n\n[이전 AI 답변 일부 생략. 같은 내용을 반복하지 말고 최신 질문에 필요한 부분만 참고.]"
)
ASSISTANT_CONTEXT_TRUNCATION_MARKER = "..."
YOUTUBE_AUDIO_TRANSCRIPTION_MODEL = "mlx-community/whisper-large-v3-turbo"
YOUTUBE_AUDIO_WORKER_STREAM_LIMIT = 8 * 1024 * 1024
YOUTUBE_TRANSCRIPTION_FALLBACK_STATUSES = {
    "transcripts_disabled",
    "no_transcript_found",
    "request_blocked",
    "transcript_parse_error",
}


def build_system_prompt(model_name: str, today: datetime | None = None) -> str:
    current_date = (today or datetime.now()).strftime("%Y-%m-%d")
    return render_prompt_profile(model_name, variables={"today": current_date})


def validate_runtime_config() -> list[str]:
    missing: list[str] = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not get_llm_task_profile("chat").model:
        missing.append("LLM_CHAT_MODEL or MODEL_NAME")
    if SESSION_INACTIVE_TTL_CONFIG_ERROR:
        missing.append(SESSION_INACTIVE_TTL_CONFIG_ERROR)
    return missing


STREAM_EDIT_INTERVAL = 1.5  # 텔레그램 메시지 수정 간격 (초)
DRAFT_STREAM_INTERVAL = 0.9
DRAFT_STREAM_START_INTERVAL = 0.18
DRAFT_STREAM_START_CHARS = 80
DRAFT_STREAM_MIN_CHARS_DELTA = 40
TYPING_ACTION_INTERVAL = 2.5
TYPING_ACTION_RETRY_INTERVAL = 1.0
TYPING_ACTION_SEND_TIMEOUT = 3.0
TELEGRAM_TEXT_LIMIT = 4000
DEFAULT_CONTEXT_PROMPT = (
    "이 자료를 한국어로 요약해줘. 너무 짧게 압축하지 말고, "
    "무엇에 관한 자료인지 먼저 밝힌 뒤 핵심 주장과 중요한 근거를 짧은 단락이나 불릿으로 정리해줘. "
    "텔레그램에서 읽기 좋게 섹션 앞에는 🔹 같은 안내용 이모지를 가끔 쓰고, "
    "위험이나 주의가 필요한 부분에는 ⚠️만 써줘. 📌, 🔎, ➡️는 쓰지 마. "
    "장식처럼 남발하지 말고 모든 문단과 불릿은 들여쓰기 없이 왼쪽 정렬해줘."
)
RESPONSE_VALIDATION_FAILURE_TEXT = "⚠️ 응답이 한국어 출력 규칙을 통과하지 못했습니다. 다시 요청해 주세요."
DRAFT_STREAM_FINAL_FLUSH_DELAY = 0.30
MIN_REASONING_STATUS_CHARS = 2
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
TELEGRAM_RESPONSE_DELIVERY_MODES = {"final", "draft", "edit"}


def parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in TRUE_ENV_VALUES:
        return True
    if normalized in FALSE_ENV_VALUES:
        return False

    logging.getLogger(__name__).warning("%s must be a boolean value; using default %s", name, default)
    return default


def parse_telegram_response_delivery() -> str:
    raw_value = os.getenv("TELEGRAM_RESPONSE_DELIVERY")
    if raw_value is not None:
        mode = raw_value.strip().lower()
        if mode in TELEGRAM_RESPONSE_DELIVERY_MODES:
            return mode
        logging.getLogger(__name__).warning(
            "TELEGRAM_RESPONSE_DELIVERY must be one of %s; using final",
            ", ".join(sorted(TELEGRAM_RESPONSE_DELIVERY_MODES)),
        )
        return "final"

    legacy_draft_streaming = os.getenv("ENABLE_TELEGRAM_DRAFT_STREAMING")
    if legacy_draft_streaming is not None:
        return "draft" if parse_bool_env("ENABLE_TELEGRAM_DRAFT_STREAMING", True) else "edit"

    return "final"


def parse_enable_thinking_for_context() -> bool:
    if os.getenv("ENABLE_THINKING_FOR_CONTEXT") is not None:
        return parse_bool_env("ENABLE_THINKING_FOR_CONTEXT", False)

    if os.getenv("DISABLE_THINKING_FOR_CONTEXT") is not None:
        return not parse_bool_env("DISABLE_THINKING_FOR_CONTEXT", False)

    return False


TELEGRAM_RESPONSE_DELIVERY = parse_telegram_response_delivery()
ENABLE_THINKING_FOR_CONTEXT = parse_enable_thinking_for_context()


@dataclass(frozen=True)
class TypingIndicator:
    stop_event: asyncio.Event
    task: asyncio.Task


@dataclass(frozen=True)
class LLMTaskProfile:
    task: str
    provider_name: str
    model: str
    base_url: str
    api_key: str
    reasoning_effort: str
    provider_data_collection: str
    require_parameters: bool
    zero_data_retention: bool
    allow_fallbacks_raw: str


_active_typing_indicator: ContextVar[TypingIndicator | None] = ContextVar(
    "active_typing_indicator",
    default=None,
)
ENABLE_AUTO_SEARCH = parse_bool_env("ENABLE_AUTO_SEARCH", True)
ENABLE_RESPONSE_VALIDATION = parse_bool_env("ENABLE_RESPONSE_VALIDATION", True)
ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION = parse_bool_env("ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION", False)


def env_task_key(task: str) -> str:
    return task.upper().replace("-", "_")


def parse_bool_value(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_ENV_VALUES:
        return True
    if normalized in FALSE_ENV_VALUES:
        return False
    return default


def resolve_direct_task_env(task: str, suffix: str) -> str:
    task_key = env_task_key(task)
    return (
        os.getenv(f"LLM_{task_key}_{suffix}")
        or os.getenv(f"{task_key}_LLM_{suffix}")
        or ""
    ).strip()


def resolve_task_env(task: str, suffix: str, default: str = "") -> str:
    raw_value = resolve_direct_task_env(task, suffix)
    if raw_value:
        return raw_value
    for fallback_task in LLM_TASK_FALLBACKS.get(task, ()):
        fallback_value = resolve_direct_task_env(fallback_task, suffix)
        if fallback_value:
            return fallback_value
    return default.strip()


def normalize_provider_name(provider_name: str) -> str:
    return provider_name.strip().lower().replace("_", "-").replace(" ", "-")


def provider_env_key(provider_name: str) -> str:
    return normalize_provider_name(provider_name).upper().replace("-", "_")


def resolve_provider_env(provider_name: str, suffix: str) -> str:
    provider_key = provider_env_key(provider_name)
    return (os.getenv(f"LLM_PROVIDER_{provider_key}_{suffix}") or "").strip()


def resolve_provider_base_url(provider_name: str) -> str:
    normalized = normalize_provider_name(provider_name)
    registered_base_url = resolve_provider_env(provider_name, "BASE_URL")
    if registered_base_url:
        return registered_base_url
    if normalized in {"openrouter", "open-router"}:
        return "https://openrouter.ai/api/v1"
    if normalized in {"omlx", "o-mlx", "local", "local-llm", "local-provider"}:
        return os.getenv("OMLX_BASE_URL") or "http://localhost:1234/v1"
    if normalize_provider_name(LLM_PROVIDER_NAME) == normalized:
        return LLM_API_BASE_URL
    return ""


def resolve_provider_api_key(provider_name: str, base_url: str) -> str:
    normalized = normalize_provider_name(provider_name)
    registered_api_key = resolve_provider_env(provider_name, "API_KEY")
    if registered_api_key:
        return registered_api_key
    if normalized in {"openrouter", "open-router"} or "openrouter.ai" in base_url.lower():
        return OPENROUTER_API_KEY or LLM_DEFAULT_API_KEY or LLM_API_KEY
    if normalized in {"omlx", "o-mlx", "local", "local-llm", "local-provider"}:
        return os.getenv("OMLX_API_KEY", "")
    if normalize_provider_name(LLM_PROVIDER_NAME) == normalized:
        return LLM_API_KEY
    return ""


def resolve_task_api_key(task: str, provider_name: str, base_url: str) -> str:
    raw_key = resolve_task_env(task, "API_KEY")
    if raw_key:
        return raw_key
    normalized_provider = normalize_provider_name(provider_name)
    if normalized_provider in {"omlx", "o-mlx", "local", "local-llm", "local-provider"}:
        return os.getenv("OMLX_API_KEY", "")
    provider_key = resolve_provider_api_key(provider_name, base_url)
    if provider_key:
        return provider_key
    if "openrouter.ai" in base_url.lower():
        return OPENROUTER_API_KEY or LLM_API_KEY
    return LLM_API_KEY


def build_llm_task_profile(task: str) -> LLMTaskProfile:
    provider_name = resolve_task_env(task, "PROVIDER_NAME", LLM_PROVIDER_NAME)
    provider_base_url = resolve_provider_base_url(provider_name)
    base_url = resolve_task_env(task, "BASE_URL", provider_base_url or LLM_API_BASE_URL)
    reasoning_effort = resolve_task_env(task, "REASONING_EFFORT", LLM_REASONING_EFFORT)
    require_parameters_raw = resolve_task_env(task, "REQUIRE_PARAMETERS")
    zdr_raw = resolve_task_env(task, "ZERO_DATA_RETENTION")
    return LLMTaskProfile(
        task=task,
        provider_name=provider_name,
        model=resolve_task_env(task, "MODEL", MODEL_NAME),
        base_url=base_url,
        api_key=resolve_task_api_key(task, provider_name, base_url),
        reasoning_effort=reasoning_effort,
        provider_data_collection=resolve_task_env(
            task,
            "PROVIDER_DATA_COLLECTION",
            LLM_PROVIDER_DATA_COLLECTION,
        ),
        require_parameters=parse_bool_value(require_parameters_raw, LLM_REQUIRE_PARAMETERS),
        zero_data_retention=parse_bool_value(zdr_raw, LLM_ZERO_DATA_RETENTION),
        allow_fallbacks_raw=resolve_task_env(task, "ALLOW_FALLBACKS", LLM_ALLOW_FALLBACKS_RAW).lower(),
    )


def get_llm_task_profile(task: str = "chat") -> LLMTaskProfile:
    if task in LLM_TASK_NAMES:
        return build_llm_task_profile(task)
    return build_llm_task_profile("chat")


def parse_positive_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value.strip())
    except ValueError:
        logging.getLogger(__name__).warning("%s must be a positive number; using default %s", name, default)
        return default

    if parsed <= 0:
        logging.getLogger(__name__).warning("%s must be greater than 0; using default %s", name, default)
        return default
    return parsed


def parse_nonnegative_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value.strip())
    except ValueError:
        logging.getLogger(__name__).warning("%s must be a non-negative number; using default %s", name, default)
        return default

    if parsed < 0:
        logging.getLogger(__name__).warning("%s must be greater than or equal to 0; using default %s", name, default)
        return default
    return parsed


def parse_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value.strip())
    except ValueError:
        logging.getLogger(__name__).warning("%s must be a positive integer; using default %s", name, default)
        return default

    if parsed <= 0:
        logging.getLogger(__name__).warning("%s must be greater than 0; using default %s", name, default)
        return default
    return parsed


def parse_optional_path_env(name: str, default: str | None = None) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        raw_value = default
    if raw_value is None:
        return None

    parsed = raw_value.strip()
    return parsed or None


AUTO_SEARCH_CLASSIFIER_TIMEOUT_SECONDS = parse_positive_float_env(
    "AUTO_SEARCH_CLASSIFIER_TIMEOUT_SECONDS",
    12.0,
)
RESPONSE_REWRITE_TIMEOUT_SECONDS = parse_positive_float_env(
    "RESPONSE_REWRITE_TIMEOUT_SECONDS",
    45.0,
)
RESPONSE_REWRITE_MAX_ATTEMPTS = parse_positive_int_env(
    "RESPONSE_REWRITE_MAX_ATTEMPTS",
    3,
)
YOUTUBE_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS = parse_nonnegative_float_env(
    "YOUTUBE_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS",
    0,
)
YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS = parse_nonnegative_float_env(
    "YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS",
    900,
)
YOUTUBE_AUDIO_TRANSCRIPTION_MAX_CONCURRENT = parse_positive_int_env(
    "YOUTUBE_AUDIO_TRANSCRIPTION_MAX_CONCURRENT",
    1,
)
INBOX_API_TIMEOUT_SECONDS = parse_positive_float_env("INBOX_API_TIMEOUT_SECONDS", 10.0)
INBOX_CONTEXT_SUMMARY_TIMEOUT_SECONDS = parse_positive_float_env(
    "INBOX_CONTEXT_SUMMARY_TIMEOUT_SECONDS",
    45.0,
)
INBOX_CONTEXT_SUMMARY_MAX_INPUT_CHARS = parse_positive_int_env(
    "INBOX_CONTEXT_SUMMARY_MAX_INPUT_CHARS",
    12_000,
)
INBOX_CONTEXT_PREVIEW_CHARS = parse_positive_int_env(
    "INBOX_CONTEXT_PREVIEW_CHARS",
    700,
)
ENABLE_INBOX_CONTEXT_PREFETCH = parse_bool_env("ENABLE_INBOX_CONTEXT_PREFETCH", True)
ENABLE_INBOX_CONTEXT_PREFETCH_SUMMARY = parse_bool_env("ENABLE_INBOX_CONTEXT_PREFETCH_SUMMARY", False)
INBOX_CONTEXT_PREFETCH_TARGET = parse_positive_int_env("INBOX_CONTEXT_PREFETCH_TARGET", 5)
INBOX_CONTEXT_PREFETCH_INTERVAL_SECONDS = parse_positive_float_env(
    "INBOX_CONTEXT_PREFETCH_INTERVAL_SECONDS",
    60.0,
)
INBOX_CONTEXT_PREFETCH_CACHE_TTL_SECONDS = parse_positive_float_env(
    "INBOX_CONTEXT_PREFETCH_CACHE_TTL_SECONDS",
    72 * 60 * 60,
)
INBOX_CONTEXT_PREFETCH_STARTUP_TARGET = parse_positive_int_env(
    "INBOX_CONTEXT_PREFETCH_STARTUP_TARGET",
    2,
)
ENABLE_INBOX_CONTEXT_PREFETCH_STARTUP_SUMMARY = parse_bool_env(
    "ENABLE_INBOX_CONTEXT_PREFETCH_STARTUP_SUMMARY",
    False,
)
INBOX_CONTEXT_PREFETCH_PERSISTENT_CACHE_PATH = parse_optional_path_env(
    "INBOX_CONTEXT_PREFETCH_PERSISTENT_CACHE_PATH",
    ".cache/inbox-context-prefetch.sqlite3",
)
INBOX_CONTEXT_PREFETCH_SUMMARY_TIMEOUT_SECONDS = parse_positive_float_env(
    "INBOX_CONTEXT_PREFETCH_SUMMARY_TIMEOUT_SECONDS",
    180.0,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(Path(__file__).resolve().parent / "bot.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)
install_log_redaction_filter()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
if ENV_FILES_LOADED:
    logger.info("Loaded env files: %s", ", ".join(ENV_FILES_LOADED))
else:
    logger.warning("No env files were loaded.")
logger.info(
    "Runtime options telegram_response_delivery=%s enable_thinking_for_context=%s youtube_audio_transcription=%s inbox_context_prefetch=%s target=%s startup_target=%s prefetch_summary=%s startup_summary=%s persistent_prefetch_cache=%s",
    TELEGRAM_RESPONSE_DELIVERY,
    ENABLE_THINKING_FOR_CONTEXT,
    ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION,
    ENABLE_INBOX_CONTEXT_PREFETCH,
    INBOX_CONTEXT_PREFETCH_TARGET,
    INBOX_CONTEXT_PREFETCH_STARTUP_TARGET,
    ENABLE_INBOX_CONTEXT_PREFETCH_SUMMARY,
    ENABLE_INBOX_CONTEXT_PREFETCH_STARTUP_SUMMARY,
    bool(INBOX_CONTEXT_PREFETCH_PERSISTENT_CACHE_PATH),
)
logger.info(
    "LLM task profiles %s",
    " ".join(
        f"{task}={get_llm_task_profile(task).model or '(unset)'}"
        for task in LLM_TASK_NAMES
    ),
)


REPLACEMENT_CHAR = "\ufffd"
LATEX_COMMAND_REPLACEMENTS = {
    r"\to": "→",
    r"\rightarrow": "→",
    r"\Rightarrow": "⇒",
    r"\leftarrow": "←",
    r"\Leftarrow": "⇐",
    r"\leftrightarrow": "↔",
    r"\Leftrightarrow": "⇔",
    r"\mapsto": "↦",
}
GENERIC_TABLE_LABELS = {
    "구분",
    "분류",
    "항목",
    "기준",
    "category",
    "categories",
    "item",
    "items",
    "label",
    "labels",
}
FORBIDDEN_RESPONSE_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
FORBIDDEN_RESPONSE_CJK_SPAN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
CJK_OMISSION_PLACEHOLDER = "[중국어/일본어 원문 표기 생략]"
EXTRACT_ONLY_SUFFIX_RE = re.compile(r"(?:^|\s)/(?:e|extract|raw)\s*$", re.IGNORECASE)
CONTEXT_HEADER_LABELS = {
    "[X Post]": "X 포스트 원문",
    "[X Article]": "X 아티클 원문",
    "[YouTube Transcript]": "YouTube 스크립트",
    "[Web Article]": "웹페이지 본문",
    "[PDF Document]": "PDF 텍스트",
}
SOURCE_EXTRACT_LABELS = {
    "x": "X 포스트 원문",
    "youtube": "YouTube 스크립트",
    "pdf": "PDF 텍스트",
    "web": "웹페이지 본문",
}
SOURCE_CONTEXT_KINDS = set(SOURCE_EXTRACT_LABELS)
CONTEXT_HEADER_SOURCE_KINDS = {
    "[X Post]": "x",
    "[X Article]": "x",
    "[YouTube Transcript]": "youtube",
    "[Web Article]": "web",
    "[PDF Document]": "pdf",
}
SOURCE_DIRECT_CONTEXT_CHARS = 20_000
SOURCE_RETRIEVAL_CONTEXT_CHARS = 20_000
SOURCE_CHUNK_CHARS = 4_500
SOURCE_CHUNK_OVERLAP_CHARS = 450
SOURCE_RETRIEVAL_MAX_CHUNKS = 5
SOURCE_MEMORY_LIMIT = 4
SOURCE_QUERY_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
SOURCE_BROAD_QUERY_RE = re.compile(
    r"(요약|정리|핵심|전체|내용|줄거리|개요|요지|summary|summarize|overview|tl;?dr|gist)",
    re.IGNORECASE,
)
SOURCE_QUERY_STOPWORDS = {
    "이거",
    "이것",
    "내용",
    "요약",
    "정리",
    "핵심",
    "전체",
    "간단히",
    "한국어",
    "알려줘",
    "해줘",
    "해주세요",
    "설명",
    "분석",
    "그리고",
    "그래서",
    "어떻게",
    "무엇",
    "뭐야",
    "관련",
    "부분",
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "what",
    "how",
    "why",
    "summary",
    "summarize",
    "overview",
    "please",
}
SOURCE_FOLLOWUP_SKIP_RE = re.compile(
    r"(다른\s*얘기|다른\s*주제|방금\s*(거|것)\s*무시|컨텍스트\s*무시|원문\s*무시|"
    r"ignore\s+(the\s+)?(previous|context|source)|new\s+topic)",
    re.IGNORECASE,
)
CONTEXT_THINKING_TRIGGER_RE = re.compile(
    r"(깊(?:게|이)|자세히|비판|검토|분석|비교|평가|판단|추론|논증|쟁점|리스크|위험|전략|"
    r"의사결정|왜|어째서|원인|가능성|타당|반박|찬반|장단점|함의|시사점|"
    r"deep|detail|critical|critique|analy[sz]e|compare|evaluate|judge|reason|infer|"
    r"argument|risk|strategy|decision|why|cause|trade[- ]?off|pros?\s+and\s+cons?)",
    re.IGNORECASE,
)
EXPLICIT_RECENCY_SIGNAL_RE = re.compile(
    r"("
    r"오늘|어제|지금|현재|최신|최근|이번\s*(주|달|분기|해)|올해|내일|방금|"
    r"출시|발표|공개|업데이트|실적|가이던스|주가|시가총액|소비자\s*반응|시장\s*반응|"
    r"today|yesterday|tomorrow|now|current|currently|latest|recent|recently|"
    r"released?|launched?|announced?|updated?|earnings|guidance|stock\s*price|market\s*cap|"
    r"consumer\s*reaction|market\s*reaction"
    r")",
    re.IGNORECASE,
)
DYNAMIC_MARKET_SIGNAL_RE = re.compile(
    r"(\$[A-Z]{1,6}\b|\b(CPI|FOMC|GDP|PCE|EPS|SEC|FDA)\b|금리|환율|인플레이션|실업률|실적발표)",
    re.IGNORECASE,
)
SOURCE_LOCAL_REFERENCE_RE = re.compile(
    r"(방금\s*(넣은|보낸|첨부|적용)|이\s*(글|영상|자료|소스|컨텍스트|내용)|원문|첨부\s*(자료|영상|글))",
    re.IGNORECASE,
)
SOURCE_LOCAL_TASK_RE = re.compile(
    r"(요약|정리|핵심|무슨\s*내용|내용인지|설명|뜻|의미|summary|summarize|overview|tl;?dr|gist)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AutoSearchDecision:
    needs_search: bool
    query: str = ""
    reason: str = ""
    source: str = "none"


@dataclass(frozen=True)
class PendingYouTubeTranscription:
    video_id: str
    youtube_url: str
    canonical_youtube_url: str | None
    user_message: str
    extract_only_requested: bool
    requested_at: float
    failure_status: str
    failure_message: str
    title: str = ""
    channel: str = ""
    duration: int | None = None


@dataclass(frozen=True)
class ExtractedContext:
    user_message: str
    content: str
    source: str
    source_kind: str
    source_url: str | None = None


@dataclass(frozen=True)
class ContextExtractionResult:
    matched: bool
    extracted: ExtractedContext | None = None


@dataclass(frozen=True)
class InboxContextSource:
    source_id: int
    source_kind: str
    source_url: str | None
    title: str
    text: str
    remaining_ready_count: int


@dataclass(frozen=True)
class PrefetchedInboxContextSource:
    source: InboxContextSource
    enhanced: bool
    cached_at: float
    initial_reply: str | None = None
    original_source: InboxContextSource | None = None


@dataclass(frozen=True)
class SourceChunk:
    index: int
    text: str
    start: int
    end: int
    terms: frozenset[str]


@dataclass(frozen=True)
class SourceMemory:
    source_id: str
    source_kind: str
    source_url: str | None
    label: str
    content: str
    total_chars: int
    chunks: tuple[SourceChunk, ...]
    created_turn_index: int
    created_at: float


def build_chat_completions_url(profile: LLMTaskProfile | None = None) -> str:
    """OpenAI-compatible base URL에서 chat/completions endpoint 생성."""
    resolved_profile = profile or get_llm_task_profile("chat")
    return f"{resolved_profile.base_url.rstrip('/')}/chat/completions"


def build_llm_headers(profile: LLMTaskProfile | None = None) -> dict[str, str]:
    resolved_profile = profile or get_llm_task_profile("chat")
    headers = {"Content-Type": "application/json"}
    if resolved_profile.api_key:
        headers["Authorization"] = f"Bearer {resolved_profile.api_key}"
    return headers


def build_llm_provider_options(profile: LLMTaskProfile | None = None) -> dict:
    resolved_profile = profile or get_llm_task_profile("chat")
    provider: dict[str, object] = {}
    if resolved_profile.provider_data_collection:
        provider["data_collection"] = resolved_profile.provider_data_collection
    if resolved_profile.require_parameters:
        provider["require_parameters"] = True
    if resolved_profile.allow_fallbacks_raw in {"1", "true", "yes", "on"}:
        provider["allow_fallbacks"] = True
    elif resolved_profile.allow_fallbacks_raw in {"0", "false", "no", "off"}:
        provider["allow_fallbacks"] = False
    return provider


def apply_llm_request_options(
    payload: dict,
    *,
    allow_reasoning: bool = True,
    profile: LLMTaskProfile | None = None,
) -> dict:
    resolved_profile = profile or get_llm_task_profile("chat")
    if allow_reasoning and resolved_profile.reasoning_effort:
        payload["reasoning"] = {"effort": resolved_profile.reasoning_effort}

    provider = build_llm_provider_options(resolved_profile)
    if provider:
        payload["provider"] = provider

    if resolved_profile.zero_data_retention:
        payload["zdr"] = True

    return payload


def disable_llm_reasoning(payload: dict) -> dict:
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    payload["reasoning"] = {"effort": "none", "exclude": True, "enabled": False}
    payload["include_reasoning"] = False
    return payload


def should_force_auto_search(user_message: str) -> bool:
    """Hard guardrail for clearly time-sensitive prompts.

    The main router is the LLM classifier below. This only catches explicit
    current-event and market-data wording where skipping search is predictably
    worse than an unnecessary lookup.
    """

    stripped = user_message.strip()
    if not stripped:
        return False
    return bool(EXPLICIT_RECENCY_SIGNAL_RE.search(stripped) or DYNAMIC_MARKET_SIGNAL_RE.search(stripped))


def is_source_local_followup(user_message: str) -> bool:
    stripped = user_message.strip()
    if not stripped:
        return False
    return bool(SOURCE_LOCAL_REFERENCE_RE.search(stripped) and SOURCE_LOCAL_TASK_RE.search(stripped))


def _format_classifier_history(session_key: SessionKey | None, max_messages: int = 6) -> str:
    if session_key is None:
        return "(none)"

    history = conversations.get(session_key, [])[-max_messages:]
    if not history:
        return "(none)"

    formatted_messages = []
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "").replace("\n", " ").strip()
        if len(content) > 500:
            content = f"{content[:500]}..."
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def build_recency_classifier_messages(user_message: str, session_key: SessionKey | None = None) -> list[dict[str, str]]:
    today = datetime.now().strftime("%Y-%m-%d")
    history_text = _format_classifier_history(session_key)
    return [
        {
            "role": "system",
            "content": (
                "You are a routing classifier for a Telegram assistant. "
                "Decide whether answering the user's latest message requires up-to-date external information. "
                "Mark needs_search=true for events, availability, prices, market conditions, financials, regulations, "
                "product releases, public or consumer reaction, company/person status, or any claim that may have changed. "
                "If uncertain, mark needs_search=true. "
                "Return only JSON with keys: needs_search, query, reason."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Today: {today}\n\n"
                f"Recent conversation:\n{history_text}\n\n"
                f"Latest user message:\n{user_message}\n\n"
                "Return JSON only. The query should be a concise web search query in the user's language when useful."
            ),
        },
    ]


def extract_json_object(text: str) -> dict | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_recency_need(user_message: str, session_key: SessionKey | None = None) -> AutoSearchDecision:
    profile = get_llm_task_profile("router")
    payload = {
        "model": profile.model,
        "messages": build_recency_classifier_messages(user_message, session_key=session_key),
        "stream": False,
        "temperature": 0,
        "max_tokens": 220,
    }
    disable_llm_reasoning(payload)

    response = requests.post(
        build_chat_completions_url(profile),
        headers=build_llm_headers(profile),
        json=payload,
        timeout=AUTO_SEARCH_CLASSIFIER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return AutoSearchDecision(False, reason="classifier response missing message content", source="classifier")

    parsed = extract_json_object(content)
    if not parsed:
        return AutoSearchDecision(False, reason="classifier response was not valid JSON", source="classifier")

    needs_search = bool(parsed.get("needs_search"))
    query = str(parsed.get("query") or "").strip()
    reason = str(parsed.get("reason") or "").strip()
    return AutoSearchDecision(
        needs_search=needs_search,
        query=query or user_message.strip(),
        reason=reason,
        source="classifier",
    )


def resolve_auto_search_decision(
    user_message: str,
    session_key: SessionKey | None = None,
) -> AutoSearchDecision:
    if not ENABLE_AUTO_SEARCH:
        return AutoSearchDecision(False, reason="auto search disabled", source="disabled")
    if tavily is None:
        return AutoSearchDecision(False, reason="missing TAVILY_API_KEY", source="missing_tavily")
    active_source_context = session_key is not None and should_apply_active_source_context(session_key, user_message)
    if active_source_context and is_source_local_followup(user_message):
        return AutoSearchDecision(False, reason="active source local follow-up", source="source_context")
    if should_force_auto_search(user_message):
        return AutoSearchDecision(True, query=user_message.strip(), reason="explicit recency signal", source="guardrail")

    try:
        return classify_recency_need(user_message, session_key=session_key)
    except Exception as exc:
        logger.warning("Auto-search classifier failed: %s", exc)
        return AutoSearchDecision(False, reason=str(exc), source="classifier_error")


def build_capture_session_id(chat_id: int, message_id: int) -> str:
    return f"tg:{chat_id}:{message_id}"


def build_capture_session_marker(session_id: str) -> str:
    return f"{CAPTURE_SESSION_ID_MARKER_PREFIX}{session_id} -->"


def find_matching_url(text: str, pattern: re.Pattern[str]) -> str | None:
    for match in GENERAL_URL_PATTERN.finditer(text):
        candidate = match.group(0)
        if pattern.search(candidate):
            return candidate

    pattern_match = pattern.search(text)
    if not pattern_match:
        return None
    return pattern_match.group(0)


def remove_url_once(text: str, url: str | None) -> str:
    if not url:
        return text.strip()
    return re.sub(r"\s{2,}", " ", text.replace(url, "", 1)).strip()


def parse_extract_only_request(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    cleaned = EXTRACT_ONLY_SUFFIX_RE.sub("", stripped).strip()
    return cleaned, cleaned != stripped


def strip_context_header(content: str) -> tuple[str | None, str]:
    stripped = content.strip()
    if not stripped:
        return None, ""

    lines = stripped.splitlines()
    first_line = lines[0].strip()
    label = CONTEXT_HEADER_LABELS.get(first_line)
    if not label:
        return None, stripped
    return label, "\n".join(lines[1:]).strip()


def format_extract_only_text(content: str, source_kind: str) -> str:
    header_label, body = strip_context_header(content)
    label = header_label or SOURCE_EXTRACT_LABELS.get(source_kind, "추출 원문")
    return f"{label}\n\n{body or content.strip()}".strip()


def infer_source_kind_from_context(content: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    return CONTEXT_HEADER_SOURCE_KINDS.get(first_line)


def extract_source_terms(text: str) -> frozenset[str]:
    terms = {
        token.casefold()
        for token in SOURCE_QUERY_TOKEN_RE.findall(text)
        if token.casefold() not in SOURCE_QUERY_STOPWORDS
    }
    return frozenset(terms)


def build_source_chunks(content: str) -> tuple[SourceChunk, ...]:
    stripped = content.strip()
    if not stripped:
        return ()

    chunks: list[SourceChunk] = []
    start = 0
    text_length = len(stripped)
    overlap = min(SOURCE_CHUNK_OVERLAP_CHARS, SOURCE_CHUNK_CHARS // 3)

    while start < text_length:
        end = min(start + SOURCE_CHUNK_CHARS, text_length)
        chunk_text = stripped[start:end].strip()
        if chunk_text:
            chunks.append(
                SourceChunk(
                    index=len(chunks),
                    text=chunk_text,
                    start=start,
                    end=end,
                    terms=extract_source_terms(chunk_text),
                )
            )
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return tuple(chunks)


def source_memory_id(source_kind: str, source_url: str | None, content: str) -> str:
    source_key = source_url or content[:80].replace("\n", " ")
    return f"{source_kind}:{source_key}:{len(content)}"


def register_source_memory(
    session_key: SessionKey,
    content: str,
    source_kind: str | None = None,
    source_url: str | None = None,
) -> SourceMemory | None:
    stripped_content = content.strip()
    resolved_source_kind = source_kind or infer_source_kind_from_context(content)
    if resolved_source_kind not in SOURCE_CONTEXT_KINDS:
        return None

    chunks = build_source_chunks(stripped_content)
    if not chunks:
        return None

    header_label, _ = strip_context_header(content)
    memory = SourceMemory(
        source_id=source_memory_id(resolved_source_kind, source_url, stripped_content),
        source_kind=resolved_source_kind,
        source_url=source_url,
        label=header_label or SOURCE_EXTRACT_LABELS.get(resolved_source_kind, "참고 자료"),
        content=stripped_content,
        total_chars=len(stripped_content),
        chunks=chunks,
        created_turn_index=len(conversations.get(session_key, [])),
        created_at=time.time(),
    )

    memories = [
        existing
        for existing in source_memories.get(session_key, [])
        if existing.source_id != memory.source_id
    ]
    memories.append(memory)
    source_memories[session_key] = memories[-SOURCE_MEMORY_LIMIT:]
    return memory


def latest_source_memory(session_key: SessionKey) -> SourceMemory | None:
    memories = source_memories.get(session_key) or []
    return memories[-1] if memories else None


def is_broad_source_query(user_message: str, query_terms: frozenset[str]) -> bool:
    stripped = user_message.strip()
    if not stripped:
        return True
    if not query_terms:
        return True
    return bool(SOURCE_BROAD_QUERY_RE.search(stripped) and len(query_terms) <= 2)


def representative_chunk_indices(chunk_count: int, max_chunks: int) -> list[int]:
    if chunk_count <= 0:
        return []
    if chunk_count <= max_chunks:
        return list(range(chunk_count))
    if max_chunks <= 1:
        return [0]

    indices = {
        round(position * (chunk_count - 1) / (max_chunks - 1))
        for position in range(max_chunks)
    }
    return sorted(indices)


def score_source_chunk(chunk: SourceChunk, query_terms: frozenset[str], normalized_query: str) -> float:
    lowered = chunk.text.casefold()
    score = 0.0
    for term in query_terms:
        count = lowered.count(term)
        if count:
            score += 2.0 + count
        if term in chunk.terms:
            score += 1.5
    if normalized_query and len(normalized_query) >= 4 and normalized_query in lowered:
        score += 5.0
    if score > 0 and chunk.index == 0:
        score += 0.25
    return score


def select_source_chunks(memory: SourceMemory, user_message: str) -> tuple[list[SourceChunk], str]:
    chunks = list(memory.chunks)
    if not chunks:
        return [], "empty"

    query_terms = extract_source_terms(user_message)
    if is_broad_source_query(user_message, query_terms):
        indices = representative_chunk_indices(len(chunks), SOURCE_RETRIEVAL_MAX_CHUNKS)
        return [chunks[index] for index in indices], "representative"

    normalized_query = " ".join(user_message.casefold().split())
    scored = [
        (score_source_chunk(chunk, query_terms, normalized_query), chunk)
        for chunk in chunks
    ]
    positive_chunks = [(score, chunk) for score, chunk in scored if score > 0]
    if not positive_chunks:
        indices = representative_chunk_indices(len(chunks), SOURCE_RETRIEVAL_MAX_CHUNKS)
        return [chunks[index] for index in indices], "representative"

    selected: list[SourceChunk] = []
    selected_indexes: set[int] = set()
    selected_chars = 0
    for score, chunk in sorted(positive_chunks, key=lambda item: (-item[0], item[1].index)):
        if chunk.index in selected_indexes:
            continue
        projected_chars = selected_chars + len(chunk.text)
        if selected and projected_chars > SOURCE_RETRIEVAL_CONTEXT_CHARS:
            continue
        selected.append(chunk)
        selected_indexes.add(chunk.index)
        selected_chars = projected_chars
        if len(selected) >= SOURCE_RETRIEVAL_MAX_CHUNKS:
            break

    return sorted(selected, key=lambda chunk: chunk.index), "relevant"


def build_retrieved_source_context(memory: SourceMemory, user_message: str) -> str:
    if memory.total_chars <= SOURCE_DIRECT_CONTEXT_CHARS:
        return memory.content

    selected_chunks, mode = select_source_chunks(memory, user_message)
    selected_numbers = ", ".join(str(chunk.index + 1) for chunk in selected_chunks) or "none"
    source_label = SOURCE_KIND_LABELS.get(memory.source_kind, memory.label)
    header_parts = [
        "[Retrieved Source Context]",
        f"Source: {source_label}",
        f"Original source chars: {memory.total_chars}",
        f"Chunks selected: {selected_numbers} / {len(memory.chunks)}",
        f"Selection mode: {mode}",
    ]
    if memory.source_url:
        header_parts.append(f"URL: {memory.source_url}")
    header_parts.extend(
        [
            "",
            "[Source Retrieval Rules]",
            "- The full source is stored in session memory as chunks; only the selected chunks are shown here.",
            "- Use the selected chunks as source evidence for the user's current question.",
            "- If a requested detail is not supported by the selected chunks, say it is not visible in the retrieved chunks.",
            "- For broad summaries, synthesize across the representative chunks without claiming exhaustive coverage of omitted chunks.",
        ]
    )

    body_parts = []
    for chunk in selected_chunks:
        body_parts.append(
            f"[Source Chunk {chunk.index + 1}/{len(memory.chunks)} chars {chunk.start}-{chunk.end}]\n"
            f"{chunk.text}"
        )

    return "\n".join(header_parts + [""] + body_parts).strip()


def build_combined_source_and_search_context(source_context: str, search_context: str) -> str:
    return (
        "[Combined Source And Web Search Context]\n"
        "[Context Combination Rules]\n"
        "- The prior source context is the user's active reference material.\n"
        "- The web search results provide current external information.\n"
        "- Use both when the user's question depends on the source and current facts.\n"
        "- If they conflict, say so explicitly and prefer web search results for time-sensitive facts.\n\n"
        "[Prior Source Context]\n"
        f"{source_context.strip()}\n\n"
        "[Current Web Search Context]\n"
        f"{search_context.strip()}"
    )


def resolve_source_context_for_request(
    session_key: SessionKey,
    user_message: str,
    search_context: str,
    source_kind: str | None = None,
    source_url: str | None = None,
    *,
    use_source_memory_for_context: bool = False,
    preserve_active_source_context: bool = False,
) -> tuple[str, str | None, str | None, bool]:
    if search_context:
        if preserve_active_source_context and should_apply_active_source_context(session_key, user_message):
            memory = latest_source_memory(session_key)
            if memory:
                source_context = build_retrieved_source_context(memory, user_message)
                return (
                    build_combined_source_and_search_context(source_context, search_context),
                    memory.source_kind,
                    memory.source_url,
                    False,
                )
        if not use_source_memory_for_context:
            active_source_sessions.discard(session_key)
            return search_context, source_kind, source_url, True

        memory = register_source_memory(session_key, search_context, source_kind, source_url)
        if memory:
            active_source_sessions.add(session_key)
        return search_context, source_kind, source_url, True

    if not should_apply_active_source_context(session_key, user_message):
        return "", source_kind, source_url, False

    memory = latest_source_memory(session_key)
    if not memory:
        return "", source_kind, source_url, False
    return build_retrieved_source_context(memory, user_message), memory.source_kind, memory.source_url, False


def build_inbox_api_url(path: str) -> str:
    if not INBOX_API_BASE_URL:
        return ""
    return f"{INBOX_API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def build_inbox_api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if INBOX_API_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {INBOX_API_ACCESS_TOKEN}"
    return headers


def parse_inbox_context_source_dict(source: dict, remaining_ready_count: int = 0) -> InboxContextSource:
    if not isinstance(source, dict):
        raise ValueError("Inbox context source response was malformed.")

    source_id = source.get("id")
    text = str(source.get("text") or "").strip()
    source_kind = str(source.get("kind") or "web").strip() or "web"
    source_url = str(source.get("url") or "").strip() or None
    title = str(source.get("title") or "").strip()
    if not isinstance(source_id, int) or not text:
        raise ValueError("Inbox context source response missed id or text.")

    return InboxContextSource(
        source_id=source_id,
        source_kind=source_kind,
        source_url=source_url,
        title=title,
        text=text,
        remaining_ready_count=remaining_ready_count,
    )


def parse_inbox_context_source_payload(data: dict) -> InboxContextSource | None:
    source = data.get("source")
    if source is None:
        return None

    try:
        remaining_ready_count = int(data.get("remaining_ready_count") or 0)
    except (TypeError, ValueError):
        remaining_ready_count = 0

    return parse_inbox_context_source_dict(source, remaining_ready_count)


def parse_inbox_context_sources_payload(data: dict) -> list[InboxContextSource]:
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Inbox context source list response was malformed.")

    ready_count = data.get("ready_count", data.get("remaining_ready_count"))
    try:
        total_ready = int(ready_count) if ready_count is not None else len(sources)
    except (TypeError, ValueError):
        total_ready = len(sources)

    parsed: list[InboxContextSource] = []
    for index, source in enumerate(sources):
        remaining_ready_count = max(total_ready - index - 1, 0)
        parsed.append(parse_inbox_context_source_dict(source, remaining_ready_count))
    return parsed


def fetch_next_inbox_context_source() -> InboxContextSource | None:
    url = build_inbox_api_url("/api/context-sources/next")
    if not url:
        raise ValueError("INBOX_API_BASE_URL is empty.")

    response = requests.get(
        url,
        headers=build_inbox_api_headers(),
        timeout=INBOX_API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return parse_inbox_context_source_payload(response.json())


def fetch_ready_inbox_context_sources(limit: int) -> list[InboxContextSource] | None:
    global inbox_context_ready_list_api_available

    if inbox_context_ready_list_api_available is False:
        return None

    url = build_inbox_api_url("/api/context-sources/ready")
    if not url:
        raise ValueError("INBOX_API_BASE_URL is empty.")

    response = requests.get(
        f"{url}?{urlencode({'limit': max(1, limit)})}",
        headers=build_inbox_api_headers(),
        timeout=INBOX_API_TIMEOUT_SECONDS,
    )
    if response.status_code in {404, 405}:
        inbox_context_ready_list_api_available = False
        logger.info("Inbox context ready list API is unavailable; falling back to /next prefetch.")
        return None

    response.raise_for_status()
    inbox_context_ready_list_api_available = True
    return parse_inbox_context_sources_payload(response.json())


def fetch_prefetchable_inbox_context_sources(limit: int) -> list[InboxContextSource]:
    if limit <= 0:
        return []

    ready_sources = fetch_ready_inbox_context_sources(limit)
    if ready_sources is not None:
        return ready_sources[:limit]

    source = fetch_next_inbox_context_source()
    return [source] if source is not None else []


def mark_inbox_context_source_consumed(source_id: int) -> None:
    url = build_inbox_api_url(f"/api/context-sources/{source_id}/consume")
    if not url:
        raise ValueError("INBOX_API_BASE_URL is empty.")

    response = requests.post(
        url,
        headers=build_inbox_api_headers(),
        json={"consumer": "telegram-llm-bot"},
        timeout=INBOX_API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def get_inbox_context_prefetch_persistent_cache() -> InboxPrefetchPersistentCache:
    return InboxPrefetchPersistentCache(
        raw_path=INBOX_CONTEXT_PREFETCH_PERSISTENT_CACHE_PATH,
        ttl_seconds=INBOX_CONTEXT_PREFETCH_CACHE_TTL_SECONDS,
        base_dir=Path(__file__).resolve().parent,
        logger=logger,
    )


def resolve_inbox_context_prefetch_persistent_cache_path() -> Path | None:
    return get_inbox_context_prefetch_persistent_cache().resolve_path()


def serialize_inbox_context_source(source: InboxContextSource) -> str:
    return json.dumps(
        {
            "id": source.source_id,
            "kind": source.source_kind,
            "url": source.source_url,
            "title": source.title,
            "text": source.text,
            "remaining_ready_count": source.remaining_ready_count,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def deserialize_inbox_context_source(payload: str) -> InboxContextSource:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("serialized inbox context source must decode to a dict")
    try:
        remaining_ready_count = int(data.get("remaining_ready_count") or 0)
    except (TypeError, ValueError):
        remaining_ready_count = 0
    return parse_inbox_context_source_dict(data, remaining_ready_count)


def build_inbox_context_prefetch_summary_signature() -> str:
    profile = get_llm_task_profile("prefetch_summary")
    signature_payload = {
        "model_name": profile.model,
        "system_prompt": build_system_prompt(profile.model),
        "context_prompt": build_context_prompt(DEFAULT_CONTEXT_PROMPT),
        "enable_thinking_for_context": ENABLE_THINKING_FOR_CONTEXT,
        "summary_enabled": ENABLE_INBOX_CONTEXT_PREFETCH_SUMMARY,
    }
    return hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_cached_youtube_audio_transcript(video_id: str) -> str | None:
    config = load_youtube_audio_transcription_config()
    path = transcript_cache_path(config, video_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("YouTube transcript cache read failed video_id=%s path=%s: %s", video_id, path, exc)
        return None
    if not text:
        return None
    return f"[YouTube Transcript]\n{text}"


def build_prefetched_inbox_context_source(
    original_source: InboxContextSource,
    hydrated_source: InboxContextSource,
    *,
    enhanced: bool,
    initial_reply: str | None,
    cached_at: float | None = None,
) -> PrefetchedInboxContextSource:
    return PrefetchedInboxContextSource(
        source=hydrated_source,
        enhanced=enhanced,
        cached_at=time.time() if cached_at is None else cached_at,
        initial_reply=initial_reply,
        original_source=original_source,
    )


def inbox_context_sources_match(left: InboxContextSource, right: InboxContextSource) -> bool:
    return (
        left.source_id == right.source_id
        and left.source_kind == right.source_kind
        and (left.source_url or "") == (right.source_url or "")
        and left.title == right.title
        and left.text == right.text
    )


def prefetched_inbox_context_source_matches(
    source: InboxContextSource,
    cached: PrefetchedInboxContextSource,
) -> bool:
    original_source = cached.original_source or cached.source
    return inbox_context_sources_match(source, original_source)


def load_persistent_prefetched_inbox_context_source(source_id: int) -> PrefetchedInboxContextSource | None:
    record = get_inbox_context_prefetch_persistent_cache().load(source_id)
    if record is None:
        return None

    initial_reply = record.initial_reply
    if initial_reply and (
        not ENABLE_INBOX_CONTEXT_PREFETCH_SUMMARY
        or record.summary_signature != build_inbox_context_prefetch_summary_signature()
    ):
        initial_reply = None

    return PrefetchedInboxContextSource(
        source=deserialize_inbox_context_source(record.hydrated_source_json),
        enhanced=record.enhanced,
        cached_at=record.cached_at,
        initial_reply=initial_reply,
        original_source=deserialize_inbox_context_source(record.original_source_json),
    )


def persist_prefetched_inbox_context_source(cached: PrefetchedInboxContextSource) -> None:
    original_source = cached.original_source or cached.source
    record = PersistentInboxPrefetchRecord(
        source_id=cached.source.source_id,
        original_source_json=serialize_inbox_context_source(original_source),
        hydrated_source_json=serialize_inbox_context_source(cached.source),
        enhanced=cached.enhanced,
        initial_reply=cached.initial_reply,
        cached_at=cached.cached_at,
        summary_signature=build_inbox_context_prefetch_summary_signature() if cached.initial_reply else "",
    )
    get_inbox_context_prefetch_persistent_cache().save(record)


def delete_persistent_prefetched_inbox_context_source(source_id: int) -> None:
    get_inbox_context_prefetch_persistent_cache().delete(source_id)


def purge_stale_persistent_inbox_context_prefetch_cache(now: float | None = None) -> None:
    get_inbox_context_prefetch_persistent_cache().purge_stale(now=now)


def clear_persistent_inbox_context_prefetch_cache() -> None:
    get_inbox_context_prefetch_persistent_cache().clear()


def build_inbox_context_summary_messages(source: InboxContextSource) -> list[dict[str, str]]:
    title = source.title or SOURCE_KIND_LABELS.get(source.source_kind, "컨텍스트 소스")
    _, body = strip_context_header(source.text)
    source_text = (body or source.text).strip()
    if len(source_text) > INBOX_CONTEXT_SUMMARY_MAX_INPUT_CHARS:
        source_text = f"{source_text[:INBOX_CONTEXT_SUMMARY_MAX_INPUT_CHARS].rstrip()}\n\n[이하 원문 생략]"

    return [
        {
            "role": "system",
            "content": (
                "You summarize a newly applied context source for a Telegram user. "
                "Answer only in Korean. Be concise and concrete. "
                "Explain what the source is about, then list the core points the user should know. "
                "Do not use web search or outside knowledge."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[Source]\n"
                f"- kind: {source.source_kind}\n"
                f"- title: {title}\n"
                f"- url: {source.source_url or '(none)'}\n\n"
                f"[Content]\n{source_text}\n\n"
                "이 컨텍스트를 막 세션에 적용했다. 사용자가 맥락을 바로 잡을 수 있게 "
                "5줄 이내로 요약해줘."
            ),
        },
    ]


def summarize_inbox_context_source(source: InboxContextSource) -> str:
    profile = get_llm_task_profile("summary")
    payload = {
        "model": profile.model,
        "messages": build_inbox_context_summary_messages(source),
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 500,
    }
    disable_llm_reasoning(payload)
    response = requests.post(
        build_chat_completions_url(profile),
        headers=build_llm_headers(profile),
        json=payload,
        timeout=INBOX_CONTEXT_SUMMARY_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("summary response missing message content")

    summary = normalize_response_text(str(content))
    if not summary:
        raise ValueError("summary response was empty")
    return summary


def generate_inbox_context_initial_reply(source: InboxContextSource) -> str:
    profile = get_llm_task_profile("prefetch_summary")
    messages = [
        {"role": "system", "content": build_system_prompt(profile.model)},
        {
            "role": "user",
            "content": build_augmented_context_message(
                build_context_prompt(DEFAULT_CONTEXT_PROMPT),
                source.text,
            ),
        },
    ]
    payload = {
        "model": profile.model,
        "messages": messages,
        "stream": False,
    }
    if source.text and not ENABLE_THINKING_FOR_CONTEXT:
        disable_llm_reasoning(payload)
    else:
        apply_llm_request_options(payload, profile=profile)

    response = requests.post(
        build_chat_completions_url(profile),
        headers=build_llm_headers(profile),
        json=payload,
        timeout=INBOX_CONTEXT_PREFETCH_SUMMARY_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("initial context reply missing message content")

    reply = normalize_response_text(str(content))
    if not reply:
        raise ValueError("initial context reply was empty")
    return reply


def build_inbox_context_preview(source: InboxContextSource) -> str:
    _, body = strip_context_header(source.text)
    preview = re.sub(r"\s+", " ", (body or source.text).strip())
    if len(preview) > INBOX_CONTEXT_PREVIEW_CHARS:
        preview = f"{preview[:INBOX_CONTEXT_PREVIEW_CHARS].rstrip()}..."
    return preview


def build_inbox_context_applied_reply(
    source: InboxContextSource,
    summary: str | None = None,
) -> str:
    title = source.title or SOURCE_KIND_LABELS.get(source.source_kind, "컨텍스트 소스")
    lines = [
        f"컨텍스트 적용됨: #{source.source_id} {title}",
    ]
    if source.source_url:
        lines.append(f"출처: {source.source_url}")
    lines.append(f"남은 준비된 컨텍스트 큐: {source.remaining_ready_count}개")

    if summary:
        lines.extend(["", "요약", summary.strip()])
    else:
        lines.extend(["", "미리보기", build_inbox_context_preview(source)])

    reply = "\n".join(lines).strip()
    if len(reply) <= TELEGRAM_TEXT_LIMIT:
        return reply
    return f"{reply[: TELEGRAM_TEXT_LIMIT - 3].rstrip()}..."


def build_inbox_context_status_reply(
    source: InboxContextSource,
    *,
    enhanced: bool = False,
) -> str:
    title = source.title or SOURCE_KIND_LABELS.get(source.source_kind, "컨텍스트 소스")
    lines = [f"컨텍스트 적용됨: #{source.source_id} {title}"]
    lines.append(f"본문: {len(source.text):,}자")
    if enhanced:
        lines.append("YouTube 원문으로 본문을 보강했습니다.")
    lines.append(f"남은 준비된 컨텍스트 큐: {source.remaining_ready_count}개")

    reply = "\n".join(lines).strip()
    if len(reply) <= TELEGRAM_TEXT_LIMIT:
        return reply
    return f"{reply[: TELEGRAM_TEXT_LIMIT - 3].rstrip()}..."


def build_inbox_context_processing_reply(source: InboxContextSource) -> str:
    title = source.title or SOURCE_KIND_LABELS.get(source.source_kind, "컨텍스트 소스")
    lines = [
        f"컨텍스트 준비 중: #{source.source_id} {title}",
        f"종류: {SOURCE_KIND_LABELS.get(source.source_kind, source.source_kind)}",
    ]
    if source.source_url:
        lines.append(f"출처: {source.source_url}")
    lines.append(f"먼저 읽은 본문: {len(source.text):,}자")
    if source.source_kind == "youtube" and source.source_url:
        lines.append("YouTube 자막을 확인해 필요하면 오디오 전사로 보강합니다.")

    reply = "\n".join(lines).strip()
    if len(reply) <= TELEGRAM_TEXT_LIMIT:
        return reply
    return f"{reply[: TELEGRAM_TEXT_LIMIT - 3].rstrip()}..."


def apply_inbox_context_source_to_session(
    session_key: SessionKey,
    source: InboxContextSource,
) -> SourceMemory:
    memory = register_source_memory(
        session_key,
        source.text,
        source.source_kind,
        source.source_url,
    )
    if memory is None:
        raise ValueError("Inbox context source could not be registered in session memory.")
    active_source_sessions.add(session_key)
    touch_session_activity(session_key)
    return memory


def _ensure_url_scheme(url: str) -> str:
    if "://" in url:
        return url
    return f"https://{url}"


def _sanitize_source_url(parsed_url) -> str:
    scheme = (parsed_url.scheme or "https").lower()
    hostname = (parsed_url.hostname or "").lower()
    if not hostname:
        return ""

    netloc = hostname
    if parsed_url.port is not None:
        netloc = f"{hostname}:{parsed_url.port}"

    sanitized = parsed_url._replace(
        scheme=scheme,
        netloc=netloc,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(sanitized)


def normalize_source_url(raw_url: str, source_kind: str) -> str | None:
    stripped = raw_url.strip()
    if not stripped:
        return None

    if source_kind == "youtube":
        match = YOUTUBE_URL_PATTERN.search(stripped)
        if not match:
            return None
        return f"https://www.youtube.com/watch?v={match.group(1)}"

    if source_kind == "x":
        match = TWEET_URL_PATTERN.search(stripped)
        if not match:
            return None
        parsed = urlparse(_ensure_url_scheme(match.group(0)))
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 3 and path_parts[1] == "status":
            username = path_parts[0]
            tweet_id = path_parts[2]
            return f"https://x.com/{username}/status/{tweet_id}"
        return f"https://x.com/i/status/{match.group(1)}"

    parsed = urlparse(_ensure_url_scheme(stripped))
    normalized = _sanitize_source_url(parsed)
    return normalized or None


def infer_source_type(context_part: str, source_kind: str | None = None) -> str:
    if source_kind:
        label = SOURCE_KIND_LABELS.get(source_kind)
        if label:
            return label

    if "[X Post]" in context_part or "[X Article]" in context_part:
        return "X 포스트"
    if "[YouTube Transcript]" in context_part:
        return "YouTube"
    if "[Web Article]" in context_part:
        return "웹 아티클"
    if "[Web Search Results]" in context_part:
        return "웹 검색"
    if "[PDF Document]" in context_part:
        return "PDF"
    return "참고 자료"


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "알 수 없음"
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분 {secs}초"
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def youtube_audio_transcription_runtime_issue() -> str | None:
    if not ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION:
        return "오디오 전사 fallback이 비활성화되어 있습니다."
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return "오디오 전사에는 ffmpeg와 ffprobe가 필요합니다."
    return None


def youtube_audio_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parent / "src"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not existing_pythonpath else f"{src_dir}{os.pathsep}{existing_pythonpath}"
    return env


def youtube_audio_worker_command(*args: str) -> list[str]:
    command = [sys.executable, "-m", "telegram_llm_bot.youtube_audio_transcription"]
    if args and args[0] in {"metadata", "transcribe"} and len(args) > 1:
        return [*command, args[0], "--", *args[1:]]
    return [*command, *args]


def parse_json_lines(text: str) -> list[dict]:
    parsed: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def fetch_youtube_audio_metadata_for_prompt(video_id: str) -> tuple[dict | None, bool, str]:
    try:
        completed = subprocess.run(
            youtube_audio_worker_command("metadata", video_id),
            cwd=Path(__file__).resolve().parent,
            env=youtube_audio_worker_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        logger.warning("YouTube audio metadata lookup failed: %s", exc)
        return None, True, ""

    json_lines = parse_json_lines(completed.stdout)
    if not json_lines:
        if completed.stderr.strip():
            logger.warning("YouTube audio metadata lookup stderr: %s", completed.stderr.strip()[-500:])
        return None, completed.returncode == 0, ""
    payload = json_lines[-1]
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
    return metadata, bool(payload.get("ok", completed.returncode == 0)), str(payload.get("reason") or "")


def build_youtube_auto_transcription_start_reply(pending: PendingYouTubeTranscription) -> str:
    return "\n".join(
        [
            "🎙️ 공개 자막을 가져오지 못해 오디오 전사를 바로 시작합니다.",
            f"사유: {pending.failure_message}",
        ]
    )


def youtube_transcript_method_label(result: YouTubeTranscriptExtractionResult) -> str:
    parts = [part for part in (result.selection, result.language_code) if part]
    if result.is_generated is not None:
        parts.append("generated" if result.is_generated else "manual")
    return "/".join(parts) or result.status


def build_pending_youtube_transcription(
    video_id: str,
    youtube_url: str,
    canonical_youtube_url: str | None,
    user_message: str,
    extract_only_requested: bool,
    yt_result: YouTubeTranscriptExtractionResult,
) -> tuple[PendingYouTubeTranscription | None, str | None]:
    if yt_result.status not in YOUTUBE_TRANSCRIPTION_FALLBACK_STATUSES:
        return None, None

    runtime_issue = youtube_audio_transcription_runtime_issue()
    if runtime_issue:
        return None, runtime_issue

    metadata, allowed, reason = fetch_youtube_audio_metadata_for_prompt(video_id)
    if not allowed:
        return None, reason or "오디오 전사 요청이 설정 한도를 넘었습니다."

    return (
        PendingYouTubeTranscription(
            video_id=video_id,
            youtube_url=youtube_url,
            canonical_youtube_url=canonical_youtube_url,
            user_message=user_message,
            extract_only_requested=extract_only_requested,
            requested_at=time.time(),
            failure_status=yt_result.status,
            failure_message=yt_result.message or "공개 스크립트/자막을 가져올 수 없습니다.",
            title=str((metadata or {}).get("title") or ""),
            channel=str((metadata or {}).get("channel") or ""),
            duration=(metadata or {}).get("duration") if isinstance((metadata or {}).get("duration"), int) else None,
        ),
        None,
    )


def get_youtube_audio_transcription_semaphore() -> asyncio.Semaphore:
    global youtube_audio_transcription_semaphore
    if youtube_audio_transcription_semaphore is None:
        youtube_audio_transcription_semaphore = asyncio.Semaphore(YOUTUBE_AUDIO_TRANSCRIPTION_MAX_CONCURRENT)
    return youtube_audio_transcription_semaphore


def normalize_message_thread_id(message_thread_id: object) -> int:
    if isinstance(message_thread_id, int):
        return message_thread_id
    return 0


def build_session_key(user_id: int, message) -> SessionKey | None:
    if message is None:
        return None

    chat_id = getattr(message, "chat_id", None)
    if not isinstance(chat_id, int):
        return None

    message_thread_id = normalize_message_thread_id(getattr(message, "message_thread_id", None))
    return (user_id, chat_id, message_thread_id)


def get_session_history(session_key: SessionKey) -> list[ChatMessage]:
    return session_histories.get(session_key) or conversations.get(session_key) or []


def format_session_key(session_key: SessionKey) -> str:
    user_id, chat_id, message_thread_id = session_key
    return f"user={user_id} chat={chat_id} thread={message_thread_id}"


def clear_session_state(session_key: SessionKey) -> None:
    conversations.pop(session_key, None)
    session_histories.pop(session_key, None)
    source_memories.pop(session_key, None)
    active_source_sessions.discard(session_key)
    session_identifiers.pop(session_key, None)
    last_activity_at_by_session.pop(session_key, None)


def touch_session_activity(session_key: SessionKey, now: float | None = None) -> None:
    last_activity_at_by_session[session_key] = now if now is not None else time.time()


def cleanup_inactive_sessions(now: float | None = None) -> list[SessionKey]:
    if SESSION_INACTIVE_TTL_SECONDS is None:
        return []

    current_time = now if now is not None else time.time()
    cleaned_up_keys: list[SessionKey] = []
    for session_key, last_activity_at in list(last_activity_at_by_session.items()):
        inactive_seconds = current_time - last_activity_at
        if inactive_seconds < SESSION_INACTIVE_TTL_SECONDS:
            continue

        clear_session_state(session_key)
        cleaned_up_keys.append(session_key)
        logger.info(
            "Cleaned up inactive session from memory: %s inactive_seconds=%s",
            format_session_key(session_key),
            int(inactive_seconds),
        )

    return cleaned_up_keys


def ensure_session_identifier(session_key: SessionKey, message) -> str | None:
    existing_identifier = session_identifiers.get(session_key)
    if existing_identifier:
        return existing_identifier

    if message is None:
        return None

    chat_id = getattr(message, "chat_id", None)
    message_id = getattr(message, "message_id", None)
    if not isinstance(chat_id, int) or not isinstance(message_id, int):
        return None

    session_identifier = build_capture_session_id(chat_id, message_id)
    session_identifiers[session_key] = session_identifier
    return session_identifier


# ──────────────────────────────────────────────
# Vault 로그 저장
# ──────────────────────────────────────────────
def save_session_to_vault(session_key: SessionKey) -> bool:
    """세션 종료 시 누적 세션 전체를 Vault/Capture에 MD로 저장"""
    if not VAULT_CAPTURE_PATH:
        return False

    history = get_session_history(session_key)
    if not history:
        return False

    logs_dir = Path(VAULT_CAPTURE_PATH).expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = logs_dir / f"{today}.md"

    # 대화를 MD로 포맷
    now = datetime.now().strftime("%H:%M")
    session_identifier = session_identifiers.get(session_key)
    session_md = f"\n\n---\n\n## AI 세션 ({now}, {get_llm_task_profile('chat').model})\n"
    if session_identifier:
        session_md += f"{build_capture_session_marker(session_identifier)}\n"
    session_md += "\n"

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        source_kind = msg.get("source_kind", "").strip() or None
        source_url = msg.get("source_url", "").strip()

        if role == "user":
            # 검색 컨텍스트가 포함된 경우, [User Question] 이후만 표시
            if "[User Question]" in content:
                parts = content.split("[User Question]")
                context_part = parts[0].strip()
                question_part = parts[1].strip() if len(parts) > 1 else ""

                source_type = infer_source_type(context_part, source_kind)
                user_block = f"**나** ({source_type} 첨부): {question_part}\n"
            else:
                if source_kind:
                    source_type = infer_source_type("", source_kind)
                    user_block = f"**나** ({source_type} 첨부): {content}\n"
                else:
                    user_block = f"**나**: {content}\n"
            if source_url:
                user_block += f"<!-- source: {source_url} -->\n"
            session_md += f"{user_block}\n"
        elif role == "assistant":
            session_md += f"**AI**: {content}\n\n"

    # 태그 추가
    tags = generate_tags(history)
    session_md += f"{tags}\n"

    # 파일이 있으면 추가, 없으면 헤더 포함 생성
    if log_file.exists():
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(session_md)
    else:
        weekday = datetime.now().strftime("%A")
        header = f"# {today} {weekday}\n"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(header + session_md)

    logger.info(f"Session saved to vault: {log_file}")
    return True


# ──────────────────────────────────────────────
# 접근 제어
# ──────────────────────────────────────────────
def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        logger.warning("Access denied because ALLOWED_USER_IDS is not configured.")
        return False
    allowed = [int(uid.strip()) for uid in ALLOWED_USER_IDS.split(",") if uid.strip()]
    return user_id in allowed


# ──────────────────────────────────────────────
# <think> 태그 제거
# ──────────────────────────────────────────────
def strip_think(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_think_text(text: str) -> str:
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return "".join(matches).strip()


def should_show_reasoning_status(reasoning_text: str) -> bool:
    normalized = re.sub(r"[\W_]+", "", reasoning_text, flags=re.UNICODE)
    return len(normalized) >= MIN_REASONING_STATUS_CHARS


def strip_markdown(text: str) -> str:
    """마크다운 서식을 평문으로 변환"""
    # 코드 블록 (```...```) → 내용만 유지
    text = re.sub(r"```\w*\n?", "", text)
    # 인라인 코드 (`...`) → 내용만 유지
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 볼드/이탤릭 (**, *, __, _)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # 헤더 (##)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 리스트 마커 (-, *, 1.)는 유지 (가독성)
    return text.strip()


def contains_replacement_char(text: str) -> bool:
    return REPLACEMENT_CHAR in text


def sanitize_replacement_chars(text: str) -> str:
    """깨진 유니코드 대체 문자는 사용자에게 보내기 전에 제거한다."""
    if not text:
        return text
    return text.replace(REPLACEMENT_CHAR, "")


def _normalize_table_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell).strip()


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [_normalize_table_cell(cell) for cell in stripped.split("|")]
    return cells if len(cells) >= 2 else []


def _is_markdown_table_separator(line: str, expected_columns: int) -> bool:
    cells = _split_markdown_table_row(line)
    if len(cells) != expected_columns:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    normalized_headers = [_normalize_table_cell(header) for header in headers]
    normalized_rows = [
        [_normalize_table_cell(cell) for cell in row[: len(normalized_headers)]]
        for row in rows
    ]

    if len(normalized_headers) >= 3:
        column_labels = [
            header or f"열 {index}"
            for index, header in enumerate(normalized_headers[1:], start=1)
        ]
        title = " vs ".join(column_labels)
        parts = [title] if title else []

        for row in normalized_rows:
            row_label = row[0] or normalized_headers[0] or "항목"
            if parts:
                parts.append("")
            parts.append(row_label)
            for column_label, value in zip(column_labels, row[1:]):
                parts.append(f"- {column_label}: {value}")
            if len(row) < len(normalized_headers):
                for column_label in column_labels[len(row) - 1 :]:
                    parts.append(f"- {column_label}: ")
        return "\n".join(parts).strip()

    if len(normalized_headers) == 2:
        left_header, right_header = normalized_headers
        left_is_generic = left_header.strip().lower() in GENERIC_TABLE_LABELS
        if left_is_generic:
            return "\n".join(f"- {row[0]}: {row[1]}" for row in normalized_rows).strip()

        parts = [f"{left_header} / {right_header}"] if left_header or right_header else []
        for row in normalized_rows:
            if parts:
                parts.append("")
            parts.append(f"{row[0]}: {row[1]}")
        return "\n".join(parts).strip()

    return "\n".join(" | ".join(row) for row in normalized_rows).strip()


def normalize_markdown_tables(text: str) -> str:
    if not text or "|" not in text:
        return text

    lines = text.splitlines()
    normalized_lines: list[str] = []
    index = 0

    while index < len(lines):
        header_cells = _split_markdown_table_row(lines[index])
        if (
            len(header_cells) >= 2
            and index + 2 < len(lines)
            and _is_markdown_table_separator(lines[index + 1], len(header_cells))
        ):
            row_index = index + 2
            rows: list[list[str]] = []
            while row_index < len(lines):
                row_cells = _split_markdown_table_row(lines[row_index])
                if len(row_cells) != len(header_cells):
                    break
                rows.append(row_cells)
                row_index += 1

            if rows:
                normalized_lines.append(_render_markdown_table(header_cells, rows))
                index = row_index
                continue

        normalized_lines.append(lines[index])
        index += 1

    return "\n".join(normalized_lines)


def _normalize_latex_fragment(fragment: str) -> tuple[str, bool]:
    normalized = fragment
    changed = False
    for latex, replacement in LATEX_COMMAND_REPLACEMENTS.items():
        if latex in normalized:
            normalized = normalized.replace(latex, replacement)
            changed = True

    if changed:
        normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized, changed


def normalize_inline_latex(text: str) -> str:
    """Telegram이 렌더링하지 못하는 흔한 LaTeX 화살표 표기를 평문 기호로 바꾼다."""
    if not text:
        return text

    normalized = text
    inline_patterns = (
        re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL),
        re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", flags=re.DOTALL),
        re.compile(r"\\\((.+?)\\\)", flags=re.DOTALL),
        re.compile(r"\\\[(.+?)\\\]", flags=re.DOTALL),
    )

    def replace_match(match: re.Match[str]) -> str:
        fragment, changed = _normalize_latex_fragment(match.group(1))
        if changed:
            return fragment
        return match.group(0)

    for pattern in inline_patterns:
        normalized = pattern.sub(replace_match, normalized)

    for latex, replacement in LATEX_COMMAND_REPLACEMENTS.items():
        normalized = re.sub(
            rf"(?<![A-Za-z0-9_\\]){re.escape(latex)}(?![A-Za-z])",
            replacement,
            normalized,
        )

    return normalized


def normalize_response_text(text: str) -> str:
    if not text:
        return text
    return normalize_plain_text_spacing(
        normalize_markdown_tables(
            normalize_inline_latex(
                sanitize_replacement_chars(
                    strip_internal_context_markers(
                        strip_markdown(
                            strip_think(text)
                        )
                    )
                )
            )
        )
    ).strip()


def strip_internal_context_markers(text: str) -> str:
    if not text:
        return text
    return text.replace(ASSISTANT_CONTEXT_TRUNCATION_NOTICE, "")


def normalize_plain_text_spacing(text: str) -> str:
    """Keep model output readable in Telegram plain-text bubbles."""
    if not text:
        return text

    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            normalized_lines.append("")
            continue

        stripped = re.sub(r"^([0-9]+[.)])\s+", r"\1 ", stripped)
        stripped = re.sub(r"^[*•]\s+", "- ", stripped)
        normalized_lines.append(stripped)

    return "\n".join(normalized_lines)


def response_validation_issue(text: str) -> str | None:
    if not text.strip():
        return "empty_response"
    if FORBIDDEN_RESPONSE_CJK_RE.search(text):
        return "contains_chinese_or_japanese_characters"
    return None


def build_validation_safe_fallback(text: str) -> str | None:
    normalized = normalize_response_text(text)
    if not normalized:
        return None

    repaired = FORBIDDEN_RESPONSE_CJK_SPAN_RE.sub(CJK_OMISSION_PLACEHOLDER, normalized)
    repaired = re.sub(
        rf"(?:{re.escape(CJK_OMISSION_PLACEHOLDER)}\s*){{2,}}",
        CJK_OMISSION_PLACEHOLDER,
        repaired,
    ).strip()
    if response_validation_issue(repaired) is not None:
        return None
    return (
        "응답 일부에 번역되지 않은 중국어/일본어 표기가 있어 해당 표기를 생략했습니다.\n\n"
        f"{repaired}"
    )


def build_response_rewrite_messages(
    original_text: str,
    user_message: str,
    search_context: str,
    issue: str,
    attempt: int = 1,
) -> list[dict[str, str]]:
    context_note = "The previous answer was based on provided context/search results." if search_context else ""
    if attempt > 1:
        system_message = (
            "You are a Korean translation repair pass. The previous translation still violated output rules. "
            "Return only the corrected final answer. "
            "Use Korean plain text only. Do not copy any Chinese Hanzi, Japanese Kanji, Hiragana, or Katakana. "
            "Use Hangul Korean and Latin technical terms only. "
            "Do not add apologies or commentary about the translation."
        )
        user_instruction = (
            "Translate the text below into Korean plain text only. "
            "Do not preserve any Chinese or Japanese characters. "
            "If a proper noun is written only in Chinese or Japanese, transliterate it into Korean or describe it in Korean."
        )
    else:
        system_message = (
            "You translate assistant responses into Korean so they comply with output rules. "
            "Preserve the meaning of the previous answer; do not answer the user request again from scratch. "
            "Return only the translated final answer. "
            "Use Korean plain text only. Do not use Chinese or Japanese characters. "
            "Do not add apologies or commentary about the translation."
        )
        user_instruction = (
            "Translate the previous answer into Korean plain text only. "
            "Preserve its meaning and structure as much as possible. "
            "Translate any Chinese or Japanese text into Korean instead of copying it."
        )

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": (
                f"Violation: {issue}\n"
                f"Translation attempt: {attempt}\n"
                f"{context_note}\n\n"
                f"Original user request:\n{user_message}\n\n"
                f"Previous answer:\n{original_text}\n\n"
                f"{user_instruction}"
            ).strip(),
        },
    ]


def rewrite_invalid_response(
    original_text: str,
    user_message: str,
    search_context: str,
    issue: str,
    attempt: int = 1,
) -> str:
    profile = get_llm_task_profile("rewrite")
    payload = {
        "model": profile.model,
        "messages": build_response_rewrite_messages(original_text, user_message, search_context, issue, attempt),
        "stream": False,
        "temperature": 0,
    }
    disable_llm_reasoning(payload)
    apply_llm_request_options(payload, allow_reasoning=False, profile=profile)

    response = requests.post(
        build_chat_completions_url(profile),
        headers=build_llm_headers(profile),
        json=payload,
        timeout=RESPONSE_REWRITE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    try:
        rewritten = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return original_text

    normalized = normalize_response_text(str(rewritten))
    return normalized or original_text


async def validate_and_rewrite_response(
    text: str,
    user_message: str,
    search_context: str,
) -> str:
    if not ENABLE_RESPONSE_VALIDATION:
        return text

    issue = response_validation_issue(text)
    if issue is None:
        return text

    candidate = text
    for attempt in range(1, RESPONSE_REWRITE_MAX_ATTEMPTS + 1):
        try:
            rewritten = await asyncio.to_thread(
                rewrite_invalid_response,
                candidate,
                user_message,
                search_context,
                issue,
                attempt,
            )
        except Exception as exc:
            logger.warning(
                "Response translation failed after validation issue=%s attempt=%s/%s: %s",
                issue,
                attempt,
                RESPONSE_REWRITE_MAX_ATTEMPTS,
                exc,
            )
            safe_fallback = build_validation_safe_fallback(candidate)
            if safe_fallback:
                logger.warning("Using CJK-omission fallback after translation exception.")
                return safe_fallback
            return RESPONSE_VALIDATION_FAILURE_TEXT

        followup_issue = response_validation_issue(rewritten)
        if followup_issue is None:
            logger.info("Response translated after validation issue=%s attempt=%s", issue, attempt)
            return rewritten

        logger.warning(
            "Response translation still failed validation issue=%s attempt=%s/%s",
            followup_issue,
            attempt,
            RESPONSE_REWRITE_MAX_ATTEMPTS,
        )
        candidate = rewritten
        issue = followup_issue

    safe_fallback = build_validation_safe_fallback(candidate)
    if safe_fallback:
        logger.warning("Using CJK-omission fallback after max translation attempts.")
        return safe_fallback

    return RESPONSE_VALIDATION_FAILURE_TEXT


def build_draft_id() -> int:
    """동일 응답 스트림 동안 재사용할 Telegram draft id를 만든다."""
    draft_id = time.monotonic_ns() & 0x7FFFFFFF
    return draft_id or 1


def chunk_text(text: str, size: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


def log_stage_metrics(
    stage: str,
    user_id: int,
    elapsed_ms: int,
    *,
    ok: bool,
    source: str | None = None,
    detail: str | None = None,
    chars: int | None = None,
    method: str | None = None,
):
    logger.info(
        "Stage metrics stage=%s user=%s source=%s method=%s ok=%s elapsed_ms=%s chars=%s detail=%s",
        stage,
        user_id,
        source,
        method,
        ok,
        elapsed_ms,
        chars,
        detail,
    )


async def update_status_message(update: Update | None, status_message, text: str):
    text = text[:TELEGRAM_TEXT_LIMIT]
    if status_message is not None:
        try:
            await status_message.edit_text(text)
            return status_message
        except BadRequest as exc:
            if "Message is not modified" in str(exc):
                return status_message
            logger.debug("status message edit failed; falling back to a new status message: %s", exc)
        except Exception as exc:
            logger.debug("status message edit failed; falling back to a new status message: %s", exc)

    if update is None or not getattr(update, "message", None):
        return status_message
    return await update.message.reply_text(text)


def build_context_ready_status(
    source_kind: str,
    content: str,
    *,
    extract_only_requested: bool = False,
    note: str = "",
) -> str:
    label = SOURCE_KIND_LABELS.get(source_kind, source_kind)
    lines = [
        f"컨텍스트 준비 완료: {label}",
        f"본문: {len(content):,}자",
    ]
    if note:
        lines.append(f"참고: {note}")
    lines.append("원문 전송 중..." if extract_only_requested else "답변 생성 중...")
    return "\n".join(lines)


async def extract_context_from_user_text(
    update: Update,
    user_id: int,
    user_text: str,
    *,
    session_key: SessionKey | None = None,
    extract_only_requested: bool = False,
) -> ContextExtractionResult:
    tweet_match = TWEET_URL_PATTERN.search(user_text)
    if tweet_match:
        tweet_url = find_matching_url(user_text, TWEET_URL_PATTERN) or tweet_match.group(0)
        status_message = await update_status_message(update, None, "📰 X 피드 읽는 중...")
        stage_started_at = asyncio.get_running_loop().time()
        tweet_context = await asyncio.to_thread(extract_tweet_from_url, tweet_url)
        log_stage_metrics(
            "extract",
            user_id,
            int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
            ok=bool(tweet_context),
            source="x",
            detail=tweet_url,
            chars=len(tweet_context) if tweet_context else 0,
        )
        if not tweet_context:
            await update_status_message(update, status_message, "⚠️ 피드를 가져올 수 없습니다.")
            return ContextExtractionResult(matched=True)

        await update_status_message(
            update,
            status_message,
            build_context_ready_status("x", tweet_context, extract_only_requested=extract_only_requested),
        )
        return ContextExtractionResult(
            matched=True,
            extracted=ExtractedContext(
                user_message=remove_url_once(user_text, tweet_url),
                content=tweet_context,
                source="x",
                source_kind="x",
                source_url=normalize_source_url(tweet_url, "x"),
            ),
        )

    yt_match = YOUTUBE_URL_PATTERN.search(user_text)
    if yt_match:
        video_id = yt_match.group(1)
        youtube_url = find_matching_url(user_text, YOUTUBE_URL_PATTERN) or yt_match.group(0)
        canonical_youtube_url = normalize_source_url(youtube_url, "youtube")
        status_message = await update_status_message(update, None, "🎬 YouTube 스크립트 확인 중...")
        stage_started_at = asyncio.get_running_loop().time()
        yt_result = await asyncio.to_thread(extract_youtube_transcript_result, video_id)
        yt_context = yt_result.content
        log_stage_metrics(
            "extract",
            user_id,
            int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
            ok=bool(yt_context),
            source="youtube",
            method=youtube_transcript_method_label(yt_result),
            detail=f"{video_id}:{yt_result.status}",
            chars=len(yt_context) if yt_context else 0,
        )
        if not yt_context:
            failure_message = yt_result.message or "스크립트를 가져올 수 없습니다."
            pending, fallback_issue = await asyncio.to_thread(
                build_pending_youtube_transcription,
                video_id,
                youtube_url,
                canonical_youtube_url,
                remove_url_once(user_text, youtube_url),
                extract_only_requested,
                yt_result,
            )
            if pending is not None:
                logger.info(
                    "YouTube audio fallback started automatically user=%s video_id=%s status=%s reason=%s",
                    user_id,
                    video_id,
                    yt_result.status,
                    failure_message,
                )
                status_message = await update_status_message(
                    update,
                    status_message,
                    build_youtube_auto_transcription_start_reply(pending),
                )
                result, elapsed_ms = await execute_youtube_audio_transcription(
                    update,
                    pending,
                    status_message=status_message,
                )
                content = result.get("content") if isinstance(result.get("content"), str) else ""
                log_stage_metrics(
                    "youtube_audio_transcription",
                    user_id,
                    elapsed_ms,
                    ok=bool(result.get("ok")),
                    source="youtube_audio",
                    detail=f"{video_id}:{result.get('status')}",
                    chars=len(content),
                )
                if not result.get("ok") or not content:
                    await update_status_message(update, status_message, build_youtube_audio_failure_reply(result))
                    return ContextExtractionResult(matched=True)
                await update_status_message(
                    update,
                    status_message,
                    build_context_ready_status(
                        "youtube",
                        content,
                        extract_only_requested=extract_only_requested,
                    ),
                )
                return ContextExtractionResult(
                    matched=True,
                    extracted=ExtractedContext(
                        user_message=pending.user_message,
                        content=content,
                        source="youtube_audio",
                        source_kind="youtube",
                        source_url=pending.canonical_youtube_url,
                    ),
                )
            if fallback_issue and ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION:
                await update_status_message(update, status_message, f"⚠️ {failure_message}\n\n{fallback_issue}")
                return ContextExtractionResult(matched=True)
            await update_status_message(update, status_message, f"⚠️ {failure_message}")
            return ContextExtractionResult(matched=True)

        await update_status_message(
            update,
            status_message,
            build_context_ready_status(
                "youtube",
                yt_context,
                extract_only_requested=extract_only_requested,
                note=yt_result.message,
            ),
        )

        return ContextExtractionResult(
            matched=True,
            extracted=ExtractedContext(
                user_message=remove_url_once(user_text, youtube_url),
                content=yt_context,
                source="youtube",
                source_kind="youtube",
                source_url=canonical_youtube_url,
            ),
        )

    url_match = GENERAL_URL_PATTERN.search(user_text)
    if not url_match:
        return ContextExtractionResult(matched=False)

    url = url_match.group(0)
    user_msg = remove_url_once(user_text, url)

    if url.lower().endswith(".pdf"):
        status_message = await update_status_message(update, None, "📄 PDF 다운로드 중...")
        stage_started_at = asyncio.get_running_loop().time()
        pdf_context = await asyncio.to_thread(extract_pdf_from_url, url)
        log_stage_metrics(
            "extract",
            user_id,
            int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
            ok=bool(pdf_context),
            source="pdf_url",
            detail=url,
            chars=len(pdf_context) if pdf_context else 0,
        )
        if not pdf_context:
            await update_status_message(update, status_message, "⚠️ PDF에서 텍스트를 추출할 수 없습니다.")
            return ContextExtractionResult(matched=True)

        await update_status_message(
            update,
            status_message,
            build_context_ready_status("pdf", pdf_context, extract_only_requested=extract_only_requested),
        )
        return ContextExtractionResult(
            matched=True,
            extracted=ExtractedContext(
                user_message=user_msg,
                content=pdf_context,
                source="pdf_url",
                source_kind="pdf",
                source_url=normalize_source_url(url, "pdf"),
            ),
        )

    status_message = await update_status_message(update, None, "📖 웹페이지 읽는 중...")
    stage_started_at = asyncio.get_running_loop().time()
    web_result = await asyncio.to_thread(extract_web_result, url)
    log_stage_metrics(
        "extract",
        user_id,
        int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
        ok=bool(web_result),
        source="web",
        detail=web_result.document_url if web_result else url,
        chars=len(web_result.content) if web_result else 0,
        method=web_result.method if web_result else None,
    )
    if not web_result:
        await update_status_message(
            update,
            status_message,
            "⚠️ 웹페이지에서 텍스트를 추출할 수 없습니다. (일부 JavaScript/iframe 기반 페이지는 여전히 지원되지 않을 수 있어요)",
        )
        return ContextExtractionResult(matched=True)

    await update_status_message(
        update,
        status_message,
        build_context_ready_status("web", web_result.content, extract_only_requested=extract_only_requested),
    )
    return ContextExtractionResult(
        matched=True,
        extracted=ExtractedContext(
            user_message=user_msg,
            content=web_result.content,
            source="web",
            source_kind="web",
            source_url=normalize_source_url(web_result.document_url or url, "web"),
        ),
    )


async def send_extract_only_reply(update: Update, session_key: SessionKey, extracted: ExtractedContext) -> None:
    output_text = format_extract_only_text(extracted.content, extracted.source_kind)
    for chunk in chunk_text(output_text):
        await update.message.reply_text(chunk)

    session_histories.setdefault(session_key, []).append(
        {
            "role": "user",
            "content": "[Extracted Context]\n\n[User Question]\n원문 추출",
            "source_kind": extracted.source_kind,
            "source_url": extracted.source_url or "",
        }
    )
    session_histories[session_key].append({"role": "assistant", "content": output_text})
    touch_session_activity(session_key)


async def read_stream_tail(stream, limit_chars: int = 8_000) -> str:
    buffer = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > limit_chars:
            del buffer[: len(buffer) - limit_chars]
    return buffer.decode("utf-8", errors="replace").strip()


async def stop_subprocess(process, timeout_seconds: float = 5.0) -> None:
    if getattr(process, "returncode", None) is not None:
        return
    with suppress(ProcessLookupError):
        process.kill()
    with suppress(Exception):
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)


def build_youtube_audio_stall_message(
    timeout_seconds: float,
    last_progress: tuple[int, int] | None,
    stderr_text: str = "",
) -> str:
    message = f"오디오 전사 진행이 {format_duration(int(timeout_seconds))} 동안 멈춰 worker를 종료했습니다."
    if last_progress:
        message += f" 마지막 진행: {last_progress[0]}/{last_progress[1]}."
    if stderr_text:
        message += f"\n최근 오류 로그: {stderr_text[-800:]}"
    return message


def build_youtube_audio_failure_reply(result: dict) -> str:
    status = str(result.get("status") or "error")
    reason = str(result.get("message") or "원인을 확인하지 못했습니다.").strip()
    if len(reason) > 1200:
        reason = f"{reason[:1200].rstrip()}..."
    return f"⚠️ 오디오 전사를 완료하지 못했습니다.\n상태: {status}\n사유: {reason}"


async def maybe_reply_text(update: Update | None, text: str) -> None:
    if update is None or not getattr(update, "message", None):
        return
    await update.message.reply_text(text)


async def run_youtube_audio_transcription_worker(
    update: Update | None,
    pending: PendingYouTubeTranscription,
    status_message=None,
) -> dict:
    process = await asyncio.create_subprocess_exec(
        *youtube_audio_worker_command("transcribe", pending.video_id),
        cwd=Path(__file__).resolve().parent,
        env=youtube_audio_worker_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=YOUTUBE_AUDIO_WORKER_STREAM_LIMIT,
    )
    result_payload: dict | None = None
    last_progress: tuple[int, int] | None = None
    stderr_task = (
        asyncio.create_task(read_stream_tail(process.stderr))
        if process.stderr is not None
        else None
    )

    try:
        assert process.stdout is not None
        while True:
            try:
                if YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS > 0:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS,
                    )
                else:
                    line = await process.stdout.readline()
            except asyncio.TimeoutError:
                await stop_subprocess(process)
                stderr_text = ""
                if stderr_task is not None:
                    with suppress(Exception):
                        stderr_text = await stderr_task
                return {
                    "ok": False,
                    "status": "stalled",
                    "message": build_youtube_audio_stall_message(
                        YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS,
                        last_progress,
                        stderr_text,
                    ),
                }
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("event") == "chunk_done":
                current = int(payload.get("index") or 0)
                total = int(payload.get("total") or 0)
                if current and total and (current, total) != last_progress:
                    last_progress = (current, total)
                    progress_text = f"🎙️ 오디오 전사 진행 중... {current}/{total}"
                    if status_message is not None:
                        await update_status_message(update, status_message, progress_text)
                    else:
                        await maybe_reply_text(update, progress_text)
                continue
            if "ok" in payload:
                result_payload = payload

        return_code = await process.wait()
        stderr_text = ""
        if stderr_task is not None:
            with suppress(Exception):
                stderr_text = await stderr_task
        if result_payload is None:
            detail = stderr_text[-800:] if stderr_text else f"worker exited with code {return_code}"
            return {"ok": False, "status": "worker_error", "message": detail}
        if return_code != 0 and result_payload.get("ok") is not True:
            if stderr_text:
                result_payload["message"] = f"{result_payload.get('message') or ''}\n{stderr_text[-800:]}".strip()
        return result_payload
    except asyncio.CancelledError:
        await stop_subprocess(process)
        if stderr_task is not None:
            stderr_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await stderr_task
        raise


async def execute_youtube_audio_transcription(
    update: Update | None,
    pending: PendingYouTubeTranscription,
    status_message=None,
) -> tuple[dict, int]:
    started_at = asyncio.get_running_loop().time()
    try:
        async with get_youtube_audio_transcription_semaphore():
            if YOUTUBE_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS > 0:
                result = await asyncio.wait_for(
                    run_youtube_audio_transcription_worker(update, pending, status_message=status_message),
                    timeout=YOUTUBE_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS,
                )
            else:
                result = await run_youtube_audio_transcription_worker(update, pending, status_message=status_message)
    except asyncio.TimeoutError:
        result = {
            "ok": False,
            "status": "timeout",
            "message": "오디오 전사 작업이 제한 시간 안에 끝나지 않았습니다.",
        }
    except Exception as exc:
        logger.exception("YouTube audio transcription worker failed: %s", exc)
        result = {
            "ok": False,
            "status": "worker_exception",
            "message": str(exc) or exc.__class__.__name__,
        }
    elapsed_ms = int((asyncio.get_running_loop().time() - started_at) * 1000)
    return result, elapsed_ms


async def enhance_inbox_youtube_context_source(
    update: Update | None,
    user_id: int,
    source: InboxContextSource,
) -> tuple[InboxContextSource, bool]:
    if source.source_kind != "youtube" or not source.source_url:
        return source, False

    match = YOUTUBE_URL_PATTERN.search(source.source_url)
    if not match:
        logger.info(
            "Inbox YouTube context skipped enhancement source_id=%s reason=missing_video_id url=%s",
            source.source_id,
            source.source_url,
        )
        return source, False

    video_id = match.group(1)
    canonical_url = normalize_source_url(source.source_url, "youtube") or source.source_url

    cached_audio_context = await asyncio.to_thread(load_cached_youtube_audio_transcript, video_id)
    if cached_audio_context:
        logger.info(
            "Inbox YouTube context enhanced from transcript cache source_id=%s video_id=%s original_chars=%s enhanced_chars=%s",
            source.source_id,
            video_id,
            len(source.text),
            len(cached_audio_context),
        )
        return replace(source, text=cached_audio_context, source_url=canonical_url), True

    stage_started_at = asyncio.get_running_loop().time()
    yt_result = await asyncio.to_thread(extract_youtube_transcript_result, video_id)
    yt_context = yt_result.content
    log_stage_metrics(
        "inbox_context_hydration",
        user_id,
        int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
        ok=bool(yt_context),
        source="youtube",
        method=youtube_transcript_method_label(yt_result),
        detail=f"source_id={source.source_id} video_id={video_id}:{yt_result.status}",
        chars=len(yt_context) if yt_context else 0,
    )
    if yt_context:
        if yt_result.message:
            await maybe_reply_text(update, f"ℹ️ {yt_result.message}")
        logger.info(
            "Inbox YouTube context enhanced from transcript source_id=%s video_id=%s original_chars=%s enhanced_chars=%s status=%s method=%s",
            source.source_id,
            video_id,
            len(source.text),
            len(yt_context),
            yt_result.status,
            youtube_transcript_method_label(yt_result),
        )
        return replace(source, text=yt_context, source_url=canonical_url), True

    pending, fallback_issue = await asyncio.to_thread(
        build_pending_youtube_transcription,
        video_id,
        source.source_url,
        canonical_url,
        DEFAULT_CONTEXT_PROMPT,
        False,
        yt_result,
    )
    if pending is None:
        if fallback_issue:
            await maybe_reply_text(
                update,
                f"ℹ️ YouTube 원문 보강은 건너뜁니다: {fallback_issue}\n"
                "먼저 읽어둔 본문을 사용합니다."
            )
        logger.info(
            "Inbox YouTube context using stored text source_id=%s video_id=%s original_chars=%s transcript_status=%s fallback_issue=%s",
            source.source_id,
            video_id,
            len(source.text),
            yt_result.status,
            fallback_issue or "",
        )
        return replace(source, source_url=canonical_url), False

    await maybe_reply_text(update, "🎙️ 공개 자막을 가져오지 못해 오디오 전사를 시작합니다.")
    result, elapsed_ms = await execute_youtube_audio_transcription(update, pending)
    content = result.get("content") if isinstance(result.get("content"), str) else ""
    log_stage_metrics(
        "inbox_context_youtube_audio_transcription",
        user_id,
        elapsed_ms,
        ok=bool(result.get("ok")),
        source="youtube_audio",
        detail=f"source_id={source.source_id} video_id={video_id}:{result.get('status')}",
        chars=len(content),
    )
    if not result.get("ok") or not content:
        await maybe_reply_text(
            update,
            f"{build_youtube_audio_failure_reply(result)}\n\n"
            "먼저 읽어둔 본문을 대신 사용합니다."
        )
        logger.info(
            "Inbox YouTube context using stored text after audio failure source_id=%s video_id=%s original_chars=%s status=%s",
            source.source_id,
            video_id,
            len(source.text),
            result.get("status"),
        )
        return replace(source, source_url=canonical_url), False

    logger.info(
        "Inbox YouTube context enhanced from audio source_id=%s video_id=%s original_chars=%s enhanced_chars=%s status=%s cached=%s",
        source.source_id,
        video_id,
        len(source.text),
        len(content),
        result.get("status"),
        result.get("cached"),
    )
    return replace(source, text=content, source_url=canonical_url), True


def purge_stale_inbox_context_prefetch_cache(now: float | None = None) -> None:
    current_time = now if now is not None else time.time()
    if not inbox_context_prefetch_cache:
        purge_stale_persistent_inbox_context_prefetch_cache(current_time)
        return
    expired_source_ids = [
        source_id
        for source_id, cached in inbox_context_prefetch_cache.items()
        if current_time - cached.cached_at > INBOX_CONTEXT_PREFETCH_CACHE_TTL_SECONDS
    ]
    for source_id in expired_source_ids:
        inbox_context_prefetch_cache.pop(source_id, None)
    purge_stale_persistent_inbox_context_prefetch_cache(current_time)


def pop_prefetched_inbox_context_source(source: InboxContextSource) -> PrefetchedInboxContextSource | None:
    purge_stale_inbox_context_prefetch_cache()
    cached = inbox_context_prefetch_cache.pop(source.source_id, None)
    if cached is not None:
        if not prefetched_inbox_context_source_matches(source, cached):
            logger.info(
                "Discarded in-memory prefetched source because Inbox source changed source_id=%s",
                source.source_id,
            )
        else:
            cached_source = replace(cached.source, remaining_ready_count=source.remaining_ready_count)
            return replace(cached, source=cached_source)

    cached = load_persistent_prefetched_inbox_context_source(source.source_id)
    if cached is None:
        return None
    if not prefetched_inbox_context_source_matches(source, cached):
        delete_persistent_prefetched_inbox_context_source(source.source_id)
        logger.info(
            "Discarded persistent prefetched source because Inbox source changed source_id=%s",
            source.source_id,
        )
        return None
    cached_source = replace(cached.source, remaining_ready_count=source.remaining_ready_count)
    return replace(cached, source=cached_source)


def should_generate_inbox_context_prefetch_summary(reason: str) -> bool:
    if not ENABLE_INBOX_CONTEXT_PREFETCH_SUMMARY:
        return False
    if reason == "startup" and not ENABLE_INBOX_CONTEXT_PREFETCH_STARTUP_SUMMARY:
        return False
    return True


async def _prefetch_inbox_context_source(source: InboxContextSource, *, reason: str) -> None:
    source_id = source.source_id
    if source_id in inbox_context_prefetch_cache:
        return

    inbox_context_prefetch_inflight.add(source_id)
    original_chars = len(source.text)
    try:
        hydrated_source, enhanced = await enhance_inbox_youtube_context_source(None, 0, source)
        initial_reply: str | None = None
        if should_generate_inbox_context_prefetch_summary(reason):
            try:
                initial_reply = await asyncio.to_thread(generate_inbox_context_initial_reply, hydrated_source)
            except Exception as exc:
                logger.warning(
                    "Inbox context prefetch summary failed source_id=%s kind=%s reason=%s: %s",
                    source_id,
                    hydrated_source.source_kind,
                    reason,
                    exc,
                )

        cached_source = build_prefetched_inbox_context_source(
            source,
            hydrated_source,
            enhanced=enhanced,
            initial_reply=initial_reply,
        )
        inbox_context_prefetch_cache[source_id] = cached_source
        await asyncio.to_thread(persist_prefetched_inbox_context_source, cached_source)
        logger.info(
            "Inbox context prefetched source_id=%s kind=%s url=%s chars=%s original_chars=%s enhanced=%s summary=%s reason=%s",
            source_id,
            hydrated_source.source_kind,
            hydrated_source.source_url or "",
            len(hydrated_source.text),
            original_chars,
            enhanced,
            bool(initial_reply),
            reason,
        )
    except Exception as exc:
        logger.warning("Inbox context prefetch failed source_id=%s reason=%s: %s", source_id, reason, exc)
    finally:
        inbox_context_prefetch_inflight.discard(source_id)
        current_task = asyncio.current_task()
        if inbox_context_prefetch_tasks.get(source_id) is current_task:
            inbox_context_prefetch_tasks.pop(source_id, None)


def ensure_inbox_context_source_prefetch_task(
    source: InboxContextSource,
    *,
    reason: str,
) -> asyncio.Task | None:
    source_id = source.source_id
    if source_id in inbox_context_prefetch_cache:
        return None

    existing_task = inbox_context_prefetch_tasks.get(source_id)
    if existing_task is not None and not existing_task.done():
        return existing_task

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    task = loop.create_task(_prefetch_inbox_context_source(source, reason=reason))
    inbox_context_prefetch_tasks[source_id] = task
    return task


async def prefetch_inbox_context_sources(*, reason: str = "scheduled", target: int | None = None) -> None:
    if not ENABLE_INBOX_CONTEXT_PREFETCH or INBOX_CONTEXT_PREFETCH_TARGET <= 0:
        return

    purge_stale_inbox_context_prefetch_cache()
    effective_target = max(1, target or INBOX_CONTEXT_PREFETCH_TARGET)
    capacity = effective_target - len(inbox_context_prefetch_cache) - len(inbox_context_prefetch_tasks)
    if capacity <= 0:
        return

    try:
        sources = await asyncio.to_thread(
            fetch_prefetchable_inbox_context_sources,
            effective_target,
        )
    except Exception as exc:
        logger.warning("Inbox context prefetch source fetch failed reason=%s: %s", reason, exc)
        return

    tasks: list[asyncio.Task] = []
    for source in sources:
        if len(inbox_context_prefetch_cache) + len(inbox_context_prefetch_tasks) >= effective_target:
            break
        if source.source_id in inbox_context_prefetch_cache:
            continue
        task = ensure_inbox_context_source_prefetch_task(source, reason=reason)
        if task is not None:
            tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def start_inbox_context_prefetch(reason: str = "manual") -> None:
    global inbox_context_prefetch_task

    if not ENABLE_INBOX_CONTEXT_PREFETCH:
        return
    if inbox_context_prefetch_task is not None and not inbox_context_prefetch_task.done():
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    inbox_context_prefetch_task = loop.create_task(prefetch_inbox_context_sources(reason=reason))


async def run_inbox_context_prefetch_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await prefetch_inbox_context_sources(reason="scheduled")


async def run_inbox_context_prefetch_startup_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await prefetch_inbox_context_sources(
        reason="startup",
        target=INBOX_CONTEXT_PREFETCH_STARTUP_TARGET,
    )


def append_prefetched_inbox_context_reply_to_history(
    session_key: SessionKey,
    source: InboxContextSource,
    final_text: str,
) -> None:
    augmented_message = build_augmented_context_message(
        build_context_prompt(DEFAULT_CONTEXT_PROMPT),
        source.text,
    )
    append_history_message(
        session_key,
        "user",
        augmented_message,
        source_kind=source.source_kind,
        source_url=source.source_url,
    )
    append_history_message(session_key, "assistant", final_text)


async def send_message_draft(update: Update, draft_id: int, text: str) -> bool:
    message = update.message
    if not message or not text:
        return False

    bot = message.get_bot()
    kwargs = {
        "chat_id": message.chat_id,
        "draft_id": draft_id,
        "text": text[:TELEGRAM_TEXT_LIMIT],
    }
    if message.message_thread_id is not None:
        kwargs["message_thread_id"] = message.message_thread_id
    return await bot.send_message_draft(**kwargs)


async def send_raw_message_draft(update: Update, draft_id: int, text: str, *, include_thread: bool) -> bool:
    message = update.message
    if not message or not text:
        return False

    payload = {
        "chat_id": message.chat_id,
        "draft_id": draft_id,
        "text": text[:TELEGRAM_TEXT_LIMIT],
    }
    if include_thread and message.message_thread_id is not None:
        payload["message_thread_id"] = message.message_thread_id
    return bool(await message.get_bot().do_api_request("sendMessageDraft", api_kwargs=payload))


async def diagnose_draft(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return
    if not update.message:
        return

    message = update.message
    checks: list[tuple[str, bool, str]] = []
    base_id = build_draft_id()

    async def run_check(label: str, call):
        try:
            ok = await call()
            checks.append((label, bool(ok), "ok" if ok else "false"))
        except Exception as e:
            checks.append((label, False, f"{type(e).__name__}: {e}"))

    await run_check(
        "ptb:auto-thread",
        lambda: send_message_draft(update, base_id, "draft test 1/4: PTB auto thread"),
    )
    await run_check(
        "ptb:no-thread",
        lambda: message.get_bot().send_message_draft(
            chat_id=message.chat_id,
            draft_id=base_id + 1,
            text="draft test 2/4: PTB no thread",
        ),
    )
    await run_check(
        "raw:auto-thread",
        lambda: send_raw_message_draft(
            update,
            base_id + 2,
            "draft test 3/4: raw auto thread",
            include_thread=True,
        ),
    )
    await run_check(
        "raw:no-thread",
        lambda: send_raw_message_draft(
            update,
            base_id + 3,
            "draft test 4/4: raw no thread",
            include_thread=False,
        ),
    )

    logger.info(
        "Draft diagnostic user=%s chat=%s thread=%s results=%s",
        user_id,
        message.chat_id,
        message.message_thread_id,
        checks,
    )
    lines = [
        "Draft 진단 완료",
        f"chat_id: {message.chat_id}",
        f"thread_id: {message.message_thread_id or 'none'}",
        "",
    ]
    lines.extend(f"- {label}: {'성공' if ok else '실패'} ({detail})" for label, ok, detail in checks)
    lines.append("")
    lines.append("Telegram 클라이언트에 draft 4개가 보이는지도 같이 확인해줘.")
    await update.message.reply_text("\n".join(lines))


async def show_reasoning_status(
    update: Update,
    draft_id: int,
    use_message_draft: bool,
    bot_msg,
    text: str = "🧠 추론 중...",
):
    if use_message_draft:
        try:
            await send_message_draft(update, draft_id, text)
            return bot_msg, use_message_draft, True, True
        except TelegramError as e:
            logger.warning("sendMessageDraft failed while showing reasoning status, falling back to edit_text: %s", e)
            use_message_draft = False

    try:
        if bot_msg is None:
            bot_msg = await update.message.reply_text(text)
        else:
            await bot_msg.edit_text(text)
        return bot_msg, use_message_draft, True, False
    except Exception:
        return bot_msg, use_message_draft, False, False


async def keep_typing_until_visible(update: Update, stop_event: asyncio.Event):
    message = update.message
    if not message:
        return

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                message.get_bot().send_chat_action(
                    chat_id=message.chat_id,
                    action=ChatAction.TYPING,
                    message_thread_id=message.message_thread_id,
                ),
                timeout=TYPING_ACTION_SEND_TIMEOUT,
            )
        except Exception as e:
            logger.debug("send_chat_action failed while keeping typing visible; retrying: %s", e)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=TYPING_ACTION_RETRY_INTERVAL)
            except asyncio.TimeoutError:
                continue
            return

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=TYPING_ACTION_INTERVAL)
        except asyncio.TimeoutError:
            continue


def start_typing_indicator(update: Update) -> TypingIndicator | None:
    if not update.message:
        return None

    stop_event = asyncio.Event()
    return TypingIndicator(
        stop_event=stop_event,
        task=asyncio.create_task(keep_typing_until_visible(update, stop_event)),
    )


async def stop_typing_indicator(indicator: TypingIndicator | None):
    if indicator is None:
        return

    indicator.stop_event.set()
    with suppress(Exception):
        await indicator.task


def trim_conversation_history(history: list[ChatMessage]) -> list[ChatMessage]:
    max_messages = MAX_HISTORY_PAIRS * 2
    if history and history[-1]["role"] == "user":
        max_messages += 1
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def compact_assistant_context(content: str, max_chars: int) -> str:
    stripped = content.strip()
    if len(stripped) <= max_chars:
        return content
    return f"{stripped[:max_chars].rstrip()}\n{ASSISTANT_CONTEXT_TRUNCATION_MARKER}"


def build_llm_context_history(history: list[ChatMessage]) -> list[ChatMessage]:
    latest_assistant_index = None
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "assistant":
            latest_assistant_index = index
            break

    payload_history: list[ChatMessage] = []
    for index, message in enumerate(history):
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "assistant":
            max_chars = (
                MAX_RECENT_ASSISTANT_CONTEXT_CHARS
                if index == latest_assistant_index
                else MAX_ASSISTANT_CONTEXT_CHARS
            )
            content = compact_assistant_context(content, max_chars)
        payload_history.append({"role": role, "content": content})
    return payload_history


def has_recent_source_context(session_key: SessionKey, max_messages: int = 8) -> bool:
    for message in reversed(conversations.get(session_key, [])[-max_messages:]):
        if message.get("role") == "user" and "[User Question]" in message.get("content", ""):
            return True
    return False


def should_apply_active_source_context(session_key: SessionKey, user_message: str) -> bool:
    stripped = user_message.strip()
    if not stripped:
        return False
    if SOURCE_FOLLOWUP_SKIP_RE.search(stripped):
        return False
    if GENERAL_URL_PATTERN.search(stripped):
        return False
    return session_key in active_source_sessions and latest_source_memory(session_key) is not None


def should_apply_source_followup_rules(session_key: SessionKey, user_message: str) -> bool:
    stripped = user_message.strip()
    if not stripped:
        return False
    if SOURCE_FOLLOWUP_SKIP_RE.search(stripped):
        return False
    if GENERAL_URL_PATTERN.search(stripped):
        return False
    if should_apply_active_source_context(session_key, stripped):
        return True
    return has_recent_source_context(session_key)


def build_source_followup_message(user_message: str) -> str:
    return (
        "[Follow-up Source Rules]\n"
        "- Treat the user's latest message as the priority lens for the previous source/context.\n"
        "- Use the previously provided source/context as the evidence base.\n"
        "- Do not use the previous assistant summary as the main frame; re-check the source/context for this lens.\n"
        "- First identify evidence relevant to the user's latest lens, then answer or analyze.\n"
        "- If the source/context does not support the requested lens, say so.\n"
        "- Answer in Korean.\n\n"
        f"[User Question]\n{user_message}"
    )


def build_augmented_context_message(user_message: str, search_context: str) -> str:
    if not search_context:
        return user_message
    return (
        f"{search_context}\n\n"
        f"[Response Rules]\n"
        f"- 반드시 한국어로만 답하세요.\n"
        f"- 원문 언어가 영어, 중국어, 일본어 또는 다른 언어여도 한국어로 번역, 요약, 설명하세요.\n"
        f"- 사용자가 명시적으로 원문 인용을 요청한 경우가 아니라면 중국어/일본어 문자를 출력하지 마세요.\n"
        f"- 사용자가 영어 답변을 명시적으로 요청한 경우에만 영어로 답하세요.\n"
        f"- Treat the provided context/search results as authoritative for current facts.\n"
        f"- If the provided context conflicts with your internal memory, prefer the provided context.\n"
        f"- Do not assert current facts that are not supported by the provided context; say they are unverified.\n"
        f"- For current-event, product, company, market, or investment answers, separate verified facts from analysis and unresolved uncertainties.\n\n"
        f"[User Question]\n{user_message}"
    )


def append_history_message(
    session_key: SessionKey,
    role: str,
    content: str,
    *,
    source_kind: str | None = None,
    source_url: str | None = None,
) -> list[ChatMessage]:
    session_history = session_histories.setdefault(session_key, [])
    session_message: ChatMessage = {"role": role, "content": content}
    if source_kind:
        session_message["source_kind"] = source_kind
    if source_url:
        session_message["source_url"] = source_url
    session_history.append(session_message)

    conversation_history = conversations.setdefault(session_key, [])
    conversation_history.append({"role": role, "content": content})
    conversations[session_key] = trim_conversation_history(conversation_history)
    touch_session_activity(session_key)
    return conversations[session_key]


def prepare_messages(
    session_key: SessionKey,
    user_message: str,
    search_context: str = "",
    *,
    source_kind: str | None = None,
    source_url: str | None = None,
    store_context_in_history: bool = True,
    model_name: str | None = None,
) -> list:

    if search_context:
        augmented_message = build_augmented_context_message(user_message, search_context)
    else:
        augmented_message = user_message
    apply_source_followup = not search_context and should_apply_source_followup_rules(session_key, user_message)
    payload_message = build_source_followup_message(user_message) if apply_source_followup else augmented_message
    history_message = augmented_message if store_context_in_history else user_message

    append_history_message(
        session_key,
        "user",
        history_message,
        source_kind=source_kind,
        source_url=source_url,
    )
    system_prompt = build_system_prompt(model_name or get_llm_task_profile("chat").model)
    if apply_source_followup or not store_context_in_history:
        payload_history = conversations[session_key][:-1] + [{"role": "user", "content": payload_message}]
    else:
        payload_history = conversations[session_key]
    return [{"role": "system", "content": system_prompt}] + build_llm_context_history(payload_history)


def build_context_prompt(user_message: str) -> str:
    stripped = user_message.strip()
    return stripped or DEFAULT_CONTEXT_PROMPT


def should_allow_thinking_for_context(user_message: str) -> bool:
    if ENABLE_THINKING_FOR_CONTEXT:
        return True
    if user_message.strip() == DEFAULT_CONTEXT_PROMPT:
        return False
    return bool(CONTEXT_THINKING_TRIGGER_RE.search(user_message))


def select_llm_task_for_reply(user_message: str, search_context: str = "") -> str:
    if not search_context:
        return "chat"
    if should_allow_thinking_for_context(user_message):
        return "context_analysis"
    return "context_summary"


def build_chat_completion_payload(
    messages: list[dict],
    search_context: str = "",
    user_message: str = "",
    *,
    task: str = "chat",
) -> dict:
    if task == "chat":
        task = select_llm_task_for_reply(user_message, search_context)
    profile = get_llm_task_profile(task)
    payload = {
        "model": profile.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if search_context and task != "context_analysis":
        disable_llm_reasoning(payload)
    else:
        apply_llm_request_options(payload, profile=profile)
    return payload


def extract_reasoning_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    reasoning_tokens = usage.get("reasoning_tokens")
    if isinstance(reasoning_tokens, int):
        return reasoning_tokens
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        nested_reasoning_tokens = completion_details.get("reasoning_tokens")
        if isinstance(nested_reasoning_tokens, int):
            return nested_reasoning_tokens
    return None


def _stream_llm_response(payload: dict, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    task = str(payload.get("_llm_task") or "chat")
    resolved_profile = get_llm_task_profile(task)
    request_payload = {key: value for key, value in payload.items() if not key.startswith("_llm_")}
    try:
        with requests.post(
            build_chat_completions_url(resolved_profile),
            headers=build_llm_headers(resolved_profile),
            json=request_payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                reasoning_token = delta.get("reasoning_content") or delta.get("reasoning", "")
                content_token = delta.get("content", "")
                usage = chunk.get("usage")

                if reasoning_token:
                    loop.call_soon_threadsafe(queue.put_nowait, ("reasoning", reasoning_token))
                if content_token:
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", content_token))
                if usage:
                    loop.call_soon_threadsafe(queue.put_nowait, ("usage", usage))

        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
    except Exception as e:
        loop.call_soon_threadsafe(queue.put_nowait, ("error", e))


async def stream_context_reply(
    update: Update,
    user_id: int,
    user_message: str,
    search_context: str,
    source: str = "context",
    source_kind: str | None = None,
    source_url: str | None = None,
    status_message=None,
):
    kwargs = {
        "source": source,
        "source_kind": source_kind,
        "source_url": source_url,
    }
    if status_message is not None:
        kwargs["status_message"] = status_message
    await stream_reply(update, user_id, build_context_prompt(user_message), search_context, **kwargs)


# ──────────────────────────────────────────────
# 스트리밍 LLM 호출 + 텔레그램 메시지 수정
# ──────────────────────────────────────────────
async def stream_reply(
    update: Update,
    user_id: int,
    user_message: str,
    search_context: str = "",
    source: str = "chat",
    source_kind: str | None = None,
    source_url: str | None = None,
    status_message=None,
):
    cleanup_inactive_sessions()
    session_key = build_session_key(user_id, update.message)
    if session_key is None:
        raise ValueError("Session key could not be determined from the incoming Telegram message.")

    touch_session_activity(session_key)
    ensure_session_identifier(session_key, update.message)
    effective_search_context, effective_source_kind, effective_source_url, store_context_in_history = (
        resolve_source_context_for_request(
            session_key,
            user_message,
            search_context,
            source_kind=source_kind,
            source_url=source_url,
            use_source_memory_for_context=source == "inbox_context",
            preserve_active_source_context=source in {"auto_search", "search_suffix", "search_command"},
        )
    )
    llm_task = select_llm_task_for_reply(user_message, effective_search_context)
    llm_profile = get_llm_task_profile(llm_task)
    messages = prepare_messages(
        session_key,
        user_message,
        effective_search_context,
        source_kind=effective_source_kind,
        source_url=effective_source_url,
        store_context_in_history=store_context_in_history,
        model_name=llm_profile.model,
    )
    request_payload = build_chat_completion_payload(
        messages,
        effective_search_context,
        user_message=user_message,
        task=llm_task,
    )
    request_payload["_llm_task"] = llm_task
    reasoning_disabled = bool(request_payload.get("chat_template_kwargs", {}).get("enable_thinking") is False)

    bot_msg = status_message
    reasoning_status_msg = None
    full_text = ""
    display_text = ""
    last_edit_time = 0
    last_streamed_text = ""
    last_streamed_length = 0
    inside_think = False
    reasoning_status_shown = False
    stream_to_telegram = TELEGRAM_RESPONSE_DELIVERY in {"draft", "edit"}
    use_message_draft = TELEGRAM_RESPONSE_DELIVERY == "draft"
    draft_visible = False
    draft_stream_disabled = False
    draft_id = build_draft_id()
    loop = asyncio.get_running_loop()
    stream_started_at = loop.time()
    first_token_at: float | None = None
    first_reasoning_at: float | None = None
    first_content_at: float | None = None
    first_visible_at: float | None = None
    stream_update_count = 0
    reasoning_chars = 0
    reasoning_used = False
    reasoning_preview = ""
    usage_data: dict | None = None
    active_typing_indicator = _active_typing_indicator.get()
    owns_typing_indicator = active_typing_indicator is None
    typing_indicator = active_typing_indicator or start_typing_indicator(update)
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    worker = asyncio.create_task(asyncio.to_thread(_stream_llm_response, request_payload, loop, queue))

    try:
        while True:
            event, payload = await queue.get()
            if event == "done":
                break
            if event == "error":
                raise payload
            if event == "usage":
                if isinstance(payload, dict):
                    usage_data = payload
                continue

            if event == "reasoning":
                if reasoning_disabled:
                    continue
                token = payload
                if not token:
                    continue

                if first_token_at is None:
                    first_token_at = loop.time()
                if first_reasoning_at is None:
                    first_reasoning_at = loop.time()

                reasoning_used = True
                reasoning_preview += token
                reasoning_chars += len(token)

                if (
                    (stream_to_telegram or TELEGRAM_RESPONSE_DELIVERY == "final")
                    and not reasoning_status_shown
                    and should_show_reasoning_status(reasoning_preview)
                ):
                    status_target = bot_msg if stream_to_telegram or bot_msg is not None else reasoning_status_msg
                    status_target, use_message_draft, shown, used_draft = await show_reasoning_status(
                        update,
                        draft_id,
                        use_message_draft if stream_to_telegram else False,
                        status_target,
                    )
                    if stream_to_telegram or bot_msg is not None:
                        bot_msg = status_target
                    else:
                        reasoning_status_msg = status_target
                    draft_visible = draft_visible or used_draft
                    if shown:
                        reasoning_status_shown = True
                        if first_visible_at is None:
                            first_visible_at = loop.time()
                        stream_update_count += 1
                continue

            token = payload
            if not token:
                continue

            if first_token_at is None:
                first_token_at = loop.time()
            if first_content_at is None:
                first_content_at = loop.time()

            full_text += token

            # <think> 블록 감지 및 스킵
            if "<think>" in full_text and "</think>" not in full_text:
                inside_think = True
                reasoning_used = True
                reasoning_preview = full_text.split("<think>", 1)[-1]
                if (
                    (stream_to_telegram or TELEGRAM_RESPONSE_DELIVERY == "final")
                    and not reasoning_status_shown
                    and should_show_reasoning_status(reasoning_preview)
                ):
                    status_target = bot_msg if stream_to_telegram or bot_msg is not None else reasoning_status_msg
                    status_target, use_message_draft, shown, used_draft = await show_reasoning_status(
                        update,
                        draft_id,
                        use_message_draft if stream_to_telegram else False,
                        status_target,
                    )
                    if stream_to_telegram or bot_msg is not None:
                        bot_msg = status_target
                    else:
                        reasoning_status_msg = status_target
                    draft_visible = draft_visible or used_draft
                    if shown:
                        reasoning_status_shown = True
                        if first_visible_at is None:
                            first_visible_at = loop.time()
                        stream_update_count += 1
                continue
            if inside_think and "</think>" in full_text:
                inside_think = False
                reasoning_chars = max(reasoning_chars, len(extract_think_text(full_text)))
                display_text = full_text.split("</think>")[-1].strip()
                continue
            if inside_think:
                continue

            display_text = normalize_response_text(full_text)

            if not display_text:
                continue
            if not stream_to_telegram:
                continue
            if draft_stream_disabled:
                continue

            # draft 또는 edit 기반 표시를 일정 간격으로 갱신한다.
            now = loop.time()
            streamed_text = display_text[:TELEGRAM_TEXT_LIMIT]
            if ENABLE_RESPONSE_VALIDATION and response_validation_issue(streamed_text):
                continue
            if streamed_text == last_streamed_text:
                continue
            if (
                use_message_draft
                and last_streamed_length > 0
                and len(streamed_text) - last_streamed_length < DRAFT_STREAM_MIN_CHARS_DELTA
            ):
                continue

            if use_message_draft and len(streamed_text) < DRAFT_STREAM_START_CHARS:
                interval = DRAFT_STREAM_START_INTERVAL
            else:
                interval = DRAFT_STREAM_INTERVAL if use_message_draft else STREAM_EDIT_INTERVAL
            if now - last_edit_time >= interval:
                if use_message_draft:
                    try:
                        await send_message_draft(update, draft_id, streamed_text)
                        draft_visible = True
                        if first_visible_at is None:
                            first_visible_at = loop.time()
                        last_edit_time = now
                        last_streamed_text = streamed_text
                        last_streamed_length = len(streamed_text)
                        stream_update_count += 1
                    except TelegramError as e:
                        if draft_visible:
                            logger.warning(
                                "sendMessageDraft failed after draft became visible; disabling draft updates for this response to avoid mixed-mode duplicates: %s",
                                e,
                            )
                            draft_stream_disabled = True
                            last_edit_time = now
                            continue
                        logger.warning("sendMessageDraft failed, falling back to edit_text: %s", e)
                        use_message_draft = False
                        bot_msg = await update.message.reply_text(streamed_text)
                        if first_visible_at is None:
                            first_visible_at = loop.time()
                        last_edit_time = now
                        last_streamed_text = streamed_text
                        last_streamed_length = len(streamed_text)
                        stream_update_count += 1
                else:
                    try:
                        if bot_msg is None:
                            bot_msg = await update.message.reply_text(streamed_text)
                        else:
                            await bot_msg.edit_text(streamed_text)
                        if first_visible_at is None:
                            first_visible_at = loop.time()
                        last_edit_time = now
                        last_streamed_text = streamed_text
                        last_streamed_length = len(streamed_text)
                        stream_update_count += 1
                    except Exception:
                        pass  # rate limit 등 무시

        # 최종 메시지 업데이트
        if full_text:
            final_text = normalize_response_text(full_text)
        else:
            final_text = "⚠️ 빈 응답"
        final_text = await validate_and_rewrite_response(final_text, user_message, effective_search_context)
        reasoning_chars = max(reasoning_chars, len(extract_think_text(full_text)))

        try:
            final_chunks = chunk_text(final_text)
            if use_message_draft:
                final_draft_text = final_chunks[0][:TELEGRAM_TEXT_LIMIT]
                if final_draft_text and final_draft_text != last_streamed_text and not draft_stream_disabled:
                    try:
                        await send_message_draft(update, draft_id, final_draft_text)
                        draft_visible = True
                        last_streamed_text = final_draft_text
                        last_streamed_length = len(final_draft_text)
                        stream_update_count += 1
                        if first_visible_at is None:
                            first_visible_at = loop.time()
                        await asyncio.sleep(DRAFT_STREAM_FINAL_FLUSH_DELAY)
                    except TelegramError as e:
                        if draft_visible:
                            logger.warning(
                                "final sendMessageDraft flush failed after draft became visible; sending final message without switching modes: %s",
                                e,
                            )
                            draft_stream_disabled = True
                        else:
                            logger.warning("final sendMessageDraft flush failed, falling back to final message only: %s", e)
                            use_message_draft = False

            if use_message_draft:
                for chunk in final_chunks:
                    await update.message.reply_text(chunk)
            else:
                if bot_msg is None:
                    bot_msg = await update.message.reply_text(final_chunks[0])
                    if first_visible_at is None:
                        first_visible_at = loop.time()
                    for chunk in final_chunks[1:]:
                        await update.message.reply_text(chunk)
                elif len(final_text) > TELEGRAM_TEXT_LIMIT:
                    await bot_msg.edit_text(final_chunks[0])
                    for chunk in final_chunks[1:]:
                        await update.message.reply_text(chunk)
                else:
                    await bot_msg.edit_text(final_text)

            if reasoning_status_msg is not None:
                with suppress(Exception):
                    await reasoning_status_msg.delete()
        except BadRequest as e:
            if "Message is not modified" not in str(e):
                raise

        append_history_message(session_key, "assistant", final_text)
        total_elapsed_ms = int((loop.time() - stream_started_at) * 1000)
        first_token_ms = int((first_token_at - stream_started_at) * 1000) if first_token_at else None
        first_reasoning_ms = int((first_reasoning_at - stream_started_at) * 1000) if first_reasoning_at else None
        first_content_ms = int((first_content_at - stream_started_at) * 1000) if first_content_at else None
        first_visible_ms = int((first_visible_at - stream_started_at) * 1000) if first_visible_at else None
        reasoning_tokens = extract_reasoning_tokens(usage_data)
        logger.info(
            "Stream metrics user=%s source=%s task=%s model=%s mode=%s reasoning_disabled=%s reasoning_used=%s reasoning_chars=%s reasoning_tokens=%s first_token_ms=%s first_reasoning_ms=%s first_content_ms=%s first_visible_ms=%s total_ms=%s updates=%s chars=%s",
            user_id,
            source,
            llm_task,
            llm_profile.model,
            "draft-disabled" if draft_stream_disabled else ("draft" if use_message_draft else ("edit" if stream_to_telegram else "final")),
            reasoning_disabled,
            reasoning_used,
            reasoning_chars,
            reasoning_tokens,
            first_token_ms,
            first_reasoning_ms,
            first_content_ms,
            first_visible_ms,
            total_elapsed_ms,
            stream_update_count,
            len(final_text),
        )

    except Exception as e:
        logger.error(f"LLM error: {e}")
        error_text = f"⚠️ {LLM_PROVIDER_NAME} 연결 실패: {e}"
        if owns_typing_indicator:
            await stop_typing_indicator(typing_indicator)
        if reasoning_status_msg is not None:
            with suppress(Exception):
                await reasoning_status_msg.delete()
        if bot_msg is not None:
            await bot_msg.edit_text(error_text)
        else:
            await update.message.reply_text(error_text)
    finally:
        with suppress(Exception):
            await worker
        if owns_typing_indicator:
            await stop_typing_indicator(typing_indicator)



# ──────────────────────────────────────────────
# Tavily 검색
# ──────────────────────────────────────────────
def search_web(query: str) -> str:
    if tavily is None:
        logger.error("Search error: missing TAVILY_API_KEY")
        return "Search failed: missing TAVILY_API_KEY"

    try:
        results = tavily.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
        )

        output_parts = ["[Web Search Results]"]

        if results.get("answer"):
            output_parts.append(f"Summary: {results['answer']}")

        for i, r in enumerate(results.get("results", []), 1):
            output_parts.append(
                f"\n[{i}] {r['title']}\n{r['content'][:300]}\nURL: {r['url']}"
            )

        return "\n".join(output_parts) if output_parts else "No results found."

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search failed: {e}"


# ──────────────────────────────────────────────
# 핸들러
# ──────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vault_status = "✅ 연결됨" if VAULT_CAPTURE_PATH else "❌ 미설정"
    access_status = "✅ 제한됨" if ALLOWED_USER_IDS else "⚠️ 미설정"
    await update.message.reply_text(
        "🤖 LLM 봇 준비 완료!\n\n"
        "💬 일반 메시지 → LLM 대화\n"
        "📥 /context → Inbox 컨텍스트 큐에서 오래된 소스 적용\n"
        "🗑️ /clear → 대화 기록 초기화 + Vault 저장\n"
        "ℹ️ /model → 현재 모델 확인\n\n"
        "숨은 단축키: /s 검색, /ctx, /raw URL, /c, /m\n\n"
        f"📂 Vault: {vault_status}\n"
        f"🔐 접근 제어: {access_status}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    cleanup_inactive_sessions()
    current_session_key = build_session_key(user_id, update.message)
    if current_session_key is not None:
        touch_session_activity(current_session_key)

    user_text = update.message.text
    if not user_text:
        return

    typing_indicator = start_typing_indicator(update)
    typing_token = _active_typing_indicator.set(typing_indicator)
    try:
        await handle_message_with_typing(
            update,
            user_id,
            current_session_key,
            user_text,
        )
    finally:
        _active_typing_indicator.reset(typing_token)
        await stop_typing_indicator(typing_indicator)


async def handle_message_with_typing(
    update: Update,
    user_id: int,
    current_session_key: SessionKey | None,
    user_text: str,
):
    routed_text, extract_only_requested = parse_extract_only_request(user_text)
    extraction_result = await extract_context_from_user_text(
        update,
        user_id,
        routed_text,
        session_key=current_session_key,
        extract_only_requested=extract_only_requested,
    )
    if extraction_result.matched:
        extracted = extraction_result.extracted
        if extracted is None:
            return
        if extract_only_requested:
            if current_session_key is not None:
                ensure_session_identifier(current_session_key, update.message)
                await send_extract_only_reply(update, current_session_key, extracted)
            else:
                for chunk in chunk_text(format_extract_only_text(extracted.content, extracted.source_kind)):
                    await update.message.reply_text(chunk)
            return

        await stream_context_reply(
            update,
            user_id,
            extracted.user_message,
            extracted.content,
            source=extracted.source,
            source_kind=extracted.source_kind,
            source_url=extracted.source_url,
        )
        return

    if extract_only_requested:
        await update.message.reply_text("사용법: /e URL 또는 URL /e")
        return

    # 메시지 끝에 /s가 있으면 검색 모드
    if user_text.rstrip().endswith("/s"):
        query = user_text.rstrip()[:-2].strip()
        if query:
            status_message = await update.message.reply_text("🔍 검색 중...")
            stage_started_at = asyncio.get_running_loop().time()
            search_results = await asyncio.to_thread(search_web, query)
            log_stage_metrics(
                "search",
                user_id,
                int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
                ok=bool(search_results),
                source="search_suffix",
                detail=query,
                chars=len(search_results) if search_results else 0,
            )
            await stream_reply(update, user_id, query, search_results, source="search_suffix", status_message=status_message)
            return

    stage_started_at = asyncio.get_running_loop().time()
    auto_search_decision = await asyncio.to_thread(
        resolve_auto_search_decision,
        user_text,
        current_session_key,
    )
    log_stage_metrics(
        "auto_search",
        user_id,
        int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
        ok=auto_search_decision.needs_search,
        source=auto_search_decision.source,
        detail=auto_search_decision.query or auto_search_decision.reason,
    )
    if auto_search_decision.needs_search:
        status_message = await update.message.reply_text("🔍 최신 정보 확인 중...")
        search_started_at = asyncio.get_running_loop().time()
        search_results = await asyncio.to_thread(search_web, auto_search_decision.query or user_text)
        log_stage_metrics(
            "search",
            user_id,
            int((asyncio.get_running_loop().time() - search_started_at) * 1000),
            ok=bool(search_results),
            source="auto_search",
            detail=auto_search_decision.query or user_text,
            chars=len(search_results) if search_results else 0,
        )
        await stream_reply(update, user_id, user_text, search_results, source="auto_search", status_message=status_message)
        return

    await stream_reply(update, user_id, user_text, source="chat")


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    cleanup_inactive_sessions()
    current_session_key = build_session_key(user_id, update.message)
    if current_session_key is not None:
        touch_session_activity(current_session_key)

    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("사용법: /s 검색할 내용")
        return

    status_message = await update.message.reply_text("🔍 검색 중...")
    stage_started_at = asyncio.get_running_loop().time()
    search_results = await asyncio.to_thread(search_web, query)
    log_stage_metrics(
        "search",
        user_id,
        int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
        ok=bool(search_results),
        source="search_command",
        detail=query,
        chars=len(search_results) if search_results else 0,
    )
    await stream_reply(update, user_id, query, search_results, source="search_command", status_message=status_message)


async def handle_inbox_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    args = " ".join(context.args) if context and context.args else ""
    if args and args.casefold() != "latest":
        await update.message.reply_text("사용법: /ctx 또는 /ctx latest")
        return

    typing_indicator = start_typing_indicator(update)
    typing_token = _active_typing_indicator.set(typing_indicator)
    try:
        await handle_inbox_context_with_typing(update, user_id)
    finally:
        _active_typing_indicator.reset(typing_token)
        await stop_typing_indicator(typing_indicator)


async def handle_inbox_context_with_typing(update: Update, user_id: int):
    cleanup_inactive_sessions()
    current_session_key = build_session_key(user_id, update.message)
    if current_session_key is None:
        await update.message.reply_text("⚠️ 현재 채팅 세션을 확인하지 못했습니다.")
        return
    touch_session_activity(current_session_key)
    ensure_session_identifier(current_session_key, update.message)

    try:
        source = await asyncio.to_thread(fetch_next_inbox_context_source)
    except Exception as exc:
        logger.error("Inbox context fetch failed: %s", exc)
        await update.message.reply_text(f"⚠️ Inbox context queue 조회 실패: {exc}")
        return

    if source is None:
        await update.message.reply_text("가져올 준비된 컨텍스트가 없어요. Inbox bot에 URL /ctx 로 먼저 넣어두면 됩니다.")
        return

    await update.message.reply_text(build_inbox_context_processing_reply(source))
    original_chars = len(source.text)
    cached_source = pop_prefetched_inbox_context_source(source)
    cached_initial_reply: str | None = None
    if cached_source is not None:
        source = cached_source.source
        enhanced = cached_source.enhanced
        cached_initial_reply = cached_source.initial_reply
        logger.info(
            "Inbox context using prefetched source source_id=%s kind=%s enhanced=%s summary=%s",
            source.source_id,
            source.source_kind,
            enhanced,
            bool(cached_initial_reply),
        )
    else:
        prefetch_task = inbox_context_prefetch_tasks.get(source.source_id)
        if prefetch_task is not None and not prefetch_task.done():
            await update.message.reply_text("미리 처리 중인 컨텍스트를 마무리하는 중입니다.")
            with suppress(Exception):
                await prefetch_task
            cached_source = pop_prefetched_inbox_context_source(source)
            if cached_source is not None:
                source = cached_source.source
                enhanced = cached_source.enhanced
                cached_initial_reply = cached_source.initial_reply
            else:
                source, enhanced = await enhance_inbox_youtube_context_source(update, user_id, source)
        else:
            source, enhanced = await enhance_inbox_youtube_context_source(update, user_id, source)

    try:
        apply_inbox_context_source_to_session(current_session_key, source)
    except Exception as exc:
        logger.error("Inbox context registration failed source_id=%s: %s", source.source_id, exc)
        await update.message.reply_text(f"⚠️ 컨텍스트를 세션에 적용하지 못했습니다: {exc}")
        return

    try:
        await asyncio.to_thread(mark_inbox_context_source_consumed, source.source_id)
    except Exception as exc:
        logger.error("Inbox context consume failed source_id=%s: %s", source.source_id, exc)
        await update.message.reply_text(
            "컨텍스트는 현재 세션에 적용됐지만 Inbox 큐의 consumed 처리는 실패했습니다.\n"
            f"source #{source.source_id}: {exc}"
        )
        return
    await asyncio.to_thread(delete_persistent_prefetched_inbox_context_source, source.source_id)

    logger.info(
        "Inbox context applied source_id=%s kind=%s url=%s chars=%s original_chars=%s enhanced=%s remaining=%s",
        source.source_id,
        source.source_kind,
        source.source_url or "",
        len(source.text),
        original_chars,
        enhanced,
        source.remaining_ready_count,
    )
    await update.message.reply_text(build_inbox_context_status_reply(source, enhanced=enhanced))
    start_inbox_context_prefetch("after_ctx")

    if cached_initial_reply:
        final_text = await validate_and_rewrite_response(
            cached_initial_reply,
            build_context_prompt(DEFAULT_CONTEXT_PROMPT),
            source.text,
        )
        for chunk in chunk_text(final_text):
            await update.message.reply_text(chunk)
        append_prefetched_inbox_context_reply_to_history(current_session_key, source, final_text)
    else:
        await stream_context_reply(
            update,
            user_id,
            DEFAULT_CONTEXT_PROMPT,
            source.text,
            source="inbox_context",
            source_kind=source.source_kind,
            source_url=source.source_url,
        )


async def handle_extract(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    cleanup_inactive_sessions()
    current_session_key = build_session_key(user_id, update.message)
    if current_session_key is not None:
        touch_session_activity(current_session_key)
        ensure_session_identifier(current_session_key, update.message)

    query = " ".join(context.args) if context and context.args else ""
    query, _ = parse_extract_only_request(query)
    if not query:
        await update.message.reply_text("사용법: /e URL 또는 URL /e")
        return

    extraction_result = await extract_context_from_user_text(
        update,
        user_id,
        query,
        session_key=current_session_key,
        extract_only_requested=True,
    )
    if not extraction_result.matched:
        await update.message.reply_text("사용법: /e URL 또는 URL /e")
        return
    if extraction_result.extracted is None:
        return

    if current_session_key is not None:
        await send_extract_only_reply(update, current_session_key, extraction_result.extracted)
    else:
        output_text = format_extract_only_text(
            extraction_result.extracted.content,
            extraction_result.extracted.source_kind,
        )
        for chunk in chunk_text(output_text):
            await update.message.reply_text(chunk)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    cleanup_inactive_sessions()
    current_session_key = build_session_key(user_id, update.message)
    if current_session_key is not None:
        touch_session_activity(current_session_key)

    doc = update.message.document
    if doc.mime_type != "application/pdf":
        return

    await update.message.reply_text("📄 PDF 읽는 중...")
    file = await doc.get_file()
    file_bytes = await file.download_as_bytearray()
    stage_started_at = asyncio.get_running_loop().time()
    pdf_context = await asyncio.to_thread(extract_pdf_text, bytes(file_bytes))
    log_stage_metrics(
        "extract",
        user_id,
        int((asyncio.get_running_loop().time() - stage_started_at) * 1000),
        ok=bool(pdf_context),
        source="pdf_upload",
        detail=doc.mime_type,
        chars=len(pdf_context) if pdf_context else 0,
    )

    if not pdf_context:
        await update.message.reply_text("⚠️ PDF에서 텍스트를 추출할 수 없습니다.")
        return

    caption = update.message.caption or ""
    caption_text, extract_only_requested = parse_extract_only_request(caption)
    if extract_only_requested:
        if current_session_key is not None:
            ensure_session_identifier(current_session_key, update.message)
            await send_extract_only_reply(
                update,
                current_session_key,
                ExtractedContext(
                    user_message=caption_text,
                    content=pdf_context,
                    source="pdf_upload",
                    source_kind="pdf",
                ),
            )
        else:
            for chunk in chunk_text(format_extract_only_text(pdf_context, "pdf")):
                await update.message.reply_text(chunk)
        return

    await stream_context_reply(update, user_id, caption, pdf_context, source="pdf_upload")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session_key = build_session_key(user_id, update.message)
    if session_key is not None:
        save_session_to_vault(session_key)
        clear_session_state(session_key)
    cleanup_inactive_sessions()
    await update.message.reply_text("🗑️ 대화 기록 초기화 완료.")


async def show_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    env_hint = Path(ENV_FILES_LOADED[-1]).name if ENV_FILES_LOADED else "process env"
    lines = ["📦 현재 LLM task profiles"]
    for task in LLM_TASK_NAMES:
        profile = get_llm_task_profile(task)
        lines.append(f"- {task}: {profile.model or '(unset)'} ({profile.provider_name or 'provider unset'})")
    lines.append(f"🔧 설정 소스: {env_hint}")
    await update.message.reply_text("\n".join(lines))


BOT_MENU_COMMANDS = [
    BotCommand("context", "Inbox 컨텍스트 가져오기"),
    BotCommand("clear", "대화 기록 초기화"),
    BotCommand("model", "현재 모델 확인"),
]


async def setup_bot_commands(app: Application) -> None:
    await app.bot.set_my_commands(BOT_MENU_COMMANDS)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    missing_config = validate_runtime_config()
    if missing_config:
        raise RuntimeError(f"Missing required configuration: {', '.join(missing_config)}")

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(setup_bot_commands).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("s", handle_search))
    app.add_handler(CommandHandler("search", handle_search))
    app.add_handler(CommandHandler("ctx", handle_inbox_context))
    app.add_handler(CommandHandler("context", handle_inbox_context))
    app.add_handler(CommandHandler("e", handle_extract))
    app.add_handler(CommandHandler("extract", handle_extract))
    app.add_handler(CommandHandler("raw", handle_extract))
    app.add_handler(CommandHandler("c", clear_history))
    app.add_handler(CommandHandler("clear", clear_history))
    app.add_handler(CommandHandler("m", show_model))
    app.add_handler(CommandHandler("model", show_model))
    app.add_handler(CommandHandler("drafttest", diagnose_draft))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if ENABLE_INBOX_CONTEXT_PREFETCH and app.job_queue is not None:
        app.job_queue.run_once(run_inbox_context_prefetch_startup_job, when=5, name="inbox-context-prefetch-startup")
        app.job_queue.run_repeating(
            run_inbox_context_prefetch_job,
            interval=INBOX_CONTEXT_PREFETCH_INTERVAL_SECONDS,
            first=INBOX_CONTEXT_PREFETCH_INTERVAL_SECONDS,
            name="inbox-context-prefetch",
        )

    logger.info("Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
