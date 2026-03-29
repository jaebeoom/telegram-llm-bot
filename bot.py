import os
import re
import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from tagger import generate_tags
from extractors import (
    TWEET_URL_PATTERN,
    YOUTUBE_URL_PATTERN,
    GENERAL_URL_PATTERN,
    extract_tweet,
    extract_pdf_text,
    extract_pdf_from_url,
    extract_youtube_transcript,
    extract_web_text,
)

load_dotenv()

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL") or os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER_NAME", "OMLX")
ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-27b")
VAULT_HAIKU_PATH = os.getenv("VAULT_HAIKU_PATH", "")

# Tavily 클라이언트
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# 대화 기록 (user_id별)
conversations: dict[int, list[dict]] = {}
last_activity: dict[int, float] = {}  # user_id → 마지막 활동 timestamp
MAX_HISTORY = 10
INACTIVITY_TIMEOUT = 3 * 60 * 60  # 3시간 (초)

SYSTEM_PROMPT = f"""You are a helpful assistant. Answer concisely and accurately.
Today's date is {datetime.now().strftime('%Y-%m-%d')}. Content shared by the user (X posts, articles, PDFs, YouTube transcripts) reflects real, current events — not fiction or speculation. Treat them as factual present-day information.
When search results are provided, use them to give up-to-date answers and cite sources when relevant.
Always respond in Korean. English technical terms are allowed.
Never use Chinese or Japanese characters in your responses.
Always respond in plain text only. Never use markdown formatting such as **, ##, `, ```, -, or any other markup syntax."""

STREAM_EDIT_INTERVAL = 1.5  # 텔레그램 메시지 수정 간격 (초)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_chat_completions_url() -> str:
    """OpenAI-compatible base URL에서 chat/completions endpoint 생성."""
    return f"{LLM_API_BASE_URL.rstrip('/')}/chat/completions"


def build_llm_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    return headers


# ──────────────────────────────────────────────
# Vault 로그 저장
# ──────────────────────────────────────────────
def save_session_to_vault(user_id: int) -> bool:
    """세션 종료 시 대화 전체를 Vault/Logs에 MD로 저장"""
    if not VAULT_HAIKU_PATH:
        return False

    history = conversations.get(user_id)
    if not history:
        return False

    logs_dir = Path(VAULT_HAIKU_PATH).expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = logs_dir / f"{today}.md"

    # 대화를 MD로 포맷
    now = datetime.now().strftime("%H:%M")
    session_md = f"\n\n---\n\n## AI 세션 ({now}, {MODEL_NAME})\n\n"

    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            # 검색 컨텍스트가 포함된 경우, [User Question] 이후만 표시
            if "[User Question]" in content:
                parts = content.split("[User Question]")
                context_part = parts[0].strip()
                question_part = parts[1].strip() if len(parts) > 1 else ""

                # 컨텍스트 소스 종류 감지
                source_type = "참고 자료"
                if "[X Post]" in context_part or "[X Article]" in context_part:
                    source_type = "X 포스트"
                elif "[YouTube Transcript]" in context_part:
                    source_type = "YouTube"
                elif "[Web Article]" in context_part:
                    source_type = "웹 아티클"
                elif "[Web Search Results]" in context_part:
                    source_type = "웹 검색"
                elif "[PDF Document]" in context_part:
                    source_type = "PDF"

                session_md += f"**나** ({source_type} 첨부): {question_part}\n\n"
            else:
                session_md += f"**나**: {content}\n\n"
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
        return True
    allowed = [int(uid.strip()) for uid in ALLOWED_USER_IDS.split(",") if uid.strip()]
    return user_id in allowed


# ──────────────────────────────────────────────
# <think> 태그 제거
# ──────────────────────────────────────────────
def strip_think(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


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



# ──────────────────────────────────────────────
# 대화 히스토리 준비
# ──────────────────────────────────────────────
async def cleanup_inactive(context: ContextTypes.DEFAULT_TYPE):
    """비활성 유저의 대화 기록 자동 정리 (job queue에서 주기적 실행)"""
    now = time.time()
    expired = [uid for uid, ts in last_activity.items() if now - ts > INACTIVITY_TIMEOUT]
    for uid in expired:
        save_session_to_vault(uid)
        conversations.pop(uid, None)
        last_activity.pop(uid, None)
        try:
            await context.bot.send_message(
                chat_id=uid,
                text="🗑️ 3시간 동안 비활성 상태여서 대화 기록을 자동 초기화했어요.",
            )
        except Exception:
            pass
    if expired:
        logger.info(f"Auto-cleared {len(expired)} inactive session(s)")


def prepare_messages(user_id: int, user_message: str, search_context: str = "") -> list:
    if user_id not in conversations:
        conversations[user_id] = []

    last_activity[user_id] = time.time()

    if search_context:
        augmented_message = (
            f"{search_context}\n\n"
            f"[User Question]\n{user_message}"
        )
    else:
        augmented_message = user_message

    conversations[user_id].append({"role": "user", "content": augmented_message})

    if len(conversations[user_id]) > MAX_HISTORY:
        conversations[user_id] = conversations[user_id][-MAX_HISTORY:]

    return [{"role": "system", "content": SYSTEM_PROMPT}] + conversations[user_id]


# ──────────────────────────────────────────────
# 스트리밍 LLM 호출 + 텔레그램 메시지 수정
# ──────────────────────────────────────────────
async def stream_reply(update: Update, user_id: int, user_message: str, search_context: str = ""):
    messages = prepare_messages(user_id, user_message, search_context)

    # 플레이스홀더 메시지 전송
    bot_msg = await update.message.reply_text("⏳")

    full_text = ""
    display_text = ""
    last_edit_time = 0
    inside_think = False
    think_notified = False

    try:
        r = requests.post(
            build_chat_completions_url(),
            headers=build_llm_headers(),
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        r.raise_for_status()

        for line in r.iter_lines():
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
                token = delta.get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

            if not token:
                continue

            full_text += token

            # <think> 블록 감지 및 스킵
            if "<think>" in full_text and "</think>" not in full_text:
                inside_think = True
                if not think_notified:
                    try:
                        await bot_msg.edit_text("🧠 생각 중...")
                    except Exception:
                        pass
                    think_notified = True
                continue
            if inside_think and "</think>" in full_text:
                inside_think = False
                display_text = full_text.split("</think>")[-1].strip()
                continue
            if inside_think:
                continue

            display_text = strip_markdown(strip_think(full_text))

            if not display_text:
                continue

            # 일정 간격마다 텔레그램 메시지 수정
            now = asyncio.get_event_loop().time()
            if now - last_edit_time >= STREAM_EDIT_INTERVAL:
                try:
                    await bot_msg.edit_text(display_text[:4000])
                    last_edit_time = now
                except Exception:
                    pass  # rate limit 등 무시

        # 최종 메시지 업데이트
        final_text = strip_markdown(strip_think(full_text)) if full_text else "⚠️ 빈 응답"

        try:
            if len(final_text) > 4000:
                await bot_msg.edit_text(final_text[:4000])
                for i in range(4000, len(final_text), 4000):
                    await update.message.reply_text(final_text[i : i + 4000])
            else:
                await bot_msg.edit_text(final_text)
        except BadRequest as e:
            if "Message is not modified" not in str(e):
                raise

        conversations[user_id].append({"role": "assistant", "content": final_text})

        # 알림 트리거: 임시 메시지 전송 후 즉시 삭제
        try:
            notify_msg = await update.message.reply_text("✅")
            await notify_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"LLM error: {e}")
        await bot_msg.edit_text(f"⚠️ {LLM_PROVIDER_NAME} 연결 실패: {e}")



# ──────────────────────────────────────────────
# Tavily 검색
# ──────────────────────────────────────────────
def search_web(query: str) -> str:
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
    vault_status = "✅ 연결됨" if VAULT_HAIKU_PATH else "❌ 미설정"
    await update.message.reply_text(
        "🤖 LLM 봇 준비 완료!\n\n"
        "💬 일반 메시지 → LLM 대화\n"
        "🔍 /s 질문 → 웹 검색 + LLM 답변\n"
        "🗑️ /c → 대화 기록 초기화 + Vault 저장\n"
        "ℹ️ /m → 현재 모델 확인\n\n"
        f"📂 Vault: {vault_status}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    user_text = update.message.text
    if not user_text:
        return

    # X/Twitter URL 감지
    tweet_match = TWEET_URL_PATTERN.search(user_text)
    if tweet_match:
        tweet_id = tweet_match.group(1)
        tweet_context = await asyncio.to_thread(extract_tweet, tweet_id)
        if tweet_context:
            # URL 제거 후 남은 텍스트를 user message로
            user_msg = TWEET_URL_PATTERN.sub("", user_text).strip()
            if user_msg:
                await stream_reply(update, user_id, user_msg, tweet_context)
            else:
                await stream_reply(update, user_id, "이 내용을 간단히 요약해줘.", tweet_context)
            return
        else:
            await update.message.reply_text("⚠️ 피드를 가져올 수 없습니다.")
            return

    # YouTube URL 감지
    yt_match = YOUTUBE_URL_PATTERN.search(user_text)
    if yt_match:
        video_id = yt_match.group(1)
        await update.message.reply_text("🎬 스크립트 추출 중...")
        yt_context = await asyncio.to_thread(extract_youtube_transcript, video_id)
        if yt_context:
            user_msg = YOUTUBE_URL_PATTERN.sub("", user_text).strip()
            if user_msg:
                await stream_reply(update, user_id, user_msg, yt_context)
            else:
                await stream_reply(update, user_id, "이 내용을 간단히 요약해줘.", yt_context)
            return
        else:
            await update.message.reply_text("⚠️ 스크립트를 가져올 수 없습니다. (스크립트가 없는 영상일 수 있어요)")
            return

    # 일반 URL 감지 (X, YouTube에 해당하지 않는 URL)
    url_match = GENERAL_URL_PATTERN.search(user_text)
    if url_match:
        url = url_match.group(0)
        user_msg = GENERAL_URL_PATTERN.sub("", user_text).strip()

        # PDF URL 감지
        if url.lower().endswith(".pdf"):
            await update.message.reply_text("📄 PDF 다운로드 중...")
            pdf_context = await asyncio.to_thread(extract_pdf_from_url, url)
            if pdf_context:
                if user_msg:
                    await stream_reply(update, user_id, user_msg, pdf_context)
                else:
                    await stream_reply(update, user_id, "이 내용을 간단히 요약해줘.", pdf_context)
                return
            else:
                await update.message.reply_text("⚠️ PDF에서 텍스트를 추출할 수 없습니다.")
                return

        # 일반 웹페이지
        await update.message.reply_text("📖 웹페이지 읽는 중...")
        web_context = await asyncio.to_thread(extract_web_text, url)
        if web_context:
            if user_msg:
                await stream_reply(update, user_id, user_msg, web_context)
            else:
                await stream_reply(update, user_id, "이 내용을 간단히 요약해줘.", web_context)
            return
        else:
            await update.message.reply_text("⚠️ 웹페이지에서 텍스트를 추출할 수 없습니다. (JavaScript 기반 페이지는 지원되지 않아요)")
            return

    # 메시지 끝에 /s가 있으면 검색 모드
    if user_text.rstrip().endswith("/s"):
        query = user_text.rstrip()[:-2].strip()
        if query:
            await update.message.reply_text("🔍 검색 중...")
            search_results = await asyncio.to_thread(search_web, query)
            await stream_reply(update, user_id, query, search_results)
            return

    await stream_reply(update, user_id, user_text)


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("사용법: /s 검색할 내용")
        return

    await update.message.reply_text("🔍 검색 중...")
    search_results = await asyncio.to_thread(search_web, query)
    await stream_reply(update, user_id, query, search_results)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    doc = update.message.document
    if doc.mime_type != "application/pdf":
        return

    await update.message.reply_text("📄 PDF 읽는 중...")
    file = await doc.get_file()
    file_bytes = await file.download_as_bytearray()
    pdf_context = await asyncio.to_thread(extract_pdf_text, bytes(file_bytes))

    if not pdf_context:
        await update.message.reply_text("⚠️ PDF에서 텍스트를 추출할 수 없습니다.")
        return

    caption = update.message.caption or ""
    if caption.strip():
        await stream_reply(update, user_id, caption.strip(), pdf_context)
    else:
        prepare_messages(user_id, pdf_context)
        await stream_reply(update, user_id, "이 내용을 간단히 요약해줘.", pdf_context)


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    save_session_to_vault(user_id)
    conversations.pop(user_id, None)
    last_activity.pop(user_id, None)
    await update.message.reply_text("🗑️ 대화 기록 초기화 완료.")


async def show_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"📦 현재 모델: {MODEL_NAME}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("s", handle_search))
    app.add_handler(CommandHandler("search", handle_search))
    app.add_handler(CommandHandler("c", clear_history))
    app.add_handler(CommandHandler("m", show_model))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 10분마다 비활성 세션 정리 체크
    app.job_queue.run_repeating(cleanup_inactive, interval=600, first=600)

    logger.info("Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
