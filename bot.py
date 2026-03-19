import os
import re
import asyncio
import json
import logging
import time
from datetime import datetime
import requests
import fitz
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from tavily import TavilyClient
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

load_dotenv()

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-27b")

# Tavily 클라이언트
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# 대화 기록 (user_id별)
conversations: dict[int, list[dict]] = {}
last_activity: dict[int, float] = {}  # user_id → 마지막 활동 timestamp
MAX_HISTORY = 10
INACTIVITY_TIMEOUT = 3 * 60 * 60  # 3시간 (초)
MAX_PDF_CHARS = 20000
MAX_TRANSCRIPT_CHARS = 20000

SYSTEM_PROMPT = f"""You are a helpful assistant. Answer concisely and accurately.
Today's date is {datetime.now().strftime('%Y-%m-%d')}. Content shared by the user (X posts, articles, PDFs, YouTube transcripts) reflects real, current events — not fiction or speculation. Treat them as factual present-day information.
When search results are provided, use them to give up-to-date answers and cite sources when relevant.
Respond in the same language the user uses.
Never use Chinese characters or words in your responses.
Always respond in plain text only. Never use markdown formatting such as **, ##, `, ```, -, or any other markup syntax."""

STREAM_EDIT_INTERVAL = 1.5  # 텔레그램 메시지 수정 간격 (초)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        conversations.pop(uid, None)
        last_activity.pop(uid, None)
        try:
            await context.bot.send_message(chat_id=uid, text="🗑️ 3시간 동안 비활성 상태여서 대화 기록을 자동 초기화했어요.")
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
            f"{LM_STUDIO_URL}/chat/completions",
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

        if len(final_text) > 4000:
            await bot_msg.edit_text(final_text[:4000])
            for i in range(4000, len(final_text), 4000):
                await update.message.reply_text(final_text[i : i + 4000])
        else:
            await bot_msg.edit_text(final_text)

        conversations[user_id].append({"role": "assistant", "content": final_text})

        # 알림 트리거: 임시 메시지 전송 후 즉시 삭제
        try:
            notify_msg = await update.message.reply_text("✅")
            await notify_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"LLM error: {e}")
        await bot_msg.edit_text(f"⚠️ LM Studio 연결 실패: {e}")


# ──────────────────────────────────────────────
# X(Twitter) 피드 추출
# ──────────────────────────────────────────────
TWEET_URL_PATTERN = re.compile(r"https?://(?:twitter\.com|x\.com)/\w+/status/(\d+)(?:\?\S*)?")



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

        # 아티클(장문 포스트) 감지 → content.blocks에서 본문 추출
        article = tweet.get("article")
        if article:
            title = article.get("title", "")
            blocks = article.get("content", {}).get("blocks", [])
            body = "\n".join(b.get("text", "") for b in blocks if b.get("text"))
            if body:
                body = body[:MAX_PDF_CHARS]  # 길이 제한 재활용
                return f"[X Article]\n작성자: {name}\n제목: {title}\nDate: {created}\n\n{body}"

        # 일반 피드
        media = tweet.get("media", {})
        if not text or text.startswith("https://t.co/"):
            media_types = [m.get("type", "unknown") for m in media.get("all", [])]
            desc = f"[미디어: {', '.join(media_types)}]" if media_types else "[텍스트 없음]"
            return f"[X Post]\n{name}: {desc}\nDate: {created}"
        return f"[X Post]\n{name}: {text}\nDate: {created}"
    except Exception as e:
        logger.error(f"X post extraction error: {e}")
        return None


# ──────────────────────────────────────────────
# PDF 텍스트 추출
# ──────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str | None:
    """pymupdf로 PDF 바이트에서 텍스트 추출"""
    try:
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


# ──────────────────────────────────────────────
# YouTube 스크립트 추출
# ──────────────────────────────────────────────
YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)


def extract_youtube_transcript(video_id: str) -> str | None:
    """youtube-transcript-api로 스크립트 추출"""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["ko", "en"])
        text = " ".join(s.text for s in transcript.snippets)
        if not text:
            return None
        return f"[YouTube Transcript]\n{text[:MAX_TRANSCRIPT_CHARS]}"
    except Exception as e:
        logger.error(f"YouTube transcript error: {e}")
        return None


# ──────────────────────────────────────────────
# 일반 웹페이지 텍스트 추출
# ──────────────────────────────────────────────
GENERAL_URL_PATTERN = re.compile(r"https?://\S+")
MAX_WEB_CHARS = 20000


def extract_pdf_from_url(url: str) -> str | None:
    """URL에서 PDF 다운로드 후 텍스트 추출"""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return extract_pdf_text(r.content)
    except Exception as e:
        logger.error(f"PDF URL extraction error: {e}")
        return None


def extract_web_text(url: str) -> str | None:
    """trafilatura로 웹페이지 본문 추출"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        if not text:
            return None
        return f"[Web Article]\nURL: {url}\n\n{text[:MAX_WEB_CHARS]}"
    except Exception as e:
        logger.error(f"Web extraction error: {e}")
        return None


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
    await update.message.reply_text(
        "🤖 LLM 봇 준비 완료!\n\n"
        "💬 일반 메시지 → LLM 대화\n"
        "🔍 /s 질문 → 웹 검색 + LLM 답변\n"
        "🗑️ /c → 대화 기록 초기화\n"
        "ℹ️ /m → 현재 모델 확인"
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