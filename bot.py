import os
import asyncio
import json
import logging
import requests
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
MAX_HISTORY = 20

SYSTEM_PROMPT = """You are a helpful assistant. Answer concisely and accurately.
When search results are provided, use them to give up-to-date answers and cite sources when relevant.
Respond in the same language the user uses."""

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


# ──────────────────────────────────────────────
# 대화 히스토리 준비
# ──────────────────────────────────────────────
def prepare_messages(user_id: int, user_message: str, search_context: str = "") -> list:
    if user_id not in conversations:
        conversations[user_id] = []

    if search_context:
        augmented_message = (
            f"[Web Search Results]\n{search_context}\n\n"
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

            display_text = strip_think(full_text)

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
        final_text = strip_think(full_text) if full_text else "⚠️ 빈 응답"

        if len(final_text) > 4000:
            await bot_msg.edit_text(final_text[:4000])
            for i in range(4000, len(final_text), 4000):
                await update.message.reply_text(final_text[i : i + 4000])
        else:
            await bot_msg.edit_text(final_text)

        conversations[user_id].append({"role": "assistant", "content": final_text})

    except Exception as e:
        logger.error(f"LLM error: {e}")
        await bot_msg.edit_text(f"⚠️ LM Studio 연결 실패: {e}")


# ──────────────────────────────────────────────
# Tavily 검색
# ──────────────────────────────────────────────
def search_web(query: str) -> str:
    try:
        results = tavily.search(
            query=query,
            max_results=5,
            search_depth="basic",
            include_answer=True,
        )

        output_parts = []

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
        "🗑️ /clear → 대화 기록 초기화\n"
        "ℹ️ /model → 현재 모델 확인"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ 접근 권한이 없습니다.")
        return

    user_text = update.message.text
    if not user_text:
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


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations.pop(user_id, None)
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
    app.add_handler(CommandHandler("clear", clear_history))
    app.add_handler(CommandHandler("model", show_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()