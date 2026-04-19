"""
Vault 태깅 모듈.
구조 태그, 출처 태그, 주제 태그를 한 곳에서 관리한다.
"""

import re

STRUCTURE_TAGS = ["#stage/capture", "#daily"]
MAX_TOPIC_TAGS = 3

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "#investment": [
        "투자", "주가", "시총", "매출", "이익률", "배당", "포트폴리오",
        "수익률", "매수", "매도", "주식", "채권", "공매도", "자사주",
        "증권사", "애널리스트", "리포트",
    ],
    "#investing/buffett": [
        "버핏", "buffett", "berkshire", "버크셔",
    ],
    "#investing/munger": [
        "멍거", "munger",
    ],
    "#investing/valuation": [
        "밸류에이션", "valuation", "적정가", "내재가치", "intrinsic value",
        "멀티플", "안전마진", "margin of safety",
    ],
    "#investing/value": [
        "가치투자", "value investing", "moat", "경제적 해자",
    ],
    "#tech/ai": [
        "에이전트", "agent", "프롬프트", "prompt",
        "딥러닝", "머신러닝", "트랜스포머", "transformer",
        "파인튜닝", "fine-tuning", "임베딩", "embedding",
        "langchain", "langgraph",
    ],
    "#tech/python": [
        "파이썬", "django", "fastapi", "flask", "pip install",
    ],
    "#cinema": [
        "영화", "감독", "촬영", "시퀀스", "cinema", "director",
        "letterboxd", "개봉", "극장",
    ],
    "#philosophy": [
        "철학", "philosophy", "비트겐슈타인", "wittgenstein",
        "로티", "rorty", "하이데거", "heidegger",
        "니체", "nietzsche", "가다머", "gadamer",
        "존재론", "인식론", "형이상학", "프래그머티즘",
    ],
    "#writing": [
        "에세이", "문체", "글쓰기", "essay", "퇴고", "산문",
    ],
}

_WORD_BOUNDARY_KEYWORDS: dict[str, str] = {
    "AI": "#tech/ai",
    "LLM": "#tech/ai",
    "GPT": "#tech/ai",
    "OMLX": "#tech/ai",
    "Claude": "#tech/ai",
    "OpenAI": "#tech/ai",
    "PER": "#investing/valuation",
    "PBR": "#investing/valuation",
    "EPS": "#investing/valuation",
    "DCF": "#investing/valuation",
    "EV/EBITDA": "#investing/valuation",
    "ROE": "#investing/valuation",
    "ETF": "#investment",
    "python": "#tech/python",
}

_WORD_RE = {
    kw: re.compile(rf"(?<![a-zA-Z0-9]){re.escape(kw)}(?![a-zA-Z0-9])")
    for kw in _WORD_BOUNDARY_KEYWORDS
}


def get_topic_tags(text: str) -> list[str]:
    """텍스트에서 주제 태그 추출. 최대 MAX_TOPIC_TAGS개."""
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for tag, keywords in TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in text_lower)
        if count > 0:
            scores[tag] = scores.get(tag, 0) + count

    for kw, tag in _WORD_BOUNDARY_KEYWORDS.items():
        if _WORD_RE[kw].search(text):
            scores[tag] = scores.get(tag, 0) + 1

    if not scores:
        return []

    sorted_tags = sorted(scores, key=lambda t: scores[t], reverse=True)

    result = []
    for tag in sorted_tags:
        if len(result) >= MAX_TOPIC_TAGS:
            break
        if tag == "#investment" and any(
            t.startswith("#investing/") for t in sorted_tags
        ):
            continue
        result.append(tag)

    return result


def generate_tags(history: list[dict], source: str = "telegram-bot") -> str:
    """대화 히스토리에서 전체 태그 문자열 생성."""
    tags = list(STRUCTURE_TAGS)
    tags.append(f"#from/{source}")

    full_text = " ".join(msg["content"] for msg in history)
    topic_tags = get_topic_tags(full_text)
    tags.extend(topic_tags)

    return " ".join(tags)
