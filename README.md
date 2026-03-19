# 텔레그램 LLM 봇

로컬 LLM(LM Studio)과 연동하는 텔레그램 봇. 웹 검색, X 피드, 유튜브 스크립트, PDF, 웹페이지 컨텍스트 주입을 지원합니다.

## 주요 기능

- LM Studio 로컬 모델과 스트리밍 대화
- Tavily 웹 검색 연동
- X(Twitter) 피드 텍스트 추출 (아티클/장문 포스트 포함)
- 유튜브 스크립트 자동 추출
- PDF 텍스트 추출 (파일 전송 및 URL 모두 지원)
- 일반 웹페이지 본문 추출
- 3시간 비활성 시 자동 세션 초기화 (알림 포함)

## 설치

```bash
cd telegram-llm-bot

# conda 환경 생성 (권장)
conda create -n telegram-llm-bot python=3.11
conda activate telegram-llm-bot

# 의존성 설치
pip install .

# .env 파일 생성
cp .env.example .env
# .env 파일 열어서 토큰/키 입력
```

## API 키 발급

### 텔레그램 봇 토큰
1. 텔레그램에서 @BotFather 검색
2. `/newbot` → 봇 이름 입력 → 유저네임 입력
3. 발급된 토큰 복사

### Tavily API 키
1. https://tavily.com 가입
2. Dashboard에서 API Key 복사
3. 무료 플랜: 월 1,000회 검색

## 실행

```bash
conda activate telegram-llm-bot
python bot.py
```

백그라운드 실행:
```bash
nohup python bot.py > bot.log 2>&1 &
```

종료:
```bash
kill $(pgrep -f bot.py)
```

## 사용법

| 명령어 | 설명 |
|--------|------|
| 일반 메시지 | LLM 대화 |
| `/s 질문` 또는 `질문 /s` | 웹 검색 + LLM 답변 |
| `/c` | 대화 기록 초기화 |
| `/m` | 현재 모델 확인 |

### 컨텍스트 주입

URL이나 파일을 보내면 자동으로 텍스트를 추출하고 요약합니다.

| 입력 | 동작 |
|------|------|
| X(Twitter) URL | 피드 텍스트 추출 (아티클 포함) |
| 유튜브 URL | 스크립트(자막) 추출 |
| `.pdf` URL | PDF 다운로드 후 텍스트 추출 |
| 일반 웹 URL | 본문 텍스트 추출 |
| PDF 파일 전송 | 텍스트 추출 |

URL과 함께 질문을 보내면 해당 컨텍스트 기반으로 답변합니다.

## 설정 (bot.py)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MAX_HISTORY` | 10 | 대화 턴 수 제한 (유저+봇 합산) |
| `INACTIVITY_TIMEOUT` | 3시간 | 비활성 시 자동 초기화 |
| `MAX_PDF_CHARS` | 20000 | PDF 추출 글자 수 제한 |
| `MAX_TRANSCRIPT_CHARS` | 20000 | 스크립트 추출 글자 수 제한 |
| `MAX_WEB_CHARS` | 20000 | 웹페이지 추출 글자 수 제한 |

## 참고

- LM Studio에서 모델을 바꿔 로드하면 봇도 자동으로 새 모델 사용
- 대화 기록은 메모리에만 저장 (봇 재시작 시 초기화)
- 검색 없이 일반 대화할 때는 Tavily API 소모 없음
- JavaScript 기반 SPA 페이지는 텍스트 추출이 안 될 수 있음
- LM Studio Context Length를 충분히 설정해야 긴 컨텍스트 처리 가능
