# telegram-llm-bot

OMLX 같은 OpenAI-compatible 로컬 LLM 서버와 연동하는 Telegram bot입니다. 웹 검색, X 포스트,
YouTube 스크립트, PDF, 일반 웹페이지를 컨텍스트로 주입해 답변할 수 있습니다.

## 주요 특징

- OMLX 등 OpenAI-compatible 로컬 모델과 스트리밍 대화
- Tavily 웹 검색 연동
- X(Twitter), YouTube, PDF, 웹페이지 컨텍스트 주입
- 3시간 비활성 시 자동 세션 정리
- 세션 초기화 시 Vault Markdown 로그 저장 옵션

## 주요 기능

- OMLX 등 OpenAI-compatible 로컬 모델과 스트리밍 대화
- Tavily 웹 검색 연동
- X(Twitter) 피드 텍스트 추출 (아티클/장문 포스트 포함)
- 유튜브 스크립트 자동 추출
- PDF 텍스트 추출 (파일 전송 및 URL 모두 지원)
- 일반 웹페이지 본문 추출
- 3시간 비활성 시 자동 세션 초기화 (알림 포함)

## 설치

```bash
conda create -n telegram-llm-bot python=3.11
conda activate telegram-llm-bot

pip install .

cp .env.example .env
```

`.env`에 필요한 토큰과 설정값을 채운 뒤 실행합니다. OMLX 서버가 OpenAI-compatible API로 서빙 중이어야 합니다.
공개 저장소 기준으로 `ALLOWED_USER_IDS`는 꼭 설정해야 하며, 비워 두면 봇이 모든 요청을 거부하도록 되어 있습니다.

예시:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
TAVILY_API_KEY=your_tavily_api_key
LLM_API_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=omlx
LLM_PROVIDER_NAME=OMLX
ALLOWED_USER_IDS=your_telegram_user_id
MODEL_NAME=Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
VAULT_HAIKU_PATH=~/path/to/your/Vault/Haiku
```

## 환경 변수

| 변수 | 설명 |
|------|------|
| `TELEGRAM_TOKEN` | BotFather에서 발급받은 Telegram bot token |
| `TAVILY_API_KEY` | Tavily search API key |
| `LLM_API_BASE_URL` | OMLX 등 OpenAI-compatible LLM 서버의 base URL. OMLX 기본 포트는 `8000`, 현재 예시는 `http://127.0.0.1:8001/v1` |
| `LLM_API_KEY` | LLM 서버 인증 키. OMLX에서 auth가 켜져 있으면 반드시 필요 |
| `LLM_PROVIDER_NAME` | 봇 메시지에 표시할 LLM 제공자 이름, 기본값 `OMLX` |
| `ALLOWED_USER_IDS` | 허용할 Telegram user id 목록, 쉼표 구분 |
| `MODEL_NAME` | 사용할 로컬 모델 이름 |
| `VAULT_HAIKU_PATH` | 세션 로그를 Markdown으로 저장할 Vault 경로, 선택 사항 |

기존 `LM_STUDIO_URL` 환경 변수도 하위 호환으로 계속 읽습니다.
`LLM_API_BASE_URL`와 `LM_STUDIO_URL`를 둘 다 설정했다면 `LLM_API_BASE_URL`가 우선합니다.

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

또는:

```bash
conda run -n telegram-llm-bot python bot.py
```

백그라운드 실행:
```bash
nohup conda run -n telegram-llm-bot python bot.py > bot.log 2>&1 &
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
| `/c` | 대화 기록 초기화, Vault 설정 시 로그 저장 |
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

## 런타임 설정 (bot.py)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MAX_HISTORY` | 10 | 대화 턴 수 제한 (유저+봇 합산) |
| `INACTIVITY_TIMEOUT` | 3시간 | 비활성 시 자동 초기화 |
| `MAX_PDF_CHARS` | 20000 | PDF 추출 글자 수 제한 |
| `MAX_TRANSCRIPT_CHARS` | 20000 | 스크립트 추출 글자 수 제한 |
| `MAX_WEB_CHARS` | 20000 | 웹페이지 추출 글자 수 제한 |

## 참고

- OMLX 기본 포트는 `8000`이지만, 커스텀 포트를 쓰는 경우 `LLM_API_BASE_URL`도 같이 맞춰야 함
- OMLX 인증이 켜져 있으면 `~/.omlx/settings.json`의 `auth.api_key` 값을 `LLM_API_KEY`에 넣어야 함
- OMLX 서버에서 현재 `MODEL_NAME`으로 지정한 모델이 로드되어 있어야 함
- 대화 기록은 메모리에만 저장 (봇 재시작 시 초기화)
- 검색 없이 일반 대화할 때는 Tavily API 소모 없음
- JavaScript 기반 SPA 페이지는 텍스트 추출이 안 될 수 있음
- OMLX 서버의 context length와 max token 설정을 충분히 잡아야 긴 컨텍스트 처리 가능
- 보안을 위해 URL 추출은 `localhost`, 사설 IP, `.local` 같은 내부 네트워크 주소를 차단함
