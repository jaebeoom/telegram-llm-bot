# telegram-llm-bot

OMLX 같은 OpenAI-compatible 로컬 LLM 서버와 연동하는 Telegram bot입니다. 웹 검색, X 포스트,
YouTube 스크립트, PDF, 일반 웹페이지를 컨텍스트로 주입해 답변할 수 있습니다.

## 주요 특징

- OMLX 등 OpenAI-compatible 로컬 모델과 스트리밍 대화
- LLM 스트리밍을 이벤트 루프와 분리해 Telegram bot 응답성을 유지
- Tavily 웹 검색 연동
- X(Twitter), YouTube, PDF, 웹페이지 컨텍스트 주입
- 3시간 비활성 시 자동 세션 정리
- 세션 초기화 시 Vault Markdown 로그 저장 옵션
- 스트리밍 중 드물게 섞일 수 있는 유니코드 대체 문자(`�`)를 Telegram 표시 전에 제거

## 주요 기능

- OMLX 등 OpenAI-compatible 로컬 모델과 스트리밍 대화
- Tavily 웹 검색 연동
- X(Twitter) 피드 텍스트 추출 (아티클/장문 포스트 포함, API 실패 시 HTML 메타데이터 fallback)
- 유튜브 스크립트 자동 추출
- PDF 텍스트 추출 (파일 전송 및 URL 모두 지원)
- 일반 웹페이지 본문 추출
- 3시간 비활성 시 자동 세션 초기화 (알림 포함)
- URL 추출 시 내부망 차단, 리다이렉트 제한, 다운로드 크기 제한

## 설치

```bash
conda create -n telegram-llm-bot python=3.11
conda activate telegram-llm-bot

pip install .

cp .env.example .env
```

`.env`에 필요한 토큰과 설정값을 채운 뒤 실행합니다. OMLX 서버가 OpenAI-compatible API로 서빙 중이어야 합니다.
프로젝트 상위 경로에 `shared_ai.env`, `.shared_ai.env`, `shared-ai.env`, `.shared-ai.env` 중 하나가 있으면 추가로 읽으며, 이 공용 파일의 값이 프로젝트 `.env`보다 우선합니다.
공개 저장소 기준으로 `ALLOWED_USER_IDS`는 꼭 설정해야 하며, 비워 두면 봇이 모든 요청을 거부하도록 되어 있습니다.
`VAULT_HAIKU_PATH`를 설정하면 대화 내용이 Markdown으로 저장되므로, 민감한 대화를 보관하지 않을 위치를 쓰거나 아예 비워 두는 편이 안전합니다.

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
| `OMLX_BASE_URL` / `OMLX_MODEL` / `OMLX_API_KEY` | 각각 `LLM_API_BASE_URL` / `MODEL_NAME` / `LLM_API_KEY`보다 우선하는 OMLX 전용 별칭 |
| `VAULT_HAIKU_PATH` | 세션 로그를 Markdown으로 저장할 Vault 경로, 선택 사항 |

기존 `LM_STUDIO_URL` 환경 변수도 하위 호환으로 계속 읽습니다.
`LLM_API_BASE_URL`와 `LM_STUDIO_URL`를 둘 다 설정했다면 `LLM_API_BASE_URL`가 우선합니다.
`/m` 명령은 현재 모델과 마지막으로 적용된 env 파일 이름을 함께 보여줍니다.
샘플링 관련 값(`temperature`, `max_tokens`, `top_p` 등)은 봇이 별도로 덮어쓰지 않고 OMLX 같은 서버 쪽 설정을 그대로 따릅니다.

## 프롬프트 프로필

시스템 프롬프트는 코드에 하드코딩하지 않고 아래 파일을 합쳐서 만듭니다.

- `prompts/base.md`: 모든 모델에 공통으로 적용할 기본 규칙
- `prompts/models/*.md`: 모델별 보정 규칙

모델별 파일은 현재 `MODEL_NAME` 또는 `OMLX_MODEL` 값과 비교해 자동 선택합니다. 파일명이나 frontmatter의 `id`, `match` 값이 모델명과 가장 잘 맞는 항목이 적용됩니다.
`prompts/base.md`에는 `{today}` 같은 템플릿 변수를 넣을 수 있고, 런타임에 실제 값으로 치환됩니다.

예시:

```md
---
id: gemma-4
match: gemma-4, gemma 4
---

Be a thoughtful collaborator, not a cheerleader.
...
```

새 모델을 튜닝하려면 `prompts/models/` 아래에 같은 형식의 `.md` 파일만 추가하면 됩니다.

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

## 보안 메모

- URL 기반 추출은 `http`와 `https`만 허용합니다.
- `localhost`, `.local`, 사설 IP, loopback, link-local, multicast, reserved 주소는 차단합니다.
- 리다이렉트는 최대 5회까지만 따라가며, 각 단계마다 다시 공개 주소인지 검증합니다.
- 원격 다운로드는 최대 15MB까지만 허용합니다.
- X URL은 일반 웹 추출 경로로 처리하지 않고, X 전용 추출기를 먼저 사용합니다.
- X API 응답이 비거나 브라우저 차단 문구만 나오는 경우에는 HTML 메타데이터 기반으로 한 번 더 fallback 합니다.
- `ALLOWED_USER_IDS`를 비워 두면 모든 요청을 거부합니다. 공개 서버에서는 반드시 설정해야 합니다.
- Vault 저장을 켜면 대화 원문이 파일로 남습니다. 운영 환경에서는 저장 경로 권한과 백업 정책을 함께 확인하는 것이 좋습니다.

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
- OMLX 서버에서 현재 `MODEL_NAME` 또는 `OMLX_MODEL`로 지정한 모델이 로드되어 있어야 함
- 상위 공용 env 파일을 수정한 뒤에는 이미 실행 중인 봇 프로세스를 재시작해야 새 설정이 반영됨
- 대화 기록은 메모리에만 저장 (봇 재시작 시 초기화)
- 검색 없이 일반 대화할 때는 Tavily API 소모 없음
- JavaScript 기반 SPA 페이지는 텍스트 추출이 안 될 수 있음
- OMLX 서버의 context length와 max token 설정을 충분히 잡아야 긴 컨텍스트 처리 가능
- OpenAI-compatible 서버가 간헐적으로 유니코드 대체 문자(`�`, U+FFFD)를 섞어 보내는 경우, 봇은 Telegram 표시 전에 해당 문자만 제거함

## 테스트

```bash
conda run -n telegram-llm-bot python -m pytest -q
```

현재 테스트에는 추출기 단위 테스트와 X URL/PDF 업로드 핸들러 회귀 테스트가 포함되어 있습니다.
