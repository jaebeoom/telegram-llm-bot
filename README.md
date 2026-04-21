# telegram-llm-bot

OMLX 같은 OpenAI-compatible 로컬 LLM 서버와 연동하는 Telegram bot입니다. 웹 검색, X 포스트,
YouTube 스크립트, PDF, 일반 웹페이지를 컨텍스트로 주입해 답변할 수 있습니다.

## 주요 특징

- OMLX 등 OpenAI-compatible 로컬 모델과 대화
- LLM 스트리밍을 이벤트 루프와 분리해 Telegram bot 응답성을 유지
- Telegram 전송 방식 선택 지원: 최종 답변 새 메시지 전송, `sendMessageDraft`, 또는 `edit_text`
- 스트리밍 완료 전까지 Telegram `typing` 액션 유지
- Tavily 웹 검색 연동
- 최신성 민감 질문은 짧은 LLM 판정 후 Tavily 검색으로 자동 보강
- X(Twitter), YouTube, PDF, 웹페이지 컨텍스트 주입
- 세션 초기화 시 Vault Markdown 로그 저장 옵션
- Vault Capture 저장 시 `## AI 세션 (...)` 헤더 바로 아래에 stable session id HTML comment marker 기록
- URL 컨텍스트(X, YouTube, PDF, 웹페이지)를 읽은 세션은 관련 `**나**` 블록 아래에 `<!-- source: ... -->` 주석 기록
- Inbox bot 컨텍스트 큐에서 오래된 ready 소스를 `/ctx`로 가져와 현재 세션에 적용
- 같은 사용자의 다른 chat/topic 세션이 메모리에서 섞이지 않도록 chat/thread 단위로 세션 분리
- 오래된 scope별 세션은 비활성 시간 기준으로 메모리에서만 정리 가능
- Telegram 표시 전에 깨진 유니코드 문자(`�`), LaTeX 화살표 표기, Markdown 표를 읽기 쉬운 평문으로 정규화
- `bot.log`에 컨텍스트 추출/검색/스트리밍 지연 메트릭 기록

## 주요 기능

- OMLX 등 OpenAI-compatible 로컬 모델과 스트리밍 대화
- Tavily 웹 검색 연동
- X(Twitter) 피드 텍스트 추출 (아티클/장문 포스트 포함, API 실패 시 HTML 메타데이터 fallback)
- 유튜브 스크립트 자동 추출
- PDF 텍스트 추출 (파일 전송 및 URL 모두 지원)
- 일반 웹페이지 본문 추출 (iframe, readability, Playwright 렌더 fallback 포함)
- URL 추출 시 내부망 차단, 리다이렉트 제한, 다운로드 크기 제한

## 설치

먼저 `uv`를 설치합니다. 예를 들어 macOS + Homebrew라면 `brew install uv`를 사용할 수 있습니다.

```bash
uv python pin 3.11
uv sync --frozen
uv run playwright install chromium

cp .env.example .env
```

`uv sync --frozen`은 프로젝트용 `.venv`를 자동으로 만들고, 커밋된 `uv.lock` 기준으로 의존성을 맞춥니다.
Python 3.11을 명시적으로 고정하려면 `uv python pin 3.11`을 사용하면 됩니다.

`.env`에 필요한 토큰과 설정값을 채운 뒤 실행합니다. OMLX 서버가 OpenAI-compatible API로 서빙 중이어야 합니다.
프로젝트 상위 경로에 `shared_ai.env`, `.shared_ai.env`, `shared-ai.env`, `.shared-ai.env` 중 하나가 있으면 추가로 읽으며, 이 공용 파일의 값이 프로젝트 `.env`보다 우선합니다.
시작 시 `TELEGRAM_TOKEN`과 `MODEL_NAME` 또는 `OMLX_MODEL`이 비어 있으면 봇은 바로 종료합니다.
공개 저장소 기준으로 `ALLOWED_USER_IDS`는 꼭 설정해야 하며, 비워 두면 봇이 모든 요청을 거부하도록 되어 있습니다.
`VAULT_CAPTURE_PATH`를 설정하면 대화 내용이 Markdown으로 저장되므로, 민감한 대화를 보관하지 않을 위치를 쓰거나 아예 비워 두는 편이 안전합니다.
현재 운영 기준 Python 타깃은 `3.11`이며, `readability-lxml`이 끌어오는 최신 `chardet`와 `requests` 조합 경고를 피하려고 `chardet<6`을 함께 고정합니다.

예시:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
TAVILY_API_KEY=your_tavily_api_key
ENABLE_AUTO_SEARCH=true
TELEGRAM_RESPONSE_DELIVERY=final
ENABLE_THINKING_FOR_CONTEXT=true
LLM_API_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=omlx
LLM_PROVIDER_NAME=OMLX
ALLOWED_USER_IDS=your_telegram_user_id
MODEL_NAME=your_loaded_local_model_name
VAULT_CAPTURE_PATH=~/path/to/your/Vault/Capture
INBOX_API_BASE_URL=http://localhost:8000
INBOX_API_ACCESS_TOKEN=your_inbox_access_token
INBOX_CONTEXT_SUMMARY_TIMEOUT_SECONDS=45
ENABLE_RESPONSE_VALIDATION=true
RESPONSE_REWRITE_TIMEOUT_SECONDS=45
RESPONSE_REWRITE_MAX_ATTEMPTS=3
```

## 환경 변수

| 변수 | 설명 |
|------|------|
| `TELEGRAM_TOKEN` | BotFather에서 발급받은 Telegram bot token. 필수 |
| `TAVILY_API_KEY` | Tavily search API key. `/s` 검색 기능을 쓸 때 필요 |
| `ENABLE_AUTO_SEARCH` | 기본값 `true`. 일반 메시지가 최신 정보가 필요한지 짧은 LLM 판정으로 확인하고 필요하면 Tavily 검색을 자동 주입 |
| `AUTO_SEARCH_CLASSIFIER_TIMEOUT_SECONDS` | 기본값 `12`. 자동 검색 필요 여부를 판정하는 비스트리밍 LLM 호출 제한 시간 |
| `TELEGRAM_RESPONSE_DELIVERY` | 기본값 `final`. `final`은 최종 답변을 새 메시지로 전송, `draft`는 `sendMessageDraft`, `edit`은 단일 메시지 `edit_text` 스트리밍 사용 |
| `LLM_API_BASE_URL` | OMLX 등 OpenAI-compatible LLM 서버의 base URL. OMLX 기본 포트는 `8000`, 현재 예시는 `http://127.0.0.1:8001/v1` |
| `LLM_API_KEY` | LLM 서버 인증 키. OMLX에서 auth가 켜져 있으면 반드시 필요 |
| `LLM_PROVIDER_NAME` | 봇 메시지에 표시할 LLM 제공자 이름, 기본값 `OMLX` |
| `ALLOWED_USER_IDS` | 허용할 Telegram user id 목록, 쉼표 구분 |
| `MODEL_NAME` | 사용할 로컬 모델 이름. `OMLX_MODEL`이 없으면 필수 |
| `OMLX_BASE_URL` / `OMLX_MODEL` / `OMLX_API_KEY` | 각각 `LLM_API_BASE_URL` / `MODEL_NAME` / `LLM_API_KEY`보다 우선하는 OMLX 전용 별칭 |
| `VAULT_CAPTURE_PATH` | 세션 로그를 Markdown으로 저장할 Vault 경로, 선택 사항. 기존 `VAULT_HAIKU_PATH`도 하위 호환 별칭으로 읽음 |
| `INBOX_API_BASE_URL` | 기본값 `http://localhost:8000`. `/ctx`가 context source를 가져올 `telegram-inbox-bot` API base URL |
| `INBOX_API_ACCESS_TOKEN` | Inbox bot의 `ACCESS_TOKEN`이 설정되어 있을 때 같은 값을 넣음. 비어 있으면 Authorization 헤더 없이 호출 |
| `INBOX_API_TIMEOUT_SECONDS` | 기본값 `10`. Inbox context source API 호출 제한 시간 |
| `INBOX_CONTEXT_SUMMARY_TIMEOUT_SECONDS` | 기본값 `45`. 하위 호환용 `/ctx` 짧은 요약 helper 제한 시간. 현재 기본 `/ctx` 응답은 직접 URL과 같은 streaming context 경로 사용 |
| `INBOX_CONTEXT_SUMMARY_MAX_INPUT_CHARS` | 기본값 `12000`. 하위 호환용 `/ctx` 짧은 요약 helper 입력 제한 |
| `INBOX_CONTEXT_PREVIEW_CHARS` | 기본값 `700`. `/ctx` preview helper가 본문 미리보기를 만들 때의 길이 제한 |
| `ENABLE_THINKING_FOR_CONTEXT` | 기본값 `true`. URL/PDF/웹검색/Inbox 컨텍스트를 주입한 답변도 서버 기본 reasoning을 유지함 |
| `ENABLE_TELEGRAM_DRAFT_STREAMING` | 하위 호환 변수. `TELEGRAM_RESPONSE_DELIVERY`가 없을 때만 읽으며, `true`는 `draft`, `false`는 `edit`로 해석 |
| `DISABLE_THINKING_FOR_CONTEXT` | 하위 호환 변수. `ENABLE_THINKING_FOR_CONTEXT`가 없을 때만 읽음 |
| `ENABLE_RESPONSE_VALIDATION` | 기본값 `true`. 최종 LLM 답변에 중국어/일본어 문자가 포함되면 전송 직전에 한국어 번역 pass를 수행 |
| `RESPONSE_REWRITE_TIMEOUT_SECONDS` | 기본값 `45`. 응답 번역 pass 1회당 제한 시간. 변수 이름은 하위 호환을 위해 유지 |
| `RESPONSE_REWRITE_MAX_ATTEMPTS` | 기본값 `3`. 한국어 번역 pass 최대 시도 횟수 |
| `ENABLE_PLAYWRIGHT_FALLBACK` | 기본값 `true`. 일반 HTML 추출이 실패하면 headless Chromium으로 렌더된 DOM을 다시 읽음 |
| `PLAYWRIGHT_RENDER_TIMEOUT_MS` | 기본값 `12000`. Playwright 렌더링 최대 대기 시간(ms) |
| `PLAYWRIGHT_IDLE_TIMEOUT_SECONDS` | 기본값 `600`. fallback 브라우저를 재사용하다가 유휴 상태로 둘 최대 시간(초) |
| `PLAYWRIGHT_MAX_QUEUE_SIZE` | 기본값 `4`. Playwright fallback 대기열 최대 길이 |
| `PLAYWRIGHT_QUEUE_WAIT_TIMEOUT_MS` | 기본값 `1000`. fallback 대기열이 가득 찼을 때 빈 자리를 기다릴 최대 시간(ms) |
| `SESSION_INACTIVE_TTL_SECONDS` | 기본값 `86400`(24시간). scope별 세션이 이 시간 이상 비활성 상태면 메모리에서만 제거. `0` 또는 빈 값이면 비활성화 |
| `ENABLE_YOUTUBE_AUDIO_TRANSCRIPTION` | 기본값 `false`. YouTube 공개 자막이 없을 때 MLX Whisper 오디오 전사 fallback 자동 사용 |
| `YOUTUBE_AUDIO_TRANSCRIPTION_MAX_SECONDS` | 기본값 `28800`(8시간). 오디오 전사를 허용할 최대 영상 길이. `0`이면 길이 제한 없음 |
| `YOUTUBE_AUDIO_TRANSCRIPTION_MAX_CONCURRENT` | 기본값 `1`. 동시에 허용할 YouTube 오디오 전사 수 |
| `YOUTUBE_AUDIO_CHUNK_SECONDS` | 기본값 `1800`(30분). 긴 오디오를 나누어 전사할 chunk 길이 |
| `YOUTUBE_AUDIO_DOWNLOAD_TIMEOUT_SECONDS` | 기본값 `900`. YouTube 오디오 다운로드/분할 제한 시간 |
| `YOUTUBE_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS` | 기본값 `0`. 전체 전사 제한 시간. `0`이면 제한 없음 |
| `YOUTUBE_AUDIO_TRANSCRIPTION_IDLE_TIMEOUT_SECONDS` | 기본값 `900`. 전사 worker가 chunk 완료나 최종 결과를 이 시간 동안 출력하지 않으면 멈춘 것으로 보고 종료 후 실패 메시지를 보냄. `0`이면 비활성화 |
| `YOUTUBE_AUDIO_CACHE_DIR` | 기본값 `.cache/youtube-audio`. 임시 오디오/chunk 저장 위치 |
| `YOUTUBE_TRANSCRIPT_CACHE_DIR` | 기본값 `.cache/youtube-transcripts`. 전사 텍스트/cache 저장 위치 |
| `YOUTUBE_AUDIO_KEEP_FILES` | 기본값 `false`. 전사 완료 후 오디오 파일을 보관할지 여부 |

기존 `LM_STUDIO_URL` 환경 변수도 하위 호환으로 계속 읽습니다.
`LLM_API_BASE_URL`와 `LM_STUDIO_URL`를 둘 다 설정했다면 `LLM_API_BASE_URL`가 우선합니다.
`/m` 명령은 현재 모델과 마지막으로 적용된 env 파일 이름을 함께 보여줍니다.
샘플링 관련 값(`temperature`, `max_tokens`, `top_p` 등)은 봇이 별도로 덮어쓰지 않고 OMLX 같은 서버 쪽 설정을 그대로 따릅니다.
`TELEGRAM_RESPONSE_DELIVERY=final`은 최종 답변을 새 메시지로 보내며, reasoning token이 감지되면 임시 `🧠 추론 중...` 메시지를 보냈다가 최종 답변 전송 후 삭제합니다. 내부 LLM 응답은 계속 스트리밍으로 읽어 timeout, usage, reasoning 메트릭을 유지합니다.
모바일에서 네이티브 draft 스트리밍을 다시 쓰고 싶다면 `TELEGRAM_RESPONSE_DELIVERY=draft`, 단일 메시지 편집 스트리밍을 쓰고 싶다면 `TELEGRAM_RESPONSE_DELIVERY=edit`로 바꿀 수 있습니다.
컨텍스트 주입 답변 속도를 우선해 reasoning을 끄고 싶다면 `ENABLE_THINKING_FOR_CONTEXT=false`로 바꿀 수 있습니다.
자동 검색은 명시적인 최신성 신호가 있으면 바로 검색하고, 그 외 일반 메시지는 로컬 LLM에 짧은 분류 요청을 보내 `needs_search=true`일 때만 검색합니다. `TAVILY_API_KEY`가 없으면 자동 검색도 비활성화됩니다.
Playwright fallback을 쓰려면 Python 패키지 설치 외에 `uv run playwright install chromium`도 한 번 실행해야 합니다.
YouTube 오디오 전사 fallback을 쓰려면 시스템에 `ffmpeg`와 `ffprobe`가 있어야 합니다. macOS에서는 `brew install ffmpeg`로 함께 설치됩니다.
`TAVILY_API_KEY`가 없으면 일반 대화는 동작하지만 `/s` 검색은 실패합니다.
`SESSION_INACTIVE_TTL_SECONDS`에 걸린 scope는 Vault에 자동 저장하지 않고, 메모리에서만 정리됩니다.

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
uv run python bot.py
```

백그라운드 실행:
```bash
nohup uv run python bot.py > nohup.out 2>&1 &
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
| `/ctx` 또는 `/ctx latest` | Inbox bot 컨텍스트 큐에서 가장 오래된 ready 소스 1개를 현재 세션에 적용하고 직접 URL과 같은 context 요약 경로로 답변한 뒤 consumed 처리 |
| `/e URL` 또는 `URL /e` | LLM 호출 없이 URL/X/YouTube/PDF/웹페이지 원문 추출. `/extract`, `/raw`도 지원. PDF 파일은 캡션에 `/e` 입력 |
| `/c` | 대화 기록 초기화, Vault 설정 시 로그 저장 |
| `/m` | 현재 모델 확인 |

### 자동 검색

`ENABLE_AUTO_SEARCH=true`이면 일반 메시지도 최신 정보가 필요한지 먼저 짧게 분류합니다.
명시적인 최신성 신호(오늘, 현재, 최신, 출시, 실적, 주가 등)가 있으면 바로 검색하고,
그 외에는 로컬 LLM에 비스트리밍 분류 요청을 보내 `needs_search=true`일 때만 Tavily 검색을 붙입니다.
Inbox 활성 컨텍스트가 있는 후속 질문도 자동 검색 분류를 거칩니다. `방금 넣은 자료 요약`처럼 명백히 source-local인 요청은 검색하지 않고, 현재 반응/최신 상태/시장 데이터처럼 외부 최신성이 필요한 요청은 검색 결과와 활성 source chunk를 함께 주입합니다.

자동 검색은 `/s`를 대체하는 편의 기능입니다. 반드시 검색을 붙이고 싶으면 기존처럼 `/s 질문` 또는 `질문 /s`를 쓰면 됩니다.
자동 검색을 끄려면 `ENABLE_AUTO_SEARCH=false`로 설정합니다.

보안상 자동 검색은 사용자의 원문 또는 분류기가 만든 짧은 검색 쿼리를 Tavily로 보냅니다.
민감한 내부 정보, 계정 정보, 비공개 투자 메모처럼 외부 검색 API로 나가면 안 되는 내용은 `/s`나 자동 검색 없이 URL/문서 컨텍스트 또는 일반 대화로 다루는 편이 안전합니다.
`TAVILY_API_KEY`가 없으면 자동 검색과 `/s` 검색은 실패하거나 비활성화되고, 일반 로컬 대화는 계속 동작합니다.

### 컨텍스트 주입

URL이나 파일을 보내면 자동으로 텍스트를 추출하고 요약합니다.

| 입력 | 동작 |
|------|------|
| X(Twitter) URL | 피드 텍스트 추출 (아티클 포함) |
| 유튜브 URL | 스크립트(자막) 추출. 공개 자막이 없고 오디오 전사 fallback이 켜져 있으면 로컬 MLX Whisper 전사 자동 실행 |
| `.pdf` URL | PDF 다운로드 후 텍스트 추출 |
| 일반 웹 URL | 본문 텍스트 추출 |
| PDF 파일 전송 | 텍스트 추출 |

URL과 함께 질문을 보내면 해당 컨텍스트 기반으로 답변합니다.
YouTube 오디오 전사는 `mlx-community/whisper-large-v3-turbo` 모델을 별도 subprocess에서 실행합니다. 평소에는 Whisper 모델을 로드하지 않고, 공개 자막 추출이 실패해 fallback이 필요할 때 자동으로 실행합니다.

### Inbox 컨텍스트 큐

`telegram-inbox-bot`에 `/ctx URL` 또는 `URL /ctx`로 넣은 소스는 일반 노트가 아니라 FIFO 컨텍스트 큐에 적재됩니다. 이 봇에서 `/ctx`를 입력하면 Inbox API의 오래된 ready 소스 1개를 가져와 현재 Telegram 세션 source memory에 붙이고, 세션 적용이 성공한 뒤 Inbox 쪽 source를 consumed 처리합니다. 적용 상태에는 title, URL, 본문 길이, 남은 큐 수가 표시되고, 실제 요약 답변은 직접 URL을 보냈을 때와 같은 streaming context 경로로 생성됩니다.

Inbox YouTube source는 LLM bot에서 같은 URL을 다시 추출해 공개 자막 결과로 갱신합니다. 공개 자막 요청이 YouTube의 `request_blocked` 등으로 실패하고 오디오 전사 fallback이 켜져 있으면, 직접 URL 주입과 같은 자동 전사 경로로 inbox의 짧은 추출본 대신 전사 전문을 적용합니다. 전사도 실패하면 실패 사유를 보낸 뒤 Inbox에 저장된 기존 컨텍스트를 사용합니다.

첫 구현은 임베딩 없이 큐 방식으로 동작합니다. 가져온 소스는 `/c`로 세션을 종료하기 전까지 현재 LLM 세션의 활성 컨텍스트로 남습니다. 활성 컨텍스트가 있을 때도 일반 후속 질문은 자동 검색 분류를 거치며, 검색이 필요한 경우 검색 결과와 source memory에서 고른 관련 chunk를 함께 사용합니다. 명시적으로 검색하려면 `/s`를 사용할 수 있습니다.

### Vault 저장 형식

`VAULT_CAPTURE_PATH`가 설정되어 있으면 `/c` 시점에 일간 Markdown 파일에 세션이 append됩니다.
기존 `## AI 세션 (HH:MM, model)` 헤더는 유지하고, 새로 저장되는 세션에는 바로 아래 줄에 stable marker를 함께 기록합니다.

```md
## AI 세션 (05:43, your_loaded_local_model_name)
<!-- capture:session-id=tg:123456789:987654 -->

**나** (YouTube 첨부): 핵심만 요약해줘.
<!-- source: https://www.youtube.com/watch?v=abcdefghijk -->
```

YouTube 공유 링크는 Vault에 남길 때 canonical `watch?v=...` 형태로 정규화됩니다.
일반 웹/PDF source URL은 query string과 fragment를 저장하지 않아 추적 파라미터나 임시 토큰이 Vault에 남지 않도록 합니다.
이 source 메타데이터는 Vault 저장용으로만 유지되며, LLM API로 보내는 대화 메시지에는 포함하지 않습니다.
기존 marker 없는 파일도 계속 유효합니다.
같은 사용자가 다른 chat/topic에서 대화하면 upstream 메모리 세션을 분리해 저장합니다.
비활성 시간 정리로 메모리에서 제거된 scope는 나중에 다시 대화하면 새 in-memory 세션으로 시작합니다.

## 보안 메모

- URL 기반 추출은 `http`와 `https`만 허용합니다.
- `localhost`, `.local`, 사설 IP, loopback, link-local, multicast, reserved 주소는 차단합니다.
- 리다이렉트는 최대 5회까지만 따라가며, 각 단계마다 다시 공개 주소인지 검증합니다.
- 원격 다운로드는 최대 15MB까지만 허용합니다.
- Playwright 렌더 fallback도 같은 공개 URL 정책을 따르며, private/local 서브리소스 요청은 차단합니다.
- X URL은 일반 웹 추출 경로로 처리하지 않고, X 전용 추출기를 먼저 사용합니다.
- X API 응답이 비거나 브라우저 차단 문구만 나오는 경우에는 HTML 메타데이터 기반으로 한 번 더 fallback 합니다.
- Vault에 기록하는 `<!-- source: ... -->` URL은 query string, fragment, embedded credential을 저장하지 않습니다.
- `ALLOWED_USER_IDS`를 비워 두면 모든 요청을 거부합니다. 공개 서버에서는 반드시 설정해야 합니다.
- Vault 저장을 켜면 대화 원문이 파일로 남습니다. 운영 환경에서는 저장 경로 권한과 백업 정책을 함께 확인하는 것이 좋습니다.

## 런타임 설정 (bot.py)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MAX_HISTORY_PAIRS` | 10 | 최근 대화 10쌍 유지. 진행 중인 현재 user 메시지 1개는 추가 허용 |
| `MAX_ASSISTANT_CONTEXT_CHARS` | 700 | LLM payload에 넣는 오래된 assistant 답변 최대 길이. 저장용 세션 로그는 원문 유지 |
| `MAX_RECENT_ASSISTANT_CONTEXT_CHARS` | 1200 | LLM payload에 넣는 가장 최근 assistant 답변 최대 길이 |
| `MAX_PDF_CHARS` | 120000 | PDF 추출 글자 수 제한 |
| `MAX_TRANSCRIPT_CHARS` | 120000 | 스크립트 추출 글자 수 제한 |
| `MAX_WEB_CHARS` | 120000 | 웹페이지 추출 글자 수 제한 |
| `SOURCE_DIRECT_CONTEXT_CHARS` | 20000 | 이 길이 이하의 URL/PDF/YouTube 컨텍스트는 원문을 직접 LLM에 주입 |
| `SOURCE_RETRIEVAL_CONTEXT_CHARS` | 20000 | 긴 source에서 질문별로 LLM에 넣을 검색 chunk 총량 |
| `SOURCE_CHUNK_CHARS` | 4500 | 긴 source를 세션 메모리에 저장할 때의 chunk 크기 |

## 참고

- OMLX 기본 포트는 `8000`이지만, 커스텀 포트를 쓰는 경우 `LLM_API_BASE_URL`도 같이 맞춰야 함
- OMLX 인증이 켜져 있으면 `~/.omlx/settings.json`의 `auth.api_key` 값을 `LLM_API_KEY`에 넣어야 함
- OMLX 서버에서 현재 `MODEL_NAME` 또는 `OMLX_MODEL`로 지정한 모델이 로드되어 있어야 함
- 상위 공용 env 파일을 수정한 뒤에는 이미 실행 중인 봇 프로세스를 재시작해야 새 설정이 반영됨
- 대화 기록은 메모리에만 저장 (봇 재시작 시 초기화)
- LLM 컨텍스트에는 최근 대화 10쌍만 유지하고 assistant 답변은 길이 제한으로 일부 생략하지만, `/c`로 Vault 저장할 때는 현재 세션 누적 대화 전체를 기록
- `/ctx` 첫 요약은 직접 URL과 동일하게 전체 컨텍스트를 LLM에 넣고, 후속 질문에서는 활성 source memory에서 관련 chunk를 골라 붙임
- 검색 없이 일반 대화할 때는 Tavily API 소모 없음
- Naver Blog 같은 iframe 기반 페이지는 내부 본문 프레임을 따라가 추출함
- 일반 HTML 추출이 실패하면 readability로 한 번 더 본문을 정리함
- 그래도 실패하면 Playwright가 headless Chromium을 lazy init으로 띄워 렌더된 DOM을 다시 추출함
- fallback 브라우저는 트래픽이 적으면 요청 시에만 실행되고, 재사용 중이더라도 기본 10분 유휴 후 자동 종료됨
- fallback 브라우저는 대기열 길이와 대기 시간을 제한해, 요청이 몰리면 렌더 fallback을 건너뛸 수 있음
- 일부 JavaScript 기반 SPA 페이지는 텍스트 추출이 안 될 수 있음
- OMLX 서버의 context length와 max token 설정을 충분히 잡아야 긴 컨텍스트 처리 가능
- OpenAI-compatible 서버가 간헐적으로 유니코드 대체 문자(`�`, U+FFFD)를 섞어 보내는 경우, 봇은 Telegram 표시 전에 제거함
- Telegram이 렌더링하지 못하는 LaTeX 화살표 표기(예: `$\rightarrow$`)는 `→` 같은 평문 기호로 치환함
- Markdown 표는 Telegram에서 가독성이 떨어지므로, 봇은 `A vs B` 비교 리스트 형태의 평문으로 바꿔서 표시함

## 테스트

```bash
uv run pytest -q
```

현재 테스트에는 추출기 단위 테스트와 X URL/PDF 업로드 핸들러 회귀 테스트가 포함되어 있습니다.
실제 구현 코드는 `src/telegram_llm_bot/` 아래에 있고, 프로젝트 루트의 `extractors.py`, `prompt_profiles.py`, `tagger.py` 같은 파일은 기존 import 호환을 위한 thin wrapper입니다.

## 운영 로그

- 애플리케이션 로그와 지연 메트릭은 프로젝트 루트의 `bot.log`에 저장됩니다.
- `Stage metrics` 로그는 YouTube/PDF/웹/X 추출 및 검색 단계의 소요 시간을 기록합니다.
- 웹 추출 `Stage metrics` 로그에는 `method=static_trafilatura`, `static_readability`, `rendered_trafilatura` 같은 실제 추출 경로가 함께 남습니다.
- `Stream metrics` 로그는 LLM 첫 토큰 시간, Telegram 첫 표시 시간, 총 스트리밍 시간, 업데이트 횟수를 기록합니다.
- reasoning이 활성화된 모델에서는 `Stream metrics` 로그에 reasoning 사용 여부와 reasoning 글자 수가 함께 기록됩니다. 서버가 usage에 reasoning token을 넣어 주는 경우 해당 값도 같이 남깁니다.
- 보안을 위해 Telegram Bot API 요청 URL 같은 저수준 HTTP 로그는 파일에 남기지 않도록 낮은 로그 레벨로 제한합니다.
