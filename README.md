# 텔레그램 LLM 봇 셋업 가이드

## 1. API 키 발급

### 텔레그램 봇 토큰
1. 텔레그램에서 @BotFather 검색
2. `/newbot` → 봇 이름 입력 → 유저네임 입력
3. 발급된 토큰 복사

### Tavily API 키
1. https://tavily.com 가입 (GitHub 로그인 가능)
2. Dashboard에서 API Key 복사
3. 무료 플랜: 월 1,000회 검색

## 2. 설치 및 실행

```bash
cd telegram-llm-bot
pip install .

# .env 파일 생성 (예제 복사 후 수정)
cp .env.example .env
# .env 파일 열어서 토큰/키 입력

# 실행
python bot.py
```

## 3. 사용법

| 명령어 | 설명 |
|--------|------|
| 일반 메시지 | LLM 대화 |
| `/s 질문` | 웹 검색 + LLM 답변 |
| `/clear` | 대화 기록 초기화 |
| `/model` | 현재 로드된 모델 확인 |

## 4. 외부에서 접속

봇은 텔레그램 서버가 중계하므로 포트포워딩/VPN 불필요.
단, **LM Studio 서버가 켜져 있어야** 함.

### Mac Mini 상시 구동 팁
- LM Studio 서버 켜둔 상태 유지
- 봇 스크립트를 백그라운드로 실행:
  ```bash
  nohup python bot.py > bot.log 2>&1 &
  ``` 
  끌 때는
  ```bash
  kill $(pgrep -f bot.py)
  ``` 
- 또는 launchd로 자동 시작 등록 가능

## 5. 참고

- LM Studio에서 모델을 바꿔 로드하면 봇도 자동으로 새 모델 사용
- 대화 기록은 메모리에만 저장 (봇 재시작 시 초기화)
- 검색 없이 일반 대화할 때는 Tavily API 소모 없음
