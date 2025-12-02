# PumpStatLocalTranslator

자체 그래픽카드를 사용한 번역 서버 (서버비 절감을 위하여)

## 백엔드 및 GPU별 모델 선택 (RTX 3080 / RTX 5090)

이 서버는 번역 전용 MT 백엔드(NLLB)와 일반 LLM 백엔드(Qwen)를 지원합니다. 기본은 LLM(Qwen)입니다. 환경 변수(.env 또는 실행 시 전달)를 통해 GPU 프로필에 맞춰 권장 모델을 자동으로 선택합니다.

- BACKEND=`llm`(기본):
  - 3080: `Qwen/Qwen2.5-7B-Instruct` (4bit 권장)
  - 5090 (32GB): `Qwen/Qwen2.5-32B-Instruct` (4bit 권장)
- BACKEND=`nllb`:
  - 3080: `facebook/nllb-200-distilled-1.3B`
  - 5090 (32GB): `facebook/nllb-200-3.3B`

우선순위는 다음과 같습니다.

1) `MODEL_NAME`가 지정되면 해당 모델 사용
2) `BACKEND` 및 `GPU_PROFILE`에 해당하는 프리셋 사용 (예: `BACKEND=nllb`+`3080` → 1.3B, `5090` → 3.3B)
3) 아무것도 없으면 `BACKEND=nllb` 기준 VRAM 자동 선택(≥20GB→3.3B, 그외→1.3B)

선택된 설정은 서버 시작 시 콘솔에 출력됩니다.

### 사용 예시

1) RTX 3080에서 실행 (LLM 기본)

```
export GPU_PROFILE=3080
uvicorn server:app --host 0.0.0.0 --port 8000
```

2) RTX 5090에서 실행 (LLM 기본)

```
export GPU_PROFILE=5090
uvicorn server:app --host 0.0.0.0 --port 8000
```

3) NLLB 백엔드를 쓰고 싶을 때

```
export BACKEND=nllb
uvicorn server:app --host 0.0.0.0 --port 8000
```

`.env` 파일을 사용하는 경우(자동 로드됨):

```
# .env 예시 (3080, LLM 기본)
GPU_PROFILE=3080
```

`GPU_PROFILE`를 지정하지 않으면 VRAM을 자동 감지하여 보수적으로 모델을 선택합니다.

- VRAM >= 30GB: 32B 4bit
- VRAM >= 18GB: 14B 4bit
- 그 외: 7B 4bit

## 빠른 실행 스크립트

- `run_server.sh`: GPU 프로파일(또는 모델)로 API 서버 실행
- `translate.sh`: 로컬 서버의 `/translate` API 호출 후 번역 결과만 출력

### 1) 서버 실행

3080 예시(NLLB 기본):

```
bash run_server.sh -g 3080
```

5090 (32GB) 예시(NLLB 기본):

```
bash run_server.sh -g 5090
```

LLM 백엔드/모델 명시 예시:

```
BACKEND=llm bash run_server.sh -m Qwen/Qwen2.5-14B-Instruct --no-4bit -p 9000 --host 0.0.0.0
```

### 2) 번역 호출

서버 기본 주소는 `http://127.0.0.1:8000` 입니다. 예시:

```
bash translate.sh -src en -dst ko -t "A stamina chart with lots of runs"
```

스페인어로 번역:

```
bash translate.sh -s en -d es -t "A stamina chart with lots of runs"
```

커스텀 URL로 호출(예: 포트 9000):

```
bash translate.sh -src en -dst ko -t "Hello" -u http://127.0.0.1:9000
```

## API

- POST `/translate`
  - body: `{ text, source, target, use_default_examples?, examples?, authors?, chart_names? }`
  - `source`/`target`: `ko` | `en` | `es`
  
- POST `/detect_lang`
  - body: `{ text }`
  - return: `{ lang: 'ko'|'en'|'es', confidence: number, scores: { ko, en, es } }`
  - 간단한 휴리스틱(한글/라틴/스페인어 구두점+불용어)으로 판단합니다.
