# turbovec-rag 아키텍처

turbovec-rag는 turbovec 벡터 검색 엔진 위에 구축된 경량 RAG 웹 애플리케이션입니다. FastAPI 느낌의 Rust 구현으로, Axum 기반 비동기 서버와 임베딩 캐시를 제공합니다.

## 구성 요소

### 1. 임베딩 (embed.rs)

외부 임베딩 API(기본: Ollama Cloud `gemma3:1b`)를 호출해 문서 청크를 밀집 벡터로 변환합니다. 요청 실패 시 재시도하며, 응답은 float32 벡터로 파싱됩니다.

- 기본 모델: `EMBED_MODEL` 환경변수로 오버라이드 가능
- 임베딩 차원: `EMBED_DIM`(기본 1536) — 인덱스 생성 시 고정
- API 호환: OpenAI `/v1/embeddings` 스펙을 따르는 모든 엔드포인트

### 2. 임베딩 캐시 (embed_cache.rs)

동일 텍스트 재임베딩을 방지하기 위해 로컬 파일 캐시를 유지합니다.

- 캐시 키: 텍스트 해시
- 저장 위치: `embed_cache.json`
- 빌드 시 히트/미스 통계를 `BuildStats`로 보고

### 3. 문서 로드 및 청킹 (rag.rs)

`rag_docs/` 디렉토리의 `.md` 파일을 읽어 마크다운을 스트rip 한 후 청킹합니다.

- 청크 크기: `CHUNK_SIZE`(기본 800자)
- 오버랩: `CHUNK_OVERLAP`(기본 200자)
- 마크다운 strip: 코드 블록, 링크, 이미지 태그 제거 후 일반 텍스트로 변환

### 4. 인덱스 (turbovec::IdMapIndex)

안정적인 외부 ID를 지원하는 `IdMapIndex`를 사용합니다.

- 비트 폭: `BIT_WIDTH`(기본 4)
- 파일 형식: `.tvim`(ID 매핑 포함) + `rag_meta.json`(청크 메타데이터)
- 빌드 정보: `rag_build_info.json`에 문서 스냅샷 저장 — UI가 변경 감지에 사용

### 5. 검색 및 답변 생성 (rag.rs)

`search()`는 쿼리를 임베딩해 `IdMapIndex::search`로 top-k 청크를 찾습니다. `generate_answer()`는 검색된 청크를 컨텍스트로 LLM(기본: Ollama Cloud)에 전달해 최종 답변을 생성합니다.

- LLM 모델: `LLM_MODEL` 환경변수
- 스트리밍: SSE(Server-Sent Events)로 클라이언트에 토큰 단위 전송

## 파일 레이아웃

```
turbovec-rag/
├── src/
│   ├── main.rs          # 서버 진입점, 라우터 구성
│   ├── routes.rs        # HTTP 핸들러 (/search, /build, /docs, ...)
│   ├── rag.rs           # RAG 파이프라인 코어 로직
│   ├── embed.rs         # 임베딩 API 클라이언트
│   └── embed_cache.rs   # 임베딩 캐시
├── static/              # 프론트엔드 정적 파일
├── build.rs            # 빌드 스크립트
└── Cargo.toml
```

## 빌드 및 실행

```bash
# 릴리즈 빌드
cargo build --release -p turbovec-rag

# 서버 실행 (기본 포트 3000)
./target/release/turbovec-rag

# 포트 변경
PORT=8080 ./target/release/turbovec-rag

# 임베딩/LLM 모델 오버라이드
EMBED_MODEL=gemma3:1b LLM_MODEL=gemma3:27b ./target/release/turbovec-rag
```

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PORT` | 3000 | 서버 리스닝 포트 |
| `EMBED_MODEL` | gemma3:1b | 임베딩용 모델명 |
| `EMBED_DIM` | 1536 | 임베딩 벡터 차원 |
| `BIT_WIDTH` | 4 | 양자화 비트 폭 (2, 3, 4) |
| `LLM_MODEL` | (기본값) | 답변 생성용 LLM |
| `CHUNK_SIZE` | 800 | 청크당 문자 수 |
| `CHUNK_OVERLAP` | 200 | 청크 간 오버랩 문자 수 |

## API 엔드포인트

- `GET /` — 프론트엔드 UI
- `POST /search` — 벡터 검색 (쿼리 → top-k 청크)
- `POST /ask` — RAG 답변 생성 (SSE 스트리밍)
- `POST /build` — 인덱스 재빌드
- `GET /docs` — 등록된 문서 목록
- `GET /health` — 헬스 체크

## 데이터 흐름

1. `rag_docs/*.md` 로드 → 마크다운 strip → 청킹
2. 각 청크 임베딩(캐시 확인) → float32 벡터
3. `IdMapIndex`에 청크 ID와 함께 추가
4. 검색 시 쿼리 임베딩 → top-k 청크 검색 → 컨텍스트 조립
5. LLM에 컨텍스트 + 질문 전달 → 스트리밍 답변

---

*turbovec-rag는 Termux/Android 환경에서도 실행 가능하도록 설계되었습니다.*