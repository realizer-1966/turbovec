# turbovec RAG 세션 요약 (2026-07-15)

## 프로젝트 분석
- turbovec: Google Research TurboQuant 알고리즘 기반 Rust 벡터 검색 라이브러리
- Cargo workspace: turbovec(Rust 코어 0.9.0) + turbovec-python(PyO3 바인딩 0.8.0)
- 핵심: 2-4bit 압축, SIMD NEON(ARM)/AVX-512BW(x86), 온라인 색인(훈련 불필요)
- 테스트: 143개 전부 통과 (8.44초, release 빌드)
- 예제 실행: dump_state(6개 설정 덤프), kernel_xtest(5000 DB/50 쿼리 top-32 검색 정상)

## RAG 데모 구축 (~/turbovec/rag_demo.py)
- 파이프라인: 로컬 Ollama nomic-embed-text(768차원) → turbovec IdMapIndex(4-bit 압축) → Ollama Cloud gemma4 LLM
- 문서: .txt/.md/.markdown 지원 (strip_markdown 함수로 마크다운 문법 제거)
- 문서 폴더: ~/turbovec/rag_docs/ (turbovec_intro.txt, turboquant_algorithm.txt, rag_concepts.txt, simd_kernel.md)
- 인덱스: rag_index.tvim + rag_meta.json
- 사용법:
  - python3 rag_demo.py build
  - python3 rag_demo.py query "질문"
  - python3 rag_demo.py query "질문" --verbose
- 성능: 임베딩 4.6초, 검색 0.3초, LLM 2.5초, 압축률 25-30%

## 환경 메모
- Termux python3.11 numpy 깨짐 (PyExc_ValueError 심볼 누락) → python3 (3.14) 사용
- Ollama 로컬: 포트 11434 막힘 → 8080 사용 (OLLAMA_HOST=127.0.0.1:8080 ollama serve)
- nomic-embed-text 모델 pull 완료
- Ollama Cloud gemma4 채팅 API 정상 작동

## npx skills (emilkowalski/skills)
- 글로벌 설치: npx skills@latest add emilkowalski/skills --yes --global
- 설치 위치: ~/.agents/skills/
- 5개 스킬: animation-vocabulary, apple-design, emil-design-eng, improve-animations, review-animations
- 제거: npx skills@latest remove <스킬명들> --yes

## LLM 수준 가이드
- 최소선: 7-8B (Qwen2.5-7B) — 단순 질답
- 권장선: 12-32B (gemma4:31b, 현재 사용 중) — 한국어 자연스러움, 지시 잘 따름
- 최적선: 70B+ (gpt-oss:120b) — 다중 청크 종합 추론 강함
- 임베딩은 nomic-embed-text(137M)면 충분

## 생성된 파일
- ~/turbovec/rag_demo.py — RAG 메인 스크립트
- ~/turbovec/rag_docs/ — 문서 폴더 (4개 파일)
- ~/turbovec/rag_index.tvim — turbovec 압축 인덱스
- ~/turbovec/rag_meta.json — 청크 메타데이터
- ~/turbovec/turbovec-dump/ — dump_state 출력 (6개 .bin 파일)