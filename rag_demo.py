#!/usr/bin/env python3
"""
turbovec RAG demo — 로컬 임베딩 + turbovec 검색 + Ollama Cloud LLM.

파이프라인:
  1. 문서를 청크로 분할
  2. 로컬 Ollama nomic-embed-text (768차원) 로 임베딩
  3. turbovec IdMapIndex (4-bit 압축) 에 저장
  4. 쿼리 → 임베딩 → turbovec top-K 검색
  5. 검색된 청크를 컨텍스트로 Ollama Cloud gemma4 에 답변 생성

사용법:
  python3 rag_demo.py build    # 문서 로드 + 인덱스 빌드
  python3 rag_demo.py query "질문"  # RAG 쿼리
  python3 rag_demo.py query "질문" --verbose  # 검색 결과도 출력
"""

import json
import os
import re
import sys
import struct
import hashlib
import time
import urllib.request

import numpy as np
import turbovec

# 지원하는 문서 확장자
SUPPORTED_EXTS = (".txt", ".md", ".markdown")

# ── 설정 ──────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
BIT_WIDTH = 4
LLM_MODEL = "gemma4"  # Ollama Cloud
CHUNK_SIZE = 300      # 문자 단위
CHUNK_OVERLAP = 50

# 로컬 Ollama (Termux는 포트 11434 막힘 → 8080 사용)
LOCAL_OLLAMA = "http://127.0.0.1:8080"
# Ollama Cloud
CLOUD_OLLAMA = "https://ollama.com"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

INDEX_PATH = os.path.join(os.path.dirname(__file__), "rag_index.tvim")
META_PATH = os.path.join(os.path.dirname(__file__), "rag_meta.json")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "rag_docs")
# ───────────────────────────────────────────────────────────────────


def api_post(url, payload, with_auth=False):
    """동기 HTTP POST (urllib 사용, 의존성 최소화)."""
    headers = {"Content-Type": "application/json"}
    if with_auth and OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def embed_texts(texts):
    """로컬 Ollama nomic-embed-text 로 배치 임베딩."""
    url = f"{LOCAL_OLLAMA}/api/embeddings"
    vectors = []
    for text in texts:
        resp = api_post(url, {"model": EMBED_MODEL, "prompt": text})
        vectors.append(resp["embedding"])
    return np.array(vectors, dtype=np.float32)


def strip_markdown(text):
    """Markdown 문법을 제거하고 본문 텍스트만 추출.

    코드 블록은 내용을 유지하되 펜스스 제거,
    헤더/링크/이미지/굵게/기울임/리스트 마커/수평선/인용 블록 마커를 평문으로 변환.
    """
    # 코드 블록 펜스 제거 (``` 또는 ~~~), 내용은 유지
    text = re.sub(r"^```[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^~~~[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^~~~$", "", text, flags=re.MULTILINE)

    # 인라인 코드 백틱 제거
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 헤더 마커 제거 (### Title -> Title)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 이미지 ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # 링크 [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 굵게/기울임 **text** / *text* / __text__ / _text_ -> text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)

    # 수평선 (---, ***, ___) 제거
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 인용 블록 마커 제거
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # 리스트 마커 제거 (-, *, +, 1.)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)

    # 연속 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """텍스트를 오버랩 청크로 분할."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def load_documents():
    """rag_docs/ 폴더의 지원 문서(.txt, .md, .markdown) 로드."""
    docs = []
    if not os.path.isdir(DOCS_PATH):
        print(f"문서 폴더가 없습니다: {DOCS_PATH}")
        print("rag_docs/ 폴더를 만들고 .txt/.md 파일을 넣은 후 다시 실행하세요.")
        sys.exit(1)
    for fname in sorted(os.listdir(DOCS_PATH)):
        if not fname.lower().endswith(SUPPORTED_EXTS):
            continue
        path = os.path.join(DOCS_PATH, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        # Markdown 파일은 문법 제거 후 평문 변환
        if fname.lower().endswith((".md", ".markdown")):
            text = strip_markdown(text)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "source": fname,
                "chunk_idx": i,
                "text": chunk,
                "id": int(hashlib.md5(f"{fname}:{i}".encode()).hexdigest()[:15], 16),
            })
        print(f"  {fname}: {len(chunks)} 청크")
    return docs


def build_index():
    """문서 로드 → 임베딩 → turbovec 인덱스 빌드 → 저장."""
    print("=== RAG 인덱스 빌드 ===")
    docs = load_documents()
    if not docs:
        print("문서가 없습니다. rag_docs/ 폴더에 .txt 파일을 넣어주세요.")
        return

    print(f"\n총 {len(docs)} 청크 임베딩 중 (로컬 {EMBED_MODEL})...")
    t0 = time.time()
    texts = [d["text"] for d in docs]
    # 배치 단위로 임베딩 (진행률 표시)
    batch_size = 16
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = embed_texts(batch)
        all_vecs.append(vecs)
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)} 완료", flush=True)
    vectors = np.vstack(all_vecs)
    t1 = time.time()
    print(f"임베딩 완료: {t1-t0:.1f}초")

    # turbovec IdMapIndex 생성
    print(f"\nturbovec IdMapIndex 생성 (dim={EMBED_DIM}, bit_width={BIT_WIDTH})...")
    ids = np.array([d["id"] for d in docs], dtype=np.uint64)
    index = turbovec.IdMapIndex(dim=EMBED_DIM, bit_width=BIT_WIDTH)
    index.add_with_ids(vectors, ids)
    print(f"  {len(index)} 벡터 인덱스됨")

    # 인덱스 저장
    index.write(INDEX_PATH)
    print(f"  인덱스 저장: {INDEX_PATH}")

    # 메타데이터 저장
    meta = {str(d["id"]): {"source": d["source"], "chunk_idx": d["chunk_idx"], "text": d["text"]} for d in docs}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  메타데이터 저장: {META_PATH}")

    # 압축률 계산
    raw_size = vectors.nbytes
    idx_size = os.path.getsize(INDEX_PATH)
    print(f"\n압축률: {raw_size/1024:.0f}KB → {idx_size/1024:.0f}KB ({idx_size/raw_size*100:.1f}%)")


def search_index(query, k=5):
    """쿼리 임베딩 → turbovec 검색 → 청크 반환."""
    # 인덱스 로드
    index = turbovec.IdMapIndex.load(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 쿼리 임베딩
    qvec = embed_texts([query])[0]  # (768,)
    qvec_2d = np.array([qvec], dtype=np.float32)

    # 검색 — scores/ids 는 (nq, k) 형태의 2D 배열
    scores, ids = index.search(qvec_2d, k)
    scores = scores[0]  # (k,)
    ids = ids[0]        # (k,)

    results = []
    for score, id_val in zip(scores, ids):
        key = str(int(id_val))
        if key in meta:
            results.append({
                "score": float(score),
                "source": meta[key]["source"],
                "chunk_idx": meta[key]["chunk_idx"],
                "text": meta[key]["text"],
            })
    return results


def generate_answer(query, contexts):
    """Ollama Cloud gemma4 로 답변 생성."""
    context_text = "\n\n".join([
        f"[{i+1}] (출처: {c['source']}, 점수: {c['score']:.4f})\n{c['text']}"
        for i, c in enumerate(contexts)
    ])

    system_prompt = (
        "당신은 주어진 컨텍스트를 기반으로 질문에 답하는 한국어 어시스턴트입니다. "
        "컨텍스트에 있는 정보만 사용하여 답변하고, 정보가 부족하면 모른다고 말하세요. "
        "답변 끝에 참조한 출처를 표시하세요."
    )

    user_prompt = f"다음 컨텍스트를 바탕으로 질문에 답변해주세요.\n\n[컨텍스트]\n{context_text}\n\n[질문]\n{query}"

    url = f"{CLOUD_OLLAMA}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    resp = api_post(url, payload, with_auth=True)
    return resp["choices"][0]["message"]["content"]


def cmd_query(query, verbose=False):
    """RAG 쿼리 실행."""
    if not os.path.exists(INDEX_PATH):
        print("인덱스가 없습니다. 먼저 'python3 rag_demo.py build' 를 실행하세요.")
        return

    print(f"=== RAG 쿼리: {query} ===\n")

    # 1. 검색
    t0 = time.time()
    results = search_index(query, k=5)
    t1 = time.time()
    print(f"검색 완료 ({t1-t0:.2f}초, {len(results)}개 청크)\n")

    if not results:
        print("관련 문서를 찾을 수 없습니다.")
        return

    if verbose:
        print("=== 검색된 청크 ===")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] 점수: {r['score']:.4f} | 출처: {r['source']} #{r['chunk_idx']}")
            print(f"   {r['text'][:200]}{'...' if len(r['text'])>200 else ''}")
        print()

    # 2. LLM 답변 생성
    print("=== LLM 답변 생성 중 (Ollama Cloud gemma4) ===")
    t0 = time.time()
    answer = generate_answer(query, results)
    t1 = time.time()
    print(f"(생성 시간: {t1-t0:.1f}초)\n")
    print(answer)


def cmd_build():
    build_index()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "build":
        cmd_build()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("사용법: python3 rag_demo.py query \"질문\" [--verbose]")
            return
        verbose = "--verbose" in sys.argv
        query = sys.argv[2]
        cmd_query(query, verbose=verbose)
    else:
        print(f"알 수 없는 명령: {cmd}")
        print("사용 가능: build, query")


if __name__ == "__main__":
    main()