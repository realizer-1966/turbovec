//! RAG index: load documents, embed, build turbovec index, search.

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};
use turbovec::IdMapIndex;

use crate::embed;
use crate::embed_cache::{self, EmbedCache};
use crate::{BIT_WIDTH, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_DIM, EMBED_MODEL, LLM_MODEL};

#[derive(Serialize, Deserialize, Clone)]
pub struct Chunk {
    pub source: String,
    pub chunk_idx: usize,
    pub text: String,
}

#[derive(Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f32,
    pub source: String,
    pub chunk_idx: usize,
    pub text: String,
}

/// Build-time document snapshot — persisted so the UI can detect
/// when rag_docs/ has changed since the last successful build.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct BuildDocInfo {
    pub name: String,
    pub size: u64,
    /// Content hash of the (post-strip-markdown) file text.
    /// Used to detect content-only changes (same size, different text).
    pub hash: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BuildInfo {
    pub docs_count: usize,
    pub docs_total_size: u64,
    pub docs: Vec<BuildDocInfo>,
}

pub struct RagIndex {
    pub index: IdMapIndex,
    pub meta: HashMap<String, Chunk>,
    pub build_info: Option<BuildInfo>,
}

impl RagIndex {
    pub fn load(index_path: &str, meta_path: &str, build_info_path: &str) -> Result<Self, String> {
        let index = IdMapIndex::load(Path::new(index_path))
            .map_err(|e| format!("load index: {e}"))?;
        let meta_str = std::fs::read_to_string(meta_path)
            .map_err(|e| format!("load meta: {e}"))?;
        let meta: HashMap<String, Chunk> = serde_json::from_str(&meta_str)
            .map_err(|e| format!("parse meta: {e}"))?;
        let build_info = std::fs::read_to_string(build_info_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        Ok(Self { index, meta, build_info })
    }

    pub fn save(&self, index_path: &str, meta_path: &str, build_info_path: &str) -> Result<(), String> {
        self.index
            .write(Path::new(index_path))
            .map_err(|e| format!("save index: {e}"))?;
        let meta_str = serde_json::to_string(&self.meta)
            .map_err(|e| format!("serialize meta: {e}"))?;
        std::fs::write(meta_path, meta_str)
            .map_err(|e| format!("save meta: {e}"))?;
        if let Some(bi) = &self.build_info {
            let s = serde_json::to_string(bi)
                .map_err(|e| format!("serialize build_info: {e}"))?;
            std::fs::write(build_info_path, s)
                .map_err(|e| format!("save build_info: {e}"))?;
        }
        Ok(())
    }

    pub fn n_chunks(&self) -> usize {
        self.meta.len()
    }
}

/// Strip Markdown syntax, keep plain text content.
pub fn strip_markdown(text: &str) -> String {
    let mut s = text.to_string();

    // Code block fences
    s = regex_replace(&s, r"^```[^\n]*\n", "");
    s = regex_replace(&s, r"^```$", "");
    s = regex_replace(&s, r"^~~~[^\n]*\n", "");
    s = regex_replace(&s, r"^~~~$", "");

    // Inline code backticks
    s = regex_replace(&s, r"`([^`]+)`", "$1");

    // Headers
    s = regex_replace(&s, r"^#{1,6}\s+", "");

    // Images ![alt](url) → alt
    s = regex_replace(&s, r"!\[([^\]]*)\]\([^)]+\)", "$1");

    // Links [text](url) → text
    s = regex_replace(&s, r"\[([^\]]+)\]\([^)]+\)", "$1");

    // Bold/italic
    s = regex_replace(&s, r"\*\*([^*]+)\*\*", "$1");
    s = regex_replace(&s, r"__([^_]+)__", "$1");

    // Horizontal rules
    s = regex_replace(&s, r"^[-*_]{3,}\s*$", "");

    // Blockquote markers
    s = regex_replace(&s, r"^>\s?", "");

    // List markers
    s = regex_replace(&s, r"^\s*[-*+]\s+", "");
    s = regex_replace(&s, r"^\s*\d+\.\s+", "");

    // HTML tags
    s = regex_replace(&s, r"<[^>]+>", "");

    // Collapse multiple blank lines
    s = regex_replace(&s, r"\n{3,}", "\n\n");

    s.trim().to_string()
}

/// Simple regex replacement using the `regex` crate.
fn regex_replace(text: &str, pattern: &str, replacement: &str) -> String {
    match Regex::new(pattern) {
        Ok(re) => {
            if pattern.starts_with('^') {
                // Multiline patterns (^) need the 'm' flag
                match Regex::new(&format!("(?m){pattern}")) {
                    Ok(re_m) => re_m.replace_all(text, replacement).to_string(),
                    Err(_) => text.to_string(),
                }
            } else {
                re.replace_all(text, replacement).to_string()
            }
        }
        Err(_) => text.to_string(),
    }
}

/// Chunk text into overlapping segments.
pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if end >= chars.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    chunks
}

/// Load all supported documents from a directory.
pub fn load_documents(docs_dir: &str) -> Result<Vec<(String, String)>, String> {
    let dir = std::fs::read_dir(docs_dir)
        .map_err(|e| format!("read docs dir: {e}"))?;
    let mut files = Vec::new();
    for entry in dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let lower = name.to_lowercase();
        if !(lower.ends_with(".txt") || lower.ends_with(".md") || lower.ends_with(".markdown")) {
            continue;
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("read {name}: {e}"))?;
        let text = if lower.ends_with(".md") || lower.ends_with(".markdown") {
            strip_markdown(&content)
        } else {
            content.trim().to_string()
        };
        if !text.is_empty() {
            files.push((name, text));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

/// List supported documents with metadata (for /api/docs).
pub fn list_documents(docs_dir: &str) -> Result<Vec<crate::routes::DocInfo>, String> {
    let dir = std::fs::read_dir(docs_dir)
        .map_err(|e| format!("read docs dir: {e}"))?;
    let mut docs = Vec::new();
    for entry in dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let lower = name.to_lowercase();
        if !(lower.ends_with(".txt") || lower.ends_with(".md") || lower.ends_with(".markdown")) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        let text = if lower.ends_with(".md") || lower.ends_with(".markdown") {
            strip_markdown(&content)
        } else {
            content.trim().to_string()
        };
        let n_chunks = chunk_text(&text, CHUNK_SIZE, CHUNK_OVERLAP).len();
        docs.push(crate::routes::DocInfo { name, size, n_chunks });
    }
    docs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(docs)
}

/// Build the RAG index with streaming progress via tokio::sync::mpsc.
///
/// Incremental: chunks whose text hashes are present in the on-disk embed
/// cache skip the Ollama embed call. Only new/changed chunks hit the network.
pub async fn build_index_streaming(
    client: &reqwest::Client,
    docs_dir: &str,
    index_path: &str,
    meta_path: &str,
    build_info_path: &str,
    ollama_local: &str,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<(RagIndex, BuildStats), String> {
    let docs = load_documents(docs_dir)?;

    // Chunk all documents
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_texts: Vec<String> = Vec::new();
    let mut chunk_ids: Vec<u64> = Vec::new();
    let mut chunk_hashes: Vec<String> = Vec::new();

    for (fname, text) in &docs {
        let parts = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP);
        let _ = tx.send(format!("파일 로드: {} ({} 청크)", fname, parts.len())).await;
        for (i, part) in parts.iter().enumerate() {
            let id = simple_hash(fname, i);
            let h = embed_cache::content_hash(part);
            chunks.push(Chunk {
                source: fname.clone(),
                chunk_idx: i,
                text: part.clone(),
            });
            chunk_texts.push(part.clone());
            chunk_ids.push(id);
            chunk_hashes.push(h);
        }
    }

    if chunk_texts.is_empty() {
        return Err("문서가 없습니다. rag_docs/ 폴더에 .txt/.md 파일을 넣어주세요.".into());
    }

    // Load on-disk embed cache (keyed by content hash)
    let cache_path = embed_cache::cache_path(index_path);
    let mut cache = EmbedCache::load(&cache_path);

    let total = chunk_texts.len();
    let cached = chunk_hashes.iter().filter(|h| cache.get(h).is_some()).count();
    let to_embed = total - cached;
    let _ = tx.send(format!(
        "임베딩 시작: 총 {} 청크 (캐시 {} · 새로 임베딩 {})",
        total, cached, to_embed
    )).await;

    // Embed each chunk, using cache when available
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
    let mut embed_hits = 0usize;
    for (i, (text, hash)) in chunk_texts.iter().zip(chunk_hashes.iter()).enumerate() {
        let v = if let Some(cached) = cache.get(hash) {
            embed_hits += 1;
            cached.clone()
        } else {
            let v = embed::embed(client, ollama_local, EMBED_MODEL, text).await?;
            if v.len() != EMBED_DIM {
                return Err(format!(
                    "embedding dimension mismatch: got {}, expected {}",
                    v.len(),
                    EMBED_DIM
                ));
            }
            cache.insert(hash.clone(), v.clone());
            v
        };
        vectors.push(v);
        let _ = tx.send(format!("임베딩 {}/{} 완료{}", i + 1, total, if embed_hits == i + 1 { " (캐시)" } else { "" })).await;
    }

    // Persist the (possibly updated) cache
    let _ = cache.save(&cache_path);

    let _ = tx.send("turbovec 인덱스 빌드 중...".into()).await;

    // Build turbovec index
    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    let mut index = IdMapIndex::new(EMBED_DIM, BIT_WIDTH)
        .map_err(|e| format!("create index: {e}"))?;
    index
        .add_with_ids(&flat, &chunk_ids)
        .map_err(|e| format!("add to index: {e}"))?;

    // Build metadata map
    let mut meta: HashMap<String, Chunk> = HashMap::new();
    for (chunk, id) in chunks.iter().zip(&chunk_ids) {
        meta.insert(id.to_string(), chunk.clone());
    }

    let raw_size = flat.len() * 4;
    let build_docs: Vec<BuildDocInfo> = docs
        .iter()
        .map(|(name, text)| {
            let size = std::fs::metadata(std::path::Path::new(docs_dir).join(name))
                .map(|m| m.len())
                .unwrap_or(0);
            BuildDocInfo {
                name: name.clone(),
                size,
                hash: embed_cache::content_hash(text),
            }
        })
        .collect();
    let rag = RagIndex {
        index,
        meta,
        build_info: Some(BuildInfo {
            docs_count: docs.len(),
            docs_total_size: raw_size as u64,
            docs: build_docs,
        }),
    };
    rag.save(index_path, meta_path, build_info_path)?;

    let idx_size = std::fs::metadata(index_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    let stats = BuildStats {
        n_documents: docs.len(),
        n_chunks: chunk_texts.len(),
        embed_dim: EMBED_DIM,
        raw_bytes: raw_size,
        index_bytes: idx_size,
        embed_hits,
        embed_misses: to_embed,
    };

    Ok((rag, stats))
}

#[derive(Serialize)]
pub struct BuildStats {
    pub n_documents: usize,
    pub n_chunks: usize,
    pub embed_dim: usize,
    pub raw_bytes: usize,
    pub index_bytes: usize,
    pub embed_hits: usize,
    pub embed_misses: usize,
}

/// Search the index for top-K chunks matching the query.
pub async fn search(
    client: &reqwest::Client,
    rag: &RagIndex,
    ollama_local: &str,
    query: &str,
    k: usize,
) -> Result<Vec<SearchResult>, String> {
    let qvec = embed::embed(client, ollama_local, EMBED_MODEL, query).await?;
    if qvec.len() != EMBED_DIM {
        return Err(format!(
            "query embedding dimension mismatch: got {}, expected {}",
            qvec.len(),
            EMBED_DIM
        ));
    }
    let qflat: Vec<f32> = qvec.clone(); // (dim,)
    let (scores, ids) = rag.index.search(&qflat, k);
    // search returns flat Vec: scores[nq*k], ids[nq*k] — take first k
    let k_actual = k.min(scores.len());
    let scores = scores[..k_actual].to_vec();
    let ids = ids[..k_actual].to_vec();

    let mut results = Vec::new();
    for (score, id) in scores.iter().zip(ids.iter()) {
        let key = format!("{}", id);
        if let Some(chunk) = rag.meta.get(&key) {
            results.push(SearchResult {
                score: *score,
                source: chunk.source.clone(),
                chunk_idx: chunk.chunk_idx,
                text: chunk.text.clone(),
            });
        }
    }
    Ok(results)
}

/// Generate an answer using the LLM.
pub async fn generate_answer(
    client: &reqwest::Client,
    ollama_cloud: &str,
    api_key: &str,
    query: &str,
    contexts: &[SearchResult],
) -> Result<String, String> {
    let context_text = contexts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "[{}] (출처: {}, 점수: {:.4})\n{}",
                i + 1,
                c.source,
                c.score,
                c.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system = "당신은 주어진 컨텍스트를 기반으로 질문에 답하는 한국어 어시스턴트입니다. 컨텍스트에 있는 정보만 사용하여 답변하고, 정보가 부족하면 모른다고 말하세요. 답변 끝에 참조한 출처를 표시하세요.";
    let user = format!(
        "다음 컨텍스트를 바탕으로 질문에 답변해주세요.\n\n[컨텍스트]\n{context_text}\n\n[질문]\n{query}"
    );

    embed::chat(client, ollama_cloud, api_key, LLM_MODEL, system, &user).await
}

/// Deterministic hash for chunk IDs (same as Python md5 approach).
fn simple_hash(fname: &str, chunk_idx: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    fname.hash(&mut hasher);
    chunk_idx.hash(&mut hasher);
    hasher.finish()
}