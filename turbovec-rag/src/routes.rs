//! HTTP routes for the RAG web server.
//! v2: scrollable areas for docs/progress/answer.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        Html, Json, Sse,
    },
    routing::{get, post},
    Router,
};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::rag::{self, BuildStats, SearchResult};
use crate::AppState;

#[derive(Deserialize)]
pub struct QueryRequest {
    pub question: String,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub answer: String,
    pub results: Vec<SearchResult>,
    pub search_time_ms: f32,
    pub llm_time_ms: f32,
}

#[derive(Serialize)]
pub struct BuildResponse {
    pub success: bool,
    pub message: String,
    pub stats: Option<BuildStats>,
}

#[derive(Serialize)]
pub struct StatusDocInfo {
    pub name: String,
    pub size: u64,
    /// "unchanged" | "changed" | "added" — relative to last build.
    /// null when no build info is available.
    pub status: Option<String>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub index_loaded: bool,
    pub n_chunks: usize,
    pub index_exists: bool,
    pub docs_total_size: u64,
    pub docs_count: usize,
    pub docs: Vec<StatusDocInfo>,
    /// Files present at build time but missing now.
    pub removed_docs: Vec<StatusDocInfo>,
    pub index_docs: Option<Vec<StatusDocInfo>>,
}

pub fn router(
    state: Arc<AppState>,
    docs_dir: Arc<String>,
    index_path: Arc<String>,
    meta_path: Arc<String>,
    build_info_path: Arc<String>,
) -> Router {
    Router::new()
        .route("/", get(index_html))
        .route("/api/status", get(status))
        .route("/api/docs", get(list_docs))
        .route("/api/build", post(build_sse))
        .route("/api/query", post(query))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(AppStateExt {
            state,
            docs_dir,
            index_path,
            meta_path,
            build_info_path,
        })
}

#[derive(Clone)]
struct AppStateExt {
    state: Arc<AppState>,
    docs_dir: Arc<String>,
    index_path: Arc<String>,
    meta_path: Arc<String>,
    build_info_path: Arc<String>,
}

async fn index_html(State(s): State<AppStateExt>) -> Html<String> {
    // Try multiple paths: relative (CWD), parent of docs_dir, absolute
    let candidates = [
        "turbovec-rag/static/index.html".to_string(),
        format!("{}/../turbovec-rag/static/index.html", s.docs_dir),
        "/data/data/com.termux/files/home/turbovec/turbovec-rag/static/index.html".to_string(),
    ];
    for path in &candidates {
        if let Ok(html) = std::fs::read_to_string(path) {
            return Html(html);
        }
    }
    Html("<html><body>index.html not found</body></html>".into())
}

async fn status(State(s): State<AppStateExt>) -> Json<StatusResponse> {
    let guard = s.state.rag.read().await;
    let (loaded, n, build_info) = match guard.as_ref() {
        Some(r) => (true, r.n_chunks(), r.build_info.clone()),
        None => (false, 0, None),
    };
    let index_exists = std::path::Path::new(&*s.index_path).exists();
    let docs = rag::list_documents(&s.docs_dir).unwrap_or_default();
    let docs_total_size: u64 = docs.iter().map(|d| d.size).sum();
    let docs_count = docs.len();

    // Build a lookup of build-time docs by name -> (size, hash)
    let build_map: HashMap<String, (u64, String)> = build_info
        .as_ref()
        .map(|b| {
            b.docs
                .iter()
                .map(|d| (d.name.clone(), (d.size, d.hash.clone())))
                .collect()
        })
        .unwrap_or_default();

    // Compute per-doc status relative to last build.
    let current_docs: Vec<StatusDocInfo> = docs
        .iter()
        .map(|d| {
            let status = if let Some((_, ref build_hash)) = build_map.get(&d.name) {
                // Compute hash of current content (post-strip-markdown)
                let path = std::path::Path::new(&*s.docs_dir).join(&d.name);
                let content = std::fs::read_to_string(&path).unwrap_or_default();
                let lower = d.name.to_lowercase();
                let text = if lower.ends_with(".md") || lower.ends_with(".markdown") {
                    rag::strip_markdown(&content)
                } else {
                    content.trim().to_string()
                };
                let cur_hash = crate::embed_cache::content_hash(&text);
                if cur_hash == *build_hash {
                    "unchanged".to_string()
                } else {
                    "changed".to_string()
                }
            } else if build_info.is_some() {
                "added".to_string()
            } else {
                // No build info at all — treat as unknown.
                String::new()
            };
            StatusDocInfo {
                name: d.name.clone(),
                size: d.size,
                status: if status.is_empty() { None } else { Some(status) },
            }
        })
        .collect();

    // Removed: present at build time, missing now.
    let removed_docs: Vec<StatusDocInfo> = if let Some(bi) = &build_info {
        let current_names: std::collections::HashSet<&str> =
            docs.iter().map(|d| d.name.as_str()).collect();
        bi.docs
            .iter()
            .filter(|d| !current_names.contains(d.name.as_str()))
            .map(|d| StatusDocInfo {
                name: d.name.clone(),
                size: d.size,
                status: Some("removed".to_string()),
            })
            .collect()
    } else {
        Vec::new()
    };

    let index_docs = build_info.as_ref().map(|b| {
        b.docs
            .iter()
            .map(|d| StatusDocInfo {
                name: d.name.clone(),
                size: d.size,
                status: None,
            })
            .collect()
    });

    Json(StatusResponse {
        index_loaded: loaded,
        n_chunks: n,
        index_exists,
        docs_total_size,
        docs_count,
        docs: current_docs,
        removed_docs,
        index_docs,
    })
}

#[derive(Serialize)]
pub struct DocInfo {
    pub name: String,
    pub size: u64,
    pub n_chunks: usize,
}

async fn list_docs(State(s): State<AppStateExt>) -> Json<Vec<DocInfo>> {
    let docs = match rag::list_documents(&s.docs_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("list_documents error: {e} (dir={})", s.docs_dir);
            Vec::new()
        }
    };
    Json(docs)
}

async fn build_sse(
    State(s): State<AppStateExt>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let (tx, rx) = mpsc::channel::<String>(64);
    let docs_dir = s.docs_dir.clone();
    let index_path = s.index_path.clone();
    let meta_path = s.meta_path.clone();
    let build_info_path = s.build_info_path.clone();
    let ollama_local = s.state.ollama_local.clone();
    let state = s.state.clone();

    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let result = rag::build_index_streaming(
            &client, &docs_dir, &index_path, &meta_path, &build_info_path, &ollama_local, &tx,
        ).await;

        match result {
            Ok((rag, stats)) => {
                *state.rag.write().await = Some(rag);
                let _ = tx.send(serde_json::to_string(&BuildResponse {
                    success: true,
                    message: format!("완료: {} 청크, {} 문서, {:.1}KB → {:.1}KB ({:.0}%) · 캐시 {}/{}",
                        stats.n_chunks, stats.n_documents,
                        stats.raw_bytes as f64 / 1024.0,
                        stats.index_bytes as f64 / 1024.0,
                        stats.index_bytes as f64 / stats.raw_bytes.max(1) as f64 * 100.0,
                        stats.embed_hits, stats.embed_hits + stats.embed_misses,
                    ),
                    stats: Some(stats),
                }).unwrap()).await;
            }
            Err(e) => {
                let _ = tx.send(serde_json::to_string(&BuildResponse {
                    success: false,
                    message: e,
                    stats: None,
                }).unwrap()).await;
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|msg| {
        Ok(Event::default().data(msg))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn query(
    State(s): State<AppStateExt>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    let guard = s.state.rag.read().await;
    let rag = guard
        .as_ref()
        .ok_or((StatusCode::BAD_REQUEST, "Index not built. Call /api/build first.".into()))?;

    let client = reqwest::Client::new();

    let t0 = std::time::Instant::now();
    let results = rag::search(&client, rag, &s.state.ollama_local, &req.question, 5)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let search_ms = t0.elapsed().as_secs_f32() * 1000.0;

    let t1 = std::time::Instant::now();
    let answer = rag::generate_answer(
        &client,
        &s.state.ollama_cloud,
        &s.state.api_key,
        &req.question,
        &results,
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let llm_ms = t1.elapsed().as_secs_f32() * 1000.0;

    Ok(Json(QueryResponse {
        answer,
        results,
        search_time_ms: search_ms,
        llm_time_ms: llm_ms,
    }))
}