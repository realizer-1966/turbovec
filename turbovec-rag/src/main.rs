//! turbovec RAG web server.
//!
//! Pipeline: local Ollama nomic-embed-text (768d) → turbovec IdMapIndex
//! (4-bit compression) → Ollama Cloud gemma4 for answer generation.
//
// Endpoints:
//   GET  /            — web UI (embedded HTML)
//   POST /api/build   — load rag_docs/ → embed → build index
//   POST /api/query   — { question } → search → LLM answer
//   GET  /api/status  — index status

mod embed;
mod rag;
mod routes;

use std::sync::Arc;
use tokio::sync::RwLock;

pub const EMBED_MODEL: &str = "nomic-embed-text";
pub const EMBED_DIM: usize = 768;
pub const BIT_WIDTH: usize = 4;
pub const LLM_MODEL: &str = "gemma4";
pub const CHUNK_SIZE: usize = 300;
pub const CHUNK_OVERLAP: usize = 50;

pub struct AppState {
    pub rag: RwLock<Option<rag::RagIndex>>,
    pub ollama_local: String,
    pub ollama_cloud: String,
    pub api_key: String,
}

#[tokio::main]
async fn main() {
    let local = std::env::var("OLLAMA_LOCAL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let cloud = std::env::var("OLLAMA_CLOUD")
        .unwrap_or_else(|_| "https://ollama.com".into());
    let api_key = std::env::var("OLLAMA_API_KEY").unwrap_or_default();
    let docs_dir = std::env::var("RAG_DOCS_DIR")
        .unwrap_or_else(|_| "rag_docs".into());
    let index_path = std::env::var("RAG_INDEX_PATH")
        .unwrap_or_else(|_| "rag_index.tvim".into());
    let meta_path = std::env::var("RAG_META_PATH")
        .unwrap_or_else(|_| "rag_meta.json".into());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let state = Arc::new(AppState {
        rag: RwLock::new(rag::RagIndex::load(&index_path, &meta_path).ok()),
        ollama_local: local.clone(),
        ollama_cloud: cloud,
        api_key,
    });

    let docs_dir = Arc::new(docs_dir);
    let index_path = Arc::new(index_path);
    let meta_path = Arc::new(meta_path);

    let app = routes::router(state, docs_dir, index_path, meta_path);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .unwrap();
    eprintln!("turbovec-rag server listening on http://0.0.0.0:{port}");
    axum::serve(listener, app).await.unwrap();
}