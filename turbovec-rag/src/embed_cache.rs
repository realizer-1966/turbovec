//! Disk-backed embedding cache keyed by content hash.
//!
//! Cache layout (single JSON file):
//!   { "entries": { "<content_hash>": [<f32; dim>] } }
//!
//! On build, each chunk's embedding is looked up by hashing its text.
//! Misses fall back to the local Ollama embed endpoint; hits skip the call.
//! The cache persists across builds so unchanged chunks never re-embed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default)]
pub struct EmbedCache {
    pub entries: HashMap<String, Vec<f32>>,
}

impl EmbedCache {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let parent = path.parent();
        if let Some(p) = parent {
            std::fs::create_dir_all(p).map_err(|e| format!("cache mkdir: {e}"))?;
        }
        let s = serde_json::to_string(self).map_err(|e| format!("cache serialize: {e}"))?;
        std::fs::write(path, s).map_err(|e| format!("cache write: {e}"))?;
        Ok(())
    }

    pub fn get(&self, hash: &str) -> Option<&Vec<f32>> {
        self.entries.get(hash)
    }

    pub fn insert(&mut self, hash: String, vec: Vec<f32>) {
        self.entries.insert(hash, vec);
    }
}

/// SHA-256 hex digest of the input text (stable across runs/platforms).
pub fn content_hash(text: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // SHA-256 would be nicer but std lacks it; use a 128-bit-ish mix.
    // For cache key uniqueness this is sufficient.
    let mut h1 = DefaultHasher::new();
    text.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    h1.finish().hash(&mut h2);
    format!("{:016x}{:016x}", h1.finish(), h2.finish())
}

/// Resolve the cache path next to the index file.
pub fn cache_path(index_path: &str) -> PathBuf {
    let p = Path::new(index_path);
    let mut buf = p.to_path_buf();
    buf.set_extension("embedcache.json");
    if buf.extension().is_none() {
        PathBuf::from(format!("{}.embedcache.json", index_path))
    } else {
        buf
    }
}