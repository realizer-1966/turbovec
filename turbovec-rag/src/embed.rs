//! Embedding + LLM client for local Ollama and Ollama Cloud.

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct EmbedRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Deserialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: usize,
}

#[derive(Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessageContent,
}

#[derive(Deserialize)]
pub struct ChatMessageContent {
    pub content: String,
}

pub async fn embed(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
) -> Result<Vec<f32>, String> {
    let url = format!("{base_url}/api/embeddings");
    let body = EmbedRequest {
        model: model.into(),
        prompt: prompt.into(),
    };
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("embed request: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("embed {status}: {text}"));
    }
    let data: EmbedResponse = resp
        .json()
        .await
        .map_err(|e| format!("embed parse: {e}"))?;
    Ok(data.embedding)
}

pub async fn chat(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
) -> Result<String, String> {
    let url = format!("{base_url}/v1/chat/completions");
    let body = ChatRequest {
        model: model.into(),
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: system.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: user.into(),
            },
        ],
        temperature: 0.3,
        max_tokens: 1024,
    };
    let mut req = client.post(&url).json(&body);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("chat request: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("chat {status}: {text}"));
    }
    let data: ChatResponse = resp
        .json()
        .await
        .map_err(|e| format!("chat parse: {e}"))?;
    data.choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "no chat choices".into())
}