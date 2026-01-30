//! Face recognition service configuration

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub inference: InferenceConfig,
    pub models: ModelsConfig,
    pub recognition: RecognitionConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub rest_port: u16,
    pub grpc_port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub device: String,
    pub num_threads: u32,
    pub model_idle_timeout: u64,
    pub batch_enabled: bool,
    pub batch_max_size: usize,
    pub batch_timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    pub detector: PathBuf,
    pub embedder: PathBuf,
    pub gender_age: PathBuf,
    pub emotion: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecognitionConfig {
    pub similarity_threshold: f32,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    #[serde(rename = "type")]
    pub storage_type: String,
    pub sqlite_path: Option<PathBuf>,
    pub postgres_url: Option<String>,
}

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn default_path() -> &'static str {
        "config.toml"
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                rest_port: 3000,
                grpc_port: 50051,
            },
            inference: InferenceConfig {
                device: "CPU".to_string(),
                num_threads: 4,
                model_idle_timeout: 300,
                batch_enabled: true,
                batch_max_size: 8,
                batch_timeout_ms: 100,
            },
            models: ModelsConfig {
                detector: PathBuf::from("models/scrfd_10g_kps.onnx"),
                embedder: PathBuf::from("models/glint360k_r100.onnx"),
                gender_age: PathBuf::from("models/genderage.onnx"),
                emotion: PathBuf::from("models/emotion_ferplus.onnx"),
            },
            recognition: RecognitionConfig {
                similarity_threshold: 0.5,
                embedding_dim: 512,
            },
            storage: StorageConfig {
                storage_type: "sqlite".to_string(),
                sqlite_path: Some(PathBuf::from("data/faces.db")),
                postgres_url: None,
            },
        }
    }
}
