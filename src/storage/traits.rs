//! Storage abstraction traits
//!
//! Defines the interface for face data persistence.
//! Implementations can be swapped between SQLite and PostgreSQL.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// A stored face record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRecord {
    /// Unique face ID (UUID)
    pub face_id: String,
    /// Person ID (user-provided identifier)
    pub person_id: String,
    /// Person name
    pub person_name: String,
    /// Category/group name (e.g., "Friends", "Family")
    pub category: Option<String>,
    /// Face embedding vector (512 floats as bytes)
    pub embedding: Vec<u8>,
    /// Optional metadata as JSON
    pub metadata: Option<String>,
    /// Creation timestamp
    pub created_at: i64,
    /// Last updated timestamp
    pub updated_at: i64,
}

impl FaceRecord {
    /// Get embedding as float vector
    pub fn get_embedding(&self) -> Vec<f32> {
        self.embedding
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(arr)
            })
            .collect()
    }
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub face_id: String,
    pub person_id: String,
    pub person_name: String,
    pub similarity: f32,
}

/// Face storage trait
/// Implementations must be thread-safe and async-compatible
#[async_trait]
pub trait FaceStorage: Send + Sync + 'static {
    /// Save a new face record
    async fn save_face(&self, record: &FaceRecord) -> Result<()>;

    /// Get a face record by face ID
    async fn get_face(&self, face_id: &str) -> Result<Option<FaceRecord>>;

    /// Get all faces for a person
    async fn get_faces_by_person(&self, person_id: &str) -> Result<Vec<FaceRecord>>;

    /// Delete a face record
    async fn delete_face(&self, face_id: &str) -> Result<bool>;

    /// Delete all faces for a person
    async fn delete_person(&self, person_id: &str) -> Result<u64>;

    /// Search for similar faces
    /// Returns faces with similarity >= threshold, sorted by similarity descending
    async fn search_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>>;

    /// Get all face records (paginated)
    async fn list_faces(&self, offset: i64, limit: i64) -> Result<Vec<FaceRecord>>;

    /// Get total face count
    async fn count_faces(&self) -> Result<i64>;

    /// Update face record
    async fn update_face(&self, record: &FaceRecord) -> Result<bool>;
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}
