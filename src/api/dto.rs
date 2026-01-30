//! REST API request/response data transfer objects

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Detect request (multipart form)
#[derive(Debug, Deserialize)]
pub struct DetectRequest {
    pub confidence_threshold: Option<f32>,
}

/// Detect response
#[derive(Debug, Serialize)]
pub struct DetectResponse {
    pub faces: Vec<FaceDto>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct FaceDto {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub landmarks: Vec<LandmarkDto>,
}

#[derive(Debug, Serialize)]
pub struct LandmarkDto {
    pub x: f32,
    pub y: f32,
}

/// Register request (multipart form)
#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub person_id: String,
    pub person_name: String,
    pub category: Option<String>,
    pub metadata: Option<String>,
}

/// Register response
#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub face_id: String,
    pub message: String,
}

/// Compare response
#[derive(Debug, Serialize)]
pub struct CompareResponse {
    pub similarity: f32,
    pub is_same_person: bool,
    pub inference_time_ms: u64,
}

/// Identify request
#[derive(Debug, Deserialize)]
pub struct IdentifyRequest {
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
}

/// Identify response - supports multiple faces
#[derive(Debug, Serialize)]
pub struct IdentifyResponse {
    pub faces: Vec<FaceIdentificationDto>,
    pub inference_time_ms: u64,
}

/// Individual face identification result
#[derive(Debug, Clone, Serialize)]
pub struct FaceIdentificationDto {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub matches: Vec<MatchDto>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchDto {
    pub person_id: String,
    pub person_name: String,
    pub face_id: String,
    pub similarity: f32,
}

/// Analyze response
#[derive(Debug, Serialize)]
pub struct AnalyzeResponse {
    pub faces: Vec<FaceAnalysisDto>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct FaceAnalysisDto {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub age: i32,
    pub gender: String,
    pub gender_confidence: f32,
    pub emotion: String,
    pub emotion_confidence: f32,
}

/// Delete response
#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub success: bool,
    pub message: String,
}

/// Health response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub healthy: bool,
    pub version: String,
    pub models_loaded: HashMap<String, bool>,
}

/// Metrics response
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub total_faces: i64,
    pub models_loaded: HashMap<String, bool>,
    pub uptime_seconds: u64,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

impl ErrorResponse {
    pub fn new(error: &str, code: &str) -> Self {
        Self {
            error: error.to_string(),
            code: code.to_string(),
        }
    }
}

/// List faces response
#[derive(Debug, Serialize)]
pub struct ListFacesResponse {
    pub faces: Vec<FaceRecordDto>,
    pub total: i64,
    pub offset: i64,
    pub limit: i64,
}

/// Face record DTO
#[derive(Debug, Serialize)]
pub struct FaceRecordDto {
    pub face_id: String,
    pub person_id: String,
    pub person_name: String,
    pub category: Option<String>,
    pub metadata: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Categories response
#[derive(Debug, Serialize)]
pub struct CategoriesResponse {
    pub categories: Vec<CategoryDto>,
}

#[derive(Debug, Serialize)]
pub struct CategoryDto {
    pub name: String,
    pub count: i64,
}

/// Stats response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_faces: i64,
    pub today_tasks: i64,
    pub success_rate: Option<f64>,
    pub avg_response_time: Option<i64>,
}

/// Recent tasks response
#[derive(Debug, Serialize)]
pub struct TaskRecordDto {
    pub id: String,
    #[serde(rename = "type")]
    pub task_type: String,
    pub result: Option<String>,
    pub image_path: Option<String>,
    pub faces_json: Option<String>,
    pub duration_ms: i64,
    pub created_at: i64,
}
