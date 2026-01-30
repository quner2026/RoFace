//! Service layer types

use serde::{Deserialize, Serialize};

/// Face detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub faces: Vec<DetectedFace>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedFace {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub landmarks: Vec<(f32, f32)>,
}

/// Face registration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResult {
    pub success: bool,
    pub face_id: String,
    pub message: String,
}

/// Face comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareResult {
    pub similarity: f32,
    pub is_same_person: bool,
    pub inference_time_ms: u64,
}

/// Face identification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifyResult {
    pub faces: Vec<FaceIdentification>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceIdentification {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub matches: Vec<IdentifyMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifyMatch {
    pub person_id: String,
    pub person_name: String,
    pub face_id: String,
    pub similarity: f32,
}

/// Face attribute analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeResult {
    pub faces: Vec<FaceAnalysis>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceAnalysis {
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

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResult {
    pub healthy: bool,
    pub version: String,
    pub models_loaded: std::collections::HashMap<String, bool>,
}
