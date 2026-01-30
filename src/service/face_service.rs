//! Face Service - Core business logic
//!
//! Orchestrates detection, embedding, and storage operations.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tracing::info;
use uuid::Uuid;

use crate::config::Config;
use crate::engine::{
    ModelPool, FaceDetector, FaceEmbedder, AttributeAnalyzer,
    detector::FaceBox,
    embedder::FaceEmbedding,
    preprocess::{decode_image, align_face},
};
use crate::storage::{FaceStorage, FaceRecord};

use super::types::*;

/// Face recognition service
#[allow(dead_code)]
pub struct FaceService<S: FaceStorage> {
    pool: Arc<ModelPool>,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    attribute: AttributeAnalyzer,
    storage: Arc<S>,
    config: Config,
}

impl<S: FaceStorage> FaceService<S> {
    /// Create a new face service
    pub fn new(pool: Arc<ModelPool>, storage: Arc<S>, config: Config) -> Self {
        let detector = FaceDetector::new(pool.clone(), config.recognition.similarity_threshold);
        let embedder = FaceEmbedder::new(pool.clone(), config.recognition.embedding_dim);
        let attribute = AttributeAnalyzer::new(pool.clone());

        Self {
            pool,
            detector,
            embedder,
            attribute,
            storage,
            config,
        }
    }

    /// Get a reference to the storage
    pub fn storage(&self) -> &Arc<S> {
        &self.storage
    }

    /// Detect faces in an image
    pub async fn detect(&self, image_data: &[u8], confidence_threshold: Option<f32>) -> Result<DetectionResult> {
        let start = Instant::now();

        // Run detection in blocking task
        let image_data = image_data.to_vec();
        let threshold = confidence_threshold.unwrap_or(self.detector.confidence_threshold());
        
        let pool = self.pool.clone();
        let faces = tokio::task::spawn_blocking(move || {
            let detector = FaceDetector::new(pool, threshold);
            detector.detect(&image_data)
        })
        .await??;

        let inference_time_ms = start.elapsed().as_millis() as u64;

        let detected_faces: Vec<DetectedFace> = faces
            .into_iter()
            .map(|f| DetectedFace {
                x1: f.x1,
                y1: f.y1,
                x2: f.x2,
                y2: f.y2,
                confidence: f.confidence,
                landmarks: f.landmarks.to_vec(),
            })
            .collect();

        Ok(DetectionResult {
            faces: detected_faces,
            inference_time_ms,
        })
    }

    /// Register a face
    pub async fn register(
        &self,
        image_data: &[u8],
        person_id: &str,
        person_name: &str,
        category: Option<String>,
        metadata: Option<String>,
    ) -> Result<RegisterResult> {
        let start = Instant::now();

        // Decode image
        let image = decode_image(image_data)?;

        // Detect faces
        let pool = self.pool.clone();
        let image_data_vec = image_data.to_vec();
        let faces: Vec<FaceBox> = tokio::task::spawn_blocking(move || {
            let detector = FaceDetector::new(pool, 0.5);
            detector.detect(&image_data_vec)
        })
        .await??;

        if faces.is_empty() {
            return Ok(RegisterResult {
                success: false,
                face_id: String::new(),
                message: "No face detected in the image".to_string(),
            });
        }

        if faces.len() > 1 {
            return Ok(RegisterResult {
                success: false,
                face_id: String::new(),
                message: format!("Multiple faces detected ({}). Please provide an image with a single face.", faces.len()),
            });
        }

        let face = &faces[0];

        // Align and embed face
        let aligned = align_face(&image, &face.landmarks)?;
        
        let pool = self.pool.clone();
        let embedding_dim = self.config.recognition.embedding_dim;
        let embedding = tokio::task::spawn_blocking(move || {
            let embedder = FaceEmbedder::new(pool, embedding_dim);
            embedder.embed(&aligned)
        })
        .await??;

        // Generate face ID
        let face_id = Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Save face image to disk
        let faces_dir = std::path::Path::new("data/faces");
        std::fs::create_dir_all(faces_dir)?;
        
        // Crop face region with some padding
        let (x1, y1) = (face.x1.max(0.0) as u32, face.y1.max(0.0) as u32);
        let (x2, y2) = ((face.x2 as u32).min(image.width()), (face.y2 as u32).min(image.height()));
        let w = x2.saturating_sub(x1);
        let h = y2.saturating_sub(y1);
        
        if w > 0 && h > 0 {
            let face_crop = image.crop_imm(x1, y1, w, h);
            let face_path = faces_dir.join(format!("{}.jpg", face_id));
            face_crop.save(&face_path)?;
            info!("Saved face image to {:?}", face_path);
        }

        // Create record
        let record = FaceRecord {
            face_id: face_id.clone(),
            person_id: person_id.to_string(),
            person_name: person_name.to_string(),
            category: category.or_else(|| Some("default".to_string())),
            embedding: embedding.to_bytes(),
            metadata,
            created_at: now,
            updated_at: now,
        };

        // Save to storage
        self.storage.save_face(&record).await?;

        let elapsed = start.elapsed().as_millis();
        info!("Registered face {} for {} in {}ms", face_id, person_name, elapsed);

        Ok(RegisterResult {
            success: true,
            face_id,
            message: "Face registered successfully".to_string(),
        })
    }

    /// Compare two faces
    pub async fn compare(&self, image1_data: &[u8], image2_data: &[u8]) -> Result<CompareResult> {
        let start = Instant::now();

        // Extract embeddings for both images
        let embedding1 = self.extract_embedding(image1_data).await?;
        let embedding2 = self.extract_embedding(image2_data).await?;

        // Compute similarity
        let similarity = embedding1.cosine_similarity(&embedding2);
        let is_same_person = similarity >= self.config.recognition.similarity_threshold;

        let inference_time_ms = start.elapsed().as_millis() as u64;

        Ok(CompareResult {
            similarity,
            is_same_person,
            inference_time_ms,
        })
    }

    /// Identify all faces in the image from the database
    pub async fn identify(
        &self,
        image_data: &[u8],
        top_k: Option<usize>,
        threshold: Option<f32>,
    ) -> Result<IdentifyResult> {
        let start = Instant::now();

        // Decode image
        let image = decode_image(image_data)?;

        // Detect all faces
        let pool = self.pool.clone();
        let image_data_vec = image_data.to_vec();
        let detected_faces: Vec<FaceBox> = tokio::task::spawn_blocking(move || {
            let detector = FaceDetector::new(pool, 0.5);
            detector.detect(&image_data_vec)
        })
        .await??;

        if detected_faces.is_empty() {
            anyhow::bail!("No face detected in the image");
        }

        let top_k = top_k.unwrap_or(5);
        let threshold = threshold.unwrap_or(self.config.recognition.similarity_threshold);
        
        // Process each detected face
        let mut face_identifications = Vec::new();
        
        for face in &detected_faces {
            // Align face
            let aligned = align_face(&image, &face.landmarks)?;
            
            // Extract embedding
            let pool = self.pool.clone();
            let embedding_dim = self.config.recognition.embedding_dim;
            let embedding = tokio::task::spawn_blocking(move || {
                let embedder = FaceEmbedder::new(pool, embedding_dim);
                embedder.embed(&aligned)
            })
            .await??;
            
            // Search for matches
            let results = self.storage
                .search_similar(&embedding.vector, threshold, top_k)
                .await?;

            let matches: Vec<IdentifyMatch> = results
                .into_iter()
                .map(|r| IdentifyMatch {
                    person_id: r.person_id,
                    person_name: r.person_name,
                    face_id: r.face_id,
                    similarity: r.similarity,
                })
                .collect();
            
            face_identifications.push(FaceIdentification {
                x1: face.x1,
                y1: face.y1,
                x2: face.x2,
                y2: face.y2,
                confidence: face.confidence,
                matches,
            });
        }

        let inference_time_ms = start.elapsed().as_millis() as u64;

        Ok(IdentifyResult {
            faces: face_identifications,
            inference_time_ms,
        })
    }

    /// Analyze face attributes
    pub async fn analyze(&self, image_data: &[u8]) -> Result<AnalyzeResult> {
        let start = Instant::now();

        // Decode image
        let image = decode_image(image_data)?;

        // Detect faces
        let pool = self.pool.clone();
        let image_data_vec = image_data.to_vec();
        let faces: Vec<FaceBox> = tokio::task::spawn_blocking(move || {
            let detector = FaceDetector::new(pool, 0.5);
            detector.detect(&image_data_vec)
        })
        .await??;

        // Analyze each face
        let mut analyses = Vec::new();

        // If no face detected but image is small (likely a pre-cropped face), analyze directly
        if faces.is_empty() {
            let (width, height) = (image.width(), image.height());
            // If image is small enough to be a face crop (typically < 500x500)
            if width > 20 && height > 20 && width < 600 && height < 600 {
                // Analyze the whole image as if it were a face
                let pool = self.pool.clone();
                let image_clone = image.clone();
                let attrs = tokio::task::spawn_blocking(move || {
                    let analyzer = AttributeAnalyzer::new(pool);
                    analyzer.analyze(&image_clone)
                })
                .await??;

                analyses.push(FaceAnalysis {
                    x1: 0.0,
                    y1: 0.0,
                    x2: width as f32,
                    y2: height as f32,
                    age: attrs.age,
                    gender: attrs.gender.as_str().to_string(),
                    gender_confidence: attrs.gender_confidence,
                    emotion: attrs.emotion.as_str().to_string(),
                    emotion_confidence: attrs.emotion_confidence,
                });
            }
        } else {
            for face in faces {
                // Align face
                let aligned = align_face(&image, &face.landmarks)?;

                // Analyze attributes
                let pool = self.pool.clone();
                let attrs = tokio::task::spawn_blocking(move || {
                    let analyzer = AttributeAnalyzer::new(pool);
                    analyzer.analyze(&aligned)
                })
                .await??;

                analyses.push(FaceAnalysis {
                    x1: face.x1,
                    y1: face.y1,
                    x2: face.x2,
                    y2: face.y2,
                    age: attrs.age,
                    gender: attrs.gender.as_str().to_string(),
                    gender_confidence: attrs.gender_confidence,
                    emotion: attrs.emotion.as_str().to_string(),
                    emotion_confidence: attrs.emotion_confidence,
                });
            }
        }

        let inference_time_ms = start.elapsed().as_millis() as u64;

        Ok(AnalyzeResult {
            faces: analyses,
            inference_time_ms,
        })
    }

    /// Delete a face
    pub async fn delete_face(&self, face_id: &str) -> Result<bool> {
        self.storage.delete_face(face_id).await
    }

    /// Get health status
    pub fn health(&self) -> HealthResult {
        let status = self.pool.get_status();
        let models_loaded: std::collections::HashMap<String, bool> = status
            .into_iter()
            .map(|(t, loaded)| (t.as_str().to_string(), loaded))
            .collect();

        HealthResult {
            healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            models_loaded,
        }
    }

    /// Extract embedding from image
    async fn extract_embedding(&self, image_data: &[u8]) -> Result<FaceEmbedding> {
        let image = decode_image(image_data)?;

        // Detect face
        let pool = self.pool.clone();
        let image_data_vec = image_data.to_vec();
        let faces: Vec<FaceBox> = tokio::task::spawn_blocking(move || {
            let detector = FaceDetector::new(pool, 0.5);
            detector.detect(&image_data_vec)
        })
        .await??;

        if faces.is_empty() {
            anyhow::bail!("No face detected in the image");
        }

        let face = &faces[0];

        // Align and embed
        let aligned = align_face(&image, &face.landmarks)?;

        let pool = self.pool.clone();
        let embedding_dim = self.config.recognition.embedding_dim;
        let embedding = tokio::task::spawn_blocking(move || {
            let embedder = FaceEmbedder::new(pool, embedding_dim);
            embedder.embed(&aligned)
        })
        .await??;

        Ok(embedding)
    }
}
