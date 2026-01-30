//! gRPC service implementation

use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::service::FaceService;
use crate::storage::FaceStorage;

// Include generated protobuf code
pub mod proto {
    include!("../proto/face.rs");
}

use proto::face_service_server::{FaceService as GrpcFaceService, FaceServiceServer};
use proto::*;

/// gRPC service implementation
pub struct GrpcHandler<S: FaceStorage> {
    service: Arc<FaceService<S>>,
}

impl<S: FaceStorage> GrpcHandler<S> {
    pub fn new(service: Arc<FaceService<S>>) -> Self {
        Self { service }
    }

    pub fn into_server(self) -> FaceServiceServer<Self> {
        FaceServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl<S: FaceStorage + 'static> GrpcFaceService for GrpcHandler<S> {
    async fn detect(
        &self,
        request: Request<DetectRequest>,
    ) -> Result<Response<DetectResponse>, Status> {
        let req = request.into_inner();
        
        let threshold = if req.confidence_threshold > 0.0 {
            Some(req.confidence_threshold)
        } else {
            None
        };

        let result = self.service.detect(&req.image_data, threshold).await
            .map_err(|e| Status::internal(e.to_string()))?;

        let faces: Vec<FaceDetection> = result.faces.into_iter().map(|f| {
            FaceDetection {
                bbox: Some(BoundingBox {
                    x1: f.x1,
                    y1: f.y1,
                    x2: f.x2,
                    y2: f.y2,
                    confidence: f.confidence,
                }),
                landmarks: f.landmarks.into_iter().map(|(x, y)| Landmark { x, y }).collect(),
            }
        }).collect();

        Ok(Response::new(DetectResponse {
            faces,
            inference_time_ms: result.inference_time_ms as i32,
        }))
    }

    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let req = request.into_inner();

        let metadata = if req.metadata.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&req.metadata).unwrap_or_default())
        };

        // gRPC doesn't have category field yet, default to None
        let category: Option<String> = None;

        let result = self.service
            .register(&req.image_data, &req.person_id, &req.person_name, category, metadata)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(RegisterResponse {
            success: result.success,
            face_id: result.face_id,
            message: result.message,
        }))
    }

    async fn compare(
        &self,
        request: Request<CompareRequest>,
    ) -> Result<Response<CompareResponse>, Status> {
        let req = request.into_inner();

        let result = self.service.compare(&req.image1_data, &req.image2_data).await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(CompareResponse {
            similarity: result.similarity,
            is_same_person: result.is_same_person,
            inference_time_ms: result.inference_time_ms as i32,
        }))
    }

    async fn identify(
        &self,
        request: Request<IdentifyRequest>,
    ) -> Result<Response<IdentifyResponse>, Status> {
        let req = request.into_inner();

        let top_k = if req.top_k > 0 { Some(req.top_k as usize) } else { None };
        let threshold = if req.threshold > 0.0 { Some(req.threshold) } else { None };

        let result = self.service.identify(&req.image_data, top_k, threshold).await
            .map_err(|e| Status::internal(e.to_string()))?;

        // Flatten all matches from all faces for gRPC compatibility
        let results: Vec<IdentifyResult> = result.faces.into_iter()
            .flat_map(|face| face.matches.into_iter())
            .map(|m| {
                IdentifyResult {
                    person_id: m.person_id,
                    person_name: m.person_name,
                    face_id: m.face_id,
                    similarity: m.similarity,
                }
            }).collect();

        Ok(Response::new(IdentifyResponse {
            results,
            inference_time_ms: result.inference_time_ms as i32,
        }))
    }

    async fn analyze(
        &self,
        request: Request<AnalyzeRequest>,
    ) -> Result<Response<AnalyzeResponse>, Status> {
        let req = request.into_inner();

        let result = self.service.analyze(&req.image_data).await
            .map_err(|e| Status::internal(e.to_string()))?;

        let faces: Vec<FaceAnalysis> = result.faces.into_iter().map(|f| {
            FaceAnalysis {
                bbox: Some(BoundingBox {
                    x1: f.x1,
                    y1: f.y1,
                    x2: f.x2,
                    y2: f.y2,
                    confidence: 1.0,
                }),
                attributes: Some(FaceAttributes {
                    age: f.age,
                    gender: f.gender,
                    gender_confidence: f.gender_confidence,
                    emotion: f.emotion,
                    emotion_confidence: f.emotion_confidence,
                }),
            }
        }).collect();

        Ok(Response::new(AnalyzeResponse {
            faces,
            inference_time_ms: result.inference_time_ms as i32,
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();

        let deleted = self.service.delete_face(&req.face_id).await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DeleteResponse {
            success: deleted,
            message: if deleted { "Face deleted" } else { "Face not found" }.to_string(),
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let health = self.service.health();

        Ok(Response::new(HealthResponse {
            healthy: health.healthy,
            version: health.version,
            models_loaded: health.models_loaded,
        }))
    }
}
