//! Axum REST API handlers

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Router,
    routing::{get, post, delete},
    extract::{Path, State, Multipart, Query, DefaultBodyLimit},
    http::StatusCode,
    response::Json,
};
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tower_http::services::ServeDir;
use tracing::error;
use serde::Deserialize;
use uuid::Uuid;

use crate::service::FaceService;
use crate::storage::{FaceStorage, SqliteStorage, TaskRecord};

use super::dto::*;

/// Application state shared across handlers
pub struct AppState<S: FaceStorage> {
    pub service: Arc<FaceService<S>>,
    pub storage: Arc<SqliteStorage>,
    pub start_time: Instant,
}

/// Create the REST API router
pub fn create_rest_router<S: FaceStorage + 'static>(state: Arc<AppState<S>>) -> Router {
    Router::new()
        // Face operations
        .route("/api/v1/detect", post(detect_handler::<S>))
        .route("/api/v1/register", post(register_handler::<S>))
        .route("/api/v1/compare", post(compare_handler::<S>))
        .route("/api/v1/identify", post(identify_handler::<S>))
        .route("/api/v1/analyze", post(analyze_handler::<S>))
        .route("/api/v1/faces/:face_id", delete(delete_handler::<S>))
        // Dashboard API endpoints
        .route("/api/v1/faces", get(list_faces_handler::<S>))
        .route("/api/v1/faces/stats", get(faces_stats_handler::<S>))
        .route("/api/v1/faces/categories", get(categories_handler::<S>))
        .route("/api/v1/tasks/recent", get(recent_tasks_handler::<S>))
        .route("/api/v1/tasks", get(list_tasks_handler::<S>))
        .route("/api/v1/tasks/:task_id", delete(delete_task_handler::<S>))
        .route("/api/v1/tasks", delete(delete_all_tasks_handler::<S>))
        // System endpoints
        .route("/health", get(health_handler::<S>))
        .route("/api/v1/health", get(health_handler::<S>))
        .route("/metrics", get(metrics_handler::<S>))
        // Admin dashboard - serve static files
        .nest_service("/admin", ServeDir::new("admin").append_index_html_on_directories(true))
        // Face images and task images - serve from data/
        .nest_service("/data", ServeDir::new("data"))
        // Middleware
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB limit for large images
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Helper to save task image and record
async fn save_task_record(
    storage: &SqliteStorage,
    task_type: &str,
    result: Option<String>,
    image_data: Option<&[u8]>,
    faces_json: Option<String>,
    duration_ms: i64,
) {
    let task_id = Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    
    // Save image to disk (with EXIF orientation applied)
    let image_path = if let Some(data) = image_data {
        let dir = std::path::Path::new("data/tasks");
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{}.jpg", task_id));
        if let Ok(img) = crate::engine::preprocess::decode_image(data) {
            let _ = img.save(&path);
            Some(format!("/data/tasks/{}.jpg", task_id))
        } else {
            // Save raw data
            let _ = std::fs::write(&path, data);
            Some(format!("/data/tasks/{}.jpg", task_id))
        }
    } else {
        None
    };
    
    let task = TaskRecord {
        id: task_id,
        task_type: task_type.to_string(),
        result,
        image_path,
        faces_json,
        duration_ms,
        created_at: now,
    };
    
    if let Err(e) = storage.save_task(&task).await {
        error!("Failed to save task record: {}", e);
    }
}

/// Query parameters for listing faces
#[derive(Debug, Deserialize)]
pub struct ListFacesQuery {
    pub offset: Option<i64>,
    pub limit: Option<i64>,
    pub search: Option<String>,
    pub category: Option<String>,
}

/// List faces handler
async fn list_faces_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    Query(query): Query<ListFacesQuery>,
) -> Result<Json<ListFacesResponse>, (StatusCode, Json<ErrorResponse>)> {
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(20).min(100);
    
    // Get faces from storage (optionally filtered by category)
    let faces = if let Some(ref category) = query.category {
        if category != "all" && !category.is_empty() {
            state.storage.list_faces_by_category(category, offset, limit).await.map_err(|e| {
                error!("Failed to list faces by category: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "LIST_FAILED")))
            })?
        } else {
            state.service.storage().list_faces(offset, limit).await.map_err(|e| {
                error!("Failed to list faces: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "LIST_FAILED")))
            })?
        }
    } else {
        state.service.storage().list_faces(offset, limit).await.map_err(|e| {
            error!("Failed to list faces: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "LIST_FAILED")))
        })?
    };
    
    let total = if let Some(ref category) = query.category {
        if category != "all" && !category.is_empty() {
            state.storage.count_faces_by_category(category).await.unwrap_or(0)
        } else {
            state.service.storage().count_faces().await.unwrap_or(0)
        }
    } else {
        state.service.storage().count_faces().await.unwrap_or(0)
    };
    
    // Filter by search if provided
    let faces: Vec<FaceRecordDto> = faces.into_iter()
        .filter(|f| {
            if let Some(ref search) = query.search {
                let search_lower = search.to_lowercase();
                f.person_name.to_lowercase().contains(&search_lower) ||
                f.person_id.to_lowercase().contains(&search_lower)
            } else {
                true
            }
        })
        .map(|f| FaceRecordDto {
            face_id: f.face_id,
            person_id: f.person_id,
            person_name: f.person_name,
            category: f.category,
            metadata: f.metadata,
            created_at: f.created_at,
            updated_at: f.updated_at,
        })
        .collect();
    
    Ok(Json(ListFacesResponse {
        faces,
        total,
        offset,
        limit,
    }))
}

/// Faces stats handler
async fn faces_stats_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
) -> Result<Json<StatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let total_faces = state.service.storage().count_faces().await.unwrap_or(0);
    
    // Get task count for today
    let tasks = state.storage.list_tasks(1000, None).await.unwrap_or_default();
    let today_start = chrono::Utc::now().date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
    let today_tasks = tasks.iter().filter(|t| t.created_at >= today_start).count() as i64;
    
    // Calculate average response time
    let avg_time = if !tasks.is_empty() {
        Some(tasks.iter().map(|t| t.duration_ms).sum::<i64>() / tasks.len() as i64)
    } else {
        None
    };
    
    Ok(Json(StatsResponse {
        total_faces,
        today_tasks,
        success_rate: Some(1.0),
        avg_response_time: avg_time,
    }))
}

/// Get all face categories
async fn categories_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
) -> Result<Json<CategoriesResponse>, (StatusCode, Json<ErrorResponse>)> {
    let category_names = state.storage.get_categories().await.map_err(|e| {
        error!("Failed to get categories: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "CATEGORIES_FAILED")))
    })?;
    
    let mut categories = Vec::new();
    
    // Get total count for "all"
    let total = state.service.storage().count_faces().await.unwrap_or(0);
    categories.push(CategoryDto {
        name: "all".to_string(),
        count: total,
    });
    
    // Get count for each category
    for name in category_names {
        let count = state.storage.count_faces_by_category(&name).await.unwrap_or(0);
        categories.push(CategoryDto {
            name,
            count,
        });
    }
    
    Ok(Json(CategoriesResponse { categories }))
}

/// Query parameters for tasks
#[derive(Debug, Deserialize)]
pub struct TasksQuery {
    pub limit: Option<i64>,
    #[serde(rename = "type")]
    pub task_type: Option<String>,
}

/// Recent tasks handler
async fn recent_tasks_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    Query(query): Query<TasksQuery>,
) -> Json<Vec<TaskRecordDto>> {
    let limit = query.limit.unwrap_or(5);
    let task_type = query.task_type.as_deref();
    
    let tasks = state.storage.list_tasks(limit, task_type).await.unwrap_or_default();
    
    Json(tasks.into_iter().map(|t| TaskRecordDto {
        id: t.id,
        task_type: t.task_type,
        result: t.result,
        image_path: t.image_path,
        faces_json: t.faces_json,
        duration_ms: t.duration_ms,
        created_at: t.created_at,
    }).collect())
}

/// List tasks handler 
async fn list_tasks_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    Query(query): Query<TasksQuery>,
) -> Json<Vec<TaskRecordDto>> {
    let limit = query.limit.unwrap_or(50);
    let task_type = query.task_type.as_deref();
    
    let tasks = state.storage.list_tasks(limit, task_type).await.unwrap_or_default();
    
    Json(tasks.into_iter().map(|t| TaskRecordDto {
        id: t.id,
        task_type: t.task_type,
        result: t.result,
        image_path: t.image_path,
        faces_json: t.faces_json,
        duration_ms: t.duration_ms,
        created_at: t.created_at,
    }).collect())
}

/// Delete a single task
async fn delete_task_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    Path(task_id): Path<String>,
) -> Result<Json<DeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let deleted = state.storage.delete_task(&task_id).await.map_err(|e| {
        error!("Failed to delete task: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "DELETE_FAILED")))
    })?;

    if deleted {
        Ok(Json(DeleteResponse {
            success: true,
            message: "Task deleted successfully".to_string(),
        }))
    } else {
        Err((StatusCode::NOT_FOUND, Json(ErrorResponse::new("Task not found", "NOT_FOUND"))))
    }
}

/// Delete all tasks
async fn delete_all_tasks_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let count = state.storage.delete_all_tasks().await.map_err(|e| {
        error!("Failed to delete all tasks: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "DELETE_FAILED")))
    })?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Deleted {} tasks", count),
        "count": count
    })))
}

/// Detect faces in an image
async fn detect_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    mut multipart: Multipart,
) -> Result<Json<DetectResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    
    // Extract image from multipart
    let mut image_data: Option<Vec<u8>> = None;
    let mut confidence_threshold: Option<f32> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "MULTIPART_ERROR")))
    })? {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "image" {
            image_data = Some(field.bytes().await.map_err(|e| {
                (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "READ_ERROR")))
            })?.to_vec());
        } else if name == "confidence_threshold" {
            let text = field.text().await.unwrap_or_default();
            confidence_threshold = text.parse().ok();
        }
    }

    let image_data = image_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing image field", "MISSING_IMAGE")))
    })?;

    // Call service
    let result = state.service.detect(&image_data, confidence_threshold).await.map_err(|e| {
        error!("Detection failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "DETECTION_FAILED")))
    })?;

    // Convert to response
    let faces: Vec<FaceDto> = result.faces.iter().map(|f| FaceDto {
        x1: f.x1,
        y1: f.y1,
        x2: f.x2,
        y2: f.y2,
        confidence: f.confidence,
        landmarks: f.landmarks.iter().map(|(x, y)| LandmarkDto { x: *x, y: *y }).collect(),
    }).collect();
    
    let duration_ms = start.elapsed().as_millis() as i64;
    
    // Save task record
    let faces_json = serde_json::to_string(&faces).ok();
    let result_str = format!("检测到 {} 张人脸", faces.len());
    save_task_record(
        &state.storage,
        "detect",
        Some(result_str),
        Some(&image_data),
        faces_json,
        duration_ms,
    ).await;

    Ok(Json(DetectResponse {
        faces,
        inference_time_ms: result.inference_time_ms,
    }))
}

/// Register a face
async fn register_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    mut multipart: Multipart,
) -> Result<Json<RegisterResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    
    let mut image_data: Option<Vec<u8>> = None;
    let mut person_id: Option<String> = None;
    let mut person_name: Option<String> = None;
    let mut category: Option<String> = None;
    let mut metadata: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "MULTIPART_ERROR")))
    })? {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "image" => {
                image_data = Some(field.bytes().await.map_err(|e| {
                    (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "READ_ERROR")))
                })?.to_vec());
            }
            "person_id" => person_id = Some(field.text().await.unwrap_or_default()),
            "person_name" | "name" => person_name = Some(field.text().await.unwrap_or_default()),
            "category" => category = Some(field.text().await.unwrap_or_default()),
            "metadata" => metadata = Some(field.text().await.unwrap_or_default()),
            _ => {}
        }
    }

    let image_data = image_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing image field", "MISSING_IMAGE")))
    })?;

    let person_id = person_id.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing person_id field", "MISSING_PERSON_ID")))
    })?;

    let person_name = person_name.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing person_name field", "MISSING_PERSON_NAME")))
    })?;

    // Default category to "default" if not provided
    let category = category.filter(|c| !c.is_empty());

    let result = state.service.register(&image_data, &person_id, &person_name, category, metadata).await.map_err(|e| {
        error!("Registration failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "REGISTRATION_FAILED")))
    })?;

    let duration_ms = start.elapsed().as_millis() as i64;
    
    // Save task record
    let result_str = format!("注册成功: {} ({})", person_name, person_id);
    save_task_record(
        &state.storage,
        "register",
        Some(result_str),
        Some(&image_data),
        None,
        duration_ms,
    ).await;

    Ok(Json(RegisterResponse {
        success: result.success,
        face_id: result.face_id,
        message: result.message,
    }))
}

/// Compare two faces
async fn compare_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    mut multipart: Multipart,
) -> Result<Json<CompareResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    
    let mut image1_data: Option<Vec<u8>> = None;
    let mut image2_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "MULTIPART_ERROR")))
    })? {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "image1" => {
                image1_data = Some(field.bytes().await.map_err(|e| {
                    (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "READ_ERROR")))
                })?.to_vec());
            }
            "image2" => {
                image2_data = Some(field.bytes().await.map_err(|e| {
                    (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "READ_ERROR")))
                })?.to_vec());
            }
            _ => {}
        }
    }

    let image1_data = image1_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing image1 field", "MISSING_IMAGE1")))
    })?;

    let image2_data = image2_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing image2 field", "MISSING_IMAGE2")))
    })?;

    let result = state.service.compare(&image1_data, &image2_data).await.map_err(|e| {
        error!("Comparison failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "COMPARISON_FAILED")))
    })?;

    let duration_ms = start.elapsed().as_millis() as i64;
    
    // Save task record
    let result_str = format!("相似度: {:.1}% ({})", 
        result.similarity * 100.0,
        if result.is_same_person { "同一人" } else { "不同人" }
    );
    save_task_record(
        &state.storage,
        "compare",
        Some(result_str),
        Some(&image1_data),
        None,
        duration_ms,
    ).await;

    Ok(Json(CompareResponse {
        similarity: result.similarity,
        is_same_person: result.is_same_person,
        inference_time_ms: result.inference_time_ms,
    }))
}

/// Identify all faces from the database
async fn identify_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    mut multipart: Multipart,
) -> Result<Json<IdentifyResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    
    let mut image_data: Option<Vec<u8>> = None;
    let mut top_k: Option<usize> = None;
    let mut threshold: Option<f32> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "MULTIPART_ERROR")))
    })? {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "image" => {
                image_data = Some(field.bytes().await.map_err(|e| {
                    (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "READ_ERROR")))
                })?.to_vec());
            }
            "top_k" => top_k = field.text().await.ok().and_then(|t| t.parse().ok()),
            "threshold" => threshold = field.text().await.ok().and_then(|t| t.parse().ok()),
            _ => {}
        }
    }

    let image_data = image_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing image field", "MISSING_IMAGE")))
    })?;

    let result = state.service.identify(&image_data, top_k, threshold).await.map_err(|e| {
        error!("Identification failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "IDENTIFICATION_FAILED")))
    })?;

    // Convert to DTO with face positions and matches
    let faces: Vec<FaceIdentificationDto> = result.faces.iter().map(|f| FaceIdentificationDto {
        x1: f.x1,
        y1: f.y1,
        x2: f.x2,
        y2: f.y2,
        confidence: f.confidence,
        matches: f.matches.iter().map(|m| MatchDto {
            person_id: m.person_id.clone(),
            person_name: m.person_name.clone(),
            face_id: m.face_id.clone(),
            similarity: m.similarity,
        }).collect(),
    }).collect();
    
    let duration_ms = start.elapsed().as_millis() as i64;
    
    // Build result string for task record
    let result_str = if faces.is_empty() {
        "未检测到人脸".to_string()
    } else {
        let identified_count = faces.iter().filter(|f| !f.matches.is_empty()).count();
        if identified_count == 0 {
            format!("检测到 {} 张人脸，未识别到匹配", faces.len())
        } else {
            let best = faces.iter()
                .filter_map(|f| f.matches.first())
                .max_by(|a, b| a.similarity.partial_cmp(&b.similarity).unwrap());
            if let Some(best) = best {
                format!("识别 {}/{} 张人脸，最佳: {} ({:.1}%)", 
                    identified_count, faces.len(), best.person_name, best.similarity * 100.0)
            } else {
                format!("检测到 {} 张人脸", faces.len())
            }
        }
    };
    
    // Save faces as JSON
    let faces_json = serde_json::to_string(&faces).ok();
    
    // Save task record
    save_task_record(
        &state.storage,
        "identify",
        Some(result_str),
        Some(&image_data),
        faces_json,
        duration_ms,
    ).await;

    Ok(Json(IdentifyResponse {
        faces,
        inference_time_ms: result.inference_time_ms,
    }))
}

/// Analyze face attributes
async fn analyze_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    mut multipart: Multipart,
) -> Result<Json<AnalyzeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    
    let mut image_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "MULTIPART_ERROR")))
    })? {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "image" {
            image_data = Some(field.bytes().await.map_err(|e| {
                (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(&e.to_string(), "READ_ERROR")))
            })?.to_vec());
        }
    }

    let image_data = image_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse::new("Missing image field", "MISSING_IMAGE")))
    })?;

    let result = state.service.analyze(&image_data).await.map_err(|e| {
        error!("Analysis failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "ANALYSIS_FAILED")))
    })?;

    let faces: Vec<FaceAnalysisDto> = result.faces.iter().map(|f| FaceAnalysisDto {
        x1: f.x1,
        y1: f.y1,
        x2: f.x2,
        y2: f.y2,
        age: f.age,
        gender: f.gender.clone(),
        gender_confidence: f.gender_confidence,
        emotion: f.emotion.clone(),
        emotion_confidence: f.emotion_confidence,
    }).collect();
    
    let duration_ms = start.elapsed().as_millis() as i64;
    
    // Save task record
    let result_str = format!("分析 {} 张人脸", faces.len());
    let faces_json = serde_json::to_string(&faces).ok();
    save_task_record(
        &state.storage,
        "analyze",
        Some(result_str),
        Some(&image_data),
        faces_json,
        duration_ms,
    ).await;

    Ok(Json(AnalyzeResponse {
        faces,
        inference_time_ms: result.inference_time_ms,
    }))
}

/// Delete a face
async fn delete_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
    Path(face_id): Path<String>,
) -> Result<Json<DeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let deleted = state.service.delete_face(&face_id).await.map_err(|e| {
        error!("Delete failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(&e.to_string(), "DELETE_FAILED")))
    })?;
    
    // Also delete face image
    let face_path = std::path::Path::new("data/faces").join(format!("{}.jpg", face_id));
    let _ = std::fs::remove_file(face_path);

    Ok(Json(DeleteResponse {
        success: deleted,
        message: if deleted { "Face deleted successfully" } else { "Face not found" }.to_string(),
    }))
}

/// Health check
async fn health_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
) -> Json<HealthResponse> {
    let health = state.service.health();
    
    Json(HealthResponse {
        healthy: health.healthy,
        version: health.version,
        models_loaded: health.models_loaded,
    })
}

/// Metrics
async fn metrics_handler<S: FaceStorage>(
    State(state): State<Arc<AppState<S>>>,
) -> Json<MetricsResponse> {
    let health = state.service.health();
    let uptime = state.start_time.elapsed().as_secs();
    
    // Get face count
    let total_faces = state.service.storage().count_faces().await.unwrap_or(0);
    
    Json(MetricsResponse {
        total_faces,
        models_loaded: health.models_loaded,
        uptime_seconds: uptime,
    })
}
