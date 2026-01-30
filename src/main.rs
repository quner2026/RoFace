//! Face Recognition Service
//!
//! High-performance face recognition service with OpenVINO acceleration.
//! Supports both REST (Axum) and gRPC (Tonic) APIs.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::net::TcpListener;
use tonic::transport::Server as TonicServer;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use roface::config::Config;
use roface::engine::ModelPool;
use roface::service::FaceService;
use roface::storage::SqliteStorage;
use roface::api::rest::{AppState, create_rest_router};
use roface::api::grpc::GrpcHandler;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("Starting Face Recognition Service v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = Config::load(Config::default_path()).unwrap_or_else(|e| {
        info!("Using default config ({})", e);
        Config::default()
    });

    info!("Configuration loaded:");
    info!("  REST port: {}", config.server.rest_port);
    info!("  gRPC port: {}", config.server.grpc_port);
    info!("  Device: {}", config.inference.device);
    info!("  Model idle timeout: {}s", config.inference.model_idle_timeout);
    info!("  Batch enabled: {}", config.inference.batch_enabled);

    // Initialize model pool
    let pool = Arc::new(ModelPool::new(
        &config.inference,
        config.models.detector.to_str().unwrap(),
        config.models.embedder.to_str().unwrap(),
        config.models.gender_age.to_str().unwrap(),
        config.models.emotion.to_str().unwrap(),
    )?);

    // Start model cleanup task
    let pool_clone = pool.clone();
    tokio::spawn(async move {
        pool_clone.start_cleanup_task().await;
    });

    // Initialize storage
    let storage_path = config.storage.sqlite_path.as_deref()
        .map(|p| p.to_str().unwrap())
        .unwrap_or("data/faces.db");
    
    let storage = Arc::new(SqliteStorage::new(storage_path).await?);
    info!("SQLite storage initialized at: {}", storage_path);

    // Create face service
    let service = Arc::new(FaceService::new(pool.clone(), storage.clone(), config.clone()));

    // Create REST app state
    let app_state = Arc::new(AppState {
        service: service.clone(),
        storage: storage.clone(),
        start_time: Instant::now(),
    });

    // Create REST router
    let rest_router = create_rest_router(app_state);

    // Start REST server
    let rest_port = config.server.rest_port;
    let _rest_handle = tokio::spawn(async move {
        let addr = format!("0.0.0.0:{}", rest_port);
        info!("REST API listening on http://{}", addr);
        
        let listener = TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, rest_router).await.unwrap();
    });

    // Start gRPC server
    let grpc_port = config.server.grpc_port;
    let grpc_handler = GrpcHandler::new(service.clone());
    
    let _grpc_handle = tokio::spawn(async move {
        let addr = format!("0.0.0.0:{}", grpc_port).parse().unwrap();
        info!("gRPC API listening on {}", addr);
        
        TonicServer::builder()
            .add_service(grpc_handler.into_server())
            .serve(addr)
            .await
            .unwrap();
    });

    info!("Face Recognition Service is ready!");
    info!("REST: http://localhost:{}/health", config.server.rest_port);
    info!("gRPC: localhost:{}", config.server.grpc_port);

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    info!("Shutdown signal received, cleaning up...");

    // Shutdown model pool
    pool.shutdown();

    info!("Goodbye!");
    Ok(())
}
