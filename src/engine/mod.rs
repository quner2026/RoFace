//! Inference engine module
//!
//! Provides OpenVINO-based inference with:
//! - Model lazy loading and auto-unloading
//! - Batch inference support
//! - Async multi-threaded execution

pub mod pool;
pub mod detector;
pub mod embedder;
pub mod attribute;
pub mod preprocess;
pub mod batch;

pub use pool::ModelPool;
pub use detector::FaceDetector;
pub use embedder::FaceEmbedder;
pub use attribute::AttributeAnalyzer;
pub use batch::BatchInferQueue;
