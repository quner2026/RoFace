//! Model Pool Manager
//!
//! Handles lazy loading and automatic unloading of models after idle timeout.
//! This is critical for 24/7 operation to minimize memory usage.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::ops::Deref;

use openvino::{Core, CompiledModel};
use parking_lot::RwLock;
use tokio::sync::Notify;
use tracing::{info, debug};

use crate::config::InferenceConfig;

/// Wrapper for OpenVINO Core that implements Send + Sync
pub struct SafeCore(Core);
unsafe impl Send for SafeCore {}
unsafe impl Sync for SafeCore {}

impl Deref for SafeCore {
    type Target = Core;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SafeCore {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Wrapper for OpenVINO CompiledModel that implements Send + Sync
#[derive(Clone)]
pub struct SafeCompiledModel(pub Arc<CompiledModel>);
unsafe impl Send for SafeCompiledModel {}
unsafe impl Sync for SafeCompiledModel {}

impl SafeCompiledModel {
    /// Create an inference request
    /// OpenVINO CompiledModel methods are thread-safe in C++, but Rust bindings
    /// require &mut self. We bypass this restriction safely.
    pub fn create_infer_request(&self) -> anyhow::Result<openvino::InferRequest> {
        unsafe {
            let ptr = std::sync::Arc::as_ptr(&self.0) as *mut CompiledModel;
            (*ptr).create_infer_request().map_err(|e| e.into())
        }
    }
}

impl Deref for SafeCompiledModel {
    type Target = CompiledModel;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A cached model with its last access time
struct CachedModel {
    compiled: SafeCompiledModel,
    last_access: Instant,
}

/// Model types that can be loaded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    Detector,
    Embedder,
    GenderAge,
    Emotion,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::Detector => "detector",
            ModelType::Embedder => "embedder",
            ModelType::GenderAge => "gender_age",
            ModelType::Emotion => "emotion",
        }
    }
}

/// Model Pool Manager with lazy loading and auto-unloading
pub struct ModelPool {
    core: Arc<RwLock<SafeCore>>,
    device: String,
    idle_timeout: Duration,
    
    // Model paths
    detector_path: String,
    embedder_path: String,
    gender_age_path: String,
    emotion_path: String,
    
    // Cached compiled models
    detector: RwLock<Option<CachedModel>>,
    embedder: RwLock<Option<CachedModel>>,
    gender_age: RwLock<Option<CachedModel>>,
    emotion: RwLock<Option<CachedModel>>,
    
    // Shutdown signal
    shutdown: Notify,
}

impl ModelPool {
    /// Create a new model pool
    pub fn new(
        config: &InferenceConfig,
        detector_path: &str,
        embedder_path: &str,
        gender_age_path: &str,
        emotion_path: &str,
    ) -> anyhow::Result<Self> {
        let core = Core::new()?;
        
        Ok(Self {
            core: Arc::new(RwLock::new(SafeCore(core))),
            device: config.device.clone(),
            idle_timeout: Duration::from_secs(config.model_idle_timeout),
            detector_path: detector_path.to_string(),
            embedder_path: embedder_path.to_string(),
            gender_age_path: gender_age_path.to_string(),
            emotion_path: emotion_path.to_string(),
            detector: RwLock::new(None),
            embedder: RwLock::new(None),
            gender_age: RwLock::new(None),
            emotion: RwLock::new(None),
            shutdown: Notify::new(),
        })
    }

    /// Get or load a model, returns a clone of the compiled model
    pub fn get_model(&self, model_type: ModelType) -> anyhow::Result<SafeCompiledModel> {
        let (cache, path) = match model_type {
            ModelType::Detector => (&self.detector, &self.detector_path),
            ModelType::Embedder => (&self.embedder, &self.embedder_path),
            ModelType::GenderAge => (&self.gender_age, &self.gender_age_path),
            ModelType::Emotion => (&self.emotion, &self.emotion_path),
        };

        // Try read lock first
        {
            let read_guard = cache.read();
            if let Some(_cached) = read_guard.as_ref() {
                // Model is loaded, update access time and return
                drop(read_guard);
                let mut write_guard = cache.write();
                if let Some(ref mut cached) = *write_guard {
                    cached.last_access = Instant::now();
                    return Ok(cached.compiled.clone());
                }
            }
        }

        // Need to load the model
        let mut write_guard = cache.write();
        
        // Double-check after acquiring write lock
        if let Some(ref mut cached) = *write_guard {
            cached.last_access = Instant::now();
            return Ok(cached.compiled.clone());
        }

        // Load the model
        info!("Loading model: {} from {}", model_type.as_str(), path);
        let start = Instant::now();
        
        // Use write lock on core to load model (Core methods like read_model require &mut self in Rust bindings)
        let mut core = self.core.write();
        let model = core.read_model_from_file(path, "")?;
        let compiled = core.compile_model(&model, self.device.as_str().into())?;
        let safe_compiled = SafeCompiledModel(Arc::new(compiled));
        
        let elapsed = start.elapsed();
        info!("Model {} loaded in {:?}", model_type.as_str(), elapsed);

        let cached_model = CachedModel {
            compiled: safe_compiled.clone(),
            last_access: Instant::now(),
        };
        
        *write_guard = Some(cached_model);
        
        Ok(safe_compiled)
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self, model_type: ModelType) -> bool {
        let cache = match model_type {
            ModelType::Detector => &self.detector,
            ModelType::Embedder => &self.embedder,
            ModelType::GenderAge => &self.gender_age,
            ModelType::Emotion => &self.emotion,
        };
        cache.read().is_some()
    }

    /// Get status of all models
    pub fn get_status(&self) -> Vec<(ModelType, bool)> {
        vec![
            (ModelType::Detector, self.is_loaded(ModelType::Detector)),
            (ModelType::Embedder, self.is_loaded(ModelType::Embedder)),
            (ModelType::GenderAge, self.is_loaded(ModelType::GenderAge)),
            (ModelType::Emotion, self.is_loaded(ModelType::Emotion)),
        ]
    }

    /// Unload a model
    fn unload_model(&self, model_type: ModelType) {
        let cache = match model_type {
            ModelType::Detector => &self.detector,
            ModelType::Embedder => &self.embedder,
            ModelType::GenderAge => &self.gender_age,
            ModelType::Emotion => &self.emotion,
        };
        
        let mut write_guard = cache.write();
        if write_guard.is_some() {
            info!("Unloading idle model: {}", model_type.as_str());
            *write_guard = None;
        }
    }

    /// Check and unload idle models
    fn cleanup_idle_models(&self) {
        let now = Instant::now();
        let models = [
            (ModelType::Detector, &self.detector),
            (ModelType::Embedder, &self.embedder),
            (ModelType::GenderAge, &self.gender_age),
            (ModelType::Emotion, &self.emotion),
        ];

        for (model_type, cache) in models {
            let should_unload = {
                let read_guard = cache.read();
                if let Some(ref cached) = *read_guard {
                    now.duration_since(cached.last_access) > self.idle_timeout
                } else {
                    false
                }
            };

            if should_unload {
                self.unload_model(model_type);
            }
        }
    }

    /// Start the background cleanup task
    pub async fn start_cleanup_task(self: Arc<Self>) {
        let check_interval = Duration::from_secs(60); // Check every minute
        
        loop {
            tokio::select! {
                _ = tokio::time::sleep(check_interval) => {
                    debug!("Running model cleanup check");
                    self.cleanup_idle_models();
                }
                _ = self.shutdown.notified() => {
                    info!("Model pool cleanup task shutting down");
                    break;
                }
            }
        }
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }
}

impl Drop for ModelPool {
    fn drop(&mut self) {
        self.shutdown.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_as_str() {
        assert_eq!(ModelType::Detector.as_str(), "detector");
        assert_eq!(ModelType::Embedder.as_str(), "embedder");
    }
}
