//! Batch Inference Queue
//!
//! Collects multiple inference requests and processes them together
//! to improve throughput on high-concurrency scenarios.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{oneshot, Notify};
use tracing::{debug, info};

/// A single inference job in the queue
pub struct InferJob<T, R> {
    pub input: T,
    pub response_tx: oneshot::Sender<anyhow::Result<R>>,
}

/// Batch inference queue that collects requests and processes them together
pub struct BatchInferQueue<T, R> {
    queue: Mutex<Vec<InferJob<T, R>>>,
    notify: Notify,
    batch_size: usize,
    batch_timeout: Duration,
    enabled: bool,
}

impl<T: Send + 'static, R: Send + 'static> BatchInferQueue<T, R> {
    /// Create a new batch inference queue
    pub fn new(batch_size: usize, batch_timeout_ms: u64, enabled: bool) -> Self {
        Self {
            queue: Mutex::new(Vec::with_capacity(batch_size)),
            notify: Notify::new(),
            batch_size,
            batch_timeout: Duration::from_millis(batch_timeout_ms),
            enabled,
        }
    }

    /// Submit a job to the queue
    /// Returns a oneshot receiver for the result
    pub fn submit(&self, input: T) -> oneshot::Receiver<anyhow::Result<R>> {
        let (tx, rx) = oneshot::channel();
        
        {
            let mut queue = self.queue.lock();
            queue.push(InferJob {
                input,
                response_tx: tx,
            });
            
            // If we've reached batch size, notify immediately
            if queue.len() >= self.batch_size {
                self.notify.notify_one();
            }
        }
        
        rx
    }

    /// Check if batching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Take all pending jobs from the queue
    pub fn take_batch(&self) -> Vec<InferJob<T, R>> {
        let mut queue = self.queue.lock();
        std::mem::take(&mut *queue)
    }

    /// Wait for a batch to be ready (either full or timeout)
    pub async fn wait_for_batch(&self) {
        let has_items = {
            let queue = self.queue.lock();
            !queue.is_empty()
        };

        if !has_items {
            // Wait for first item
            self.notify.notified().await;
        }

        // Wait for batch to fill or timeout
        tokio::select! {
            _ = self.notify.notified() => {
                debug!("Batch ready (full)");
            }
            _ = tokio::time::sleep(self.batch_timeout) => {
                debug!("Batch ready (timeout)");
            }
        }
    }

    /// Get current queue size
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }
}

/// Helper trait for batch processing
pub trait BatchProcessor<T, R> {
    /// Process a batch of inputs
    fn process_batch(&self, inputs: Vec<T>) -> Vec<anyhow::Result<R>>;
}

/// Run the batch processing loop
pub async fn run_batch_loop<T, R, P>(
    queue: Arc<BatchInferQueue<T, R>>,
    processor: Arc<P>,
) where
    T: Send + 'static,
    R: Send + 'static,
    P: BatchProcessor<T, R> + Send + Sync + 'static,
{
    info!("Starting batch inference loop");
    
    loop {
        queue.wait_for_batch().await;
        
        let jobs = queue.take_batch();
        if jobs.is_empty() {
            continue;
        }

        let batch_size = jobs.len();
        debug!("Processing batch of {} items", batch_size);

        // Extract inputs
        let inputs: Vec<T> = jobs.iter().map(|j| {
            // This is a workaround - in real impl we'd use proper ownership
            unsafe { std::ptr::read(&j.input) }
        }).collect();
        
        // Process batch
        let processor_clone = processor.clone();
        let results = tokio::task::spawn_blocking(move || {
            processor_clone.process_batch(inputs)
        }).await;

        // Send results
        match results {
            Ok(results) => {
                for (job, result) in jobs.into_iter().zip(results) {
                    let _ = job.response_tx.send(result);
                }
            }
            Err(e) => {
                for job in jobs {
                    let _ = job.response_tx.send(Err(anyhow::anyhow!("Batch processing failed: {}", e)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_queue_submit() {
        let queue: BatchInferQueue<i32, i32> = BatchInferQueue::new(4, 100, true);
        
        let rx = queue.submit(42);
        assert_eq!(queue.len(), 1);
        
        // Clean up
        drop(rx);
    }
}
