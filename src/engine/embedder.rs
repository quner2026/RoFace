//! ArcFace Face Embedder
//!
//! High-accuracy face embedding using R100+Glint360K model.
//! Outputs 512-dimensional feature vectors.

use std::sync::Arc;

use image::DynamicImage;
use ndarray::Array4;
use openvino::{Tensor, ElementType, Shape};
use anyhow::Result;

use super::pool::{ModelPool, ModelType};
use super::preprocess::{align_face, EMBEDDER_INPUT_SIZE};
use super::detector::FaceBox;

/// Face embedding result
#[derive(Debug, Clone)]
pub struct FaceEmbedding {
    pub vector: Vec<f32>,
    pub norm: f32,
}

impl FaceEmbedding {
    /// Create a new normalized embedding
    pub fn new(mut vector: Vec<f32>) -> Self {
        // L2 normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }
        Self { vector, norm }
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &FaceEmbedding) -> f32 {
        // Since both vectors are normalized, dot product = cosine similarity
        self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Convert to bytes for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        self.vector.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() % 4 != 0 {
            anyhow::bail!("Invalid embedding bytes length");
        }
        
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(arr)
            })
            .collect();
        
        Ok(Self::new(vector))
    }
}

/// ArcFace Face Embedder
pub struct FaceEmbedder {
    pool: Arc<ModelPool>,
    embedding_dim: usize,
}

impl FaceEmbedder {
    /// Create a new face embedder
    pub fn new(pool: Arc<ModelPool>, embedding_dim: usize) -> Self {
        Self {
            pool,
            embedding_dim,
        }
    }

    /// Extract embedding from an aligned face image
    pub fn embed(&self, aligned_face: &DynamicImage) -> Result<FaceEmbedding> {
        // Resize to 112x112
        let (target_w, target_h) = EMBEDDER_INPUT_SIZE;
        let resized = aligned_face.resize_exact(
            target_w,
            target_h,
            image::imageops::FilterType::Lanczos3,
        );
        
        // Convert to tensor
        let input_tensor = self.image_to_tensor(&resized);
        
        // Get model
        let model = self.pool.get_model(ModelType::Embedder)?;
        
        // Run inference
        let mut request = model.create_infer_request()?;
        
        // Set input
        let input_shape = Shape::new(&[1, 3, target_h as i64, target_w as i64])?;
        let mut input = Tensor::new(ElementType::F32, &input_shape)?;
        
        let input_data = input_tensor.as_slice().unwrap();
        unsafe {
            let tensor_data = input.get_raw_data_mut()?.as_mut_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                tensor_data,
                input_data.len(),
            );
        }
        
        request.set_input_tensor(&input)?;
        
        // Infer
        request.infer()?;
        
        // Get output
        let output = request.get_output_tensor()?;
        let output_shape = output.get_shape()?;
        let output_dims: Vec<i64> = output_shape.get_dimensions().to_vec();
        
        let output_len = output_dims.iter().product::<i64>() as usize;
        let output_data: Vec<f32> = unsafe {
            let ptr = output.get_raw_data()?.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, output_len).to_vec()
        };
        
        Ok(FaceEmbedding::new(output_data))
    }

    /// Extract embedding from raw image data with face detection
    pub fn embed_from_detection(
        &self,
        image: &DynamicImage,
        face: &FaceBox,
    ) -> Result<FaceEmbedding> {
        // Align face using landmarks
        let aligned = align_face(image, &face.landmarks)?;
        self.embed(&aligned)
    }

    /// Extract embeddings from multiple faces (batch)
    pub fn embed_batch(&self, aligned_faces: &[DynamicImage]) -> Result<Vec<FaceEmbedding>> {
        if aligned_faces.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = aligned_faces.len();
        let (target_w, target_h) = EMBEDDER_INPUT_SIZE;
        
        // Prepare batch tensor
        let mut batch_tensor = Array4::<f32>::zeros((
            batch_size,
            3,
            target_h as usize,
            target_w as usize,
        ));
        
        for (i, face) in aligned_faces.iter().enumerate() {
            let resized = face.resize_exact(
                target_w,
                target_h,
                image::imageops::FilterType::Lanczos3,
            );
            let single_tensor = self.image_to_tensor(&resized);
            
            for c in 0..3 {
                for h in 0..target_h as usize {
                    for w in 0..target_w as usize {
                        batch_tensor[[i, c, h, w]] = single_tensor[[0, c, h, w]];
                    }
                }
            }
        }
        
        // Get model
        let model = self.pool.get_model(ModelType::Embedder)?;
        
        // Run inference
        let mut request = model.create_infer_request()?;
        
        let input_shape = Shape::new(&[
            batch_size as i64,
            3,
            target_h as i64,
            target_w as i64,
        ])?;
        let mut input = Tensor::new(ElementType::F32, &input_shape)?;
        
        let input_data = batch_tensor.as_slice().unwrap();
        unsafe {
            let tensor_data = input.get_raw_data_mut()?.as_mut_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                tensor_data,
                input_data.len(),
            );
        }
        
        request.set_input_tensor(&input)?;
        request.infer()?;
        
        // Get outputs
        let output = request.get_output_tensor()?;
        let output_data: Vec<f32> = unsafe {
            let ptr = output.get_raw_data()?.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, batch_size * self.embedding_dim).to_vec()
        };
        
        // Split into individual embeddings
        let embeddings: Vec<FaceEmbedding> = output_data
            .chunks_exact(self.embedding_dim)
            .map(|chunk| FaceEmbedding::new(chunk.to_vec()))
            .collect();
        
        Ok(embeddings)
    }

    /// Convert image to NCHW tensor with InsightFace normalization
    /// Note: InsightFace models expect BGR order, not RGB!
    fn image_to_tensor(&self, image: &DynamicImage) -> Array4<f32> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                // InsightFace uses BGR order with normalization: (pixel - 127.5) / 128.0
                // Channel 0 = B, Channel 1 = G, Channel 2 = R
                tensor[[0, 0, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0; // B
                tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0; // G
                tensor[[0, 2, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0; // R
            }
        }
        
        tensor
    }
}

/// Compute cosine similarity between two embedding vectors
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_normalization() {
        let embedding = FaceEmbedding::new(vec![3.0, 4.0]);
        // Norm should be 5, so normalized vector is [0.6, 0.8]
        assert!((embedding.vector[0] - 0.6).abs() < 1e-6);
        assert!((embedding.vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_serialization() {
        let original = FaceEmbedding::new(vec![1.0, 2.0, 3.0, 4.0]);
        let bytes = original.to_bytes();
        let restored = FaceEmbedding::from_bytes(&bytes).unwrap();
        
        for (a, b) in original.vector.iter().zip(restored.vector.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
