//! SCRFD Face Detector
//!
//! High-accuracy face detection using InsightFace SCRFD model.
//! Outputs bounding boxes and 5-point landmarks.

use std::sync::Arc;

use image::GenericImageView;
use openvino::{InferRequest, Tensor, ElementType, Shape};
use anyhow::{Result, Context};

use super::pool::{ModelPool, ModelType};
use super::preprocess::{preprocess_for_detection, ResizeInfo, DETECTOR_INPUT_SIZE};

/// Face detection result
#[derive(Debug, Clone)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub landmarks: [(f32, f32); 5],
}

/// SCRFD Face Detector
pub struct FaceDetector {
    pool: Arc<ModelPool>,
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl FaceDetector {
    /// Create a new face detector
    pub fn new(pool: Arc<ModelPool>, confidence_threshold: f32) -> Self {
        Self {
            pool,
            confidence_threshold,
            nms_threshold: 0.4,
        }
    }

    /// Get the confidence threshold
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// Detect faces in an image
    pub fn detect(&self, image_data: &[u8]) -> Result<Vec<FaceBox>> {
        // Decode image with EXIF orientation handling
        let image = super::preprocess::decode_image(image_data)
            .context("Failed to decode image")?;
        
        let (orig_w, orig_h) = image.dimensions();
        let resize_info = ResizeInfo::new((orig_w, orig_h), DETECTOR_INPUT_SIZE);
        
        // Preprocess
        let input_tensor = preprocess_for_detection(&image)?;
        
        // Get model
        let model = self.pool.get_model(ModelType::Detector)?;
        
        // Run inference
        let mut request = model.create_infer_request()?;
        
        // Set input
        let input_shape = Shape::new(&[1, 3, DETECTOR_INPUT_SIZE.1 as i64, DETECTOR_INPUT_SIZE.0 as i64])?;
        let mut input = Tensor::new(ElementType::F32, &input_shape)?;
        
        // Copy data to tensor
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
        
        // Parse outputs (SCRFD/InsightFace format)
        let detections = self.parse_insightface_outputs(&request, &resize_info)?;
        
        // Apply NMS
        let final_detections = self.nms(detections);
        
        tracing::info!("Detected {} faces after NMS", final_detections.len());
        
        Ok(final_detections)
    }

    /// Parse InsightFace SCRFD model outputs
    /// 
    /// InsightFace det_10g.onnx model has 9 outputs:
    /// - outputs 0-2: scores for stride 8, 16, 32
    /// - outputs 3-5: bbox_preds for stride 8, 16, 32
    /// - outputs 6-8: kps_preds for stride 8, 16, 32
    fn parse_insightface_outputs(&self, request: &InferRequest, resize_info: &ResizeInfo) -> Result<Vec<FaceBox>> {
        let mut all_boxes = Vec::new();
        
        // Count outputs to determine model type
        let mut output_count = 0;
        for i in 0..20 {
            if request.get_output_tensor_by_index(i).is_ok() {
                output_count += 1;
            } else {
                break;
            }
        }
        
        tracing::info!("SCRFD model has {} outputs, threshold: {}", output_count, self.confidence_threshold);
        
        // Determine model configuration based on output count
        let (fmc, use_kps, num_anchors) = match output_count {
            6 => (3, false, 2),  // 3 strides, no kps, 2 anchors
            9 => (3, true, 2),   // 3 strides, with kps, 2 anchors
            10 => (5, false, 1), // 5 strides, no kps, 1 anchor
            15 => (5, true, 1),  // 5 strides, with kps, 1 anchor
            _ => {
                tracing::warn!("Unknown SCRFD output count: {}, trying default", output_count);
                (3, true, 2)
            }
        };
        
        let strides: Vec<i32> = if fmc == 3 {
            vec![8, 16, 32]
        } else {
            vec![8, 16, 32, 64, 128]
        };
        
        let (input_h, input_w) = (DETECTOR_INPUT_SIZE.1 as i32, DETECTOR_INPUT_SIZE.0 as i32);
        
        for (idx, &stride) in strides.iter().enumerate() {
            // Get outputs for this stride
            let scores_tensor = request.get_output_tensor_by_index(idx)?;
            let bbox_tensor = request.get_output_tensor_by_index(idx + fmc)?;
            let kps_tensor = if use_kps {
                Some(request.get_output_tensor_by_index(idx + fmc * 2)?)
            } else {
                None
            };
            
            // Get shapes
            let score_shape = scores_tensor.get_shape()?;
            let score_dims: Vec<i64> = score_shape.get_dimensions().to_vec();
            
            tracing::debug!("Stride {}: score shape {:?}", stride, score_dims);
            
            // Read raw data
            let scores: Vec<f32> = self.read_tensor_f32(&scores_tensor)?;
            let bboxes: Vec<f32> = self.read_tensor_f32(&bbox_tensor)?;
            let kps: Option<Vec<f32>> = kps_tensor.as_ref().map(|t| self.read_tensor_f32(t)).transpose()?;
            
            // Calculate feature map size
            let feat_h = input_h / stride;
            let feat_w = input_w / stride;
            
            // Generate anchor centers
            let mut anchor_centers: Vec<(f32, f32)> = Vec::new();
            for y in 0..feat_h {
                for x in 0..feat_w {
                    let cx = x as f32 * stride as f32;
                    let cy = y as f32 * stride as f32;
                    for _ in 0..num_anchors {
                        anchor_centers.push((cx, cy));
                    }
                }
            }
            
            let num_anchors_total = anchor_centers.len();
            tracing::debug!("Stride {}: {} anchors, scores len: {}, bboxes len: {}", 
                stride, num_anchors_total, scores.len(), bboxes.len());
            
            // Parse detections
            for (i, &(cx, cy)) in anchor_centers.iter().enumerate() {
                // Get score
                let score = if i < scores.len() {
                    scores[i]
                } else {
                    continue;
                };
                
                if score < self.confidence_threshold {
                    continue;
                }
                
                // Get bbox predictions (distance format: left, top, right, bottom)
                let bbox_idx = i * 4;
                if bbox_idx + 3 >= bboxes.len() {
                    continue;
                }
                
                let left = bboxes[bbox_idx] * stride as f32;
                let top = bboxes[bbox_idx + 1] * stride as f32;
                let right = bboxes[bbox_idx + 2] * stride as f32;
                let bottom = bboxes[bbox_idx + 3] * stride as f32;
                
                // Convert to x1, y1, x2, y2
                let x1 = cx - left;
                let y1 = cy - top;
                let x2 = cx + right;
                let y2 = cy + bottom;
                
                // Get landmarks
                let mut landmarks = [(0.0f32, 0.0f32); 5];
                if let Some(ref kps_data) = kps {
                    let kps_idx = i * 10;
                    if kps_idx + 9 < kps_data.len() {
                        for j in 0..5 {
                            let lx = cx + kps_data[kps_idx + j * 2] * stride as f32;
                            let ly = cy + kps_data[kps_idx + j * 2 + 1] * stride as f32;
                            
                            // Convert to original image coordinates
                            let (orig_lx, orig_ly) = resize_info.to_original(lx, ly);
                            landmarks[j] = (orig_lx, orig_ly);
                        }
                    }
                }
                
                // Convert bbox to original coordinates
                let (orig_x1, orig_y1) = resize_info.to_original(x1, y1);
                let (orig_x2, orig_y2) = resize_info.to_original(x2, y2);
                
                // Clamp to image bounds
                let orig_x1 = orig_x1.max(0.0).min(resize_info.original_width as f32);
                let orig_y1 = orig_y1.max(0.0).min(resize_info.original_height as f32);
                let orig_x2 = orig_x2.max(0.0).min(resize_info.original_width as f32);
                let orig_y2 = orig_y2.max(0.0).min(resize_info.original_height as f32);
                
                all_boxes.push(FaceBox {
                    x1: orig_x1,
                    y1: orig_y1,
                    x2: orig_x2,
                    y2: orig_y2,
                    confidence: score,
                    landmarks,
                });
            }
            
            tracing::debug!("Stride {} found {} faces", stride, all_boxes.len());
        }
        
        tracing::info!("Total {} faces before NMS", all_boxes.len());
        
        Ok(all_boxes)
    }
    
    /// Read tensor data as f32 vector
    fn read_tensor_f32(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        let shape = tensor.get_shape()?;
        let dims: Vec<i64> = shape.get_dimensions().to_vec();
        let total_elements: i64 = dims.iter().product();
        
        let data: Vec<f32> = unsafe {
            let ptr = tensor.get_raw_data()?.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, total_elements as usize).to_vec()
        };
        
        Ok(data)
    }

    /// Non-maximum suppression
    fn nms(&self, mut boxes: Vec<FaceBox>) -> Vec<FaceBox> {
        if boxes.is_empty() {
            return boxes;
        }
        
        // Sort by confidence (descending)
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut keep = Vec::new();
        let mut suppressed = vec![false; boxes.len()];
        
        for i in 0..boxes.len() {
            if suppressed[i] {
                continue;
            }
            
            keep.push(boxes[i].clone());
            
            for j in (i + 1)..boxes.len() {
                if suppressed[j] {
                    continue;
                }
                
                let iou = self.compute_iou(&boxes[i], &boxes[j]);
                if iou > self.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }
        
        keep
    }

    /// Compute intersection over union
    fn compute_iou(&self, a: &FaceBox, b: &FaceBox) -> f32 {
        let x1 = a.x1.max(b.x1);
        let y1 = a.y1.max(b.y1);
        let x2 = a.x2.min(b.x2);
        let y2 = a.y2.min(b.y2);
        
        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        
        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        
        let union = area_a + area_b - intersection;
        
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_calculation() {
        let a = FaceBox {
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0,
            confidence: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        let b = FaceBox {
            x1: 5.0, y1: 5.0, x2: 15.0, y2: 15.0,
            confidence: 0.8,
            landmarks: [(0.0, 0.0); 5],
        };
        
        let pool = Arc::new(ModelPool::new(
            &crate::config::InferenceConfig {
                device: "CPU".to_string(),
                num_threads: 1,
                model_idle_timeout: 300,
                batch_enabled: false,
                batch_max_size: 1,
                batch_timeout_ms: 100,
            },
            "", "", "", "",
        ).unwrap());
        
        let detector = FaceDetector::new(pool, 0.5);
        let iou = detector.compute_iou(&a, &b);
        
        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        // IOU: 25/175 ≈ 0.143
        assert!((iou - 0.143).abs() < 0.01);
    }
}
