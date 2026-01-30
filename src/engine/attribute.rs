//! Face Attribute Analyzer
//!
//! Analyzes face attributes including:
//! - Age prediction
//! - Gender classification
//! - Emotion recognition

use std::sync::Arc;

use image::DynamicImage;
use ndarray::Array4;
use openvino::{Tensor, ElementType, Shape};
use anyhow::Result;

use super::pool::{ModelPool, ModelType};
use super::preprocess::ATTRIBUTE_INPUT_SIZE;

/// Gender classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
}

impl Gender {
    pub fn as_str(&self) -> &'static str {
        match self {
            Gender::Male => "male",
            Gender::Female => "female",
        }
    }
}

/// Emotion classification result
/// FER+ model uses 8 emotion classes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    Neutral,
    Happy,
    Surprise,
    Sad,
    Angry,
    Disgust,
    Fear,
    Contempt,
}

impl Emotion {
    pub fn as_str(&self) -> &'static str {
        match self {
            Emotion::Neutral => "neutral",
            Emotion::Happy => "happy",
            Emotion::Surprise => "surprise",
            Emotion::Sad => "sad",
            Emotion::Angry => "angry",
            Emotion::Disgust => "disgust",
            Emotion::Fear => "fear",
            Emotion::Contempt => "contempt",
        }
    }

    /// FER+ emotion model output order:
    /// 0: neutral, 1: happiness, 2: surprise, 3: sadness,
    /// 4: anger, 5: disgust, 6: fear, 7: contempt
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Emotion::Neutral,
            1 => Emotion::Happy,
            2 => Emotion::Surprise,
            3 => Emotion::Sad,
            4 => Emotion::Angry,
            5 => Emotion::Disgust,
            6 => Emotion::Fear,
            7 => Emotion::Contempt,
            _ => Emotion::Neutral,
        }
    }
}

/// Face attribute analysis result
#[derive(Debug, Clone)]
pub struct FaceAttributes {
    pub age: i32,
    pub gender: Gender,
    pub gender_confidence: f32,
    pub emotion: Emotion,
    pub emotion_confidence: f32,
}

/// Face Attribute Analyzer
pub struct AttributeAnalyzer {
    pool: Arc<ModelPool>,
}

impl AttributeAnalyzer {
    /// Create a new attribute analyzer
    pub fn new(pool: Arc<ModelPool>) -> Self {
        Self { pool }
    }

    /// Analyze attributes from an aligned face image
    pub fn analyze(&self, aligned_face: &DynamicImage) -> Result<FaceAttributes> {
        // Get age and gender
        let (age, gender, gender_conf) = self.analyze_gender_age(aligned_face)?;
        
        // Get emotion
        let (emotion, emotion_conf) = self.analyze_emotion(aligned_face)?;
        
        Ok(FaceAttributes {
            age,
            gender,
            gender_confidence: gender_conf,
            emotion,
            emotion_confidence: emotion_conf,
        })
    }

    /// Analyze age and gender
    fn analyze_gender_age(&self, face: &DynamicImage) -> Result<(i32, Gender, f32)> {
        let (target_w, target_h) = ATTRIBUTE_INPUT_SIZE;
        let resized = face.resize_exact(
            target_w,
            target_h,
            image::imageops::FilterType::Lanczos3,
        );
        
        let input_tensor = self.image_to_tensor(&resized);
        
        // Get model
        let model = self.pool.get_model(ModelType::GenderAge)?;
        
        // Run inference
        let mut request = model.create_infer_request()?;
        
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
        request.infer()?;
        
        // Parse output
        // InsightFace GenderAge model outputs: [female_logit, male_logit, age_scale]
        // See: https://github.com/deepinsight/insightface
        let output = request.get_output_tensor()?;
        let output_shape = output.get_shape()?;
        let output_len = output_shape.get_dimensions().iter().product::<i64>() as usize;
        
        let output_data: Vec<f32> = unsafe {
            let ptr = output.get_raw_data()?.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, output_len).to_vec()
        };
        
        // InsightFace genderage model output format: [female_logit, male_logit, age_scale]
        // - output[0]: Female logit
        // - output[1]: Male logit  
        // - output[2]: Age scale (0.0 ~ 1.0), multiply by 100 to get actual age
        tracing::debug!("GenderAge model raw output: {:?} (len={})", output_data, output_data.len());
        
        let (gender, gender_conf, age) = if output_data.len() == 3 {
            // Format: [female_logit, male_logit, age_scale]
            let female_logit = output_data[0];
            let male_logit = output_data[1];
            let age_scale = output_data[2];
            
            // Gender: compare logits directly (no softmax needed for decision)
            // But compute confidence using softmax for proper probability
            let gender_logits = vec![female_logit, male_logit];
            let gender_probs = self.softmax(&gender_logits);
            let female_prob = gender_probs[0];
            let male_prob = gender_probs[1];
            
            tracing::debug!("Gender probs - female: {:.3}, male: {:.3}, age_scale: {:.3}", 
                female_prob, male_prob, age_scale);
            
            let (gender, conf) = if male_logit > female_logit {
                (Gender::Male, male_prob)
            } else {
                (Gender::Female, female_prob)
            };
            
            // Age: age_scale is normalized [0, 1], multiply by 100 to get actual age
            let age = (age_scale * 100.0).round() as i32;
            
            (gender, conf, age)
            
        } else if output_data.len() == 2 {
            // Format: [gender_val, age_factor]
            // gender_val: <0 = male, >0 = female (signed logit)
            let gender_val = output_data[0];
            let age_factor = output_data[1];
            
            // Convert signed logit to probability using sigmoid
            let sigmoid = 1.0 / (1.0 + (-gender_val).exp());
            let (gender, conf) = if sigmoid > 0.5 {
                (Gender::Female, sigmoid)
            } else {
                (Gender::Male, 1.0 - sigmoid)
            };
            
            let age = if age_factor > 1.0 && age_factor < 120.0 {
                age_factor.round() as i32
            } else {
                (age_factor * 100.0).round() as i32
            };
            
            (gender, conf, age)
            
        } else {
            tracing::warn!("Unexpected genderage output length: {}", output_data.len());
            (Gender::Male, 0.5, 25)
        };
        
        // Clamp age to reasonable range
        let age = age.clamp(1, 100);
        
        tracing::debug!("Parsed: age={}, gender={:?}, conf={:.3}", age, gender, gender_conf);
        
        Ok((age, gender, gender_conf))
    }

    /// Analyze emotion
    fn analyze_emotion(&self, face: &DynamicImage) -> Result<(Emotion, f32)> {
        let (target_w, target_h) = (64, 64); // Common emotion model input size
        let resized = face.resize_exact(
            target_w,
            target_h,
            image::imageops::FilterType::Lanczos3,
        );
        
        // Convert to grayscale for some emotion models, or RGB for others
        let input_tensor = self.image_to_tensor_grayscale(&resized);
        
        // Get model
        let model = self.pool.get_model(ModelType::Emotion)?;
        
        // Run inference
        let mut request = model.create_infer_request()?;
        
        // Emotion models often use grayscale input
        let input_shape = Shape::new(&[1, 1, target_h as i64, target_w as i64])?;
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
        request.infer()?;
        
        // Parse output (8 emotion classes for FER+)
        // FER+ classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
        let output = request.get_output_tensor()?;
        let output_shape = output.get_shape()?;
        let output_len = output_shape.get_dimensions().iter().product::<i64>() as usize;
        
        let output_data: Vec<f32> = unsafe {
            let ptr = output.get_raw_data()?.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, output_len).to_vec()
        };
        
        tracing::debug!("Emotion model raw output: {:?}", output_data);
        
        // Apply softmax and find max
        let softmax = self.softmax(&output_data);
        
        tracing::debug!("Emotion softmax: {:?}", softmax);
        
        let (max_idx, max_conf) = softmax
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &conf)| (idx, conf))
            .unwrap_or((0, 0.0));
        
        let emotion = Emotion::from_index(max_idx);
        
        Ok((emotion, max_conf))
    }

    /// Convert image to NCHW tensor for GenderAge model
    /// This model expects RGB order with normalization: (x - 127.5) / 128.0
    fn image_to_tensor(&self, image: &DynamicImage) -> Array4<f32> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                // GenderAge model expects RGB order with normalization: (x - 127.5) / 128.0
                // Channel 0 = R, Channel 1 = G, Channel 2 = B
                tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0; // R
                tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0; // G
                tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0; // B
            }
        }
        
        tensor
    }

    /// Convert image to grayscale tensor for Emotion model
    /// FER+ model expects raw pixel values [0, 255], NO normalization!
    fn image_to_tensor_grayscale(&self, image: &DynamicImage) -> Array4<f32> {
        let gray = image.to_luma8();
        let (width, height) = gray.dimensions();
        
        let mut tensor = Array4::<f32>::zeros((1, 1, height as usize, width as usize));
        
        for y in 0..height {
            for x in 0..width {
                let pixel = gray.get_pixel(x, y);
                // FER+ emotion model expects raw pixel values [0, 255], not normalized!
                tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32;
            }
        }
        
        tensor
    }

    /// Softmax function
    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|v| v / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gender_as_str() {
        assert_eq!(Gender::Male.as_str(), "male");
        assert_eq!(Gender::Female.as_str(), "female");
    }

    #[test]
    fn test_emotion_from_index() {
        assert_eq!(Emotion::from_index(0), Emotion::Neutral);
        assert_eq!(Emotion::from_index(1), Emotion::Happy);
        assert_eq!(Emotion::from_index(2), Emotion::Surprise);
        assert_eq!(Emotion::from_index(7), Emotion::Contempt);
        assert_eq!(Emotion::from_index(100), Emotion::Neutral); // Out of range
    }

    #[test]
    fn test_softmax() {
        let analyzer = AttributeAnalyzer::new(Arc::new(
            ModelPool::new(
                &crate::config::InferenceConfig {
                    device: "CPU".to_string(),
                    num_threads: 1,
                    model_idle_timeout: 300,
                    batch_enabled: false,
                    batch_max_size: 1,
                    batch_timeout_ms: 100,
                },
                "", "", "", "",
            ).unwrap()
        ));
        
        let input = vec![1.0, 2.0, 3.0];
        let result = analyzer.softmax(&input);
        
        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Largest input should have largest output
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }
}
