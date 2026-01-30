//! Image preprocessing utilities for face recognition

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;
use anyhow::Result;

/// Standard input size for face detection (SCRFD)
pub const DETECTOR_INPUT_SIZE: (u32, u32) = (640, 640);

/// Standard input size for face embedding (ArcFace)
pub const EMBEDDER_INPUT_SIZE: (u32, u32) = (112, 112);

/// Standard input size for attribute models
pub const ATTRIBUTE_INPUT_SIZE: (u32, u32) = (96, 96);

/// Preprocess image for detection model
/// Resizes to 640x640 and normalizes to [-1, 1] or [0, 1] depending on model
pub fn preprocess_for_detection(image: &DynamicImage) -> Result<Array4<f32>> {
    let (target_w, target_h) = DETECTOR_INPUT_SIZE;
    
    // Resize with aspect ratio preservation and padding
    let resized = resize_with_padding(image, target_w, target_h);
    
    // Convert to NCHW format with normalization
    let tensor = image_to_nchw(&resized, true);
    
    Ok(tensor)
}

/// Preprocess aligned face for embedding model
/// Input should be a cropped and aligned face image
pub fn preprocess_for_embedding(face_image: &DynamicImage) -> Result<Array4<f32>> {
    let (target_w, target_h) = EMBEDDER_INPUT_SIZE;
    
    // Resize to 112x112
    let resized = face_image.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    
    // Convert to NCHW with normalization
    let tensor = image_to_nchw(&resized, true);
    
    Ok(tensor)
}

/// Preprocess face for attribute recognition
pub fn preprocess_for_attribute(face_image: &DynamicImage) -> Result<Array4<f32>> {
    let (target_w, target_h) = ATTRIBUTE_INPUT_SIZE;
    
    let resized = face_image.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    let tensor = image_to_nchw(&resized, false); // Some attribute models use [0, 1]
    
    Ok(tensor)
}

/// Resize image with padding to maintain aspect ratio
fn resize_with_padding(image: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    let (orig_w, orig_h) = image.dimensions();
    
    // Calculate scale factor
    let scale = f32::min(
        target_w as f32 / orig_w as f32,
        target_h as f32 / orig_h as f32,
    );
    
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    
    // Resize the image
    let resized = image.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);
    
    // Create padded image
    let mut padded = ImageBuffer::from_pixel(target_w, target_h, Rgb([0u8, 0, 0]));
    
    // Calculate padding offset (center the image)
    let offset_x = (target_w - new_w) / 2;
    let offset_y = (target_h - new_h) / 2;
    
    // Copy resized image to padded canvas
    let rgb_image = resized.to_rgb8();
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = rgb_image.get_pixel(x, y);
            padded.put_pixel(x + offset_x, y + offset_y, *pixel);
        }
    }
    
    DynamicImage::ImageRgb8(padded)
}

/// Convert image to NCHW tensor format
/// All InsightFace models expect BGR order with normalization
/// normalize: if true, normalize to [-1, 1], otherwise [0, 1]
fn image_to_nchw(image: &DynamicImage, normalize: bool) -> Array4<f32> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    
    let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            // InsightFace uses BGR order (swapRB in cv2.dnn.blobFromImage)
            let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            
            if normalize {
                // Normalize to [-1, 1] with BGR order
                tensor[[0, 0, y as usize, x as usize]] = (b - 127.5) / 128.0; // B
                tensor[[0, 1, y as usize, x as usize]] = (g - 127.5) / 128.0; // G
                tensor[[0, 2, y as usize, x as usize]] = (r - 127.5) / 128.0; // R
            } else {
                // Normalize to [0, 1] with BGR order
                tensor[[0, 0, y as usize, x as usize]] = b / 255.0;
                tensor[[0, 1, y as usize, x as usize]] = g / 255.0;
                tensor[[0, 2, y as usize, x as usize]] = r / 255.0;
            }
        }
    }
    
    tensor
}

/// Extract face region from image given bounding box
pub fn crop_face(
    image: &DynamicImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    margin: f32,
) -> DynamicImage {
    let (img_w, img_h) = image.dimensions();
    
    // Add margin
    let w = x2 - x1;
    let h = y2 - y1;
    let margin_x = w * margin;
    let margin_y = h * margin;
    
    let x1 = (x1 - margin_x).max(0.0) as u32;
    let y1 = (y1 - margin_y).max(0.0) as u32;
    let x2 = (x2 + margin_x).min(img_w as f32) as u32;
    let y2 = (y2 + margin_y).min(img_h as f32) as u32;
    
    image.crop_imm(x1, y1, x2 - x1, y2 - y1)
}

/// Align face using 5-point landmarks
/// Standard destination points for 112x112 aligned face
pub fn align_face(
    image: &DynamicImage,
    landmarks: &[(f32, f32); 5],
) -> Result<DynamicImage> {
    // Standard destination points for 112x112 aligned face (InsightFace standard)
    let dst_points: [(f32, f32); 5] = [
        (38.2946, 51.6963),  // left eye
        (73.5318, 51.5014),  // right eye
        (56.0252, 71.7366),  // nose
        (41.5493, 92.3655),  // left mouth
        (70.7299, 92.2041),  // right mouth
    ];
    
    // Calculate affine transform matrix using least squares
    let transform = estimate_affine_transform(landmarks, &dst_points)?;
    
    // Apply transformation
    let aligned = apply_affine_transform(image, &transform, 112, 112);
    
    Ok(aligned)
}

/// Estimate similarity transformation matrix from source to destination points
/// Uses the Umeyama algorithm for 2D similarity transformation
/// Reference: "Least-squares estimation of transformation parameters between two point patterns"
fn estimate_affine_transform(
    src: &[(f32, f32); 5],
    dst: &[(f32, f32); 5],
) -> Result<[[f32; 3]; 2]> {
    // Umeyama algorithm for similarity transform estimation
    // Similarity transform: dst = scale * R * src + t
    
    let n = 5.0f32;
    
    // Step 1: Compute centroids
    let (mut src_cx, mut src_cy) = (0.0f32, 0.0f32);
    let (mut dst_cx, mut dst_cy) = (0.0f32, 0.0f32);
    
    for i in 0..5 {
        src_cx += src[i].0;
        src_cy += src[i].1;
        dst_cx += dst[i].0;
        dst_cy += dst[i].1;
    }
    src_cx /= n;
    src_cy /= n;
    dst_cx /= n;
    dst_cy /= n;
    
    // Step 2: Compute centered points, variances and covariance
    let mut var_src = 0.0f32;
    
    // Covariance matrix elements (Sigma = sum of (src_centered) * (dst_centered)^T / n)
    // Note: For Umeyama, we compute Sigma = 1/n * sum( (src_i - src_mean) * (dst_i - dst_mean)^T )
    // But since we want to map src -> dst, we use: Sigma = 1/n * sum( (dst_i - dst_mean) * (src_i - src_mean)^T )
    let mut sigma_00 = 0.0f32; // sum of dx * sx
    let mut sigma_01 = 0.0f32; // sum of dx * sy
    let mut sigma_10 = 0.0f32; // sum of dy * sx  
    let mut sigma_11 = 0.0f32; // sum of dy * sy
    
    for i in 0..5 {
        let sx = src[i].0 - src_cx;
        let sy = src[i].1 - src_cy;
        let dx = dst[i].0 - dst_cx;
        let dy = dst[i].1 - dst_cy;
        
        var_src += sx * sx + sy * sy;
        
        // Covariance: Sigma = (1/n) * D^T * S where D is centered dst, S is centered src
        sigma_00 += dx * sx;
        sigma_01 += dx * sy;
        sigma_10 += dy * sx;
        sigma_11 += dy * sy;
    }
    
    var_src /= n;
    sigma_00 /= n;
    sigma_01 /= n;
    sigma_10 /= n;
    sigma_11 /= n;
    
    // Step 3: 2x2 SVD using closed-form solution
    // For 2x2 matrix M = [[a, b], [c, d]], SVD gives U, S, V such that M = U * S * V^T
    let a = sigma_00;
    let b = sigma_01;
    let c = sigma_10;
    let d = sigma_11;
    
    // Compute SVD components for 2x2 matrix
    // Using the formulas from https://scicomp.stackexchange.com/questions/8899
    let e = (a + d) / 2.0;
    let f = (a - d) / 2.0;
    let g = (c + b) / 2.0;
    let h = (c - b) / 2.0;
    
    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();
    
    let s1 = q + r;  // First singular value
    let s2 = (q - r).abs();  // Second singular value
    
    // Compute rotation angles
    let a1 = h.atan2(e);
    let a2 = g.atan2(f);
    
    let theta = (a2 - a1) / 2.0;  // V rotation angle
    let phi = (a2 + a1) / 2.0;    // U rotation angle
    
    // Determinant of covariance matrix
    let det_sigma = a * d - b * c;
    
    // Step 4: Compute rotation matrix R = U * S_mod * V^T
    // where S_mod = diag(1, sign(det)) to prevent reflection
    let (r00, r01, r10, r11) = if det_sigma >= 0.0 {
        // No reflection: R = U * V^T
        // R = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]] * [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
        let angle = phi - theta;
        (angle.cos(), -angle.sin(), angle.sin(), angle.cos())
    } else {
        // Need reflection: R = U * diag(1, -1) * V^T
        let angle = phi + theta;
        (angle.cos(), angle.sin(), angle.sin(), -angle.cos())
    };
    
    // Step 5: Compute scale
    // scale = trace(S * D) / var_src where D = diag(1, sign(det))
    let trace_sd = if det_sigma >= 0.0 {
        s1 + s2
    } else {
        s1 - s2
    };
    
    let scale = if var_src > 1e-10 {
        trace_sd / var_src
    } else {
        1.0
    };
    
    // Step 6: Compute translation
    // t = dst_mean - scale * R * src_mean
    let tx = dst_cx - scale * (r00 * src_cx + r01 * src_cy);
    let ty = dst_cy - scale * (r10 * src_cx + r11 * src_cy);
    
    // Return the 2x3 affine transformation matrix
    // [scale*R | t]
    Ok([
        [scale * r00, scale * r01, tx],
        [scale * r10, scale * r11, ty],
    ])
}

/// Apply affine transformation to image
fn apply_affine_transform(
    image: &DynamicImage,
    transform: &[[f32; 3]; 2],
    out_width: u32,
    out_height: u32,
) -> DynamicImage {
    let rgb = image.to_rgb8();
    let mut output = ImageBuffer::from_pixel(out_width, out_height, Rgb([0u8, 0, 0]));
    
    // Compute inverse transform for backward mapping
    let det = transform[0][0] * transform[1][1] - transform[0][1] * transform[1][0];
    let inv = [
        [transform[1][1] / det, -transform[0][1] / det],
        [-transform[1][0] / det, transform[0][0] / det],
    ];
    
    for y in 0..out_height {
        for x in 0..out_width {
            // Apply inverse transform
            let dx = x as f32 - transform[0][2];
            let dy = y as f32 - transform[1][2];
            
            let src_x = inv[0][0] * dx + inv[0][1] * dy;
            let src_y = inv[1][0] * dx + inv[1][1] * dy;
            
            // Bilinear interpolation
            if src_x >= 0.0 && src_x < (rgb.width() - 1) as f32 
                && src_y >= 0.0 && src_y < (rgb.height() - 1) as f32 
            {
                let x0 = src_x as u32;
                let y0 = src_y as u32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                
                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;
                
                let p00 = rgb.get_pixel(x0, y0);
                let p01 = rgb.get_pixel(x0, y1);
                let p10 = rgb.get_pixel(x1, y0);
                let p11 = rgb.get_pixel(x1, y1);
                
                let mut pixel = [0u8; 3];
                for c in 0..3 {
                    let v00 = p00[c] as f32;
                    let v01 = p01[c] as f32;
                    let v10 = p10[c] as f32;
                    let v11 = p11[c] as f32;
                    
                    let v = v00 * (1.0 - fx) * (1.0 - fy)
                        + v10 * fx * (1.0 - fy)
                        + v01 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                    
                    pixel[c] = v.clamp(0.0, 255.0) as u8;
                }
                
                output.put_pixel(x, y, Rgb(pixel));
            }
        }
    }
    
    DynamicImage::ImageRgb8(output)
}

/// Decode image from bytes with EXIF orientation handling
/// This ensures images are correctly oriented regardless of how they were captured
pub fn decode_image(data: &[u8]) -> Result<DynamicImage> {
    let image = image::load_from_memory(data)?;
    
    // Try to read EXIF orientation and apply rotation
    let oriented_image = apply_exif_orientation(data, image);
    
    Ok(oriented_image)
}

/// Apply EXIF orientation to correct image rotation
/// Mobile phones often store images with EXIF orientation tags instead of rotating pixels
fn apply_exif_orientation(data: &[u8], image: DynamicImage) -> DynamicImage {
    use std::io::Cursor;
    
    // Try to read EXIF data
    let orientation = match exif::Reader::new().read_from_container(&mut Cursor::new(data)) {
        Ok(exif_data) => {
            exif_data
                .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
                .and_then(|field| field.value.get_uint(0))
                .unwrap_or(1) as u8
        }
        Err(_) => 1, // No EXIF or error reading, assume normal orientation
    };
    
    // Apply transformation based on EXIF orientation value
    // See: https://exiftool.org/TagNames/EXIF.html (Orientation)
    // 1 = Normal (0° rotation)
    // 2 = Flipped horizontally
    // 3 = Rotated 180°
    // 4 = Flipped vertically
    // 5 = Rotated 90° CCW and flipped horizontally
    // 6 = Rotated 90° CW (270° CCW)
    // 7 = Rotated 90° CW and flipped horizontally
    // 8 = Rotated 90° CCW (270° CW)
    match orientation {
        1 => image, // Normal, no transformation needed
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.rotate90().fliph(),
        6 => image.rotate90(),
        7 => image.rotate270().fliph(),
        8 => image.rotate270(),
        _ => image, // Unknown orientation, return as-is
    }
}

/// Calculate resize info for detection post-processing
pub struct ResizeInfo {
    pub scale: f32,
    pub offset_x: u32,
    pub offset_y: u32,
    pub original_width: u32,
    pub original_height: u32,
}

impl ResizeInfo {
    pub fn new(original: (u32, u32), target: (u32, u32)) -> Self {
        let (orig_w, orig_h) = original;
        let (target_w, target_h) = target;
        
        let scale = f32::min(
            target_w as f32 / orig_w as f32,
            target_h as f32 / orig_h as f32,
        );
        
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;
        
        Self {
            scale,
            offset_x: (target_w - new_w) / 2,
            offset_y: (target_h - new_h) / 2,
            original_width: orig_w,
            original_height: orig_h,
        }
    }

    /// Convert detection coordinates back to original image space
    pub fn to_original(&self, x: f32, y: f32) -> (f32, f32) {
        let x = (x - self.offset_x as f32) / self.scale;
        let y = (y - self.offset_y as f32) / self.scale;
        (x, y)
    }
}
