//! Image utility functions

use image::DynamicImage;
use anyhow::Result;

/// Decode image from bytes (JPEG, PNG, etc.)
pub fn decode_image(data: &[u8]) -> Result<DynamicImage> {
    let img = image::load_from_memory(data)?;
    Ok(img)
}

/// Encode image to JPEG bytes
pub fn encode_jpeg(image: &DynamicImage, _quality: u8) -> Result<Vec<u8>> {
    let mut buffer = std::io::Cursor::new(Vec::new());
    image.write_to(&mut buffer, image::ImageFormat::Jpeg)?;
    Ok(buffer.into_inner())
}

/// Encode image to PNG bytes
pub fn encode_png(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut buffer = std::io::Cursor::new(Vec::new());
    image.write_to(&mut buffer, image::ImageFormat::Png)?;
    Ok(buffer.into_inner())
}

/// Resize image maintaining aspect ratio
pub fn resize_with_aspect(image: &DynamicImage, max_width: u32, max_height: u32) -> DynamicImage {
    image.resize(max_width, max_height, image::imageops::FilterType::Lanczos3)
}
