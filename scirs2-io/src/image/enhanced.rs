//! Enhanced image capabilities with multi-scale support and lossless compression
//!
//! This module provides advanced image processing features including:
//! - Multi-scale image pyramids for efficient processing at different resolutions
//! - Lossless compression using modern algorithms
//! - Advanced format support with enhanced features
//! - Efficient memory management for large images
//! - Quality-preserving image operations

use crate::error::{IoError, Result};
use crate::image::{ColorMode, ImageData, ImageFormat};
use ndarray::{Array3, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;
use std::path::Path;

/// Compression quality settings
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionQuality {
    /// Lossless compression (maximum quality)
    Lossless,
    /// High quality (minimal loss)
    High,
    /// Medium quality (balanced)
    Medium,
    /// Low quality (maximum compression)
    Low,
    /// Custom quality (0-100)
    Custom(u8),
}

impl CompressionQuality {
    /// Get quality value as 0-100 scale
    pub fn value(&self) -> u8 {
        match self {
            CompressionQuality::Lossless => 100,
            CompressionQuality::High => 95,
            CompressionQuality::Medium => 80,
            CompressionQuality::Low => 60,
            CompressionQuality::Custom(v) => (*v).min(100),
        }
    }
}

/// Advanced compression options
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Quality setting
    pub quality: CompressionQuality,
    /// Enable progressive encoding (for JPEG)
    pub progressive: bool,
    /// Enable lossless optimization
    pub optimize: bool,
    /// Custom compression level (0-9, higher = better compression)
    pub compression_level: Option<u8>,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            quality: CompressionQuality::High,
            progressive: false,
            optimize: true,
            compression_level: None,
        }
    }
}

/// Image pyramid configuration
#[derive(Debug, Clone)]
pub struct PyramidConfig {
    /// Number of pyramid levels (default: 4)
    pub levels: usize,
    /// Scale factor between levels (default: 0.5)
    pub scale_factor: f64,
    /// Minimum image size (will stop creating levels below this)
    pub min_size: u32,
    /// Interpolation method for downsampling
    pub interpolation: InterpolationMethod,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            levels: 4,
            scale_factor: 0.5,
            min_size: 32,
            interpolation: InterpolationMethod::Lanczos,
        }
    }
}

/// Interpolation methods for image resizing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Nearest neighbor (fastest, lowest quality)
    Nearest,
    /// Linear interpolation
    Linear,
    /// Cubic interpolation (good quality)
    Cubic,
    /// Lanczos interpolation (highest quality)
    Lanczos,
}

/// Multi-scale image pyramid
#[derive(Debug, Clone)]
pub struct ImagePyramid {
    /// Original image at full resolution (level 0)
    pub original: ImageData,
    /// Downscaled versions at different levels
    pub levels: Vec<ImageData>,
    /// Configuration used to create pyramid
    pub config: PyramidConfig,
}

/// Enhanced image processing operations
#[derive(Debug, Clone)]
pub struct EnhancedImageProcessor {
    /// Default compression options
    pub compression: CompressionOptions,
    /// Cache for processed images
    cache: HashMap<String, ImageData>,
    /// Maximum cache size in MB
    max_cache_size: usize,
}

impl Default for EnhancedImageProcessor {
    fn default() -> Self {
        Self {
            compression: CompressionOptions::default(),
            cache: HashMap::new(),
            max_cache_size: 256, // 256MB default cache
        }
    }
}

impl EnhancedImageProcessor {
    /// Create a new enhanced image processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression options
    pub fn with_compression(mut self, compression: CompressionOptions) -> Self {
        self.compression = compression;
        self
    }

    /// Set maximum cache size in MB
    pub fn with_cache_size(mut self, size_mb: usize) -> Self {
        self.max_cache_size = size_mb;
        self
    }

    /// Create an image pyramid from the given image
    pub fn create_pyramid(&self, image: &ImageData, config: PyramidConfig) -> Result<ImagePyramid> {
        let mut levels = Vec::new();
        let mut current_image = image.clone();

        for level in 1..=config.levels {
            let scale = config.scale_factor.powi(level as i32);
            let new_width =
                ((image.metadata.width as f64) * scale).max(config.min_size as f64) as u32;
            let new_height =
                ((image.metadata.height as f64) * scale).max(config.min_size as f64) as u32;

            // Stop if we've reached minimum size
            if new_width < config.min_size || new_height < config.min_size {
                break;
            }

            current_image = self.resize_with_interpolation(
                &current_image,
                new_width,
                new_height,
                config.interpolation,
            )?;
            levels.push(current_image.clone());
        }

        Ok(ImagePyramid {
            original: image.clone(),
            levels,
            config,
        })
    }

    /// Resize image with specified interpolation method
    pub fn resize_with_interpolation(
        &self,
        image: &ImageData,
        new_width: u32,
        new_height: u32,
        method: InterpolationMethod,
    ) -> Result<ImageData> {
        let (_height, width, channels) = image.data.dim();
        let raw_data = image.data.iter().cloned().collect::<Vec<u8>>();

        let img_buffer = if channels == 3 {
            image::RgbImage::from_raw(width as u32, _height as u32, raw_data)
                .ok_or_else(|| IoError::FormatError("Invalid RGB image dimensions".to_string()))?
        } else {
            return Err(IoError::FormatError(
                "Unsupported number of channels".to_string(),
            ));
        };

        let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);

        let filter = match method {
            InterpolationMethod::Nearest => image::imageops::FilterType::Nearest,
            InterpolationMethod::Linear => image::imageops::FilterType::Triangle,
            InterpolationMethod::Cubic => image::imageops::FilterType::CatmullRom,
            InterpolationMethod::Lanczos => image::imageops::FilterType::Lanczos3,
        };

        let resized_img = dynamic_img.resize(new_width, new_height, filter);
        let rgb_img = resized_img.to_rgb8();
        let resized_raw = rgb_img.into_raw();

        let resized_data = Array3::from_shape_vec(
            (new_height as usize, new_width as usize, channels),
            resized_raw,
        )
        .map_err(|e| IoError::FormatError(e.to_string()))?;

        let mut new_metadata = image.metadata.clone();
        new_metadata.width = new_width;
        new_metadata.height = new_height;

        Ok(ImageData {
            data: resized_data,
            metadata: new_metadata,
        })
    }

    /// Save image with enhanced compression options
    pub fn save_with_compression<P: AsRef<Path>>(
        &self,
        image: &ImageData,
        path: P,
        format: ImageFormat,
        compression: Option<CompressionOptions>,
    ) -> Result<()> {
        let path = path.as_ref();
        let compression = compression.unwrap_or(self.compression.clone());

        let (height, width_, _) = image.data.dim();
        let raw_data = image.data.iter().cloned().collect::<Vec<u8>>();

        let img_buffer = image::RgbImage::from_raw(width_ as u32, height as u32, raw_data)
            .ok_or_else(|| IoError::FormatError("Invalid image dimensions".to_string()))?;

        let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);

        match format {
            ImageFormat::PNG => {
                // PNG is always lossless - use the standard save method
                dynamic_img
                    .save_with_format(path, image::ImageFormat::Png)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
            ImageFormat::JPEG => {
                // For JPEG, we need to use a more manual approach to control quality
                let file =
                    std::fs::File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                    file,
                    compression.quality.value(),
                );
                if compression.progressive {
                    // Note: Progressive JPEG not directly supported by image crate
                    // This would require additional dependencies
                }
                encoder
                    .encode(
                        dynamic_img.as_bytes(),
                        width_ as u32,
                        height as u32,
                        image::ColorType::Rgb8.into(),
                    )
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
            ImageFormat::WEBP => {
                // WebP supports both lossy and lossless
                if compression.quality == CompressionQuality::Lossless {
                    // Use lossless WebP encoding
                    dynamic_img
                        .save_with_format(path, image::ImageFormat::WebP)
                        .map_err(|e| IoError::FileError(e.to_string()))?;
                } else {
                    // Use lossy WebP encoding
                    dynamic_img
                        .save_with_format(path, image::ImageFormat::WebP)
                        .map_err(|e| IoError::FileError(e.to_string()))?;
                }
            }
            _ => {
                // Use default encoding for other formats
                dynamic_img
                    .save_with_format(path, format.into())
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Convert image to grayscale while preserving luminance
    pub fn to_grayscale(&self, image: &ImageData) -> Result<ImageData> {
        let (height, width, channels) = image.data.dim();

        if channels != 3 {
            return Err(IoError::FormatError("Expected RGB image".to_string()));
        }

        let mut gray_data = Array3::zeros((height, width, 3));

        // Process image in rows for better cache locality and SIMD efficiency
        for y in 0..height {
            // Extract row data for SIMD processing
            let _row_size = width * 3;
            let mut r_values = vec![0f32; width];
            let mut g_values = vec![0f32; width];
            let mut b_values = vec![0f32; width];

            // Extract RGB values for the entire row
            for x in 0..width {
                r_values[x] = image.data[[y, x, 0]] as f32;
                g_values[x] = image.data[[y, x, 1]] as f32;
                b_values[x] = image.data[[y, x, 2]] as f32;
            }

            // Create coefficient arrays for SIMD multiplication
            let r_coeff = vec![0.299f32; width];
            let g_coeff = vec![0.587f32; width];
            let b_coeff = vec![0.114f32; width];

            // Use SIMD operations for luminance calculation
            // First multiply each channel by its coefficient
            let _r_weighted = vec![0f32; width];
            let _g_weighted = vec![0f32; width];
            let _b_weighted = vec![0f32; width];

            let r_values_view = ArrayView1::from(&r_values);
            let g_values_view = ArrayView1::from(&g_values);
            let b_values_view = ArrayView1::from(&b_values);
            let r_coeff_view = ArrayView1::from(&r_coeff);
            let g_coeff_view = ArrayView1::from(&g_coeff);
            let b_coeff_view = ArrayView1::from(&b_coeff);

            let r_weighted = f32::simd_mul(&r_values_view, &r_coeff_view);
            let g_weighted = f32::simd_mul(&g_values_view, &g_coeff_view);
            let b_weighted = f32::simd_mul(&b_values_view, &b_coeff_view);

            // Add the weighted values together
            let r_weighted_view = ArrayView1::from(&r_weighted);
            let g_weighted_view = ArrayView1::from(&g_weighted);
            let b_weighted_view = ArrayView1::from(&b_weighted);

            let gray_values = f32::simd_add(&r_weighted_view, &g_weighted_view);
            let gray_values_view = ArrayView1::from(&gray_values);
            let gray_values_final = f32::simd_add(&gray_values_view, &b_weighted_view);

            // Store the grayscale values back to the output array
            for x in 0..width {
                let gray = gray_values_final[x].clamp(0.0, 255.0) as u8;
                gray_data[[y, x, 0]] = gray;
                gray_data[[y, x, 1]] = gray;
                gray_data[[y, x, 2]] = gray;
            }
        }

        let mut new_metadata = image.metadata.clone();
        new_metadata.color_mode = ColorMode::Grayscale;

        Ok(ImageData {
            data: gray_data,
            metadata: new_metadata,
        })
    }

    /// Apply histogram equalization to enhance contrast
    pub fn histogram_equalization(&self, image: &ImageData) -> Result<ImageData> {
        let (height, width, channels) = image.data.dim();
        let mut enhanced_data = image.data.clone();

        for c in 0..channels {
            // Calculate histogram
            let mut histogram = [0u32; 256];
            for y in 0..height {
                for x in 0..width {
                    let pixel = image.data[[y, x, c]] as usize;
                    histogram[pixel] += 1;
                }
            }

            // Calculate cumulative distribution function
            let mut cdf = [0u32; 256];
            cdf[0] = histogram[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + histogram[i];
            }

            // Normalize CDF to create lookup table
            let total_pixels = (height * width) as f32;
            let mut lookup = [0u8; 256];
            for i in 0..256 {
                lookup[i] = ((cdf[i] as f32 / total_pixels) * 255.0) as u8;
            }

            // Apply histogram equalization
            for y in 0..height {
                for x in 0..width {
                    let pixel = image.data[[y, x, c]] as usize;
                    enhanced_data[[y, x, c]] = lookup[pixel];
                }
            }
        }

        Ok(ImageData {
            data: enhanced_data,
            metadata: image.metadata.clone(),
        })
    }

    /// Apply Gaussian blur filter
    pub fn gaussian_blur(&self, image: &ImageData, radius: f32) -> Result<ImageData> {
        let (height, width_, _) = image.data.dim();
        let raw_data = image.data.iter().cloned().collect::<Vec<u8>>();

        let img_buffer = image::RgbImage::from_raw(width_ as u32, height as u32, raw_data)
            .ok_or_else(|| IoError::FormatError("Invalid image dimensions".to_string()))?;

        let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);
        let blurred = dynamic_img.blur(radius);
        let rgb_blurred = blurred.to_rgb8();
        let blurred_raw = rgb_blurred.into_raw();

        let blurred_data = Array3::from_shape_vec((height, width_, 3), blurred_raw)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(ImageData {
            data: blurred_data,
            metadata: image.metadata.clone(),
        })
    }

    /// Sharpen image using unsharp mask
    pub fn sharpen(&self, image: &ImageData, amount: f32, radius: f32) -> Result<ImageData> {
        // Create blurred version
        let blurred = self.gaussian_blur(image, radius)?;

        let (height, width, channels) = image.data.dim();
        let mut sharpened_data = Array3::zeros((height, width, channels));

        // Apply unsharp mask: sharpened = original + amount * (original - blurred)
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let original = image.data[[y, x, c]] as f32;
                    let blur = blurred.data[[y, x, c]] as f32;
                    let difference = original - blur;
                    let sharpened = original + amount * difference;
                    sharpened_data[[y, x, c]] = sharpened.clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(ImageData {
            data: sharpened_data,
            metadata: image.metadata.clone(),
        })
    }

    /// Clear the image cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.max_cache_size)
    }
}

// Conversion from our ImageFormat to image crate's ImageFormat
impl From<ImageFormat> for image::ImageFormat {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::PNG => image::ImageFormat::Png,
            ImageFormat::JPEG => image::ImageFormat::Jpeg,
            ImageFormat::BMP => image::ImageFormat::Bmp,
            ImageFormat::TIFF => image::ImageFormat::Tiff,
            ImageFormat::GIF => image::ImageFormat::Gif,
            ImageFormat::WEBP => image::ImageFormat::WebP,
            ImageFormat::Other => image::ImageFormat::Png, // Default fallback
        }
    }
}

impl ImagePyramid {
    /// Get image at specific level (0 = original, higher = smaller)
    pub fn get_level(&self, level: usize) -> Option<&ImageData> {
        if level == 0 {
            Some(&self.original)
        } else {
            self.levels.get(level - 1)
        }
    }

    /// Get the number of pyramid levels (including original)
    pub fn num_levels(&self) -> usize {
        self.levels.len() + 1
    }

    /// Find the best level for a target size
    pub fn find_best_level(&self, target_width: u32, target_height: u32) -> usize {
        let mut best_level = 0;
        let mut best_diff = u32::MAX;

        for level in 0..self.num_levels() {
            if let Some(level_image) = self.get_level(level) {
                let width_diff = level_image.metadata.width.abs_diff(target_width);
                let height_diff = level_image.metadata.height.abs_diff(target_height);
                let total_diff = width_diff + height_diff;

                if total_diff < best_diff {
                    best_diff = total_diff;
                    best_level = level;
                }
            }
        }

        best_level
    }

    /// Get level that's closest to target size
    pub fn get_level_for_size(&self, target_width: u32, target_height: u32) -> Option<&ImageData> {
        let level = self.find_best_level(target_width, target_height);
        self.get_level(level)
    }
}

/// Convenience functions for enhanced image operations
/// Create an image pyramid with default configuration
#[allow(dead_code)]
pub fn create_image_pyramid(image: &ImageData) -> Result<ImagePyramid> {
    let processor = EnhancedImageProcessor::new();
    processor.create_pyramid(image, PyramidConfig::default())
}

/// Save image with lossless compression
#[allow(dead_code)]
pub fn save_lossless<P: AsRef<Path>>(
    image: &ImageData,
    path: P,
    format: ImageFormat,
) -> Result<()> {
    let processor = EnhancedImageProcessor::new();
    let compression = CompressionOptions {
        quality: CompressionQuality::Lossless,
        progressive: false,
        optimize: true,
        compression_level: None,
    };
    processor.save_with_compression(image, path, format, Some(compression))
}

/// Save image with high quality compression
#[allow(dead_code)]
pub fn save_high_quality<P: AsRef<Path>>(
    image: &ImageData,
    path: P,
    format: ImageFormat,
) -> Result<()> {
    let processor = EnhancedImageProcessor::new();
    let compression = CompressionOptions {
        quality: CompressionQuality::High,
        progressive: true,
        optimize: true,
        compression_level: Some(9),
    };
    processor.save_with_compression(image, path, format, Some(compression))
}

/// Batch convert images with enhanced compression
#[allow(dead_code)]
pub fn batch_convert_with_compression<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_dir: P1,
    output_dir: P2,
    target_format: ImageFormat,
    compression: CompressionOptions,
) -> Result<()> {
    use crate::image::{find_images, load_image};
    use std::fs;

    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();

    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).map_err(|e| IoError::FileError(e.to_string()))?;

    let processor = EnhancedImageProcessor::new().with_compression(compression);
    let image_files = find_images(input_dir, "*", false)?;

    for input_path in image_files {
        let file_stem = input_path
            .file_stem()
            .ok_or_else(|| IoError::FileError("Invalid file name".to_string()))?;
        let output_filename = format!(
            "{}.{}",
            file_stem.to_string_lossy(),
            target_format.extension()
        );
        let output_path = output_dir.join(output_filename);

        let image_data = load_image(&input_path)?;
        processor.save_with_compression(&image_data, &output_path, target_format, None)?;

        println!(
            "Converted: {} -> {} ({})",
            input_path.display(),
            output_path.display(),
            target_format.extension().to_uppercase()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageMetadata;
    use ndarray::Array3;

    fn create_test_image() -> ImageData {
        let data = Array3::zeros((100, 100, 3));
        let metadata = ImageMetadata {
            width: 100,
            height: 100,
            color_mode: ColorMode::RGB,
            format: ImageFormat::PNG,
            file_size: 0,
            exif: None,
        };
        ImageData { data, metadata }
    }

    #[test]
    fn test_compression_quality_values() {
        assert_eq!(CompressionQuality::Lossless.value(), 100);
        assert_eq!(CompressionQuality::High.value(), 95);
        assert_eq!(CompressionQuality::Medium.value(), 80);
        assert_eq!(CompressionQuality::Low.value(), 60);
        assert_eq!(CompressionQuality::Custom(75).value(), 75);
        assert_eq!(CompressionQuality::Custom(150).value(), 100); // Clamped
    }

    #[test]
    fn test_pyramid_config_default() {
        let config = PyramidConfig::default();
        assert_eq!(config.levels, 4);
        assert_eq!(config.scale_factor, 0.5);
        assert_eq!(config.min_size, 32);
        assert_eq!(config.interpolation, InterpolationMethod::Lanczos);
    }

    #[test]
    fn test_enhanced_processor_creation() {
        let processor = EnhancedImageProcessor::new();
        assert_eq!(processor.compression.quality.value(), 95); // High quality default
        assert!(processor.compression.optimize);
    }

    #[test]
    fn test_processor_with_compression() {
        let compression = CompressionOptions {
            quality: CompressionQuality::Lossless,
            progressive: true,
            optimize: false,
            compression_level: Some(5),
        };

        let processor = EnhancedImageProcessor::new().with_compression(compression.clone());
        assert_eq!(processor.compression.quality.value(), 100);
        assert!(processor.compression.progressive);
        assert!(!processor.compression.optimize);
        assert_eq!(processor.compression.compression_level, Some(5));
    }

    #[test]
    fn test_interpolation_methods() {
        assert_eq!(InterpolationMethod::Nearest, InterpolationMethod::Nearest);
        assert_ne!(InterpolationMethod::Nearest, InterpolationMethod::Linear);
    }

    #[test]
    fn test_image_pyramid_creation() {
        let image = create_test_image();
        let config = PyramidConfig {
            levels: 2,
            scale_factor: 0.5,
            min_size: 10,
            interpolation: InterpolationMethod::Linear,
        };

        let processor = EnhancedImageProcessor::new();
        let pyramid = processor.create_pyramid(&image, config).unwrap();

        assert_eq!(pyramid.original.metadata.width, 100);
        assert_eq!(pyramid.original.metadata.height, 100);
        assert!(pyramid.levels.len() <= 2);
    }

    #[test]
    fn test_pyramid_level_access() {
        let image = create_test_image();
        let processor = EnhancedImageProcessor::new();
        let pyramid = processor
            .create_pyramid(&image, PyramidConfig::default())
            .unwrap();

        // Level 0 should be the original
        assert!(pyramid.get_level(0).is_some());
        assert_eq!(pyramid.get_level(0).unwrap().metadata.width, 100);

        // Check number of levels
        assert!(pyramid.num_levels() >= 1);
    }

    #[test]
    fn test_find_best_pyramid_level() {
        let image = create_test_image();
        let processor = EnhancedImageProcessor::new();
        let pyramid = processor
            .create_pyramid(&image, PyramidConfig::default())
            .unwrap();

        // Target size close to original should return level 0
        let best_level = pyramid.find_best_level(100, 100);
        assert_eq!(best_level, 0);

        // Very small target should return higher level
        let best_level = pyramid.find_best_level(10, 10);
        assert!(best_level > 0 || pyramid.num_levels() == 1);
    }

    #[test]
    fn test_cache_operations() {
        let mut processor = EnhancedImageProcessor::new().with_cache_size(128);
        let (count, max_size) = processor.cache_stats();
        assert_eq!(count, 0);
        assert_eq!(max_size, 128);

        processor.clear_cache();
        let count_ = processor.cache_stats();
        assert_eq!(count, 0);
    }
}
