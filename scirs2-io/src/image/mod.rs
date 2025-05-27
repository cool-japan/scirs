//! Image file format support
//!
//! This module provides functionality for reading and writing common image formats.
//! It supports basic operations like loading, saving, and converting between formats.
//!
//! Features:
//! - Reading and writing common image formats (PNG, JPEG, BMP, TIFF)
//! - Basic EXIF metadata extraction (currently limited)
//! - Conversion between different image formats
//! - Basic image properties and information
//! - Image sequence handling and animations (GIF, sequence of images)

use chrono::{DateTime, Utc};
use image::AnimationDecoder;
use ndarray::Array3;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use crate::error::{IoError, Result};

/// Image color mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Grayscale (single channel)
    Grayscale,
    /// RGB color (3 channels)
    RGB,
    /// RGBA color with alpha (4 channels)
    RGBA,
    /// CMYK color (4 channels)
    CMYK,
}

/// Image format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// PNG format
    PNG,
    /// JPEG format
    JPEG,
    /// BMP format
    BMP,
    /// TIFF format
    TIFF,
    /// GIF format
    GIF,
    /// WebP format
    WEBP,
    /// Other/Unknown format
    Other,
}

impl ImageFormat {
    /// Get format from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "png" => ImageFormat::PNG,
            "jpg" | "jpeg" => ImageFormat::JPEG,
            "bmp" => ImageFormat::BMP,
            "tiff" | "tif" => ImageFormat::TIFF,
            "gif" => ImageFormat::GIF,
            "webp" => ImageFormat::WEBP,
            _ => ImageFormat::Other,
        }
    }

    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::PNG => "png",
            ImageFormat::JPEG => "jpg",
            ImageFormat::BMP => "bmp",
            ImageFormat::TIFF => "tiff",
            ImageFormat::GIF => "gif",
            ImageFormat::WEBP => "webp",
            ImageFormat::Other => "unknown",
        }
    }
}

/// Basic image metadata
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Color mode/channels
    pub color_mode: ColorMode,
    /// File format
    pub format: ImageFormat,
    /// File size in bytes
    pub file_size: u64,
    /// EXIF metadata (if available)
    pub exif: Option<ExifMetadata>,
}

/// GPS coordinates extracted from EXIF
#[derive(Debug, Clone, Default)]
pub struct GpsCoordinates {
    /// Latitude in decimal degrees
    pub latitude: Option<f64>,
    /// Longitude in decimal degrees  
    pub longitude: Option<f64>,
    /// Altitude in meters
    pub altitude: Option<f64>,
}

/// Camera settings from EXIF
#[derive(Debug, Clone, Default)]
pub struct CameraSettings {
    /// Camera make
    pub make: Option<String>,
    /// Camera model
    pub model: Option<String>,
    /// Lens model
    pub lens_model: Option<String>,
    /// ISO sensitivity
    pub iso: Option<u32>,
    /// Aperture (f-number)
    pub aperture: Option<f64>,
    /// Shutter speed
    pub shutter_speed: Option<f64>,
    /// Focal length in mm
    pub focal_length: Option<f64>,
    /// Flash fired
    pub flash: Option<bool>,
    /// White balance setting
    pub white_balance: Option<String>,
    /// Exposure mode
    pub exposure_mode: Option<String>,
    /// Metering mode
    pub metering_mode: Option<String>,
}

/// EXIF metadata
#[derive(Debug, Clone, Default)]
pub struct ExifMetadata {
    /// Date and time photo was taken
    pub datetime: Option<DateTime<Utc>>,
    /// GPS coordinates
    pub gps: Option<GpsCoordinates>,
    /// Camera settings
    pub camera: CameraSettings,
    /// Image orientation
    pub orientation: Option<u32>,
    /// Software used
    pub software: Option<String>,
    /// Copyright information
    pub copyright: Option<String>,
    /// Artist/photographer
    pub artist: Option<String>,
    /// Image description
    pub description: Option<String>,
    /// Raw EXIF tags
    pub raw_tags: std::collections::HashMap<String, String>,
}

/// Image data container
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Image data as ndarray
    pub data: Array3<u8>,
    /// Image metadata
    pub metadata: ImageMetadata,
}

/// Animation/sequence data
#[derive(Debug, Clone)]
pub struct AnimationData {
    /// Frames as Vec of ImageData
    pub frames: Vec<ImageData>,
    /// Frame delays in milliseconds
    pub delays: Vec<u32>,
    /// Loop count (0 = infinite)
    pub loop_count: u32,
}

/// Load image from file
///
/// # Arguments
/// * `path` - Path to image file
///
/// # Returns
/// Result containing ImageData with pixel data and metadata
///
/// # Example
/// ```no_run
/// use scirs2_io::image::load_image;
///
/// let image_data = load_image("photo.jpg")?;
/// println!("Image size: {}x{}", image_data.metadata.width, image_data.metadata.height);
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<ImageData> {
    let path = path.as_ref();
    let img = image::open(path).map_err(|e| IoError::FileError(e.to_string()))?;

    let width = img.width();
    let height = img.height();
    let format =
        ImageFormat::from_extension(path.extension().and_then(|ext| ext.to_str()).unwrap_or(""));

    let file_size = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    // Convert to RGB
    let rgb_img = img.to_rgb8();
    let raw_data = rgb_img.into_raw();

    // Convert to ndarray
    let data = Array3::from_shape_vec((height as usize, width as usize, 3), raw_data)
        .map_err(|e| IoError::FormatError(e.to_string()))?;

    // Try to read EXIF metadata
    let exif = read_exif_metadata(path)?;

    let metadata = ImageMetadata {
        width,
        height,
        color_mode: ColorMode::RGB,
        format,
        file_size,
        exif,
    };

    Ok(ImageData { data, metadata })
}

/// Save image to file
///
/// # Arguments
/// * `image_data` - ImageData to save
/// * `path` - Output file path
/// * `format` - Output format (optional, inferred from path if None)
///
/// # Example
/// ```no_run
/// use scirs2_io::image::{load_image, save_image, ImageFormat};
///
/// let image_data = load_image("input.jpg")?;
/// save_image(&image_data, "output.png", Some(ImageFormat::PNG))?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn save_image<P: AsRef<Path>>(
    image_data: &ImageData,
    path: P,
    format: Option<ImageFormat>,
) -> Result<()> {
    let path = path.as_ref();
    let format = format.unwrap_or_else(|| {
        ImageFormat::from_extension(path.extension().and_then(|ext| ext.to_str()).unwrap_or(""))
    });

    let (height, width, _) = image_data.data.dim();
    let raw_data = image_data.data.iter().cloned().collect::<Vec<u8>>();

    let img_buffer = image::RgbImage::from_raw(width as u32, height as u32, raw_data)
        .ok_or_else(|| IoError::FormatError("Invalid image dimensions".to_string()))?;

    let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);

    match format {
        ImageFormat::PNG => dynamic_img.save_with_format(path, image::ImageFormat::Png),
        ImageFormat::JPEG => dynamic_img.save_with_format(path, image::ImageFormat::Jpeg),
        ImageFormat::BMP => dynamic_img.save_with_format(path, image::ImageFormat::Bmp),
        ImageFormat::TIFF => dynamic_img.save_with_format(path, image::ImageFormat::Tiff),
        ImageFormat::GIF => dynamic_img.save_with_format(path, image::ImageFormat::Gif),
        ImageFormat::WEBP => dynamic_img.save_with_format(path, image::ImageFormat::WebP),
        ImageFormat::Other => return Err(IoError::FormatError("Unknown format".to_string())),
    }
    .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

/// Convert image to different format
///
/// # Arguments
/// * `input_path` - Input image path
/// * `output_path` - Output image path
/// * `target_format` - Target format
///
/// # Example
/// ```no_run
/// use scirs2_io::image::{convert_image, ImageFormat};
///
/// convert_image("photo.jpg", "photo.png", ImageFormat::PNG)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn convert_image<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    target_format: ImageFormat,
) -> Result<()> {
    let image_data = load_image(input_path)?;
    save_image(&image_data, output_path, Some(target_format))
}

/// Resize image
///
/// # Arguments
/// * `image_data` - Input image data
/// * `new_width` - New width in pixels
/// * `new_height` - New height in pixels
///
/// # Returns
/// New ImageData with resized image
///
/// # Example
/// ```no_run
/// use scirs2_io::image::{load_image, resize_image};
///
/// let image_data = load_image("large_photo.jpg")?;
/// let resized = resize_image(&image_data, 800, 600)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn resize_image(image_data: &ImageData, new_width: u32, new_height: u32) -> Result<ImageData> {
    let (height, width, _) = image_data.data.dim();
    let raw_data = image_data.data.iter().cloned().collect::<Vec<u8>>();

    let img_buffer = image::RgbImage::from_raw(width as u32, height as u32, raw_data)
        .ok_or_else(|| IoError::FormatError("Invalid image dimensions".to_string()))?;

    let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);
    let resized_img =
        dynamic_img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
    let rgb_img = resized_img.to_rgb8();
    let resized_raw = rgb_img.into_raw();

    let resized_data =
        Array3::from_shape_vec((new_height as usize, new_width as usize, 3), resized_raw)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

    let mut new_metadata = image_data.metadata.clone();
    new_metadata.width = new_width;
    new_metadata.height = new_height;

    Ok(ImageData {
        data: resized_data,
        metadata: new_metadata,
    })
}

/// Get basic image information without loading full data
///
/// # Arguments
/// * `path` - Path to image file
///
/// # Returns
/// ImageMetadata with basic information
///
/// # Example
/// ```no_run
/// use scirs2_io::image::get_image_info;
///
/// let info = get_image_info("photo.jpg")?;
/// println!("Image: {}x{} pixels", info.width, info.height);
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn get_image_info<P: AsRef<Path>>(path: P) -> Result<ImageMetadata> {
    let path = path.as_ref();
    let reader = image::io::Reader::open(path).map_err(|e| IoError::FileError(e.to_string()))?;

    let reader = reader
        .with_guessed_format()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let dimensions = reader
        .into_dimensions()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let format =
        ImageFormat::from_extension(path.extension().and_then(|ext| ext.to_str()).unwrap_or(""));

    let file_size = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    // Try to read EXIF metadata
    let exif = read_exif_metadata(path)?;

    Ok(ImageMetadata {
        width: dimensions.0,
        height: dimensions.1,
        color_mode: ColorMode::RGB, // Default assumption
        format,
        file_size,
        exif,
    })
}

/// Load animated image/GIF
///
/// # Arguments
/// * `path` - Path to animated image file
///
/// # Returns
/// AnimationData with all frames and timing information
///
/// # Example
/// ```no_run
/// use scirs2_io::image::load_animation;
///
/// let animation = load_animation("animated.gif")?;
/// println!("Animation has {} frames", animation.frames.len());
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn load_animation<P: AsRef<Path>>(path: P) -> Result<AnimationData> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let decoder = image::codecs::gif::GifDecoder::new(reader)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let mut frames = Vec::new();
    let mut delays = Vec::new();

    for frame_result in decoder.into_frames() {
        let frame = frame_result.map_err(|e| IoError::FileError(e.to_string()))?;
        let delay = frame.delay().numer_denom_ms().0;
        let image = frame.into_buffer();

        let width = image.width();
        let height = image.height();
        let raw_data = image.into_raw();

        // Convert to RGB if needed
        let rgb_data = if raw_data.len() == (width * height * 4) as usize {
            // RGBA to RGB conversion
            raw_data
                .chunks(4)
                .flat_map(|rgba| &rgba[..3])
                .cloned()
                .collect()
        } else {
            raw_data
        };

        let data = Array3::from_shape_vec((height as usize, width as usize, 3), rgb_data)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

        let metadata = ImageMetadata {
            width,
            height,
            color_mode: ColorMode::RGB,
            format: ImageFormat::GIF,
            file_size: 0, // Unknown for individual frames
            exif: None,
        };

        frames.push(ImageData { data, metadata });
        delays.push(delay);
    }

    Ok(AnimationData {
        frames,
        delays,
        loop_count: 0, // Assume infinite loop
    })
}

/// Read EXIF metadata from image file
///
/// # Arguments
/// * `path` - Path to image file
///
/// # Returns
/// Optional ExifMetadata if EXIF data exists and can be read
///
/// # Example
/// ```no_run
/// use scirs2_io::image::read_exif_metadata;
///
/// if let Some(exif) = read_exif_metadata("photo.jpg")? {
///     if let Some(gps) = exif.gps {
///         println!("GPS: {}, {}", gps.latitude.unwrap_or(0.0), gps.longitude.unwrap_or(0.0));
///     }
/// }
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn read_exif_metadata<P: AsRef<Path>>(_path: P) -> Result<Option<ExifMetadata>> {
    // TODO: Implement EXIF reading functionality
    // Currently disabled due to dependency issues with kamadak-exif
    Ok(None)
}

/// Extract all images from a directory matching pattern
///
/// # Arguments
/// * `dir_path` - Directory to search
/// * `pattern` - File pattern (e.g., "*.jpg")
/// * `recursive` - Whether to search subdirectories
///
/// # Returns
/// Vector of image file paths
///
/// # Example
/// ```no_run
/// use scirs2_io::image::find_images;
///
/// let images = find_images("./photos", "*.{jpg,png}", true)?;
/// for image_path in images {
///     println!("Found: {}", image_path.display());
/// }
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn find_images<P: AsRef<Path>>(
    dir_path: P,
    pattern: &str,
    recursive: bool,
) -> Result<Vec<std::path::PathBuf>> {
    let search_pattern = if recursive {
        format!("{}/**/{}", dir_path.as_ref().display(), pattern)
    } else {
        format!("{}/{}", dir_path.as_ref().display(), pattern)
    };

    let paths = glob::glob(&search_pattern)
        .map_err(|e| IoError::FileError(e.to_string()))?
        .filter_map(|entry| entry.ok())
        .filter(|path| {
            let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
            matches!(
                ext.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "tif" | "gif" | "webp"
            )
        })
        .collect();

    Ok(paths)
}

/// Batch process images in directory
///
/// # Arguments
/// * `input_dir` - Input directory path
/// * `output_dir` - Output directory path
/// * `processor` - Function to process each image
///
/// # Example
/// ```no_run
/// use scirs2_io::image::{batch_process_images, ImageData, resize_image};
///
/// batch_process_images(
///     "./input",
///     "./output",
///     |image_data| resize_image(image_data, 800, 600)
/// )?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn batch_process_images<P1, P2, F>(input_dir: P1, output_dir: P2, processor: F) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    F: Fn(&ImageData) -> Result<ImageData>,
{
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();

    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).map_err(|e| IoError::FileError(e.to_string()))?;

    let image_files = find_images(input_dir, "*", false)?;

    for input_path in image_files {
        let file_name = input_path
            .file_name()
            .ok_or_else(|| IoError::FileError("Invalid file name".to_string()))?;
        let output_path = output_dir.join(file_name);

        let image_data = load_image(&input_path)?;
        let processed_data = processor(&image_data)?;
        save_image(&processed_data, &output_path, None)?;

        println!(
            "Processed: {} -> {}",
            input_path.display(),
            output_path.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("jpg"), ImageFormat::JPEG);
        assert_eq!(ImageFormat::from_extension("png"), ImageFormat::PNG);
        assert_eq!(ImageFormat::from_extension("unknown"), ImageFormat::Other);
    }

    #[test]
    fn test_image_format_extension() {
        assert_eq!(ImageFormat::JPEG.extension(), "jpg");
        assert_eq!(ImageFormat::PNG.extension(), "png");
    }

    #[test]
    fn test_color_mode() {
        let grayscale = ColorMode::Grayscale;
        let rgb = ColorMode::RGB;
        assert_ne!(grayscale, rgb);
    }

    #[test]
    fn test_metadata_creation() {
        let metadata = ImageMetadata {
            width: 800,
            height: 600,
            color_mode: ColorMode::RGB,
            format: ImageFormat::JPEG,
            file_size: 1024,
            exif: None,
        };

        assert_eq!(metadata.width, 800);
        assert_eq!(metadata.height, 600);
    }
}
