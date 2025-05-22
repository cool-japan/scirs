//! Image file format support
//!
//! This module provides functionality for reading and writing common image formats.
//! It supports basic operations like loading, saving, and converting between formats.
//!
//! Features:
//! - Reading and writing common image formats (PNG, JPEG, BMP, TIFF)
//! - EXIF metadata extraction from JPEG and TIFF files
//! - GPS data extraction from EXIF
//! - Camera settings and technical metadata
//! - Conversion between different image formats
//! - Basic image properties and information
//! - Image sequence handling and animations (GIF, sequence of images)

use image::{AnimationDecoder, ImageDecoder};
use ndarray::{Array2, Array3, Array4};
use regex::Regex;
use std::fs;
use std::path::Path;
use std::io::BufReader;
use chrono::{DateTime, NaiveDateTime, Utc};
// extern crate kamadak_exif; // Temporarily disabled

use crate::error::{IoError, Result};

/// Image color mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Grayscale (single channel)
    Grayscale,
    /// RGB (3 channels)
    RGB,
    /// RGBA (4 channels with alpha/transparency)
    RGBA,
}

impl ColorMode {
    /// Get the number of channels for this color mode
    pub fn channels(&self) -> usize {
        match self {
            ColorMode::Grayscale => 1,
            ColorMode::RGB => 3,
            ColorMode::RGBA => 4,
        }
    }

    /// Get a string representation of the color mode
    pub fn as_str(&self) -> &'static str {
        match self {
            ColorMode::Grayscale => "grayscale",
            ColorMode::RGB => "rgb",
            ColorMode::RGBA => "rgba",
        }
    }
}

/// GPS coordinates from EXIF data
#[derive(Debug, Clone)]
pub struct GpsCoordinates {
    /// Latitude in decimal degrees
    pub latitude: f64,
    /// Longitude in decimal degrees
    pub longitude: f64,
    /// Altitude in meters (if available)
    pub altitude: Option<f64>,
    /// Latitude reference (N/S)
    pub latitude_ref: String,
    /// Longitude reference (E/W)
    pub longitude_ref: String,
}

/// Camera settings from EXIF data
#[derive(Debug, Clone, Default)]
pub struct CameraSettings {
    /// Camera make (manufacturer)
    pub make: Option<String>,
    /// Camera model
    pub model: Option<String>,
    /// Lens model
    pub lens_model: Option<String>,
    /// ISO speed
    pub iso: Option<u32>,
    /// Aperture value (f-number)
    pub aperture: Option<f64>,
    /// Shutter speed in seconds
    pub shutter_speed: Option<f64>,
    /// Focal length in mm
    pub focal_length: Option<f64>,
    /// Flash fired (true/false)
    pub flash: Option<bool>,
    /// White balance setting
    pub white_balance: Option<String>,
    /// Exposure mode
    pub exposure_mode: Option<String>,
    /// Metering mode
    pub metering_mode: Option<String>,
}

/// EXIF metadata extracted from image files
#[derive(Debug, Clone, Default)]
pub struct ExifMetadata {
    /// Date and time when image was taken
    pub datetime: Option<DateTime<Utc>>,
    /// GPS coordinates (if available)
    pub gps: Option<GpsCoordinates>,
    /// Camera settings
    pub camera: CameraSettings,
    /// Image orientation (rotation)
    pub orientation: Option<u32>,
    /// Software used to create/edit the image
    pub software: Option<String>,
    /// Copyright information
    pub copyright: Option<String>,
    /// Artist/photographer name
    pub artist: Option<String>,
    /// Image description/title
    pub description: Option<String>,
    /// All EXIF tags as key-value pairs
    pub raw_tags: std::collections::HashMap<String, String>,
}

/// Image metadata
#[derive(Debug, Clone, Default)]
pub struct ImageMetadata {
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Color mode
    pub color_mode: Option<ColorMode>,
    /// Bits per channel
    pub bits_per_channel: Option<u8>,
    /// Dots per inch (resolution)
    pub dpi: Option<(u32, u32)>,
    /// Original format
    pub format: Option<String>,
    /// EXIF metadata (for JPEG and TIFF files)
    pub exif: Option<ExifMetadata>,
    /// Additional metadata as key-value pairs
    pub custom: std::collections::HashMap<String, String>,
}

/// Metadata for image sequences and animations
#[derive(Debug, Clone, Default)]
pub struct AnimationMetadata {
    /// Number of frames in the sequence
    pub frame_count: usize,
    /// Width of each frame in pixels
    pub width: usize,
    /// Height of each frame in pixels
    pub height: usize,
    /// Color mode
    pub color_mode: Option<ColorMode>,
    /// Frame delay in milliseconds (for animated formats)
    pub frame_delays: Vec<u32>,
    /// Whether the animation should loop
    pub loop_forever: bool,
    /// Number of times to loop (0 = infinite)
    pub loop_count: u32,
    /// Original format
    pub format: Option<String>,
    /// Additional metadata as key-value pairs
    pub custom: std::collections::HashMap<String, String>,
}

/// Supported image formats
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
    /// GIF format (including animated GIFs)
    GIF,
    /// Detect format from file extension or content
    Auto,
}

impl ImageFormat {
    /// Get the extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::PNG => "png",
            ImageFormat::JPEG => "jpg",
            ImageFormat::BMP => "bmp",
            ImageFormat::TIFF => "tiff",
            ImageFormat::GIF => "gif",
            ImageFormat::Auto => "",
        }
    }

    /// Get format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(ImageFormat::PNG),
            "jpg" | "jpeg" => Some(ImageFormat::JPEG),
            "bmp" => Some(ImageFormat::BMP),
            "tiff" | "tif" => Some(ImageFormat::TIFF),
            "gif" => Some(ImageFormat::GIF),
            _ => None,
        }
    }
}

/// Read an image from a file
///
/// # Arguments
///
/// * `path` - Path to the image file
/// * `format` - Optional image format (if not specified, it will be detected from the file extension)
///
/// # Returns
///
/// * `Result<(Array3<u8>, ImageMetadata)>` - 3D array of image data and metadata
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array3;
/// use scirs2_io::image::{read_image, ImageFormat};
///
/// // Read an image with automatic format detection
/// let (img_data, metadata) = read_image("image.png", None).unwrap();
/// println!("Image dimensions: {}x{}", metadata.width, metadata.height);
///
/// // Read with explicit format
/// let (img_data, _) = read_image("image.dat", Some(ImageFormat::PNG)).unwrap();
/// ```
pub fn read_image<P: AsRef<Path>>(
    path: P,
    format: Option<ImageFormat>,
) -> Result<(Array3<u8>, ImageMetadata)> {
    let path = path.as_ref();

    // Determine format from extension if not provided
    let format = match format {
        Some(fmt) if fmt != ImageFormat::Auto => fmt,
        _ => {
            // Try to get format from extension
            path.extension()
                .and_then(|ext| ext.to_str())
                .and_then(ImageFormat::from_extension)
                .ok_or_else(|| {
                    IoError::FormatError(format!(
                        "Could not determine image format from extension: {:?}",
                        path
                    ))
                })?
        }
    };

    // Use the image crate to read the image
    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => return Err(IoError::FileError(format!("Failed to read image: {}", e))),
    };

    // Extract basic metadata
    let mut metadata = ImageMetadata {
        width: img.width() as usize,
        height: img.height() as usize,
        format: Some(format.extension().to_string()),
        ..Default::default()
    };

    // Convert to ndarray based on image type
    let array_data = match img {
        image::DynamicImage::ImageLuma8(gray_img) => {
            metadata.color_mode = Some(ColorMode::Grayscale);
            metadata.bits_per_channel = Some(8);

            let raw = gray_img.into_raw();

            // Convert to 3D array with one channel
            let mut array = Array3::zeros((metadata.height, metadata.width, 1));

            for y in 0..metadata.height {
                for x in 0..metadata.width {
                    let idx = y * metadata.width + x;
                    array[[y, x, 0]] = raw[idx];
                }
            }

            array
        }
        image::DynamicImage::ImageRgb8(rgb_img) => {
            metadata.color_mode = Some(ColorMode::RGB);
            metadata.bits_per_channel = Some(8);

            let raw = rgb_img.into_raw();

            // Convert to 3D array with three channels
            let mut array = Array3::zeros((metadata.height, metadata.width, 3));

            for y in 0..metadata.height {
                for x in 0..metadata.width {
                    let idx = (y * metadata.width + x) * 3;
                    array[[y, x, 0]] = raw[idx];
                    array[[y, x, 1]] = raw[idx + 1];
                    array[[y, x, 2]] = raw[idx + 2];
                }
            }

            array
        }
        image::DynamicImage::ImageRgba8(rgba_img) => {
            metadata.color_mode = Some(ColorMode::RGBA);
            metadata.bits_per_channel = Some(8);

            let raw = rgba_img.into_raw();

            // Convert to 3D array with four channels
            let mut array = Array3::zeros((metadata.height, metadata.width, 4));

            for y in 0..metadata.height {
                for x in 0..metadata.width {
                    let idx = (y * metadata.width + x) * 4;
                    array[[y, x, 0]] = raw[idx];
                    array[[y, x, 1]] = raw[idx + 1];
                    array[[y, x, 2]] = raw[idx + 2];
                    array[[y, x, 3]] = raw[idx + 3];
                }
            }

            array
        }
        _ => {
            // For other formats, convert to RGB
            let rgb_img = img.to_rgb8();
            metadata.color_mode = Some(ColorMode::RGB);
            metadata.bits_per_channel = Some(8);

            let raw = rgb_img.into_raw();

            // Convert to 3D array with three channels
            let mut array = Array3::zeros((metadata.height, metadata.width, 3));

            for y in 0..metadata.height {
                for x in 0..metadata.width {
                    let idx = (y * metadata.width + x) * 3;
                    array[[y, x, 0]] = raw[idx];
                    array[[y, x, 1]] = raw[idx + 1];
                    array[[y, x, 2]] = raw[idx + 2];
                }
            }

            array
        }
    };

    Ok((array_data, metadata))
}

/// Write an image to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `data` - 3D array of image data
/// * `format` - Optional image format (if not specified, it will be detected from the file extension)
/// * `metadata` - Optional metadata
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array3;
/// use scirs2_io::image::{write_image, ImageFormat, ImageMetadata};
///
/// // Create a simple RGB image (3x2 pixels)
/// let mut img_data = Array3::zeros((2, 3, 3));
/// // Red pixel at (0,0)
/// img_data[[0, 0, 0]] = 255;
/// // Green pixel at (0,1)
/// img_data[[0, 1, 1]] = 255;
/// // Blue pixel at (0,2)
/// img_data[[0, 2, 2]] = 255;
///
/// // Write as PNG
/// write_image("output.png", &img_data, None, None).unwrap();
///
/// // Write as JPEG with explicit format
/// write_image("output.dat", &img_data, Some(ImageFormat::JPEG), None).unwrap();
/// ```
pub fn write_image<P: AsRef<Path>>(
    path: P,
    data: &Array3<u8>,
    format: Option<ImageFormat>,
    _metadata: Option<&ImageMetadata>,
) -> Result<()> {
    let path = path.as_ref();

    // Validate dimensions
    let shape = data.shape();
    if shape.len() != 3 {
        return Err(IoError::FormatError(
            "Image data must be a 3D array".to_string(),
        ));
    }

    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    // Determine color mode based on number of channels
    let color_mode = match channels {
        1 => ColorMode::Grayscale,
        3 => ColorMode::RGB,
        4 => ColorMode::RGBA,
        _ => {
            return Err(IoError::FormatError(format!(
                "Unsupported number of channels: {}",
                channels
            )))
        }
    };

    // Determine format from extension if not provided
    let format = match format {
        Some(fmt) if fmt != ImageFormat::Auto => fmt,
        _ => {
            // Try to get format from extension
            path.extension()
                .and_then(|ext| ext.to_str())
                .and_then(ImageFormat::from_extension)
                .ok_or_else(|| {
                    IoError::FormatError(format!(
                        "Could not determine image format from extension: {:?}",
                        path
                    ))
                })?
        }
    };

    // Convert ndarray to image buffer
    let img_buffer: image::DynamicImage = match color_mode {
        ColorMode::Grayscale => {
            let mut buffer = vec![0u8; width * height];

            for y in 0..height {
                for x in 0..width {
                    buffer[y * width + x] = data[[y, x, 0]];
                }
            }

            let gray_img = image::GrayImage::from_raw(width as u32, height as u32, buffer)
                .ok_or_else(|| {
                    IoError::FormatError("Failed to create grayscale image".to_string())
                })?;
            image::DynamicImage::ImageLuma8(gray_img)
        }
        ColorMode::RGB => {
            let mut buffer = vec![0u8; width * height * 3];

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    buffer[idx] = data[[y, x, 0]];
                    buffer[idx + 1] = data[[y, x, 1]];
                    buffer[idx + 2] = data[[y, x, 2]];
                }
            }

            let rgb_img = image::RgbImage::from_raw(width as u32, height as u32, buffer)
                .ok_or_else(|| IoError::FormatError("Failed to create RGB image".to_string()))?;
            image::DynamicImage::ImageRgb8(rgb_img)
        }
        ColorMode::RGBA => {
            let mut buffer = vec![0u8; width * height * 4];

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 4;
                    buffer[idx] = data[[y, x, 0]];
                    buffer[idx + 1] = data[[y, x, 1]];
                    buffer[idx + 2] = data[[y, x, 2]];
                    buffer[idx + 3] = data[[y, x, 3]];
                }
            }

            let rgba_img = image::RgbaImage::from_raw(width as u32, height as u32, buffer)
                .ok_or_else(|| IoError::FormatError("Failed to create RGBA image".to_string()))?;
            image::DynamicImage::ImageRgba8(rgba_img)
        }
    };

    // Save the image
    match format {
        ImageFormat::PNG => {
            img_buffer
                .save_with_format(path, image::ImageFormat::Png)
                .map_err(|e| IoError::FileError(format!("Failed to save PNG image: {}", e)))?;
        }
        ImageFormat::JPEG => {
            img_buffer
                .save_with_format(path, image::ImageFormat::Jpeg)
                .map_err(|e| IoError::FileError(format!("Failed to save JPEG image: {}", e)))?;
        }
        ImageFormat::BMP => {
            img_buffer
                .save_with_format(path, image::ImageFormat::Bmp)
                .map_err(|e| IoError::FileError(format!("Failed to save BMP image: {}", e)))?;
        }
        ImageFormat::TIFF => {
            img_buffer
                .save_with_format(path, image::ImageFormat::Tiff)
                .map_err(|e| IoError::FileError(format!("Failed to save TIFF image: {}", e)))?;
        }
        ImageFormat::GIF => {
            img_buffer
                .save_with_format(path, image::ImageFormat::Gif)
                .map_err(|e| IoError::FileError(format!("Failed to save GIF image: {}", e)))?;
        }
        ImageFormat::Auto => {
            // This shouldn't happen as we've already resolved the format
            return Err(IoError::FormatError(
                "Automatic format detection failed".to_string(),
            ));
        }
    }

    Ok(())
}

/// Read image metadata without loading the full image
///
/// # Arguments
///
/// * `path` - Path to the image file
///
/// # Returns
///
/// * `Result<ImageMetadata>` - Image metadata
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::read_image_metadata;
///
/// let metadata = read_image_metadata("image.png").unwrap();
/// println!("Image dimensions: {}x{}", metadata.width, metadata.height);
/// ```
pub fn read_image_metadata<P: AsRef<Path>>(path: P) -> Result<ImageMetadata> {
    let path = path.as_ref();

    // Open image reader without fully decoding the image
    let reader = match image::io::Reader::open(path) {
        Ok(reader) => reader,
        Err(e) => return Err(IoError::FileError(format!("Failed to open image: {}", e))),
    };

    // Try to read dimensions without decoding
    let dimensions = match reader.into_dimensions() {
        Ok(dim) => dim,
        Err(e) => {
            return Err(IoError::FormatError(format!(
                "Failed to read image dimensions: {}",
                e
            )))
        }
    };

    // Get format from extension
    let format = path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(ImageFormat::from_extension)
        .map(|fmt| fmt.extension().to_string());

    // Create metadata with basic information
    let metadata = ImageMetadata {
        width: dimensions.0 as usize,
        height: dimensions.1 as usize,
        format,
        ..Default::default()
    };

    Ok(metadata)
}

/// Convert between image formats
///
/// # Arguments
///
/// * `input_path` - Path to the input image file
/// * `output_path` - Path to the output image file
/// * `output_format` - Optional output format (if not specified, it will be detected from the output file extension)
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::{convert_image, ImageFormat};
///
/// // Convert PNG to JPEG
/// convert_image("input.png", "output.jpg", None).unwrap();
///
/// // Convert to BMP with explicit format
/// convert_image("input.png", "output.dat", Some(ImageFormat::BMP)).unwrap();
/// ```
pub fn convert_image<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    output_format: Option<ImageFormat>,
) -> Result<()> {
    // Read input image
    let (data, metadata) = read_image(input_path, None)?;

    // Write to output format
    write_image(output_path, &data, output_format, Some(&metadata))
}

/// Extract EXIF metadata from an image file
///
/// # Arguments
///
/// * `path` - Path to the image file
///
/// # Returns
///
/// * `Result<Option<ExifMetadata>>` - EXIF metadata if present, None if not available
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::read_exif_metadata;
///
/// // Read EXIF data from a JPEG file
/// let exif = read_exif_metadata("photo.jpg").unwrap();
/// if let Some(metadata) = exif {
///     println!("Camera: {:?} {:?}", metadata.camera.make, metadata.camera.model);
///     if let Some(gps) = metadata.gps {
///         println!("Location: {}, {}", gps.latitude, gps.longitude);
///     }
/// }
/// ```
pub fn read_exif_metadata<P: AsRef<Path>>(path: P) -> Result<Option<ExifMetadata>> {
    let file = std::fs::File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut bufreader = BufReader::new(&file);
    
    // Try to read EXIF data
    match kamadak_exif::Reader::new().read_from_container(&mut bufreader) {
        Ok(exif_data) => {
            let mut metadata = ExifMetadata::default();
            let mut raw_tags = std::collections::HashMap::new();
            
            // Extract all EXIF fields
            for field in exif_data.fields() {
                let tag_name = format!("{}", field.tag);
                let value = format!("{}", field.display_value().with_unit(&exif_data));
                raw_tags.insert(tag_name.clone(), value.clone());
                
                // Parse specific important fields
                match field.tag {
                    kamadak_exif::Tag::DateTime => {
                        metadata.datetime = parse_exif_datetime(&value);
                    }
                    kamadak_exif::Tag::DateTimeOriginal => {
                        if metadata.datetime.is_none() {
                            metadata.datetime = parse_exif_datetime(&value);
                        }
                    }
                    kamadak_exif::Tag::Make => {
                        metadata.camera.make = Some(value);
                    }
                    kamadak_exif::Tag::Model => {
                        metadata.camera.model = Some(value);
                    }
                    kamadak_exif::Tag::LensModel => {
                        metadata.camera.lens_model = Some(value);
                    }
                    kamadak_exif::Tag::PhotographicSensitivity => {
                        metadata.camera.iso = value.parse().ok();
                    }
                    kamadak_exif::Tag::FNumber => {
                        metadata.camera.aperture = parse_rational_value(&value);
                    }
                    kamadak_exif::Tag::ExposureTime => {
                        metadata.camera.shutter_speed = parse_rational_value(&value);
                    }
                    kamadak_exif::Tag::FocalLength => {
                        metadata.camera.focal_length = parse_rational_value(&value);
                    }
                    kamadak_exif::Tag::Flash => {
                        metadata.camera.flash = Some(value.contains("fired") || value.contains("Fired"));
                    }
                    kamadak_exif::Tag::WhiteBalance => {
                        metadata.camera.white_balance = Some(value);
                    }
                    kamadak_exif::Tag::ExposureMode => {
                        metadata.camera.exposure_mode = Some(value);
                    }
                    kamadak_exif::Tag::MeteringMode => {
                        metadata.camera.metering_mode = Some(value);
                    }
                    kamadak_exif::Tag::Orientation => {
                        metadata.orientation = value.parse().ok();
                    }
                    kamadak_exif::Tag::Software => {
                        metadata.software = Some(value);
                    }
                    kamadak_exif::Tag::Copyright => {
                        metadata.copyright = Some(value);
                    }
                    kamadak_exif::Tag::Artist => {
                        metadata.artist = Some(value);
                    }
                    kamadak_exif::Tag::ImageDescription => {
                        metadata.description = Some(value);
                    }
                    _ => {} // Store in raw_tags only
                }
            }
            
            // Extract GPS data if available
            metadata.gps = extract_gps_data(&exif_data);
            metadata.raw_tags = raw_tags;
            
            Ok(Some(metadata))
        }
        Err(_) => {
            // No EXIF data or unsupported format
            Ok(None)
        }
    }
}

/// Extract GPS coordinates from EXIF data
fn extract_gps_data(exif_data: &kamadak_exif::Exif) -> Option<GpsCoordinates> {
    let mut lat: Option<f64> = None;
    let mut lon: Option<f64> = None;
    let mut alt: Option<f64> = None;
    let mut lat_ref = String::new();
    let mut lon_ref = String::new();
    
    for field in exif_data.fields() {
        match field.tag {
            kamadak_exif::Tag::GPSLatitude => {
                lat = parse_gps_coordinate(&field);
            }
            kamadak_exif::Tag::GPSLongitude => {
                lon = parse_gps_coordinate(&field);
            }
            kamadak_exif::Tag::GPSAltitude => {
                alt = parse_rational_value(&format!("{}", field.display_value().with_unit(exif_data)));
            }
            kamadak_exif::Tag::GPSLatitudeRef => {
                lat_ref = format!("{}", field.display_value().with_unit(exif_data));
            }
            kamadak_exif::Tag::GPSLongitudeRef => {
                lon_ref = format!("{}", field.display_value().with_unit(exif_data));
            }
            _ => {}
        }
    }
    
    if let (Some(latitude), Some(longitude)) = (lat, lon) {
        // Apply sign based on hemisphere
        let final_lat = if lat_ref.contains('S') { -latitude } else { latitude };
        let final_lon = if lon_ref.contains('W') { -longitude } else { longitude };
        
        Some(GpsCoordinates {
            latitude: final_lat,
            longitude: final_lon,
            altitude: alt,
            latitude_ref: lat_ref,
            longitude_ref: lon_ref,
        })
    } else {
        None
    }
}

/// Parse GPS coordinate from EXIF field (degrees, minutes, seconds)
fn parse_gps_coordinate(field: &kamadak_exif::Field) -> Option<f64> {
    if let kamadak_exif::Value::Rational(ref rationals) = field.value {
        if rationals.len() >= 3 {
            let degrees = rationals[0].to_f64();
            let minutes = rationals[1].to_f64();
            let seconds = rationals[2].to_f64();
            
            Some(degrees + minutes / 60.0 + seconds / 3600.0)
        } else {
            None
        }
    } else {
        None
    }
}

/// Parse EXIF datetime string
fn parse_exif_datetime(datetime_str: &str) -> Option<DateTime<Utc>> {
    // EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
    let clean_str = datetime_str.trim().replace(':', "-");
    if let Some(space_pos) = clean_str.find(' ') {
        let date_part = &clean_str[..space_pos];
        let time_part = &clean_str[space_pos + 1..];
        let iso_str = format!("{}T{}Z", date_part, time_part);
        
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(&iso_str[..19], "%Y-%m-%dT%H:%M:%S") {
            Some(DateTime::from_naive_utc_and_offset(naive_dt, Utc))
        } else {
            None
        }
    } else {
        None
    }
}

/// Parse rational value from EXIF field display string
fn parse_rational_value(value_str: &str) -> Option<f64> {
    let clean_str = value_str.trim();
    
    // Handle fractions like "1/60" or "1/60 sec"
    if clean_str.contains('/') {
        let parts: Vec<&str> = clean_str.split_whitespace().collect();
        if let Some(fraction) = parts.first() {
            let frac_parts: Vec<&str> = fraction.split('/').collect();
            if frac_parts.len() == 2 {
                if let (Ok(num), Ok(den)) = (frac_parts[0].parse::<f64>(), frac_parts[1].parse::<f64>()) {
                    if den != 0.0 {
                        return Some(num / den);
                    }
                }
            }
        }
    }
    
    // Handle decimal values like "2.8" or "2.8 f"
    let parts: Vec<&str> = clean_str.split_whitespace().collect();
    if let Some(number_part) = parts.first() {
        number_part.parse().ok()
    } else {
        None
    }
}

/// Read enhanced image metadata including EXIF data
///
/// # Arguments
///
/// * `path` - Path to the image file
///
/// # Returns
///
/// * `Result<ImageMetadata>` - Enhanced metadata including EXIF data
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::read_enhanced_metadata;
///
/// let metadata = read_enhanced_metadata("photo.jpg").unwrap();
/// println!("Image: {}x{}", metadata.width, metadata.height);
/// 
/// if let Some(exif) = &metadata.exif {
///     if let Some(camera_make) = &exif.camera.make {
///         println!("Camera: {}", camera_make);
///     }
///     if let Some(gps) = &exif.gps {
///         println!("GPS: {}, {}", gps.latitude, gps.longitude);
///     }
/// }
/// ```
pub fn read_enhanced_metadata<P: AsRef<Path>>(path: P) -> Result<ImageMetadata> {
    // Get basic metadata
    let mut metadata = read_image_metadata(&path)?;
    
    // Try to add EXIF metadata
    metadata.exif = read_exif_metadata(&path)?;
    
    Ok(metadata)
}

/// Get a grayscale view of an image
///
/// # Arguments
///
/// * `image` - 3D array of image data
///
/// # Returns
///
/// * `Array2<u8>` - 2D array of grayscale values
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::{read_image, get_grayscale};
///
/// let (img_data, _) = read_image("image.png", None).unwrap();
/// let gray_img = get_grayscale(&img_data);
/// println!("Grayscale dimensions: {:?}", gray_img.shape());
/// ```
pub fn get_grayscale(image: &Array3<u8>) -> Array2<u8> {
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    let mut gray = Array2::zeros((height, width));

    match channels {
        1 => {
            // Already grayscale, just copy
            for y in 0..height {
                for x in 0..width {
                    gray[[y, x]] = image[[y, x, 0]];
                }
            }
        }
        3 | 4 => {
            // RGB or RGBA, convert to grayscale using luminance formula
            // Y = 0.299 R + 0.587 G + 0.114 B
            for y in 0..height {
                for x in 0..width {
                    let r = image[[y, x, 0]] as f32;
                    let g = image[[y, x, 1]] as f32;
                    let b = image[[y, x, 2]] as f32;

                    let gray_value = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                    gray[[y, x]] = gray_value;
                }
            }
        }
        _ => {
            // Unsupported number of channels, just use the first channel
            for y in 0..height {
                for x in 0..width {
                    gray[[y, x]] = image[[y, x, 0]];
                }
            }
        }
    }

    gray
}

/// Read all frames from an animated GIF
///
/// # Arguments
///
/// * `path` - Path to the GIF file
///
/// # Returns
///
/// * `Result<(Array4<u8>, AnimationMetadata)>` - 4D array of image data (frames, height, width, channels) and animation metadata
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::read_animated_gif;
///
/// let (frames, metadata) = read_animated_gif("animation.gif").unwrap();
/// println!("Loaded {} frames of {}x{}", metadata.frame_count, metadata.width, metadata.height);
/// ```
pub fn read_animated_gif<P: AsRef<Path>>(path: P) -> Result<(Array4<u8>, AnimationMetadata)> {
    let path = path.as_ref();

    // Open the file
    let file = fs::File::open(path)
        .map_err(|e| IoError::FileError(format!("Failed to open GIF file: {}", e)))?;

    // Decode the GIF
    // First check that the file is a valid GIF
    let _decoder = image::codecs::gif::GifDecoder::new(file)
        .map_err(|e| IoError::FormatError(format!("Failed to decode GIF: {}", e)))?;

    // GIF always has delay specified in 1/100ths of a second
    // Extract metadata
    let mut metadata = AnimationMetadata {
        format: Some("gif".to_string()),
        ..Default::default()
    };

    // We can't easily get GIF specific info like looping from the decoder,
    // so we'll collect frame delays from the frames themselves
    metadata.loop_forever = true; // Assume GIFs loop by default

    // Create animation frame iterator
    let frames = image::codecs::gif::GifDecoder::new(
        fs::File::open(path)
            .map_err(|e| IoError::FileError(format!("Failed to open GIF file: {}", e)))?,
    )
    .map_err(|e| IoError::FormatError(format!("Failed to decode GIF: {}", e)))?;

    // Get dimensions
    let (width, height) = frames.dimensions();
    metadata.width = width as usize;
    metadata.height = height as usize;

    // Get frames
    let frames = frames
        .into_frames()
        .collect_frames()
        .map_err(|e| IoError::FormatError(format!("Failed to collect frames: {}", e)))?;

    metadata.frame_count = frames.len();

    // Update frame delays if we didn't get them earlier
    if metadata.frame_delays.is_empty() {
        for frame in &frames {
            metadata
                .frame_delays
                .push(frame.delay().numer_denom_ms().0.max(10));
        }
    }

    // Allocate space for all frames
    let mut frames_array = Array4::zeros((
        frames.len(),
        metadata.height,
        metadata.width,
        4, // GIF supports RGBA
    ));

    // Process each frame
    for (i, frame) in frames.iter().enumerate() {
        let buffer = frame.buffer();

        // GIF frames are already RGBA
        let rgba = buffer;
        metadata.color_mode = Some(ColorMode::RGBA);

        // Copy data to our array
        for y in 0..metadata.height {
            for x in 0..metadata.width {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                frames_array[[i, y, x, 0]] = pixel[0];
                frames_array[[i, y, x, 1]] = pixel[1];
                frames_array[[i, y, x, 2]] = pixel[2];
                frames_array[[i, y, x, 3]] = pixel[3];
            }
        }
    }

    Ok((frames_array, metadata))
}

/// Create an animated GIF from a sequence of images
///
/// # Arguments
///
/// * `path` - Path to save the GIF file
/// * `frames` - 4D array of frame data (frames, height, width, channels)
/// * `metadata` - Animation metadata, including frame delays
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array3, Array4};
/// use scirs2_io::image::{AnimationMetadata, create_animated_gif};
///
/// // Create a 2-frame animation (2 frames, 2x2 pixels, RGBA)
/// let mut frames = Array4::zeros((2, 2, 2, 4));
///
/// // Frame 1: red pixel at (0,0)
/// frames[[0, 0, 0, 0]] = 255;
///
/// // Frame 2: blue pixel at (0,0)
/// frames[[1, 0, 0, 2]] = 255;
///
/// // Create metadata with 100ms delay between frames
/// let metadata = AnimationMetadata {
///     frame_count: 2,
///     width: 2,
///     height: 2,
///     frame_delays: vec![100, 100],
///     loop_forever: true,
///     ..Default::default()
/// };
///
/// create_animated_gif("animation.gif", &frames, &metadata).unwrap();
/// ```
pub fn create_animated_gif<P: AsRef<Path>>(
    path: P,
    frames: &Array4<u8>,
    metadata: &AnimationMetadata,
) -> Result<()> {
    let path = path.as_ref();

    // Validate dimensions
    let shape = frames.shape();
    if shape.len() != 4 {
        return Err(IoError::FormatError(
            "Frame data must be a 4D array".to_string(),
        ));
    }

    let num_frames = shape[0];
    let height = shape[1];
    let width = shape[2];
    let channels = shape[3];

    // Verify frame count matches metadata
    if num_frames != metadata.frame_count {
        return Err(IoError::FormatError(format!(
            "Frame count mismatch: {} in array vs {} in metadata",
            num_frames, metadata.frame_count
        )));
    }

    // Ensure we have delay information for each frame
    if metadata.frame_delays.len() != num_frames {
        return Err(IoError::FormatError(format!(
            "Frame delay count mismatch: {} delays vs {} frames",
            metadata.frame_delays.len(),
            num_frames
        )));
    }

    // Create output file
    let mut file = fs::File::create(path)
        .map_err(|e| IoError::FileError(format!("Failed to create GIF file: {}", e)))?;

    // Create encoder
    let mut encoder = image::codecs::gif::GifEncoder::new_with_speed(&mut file, 10); // Speed 1-30, higher is faster

    // Set repeat behavior
    if metadata.loop_forever || metadata.loop_count == 0 {
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .map_err(|e| IoError::FormatError(format!("Failed to set GIF to loop: {}", e)))?;
    } else if metadata.loop_count > 1 {
        encoder
            .set_repeat(image::codecs::gif::Repeat::Finite(
                metadata.loop_count as u16,
            ))
            .map_err(|e| IoError::FormatError(format!("Failed to set GIF loop count: {}", e)))?;
    }

    // Convert and write each frame
    for i in 0..num_frames {
        // Convert frame to image buffer
        let mut buffer = image::RgbaImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let r = if channels > 0 {
                    frames[[i, y, x, 0]]
                } else {
                    0
                };
                let g = if channels > 1 {
                    frames[[i, y, x, 1]]
                } else {
                    0
                };
                let b = if channels > 2 {
                    frames[[i, y, x, 2]]
                } else {
                    0
                };
                let a = if channels > 3 {
                    frames[[i, y, x, 3]]
                } else {
                    255
                };

                let pixel = image::Rgba([r, g, b, a]);
                buffer.put_pixel(x as u32, y as u32, pixel);
            }
        }

        // Create frame with the appropriate delay
        let delay_ms = metadata.frame_delays[i];
        let frame = image::Frame::from_parts(
            buffer,
            0,
            0, // left, top
            image::Delay::from_numer_denom_ms(delay_ms, 1),
        );

        // Add the frame to the GIF
        encoder
            .encode_frame(frame)
            .map_err(|e| IoError::FormatError(format!("Failed to encode frame {}: {}", i, e)))?;
    }

    Ok(())
}

/// Read a sequence of image files into a 4D array
///
/// # Arguments
///
/// * `pattern` - Path pattern (e.g., "frames/frame_{:04d}.png", "frames/frame_*.png")
/// * `format` - Optional image format
///
/// # Returns
///
/// * `Result<(Array4<u8>, AnimationMetadata)>` - 4D array of image data (frames, height, width, channels) and metadata
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::{read_image_sequence, ImageFormat};
///
/// // Read named sequence: frame_0001.png, frame_0002.png, etc.
/// let (frames, metadata) = read_image_sequence("frames/frame_{:04d}.png", None).unwrap();
///
/// // Read using glob pattern
/// let (frames, metadata) = read_image_sequence("frames/frame_*.png", None).unwrap();
///
/// println!("Loaded {} frames of {}x{}", metadata.frame_count, metadata.width, metadata.height);
/// ```
pub fn read_image_sequence<P: AsRef<Path>>(
    pattern: P,
    format: Option<ImageFormat>,
) -> Result<(Array4<u8>, AnimationMetadata)> {
    let pattern = pattern.as_ref().to_string_lossy().to_string();

    // Determine if this is a format string or glob pattern
    let file_paths = if pattern.contains("*") {
        // Glob pattern
        glob::glob(&pattern)
            .map_err(|e| IoError::FileError(format!("Invalid glob pattern: {}", e)))?
            .map(|res| res.map_err(|e| IoError::FileError(format!("Glob error: {}", e))))
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else if pattern.contains("{}") || pattern.contains("{:") {
        // Format string with placeholder
        // Extract the format pattern
        let re = Regex::new(r"\{(?::([^}]+))?\}").map_err(|_| {
            IoError::FormatError("Invalid format placeholder in pattern".to_string())
        })?;

        // Ensure the pattern has valid captures
        re.captures(&pattern).ok_or_else(|| {
            IoError::FormatError("No valid format placeholder found in pattern".to_string())
        })?;

        // Replace the placeholder with a regex to extract frame numbers
        let pattern_regex = re.replace(&pattern, r"(\d+)");
        let re = Regex::new(&format!("^{}$", pattern_regex))
            .map_err(|_| IoError::FormatError("Failed to create regex for pattern".to_string()))?;

        // Get directory for the pattern
        let dir_path = if let Some(parent) = Path::new(&pattern).parent() {
            parent.to_path_buf()
        } else {
            Path::new(".").to_path_buf()
        };

        // Scan directory for matching files
        let entries = fs::read_dir(&dir_path)
            .map_err(|e| IoError::FileError(format!("Failed to read directory: {}", e)))?
            .map(|res| res.map_err(|e| IoError::FileError(format!("Directory read error: {}", e))))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Find entries that match our pattern and extract frame number
        let mut numbered_entries = Vec::new();
        for entry in entries {
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(caps) = re.captures(file_name) {
                    if let Some(num_str) = caps.get(1) {
                        if let Ok(num) = num_str.as_str().parse::<usize>() {
                            numbered_entries.push((num, path));
                        }
                    }
                }
            }
        }

        // Sort by frame number
        numbered_entries.sort_by_key(|&(num, _)| num);

        // Extract just the paths
        numbered_entries.into_iter().map(|(_, path)| path).collect()
    } else {
        // Treat as a single file (maybe it's an animated format like GIF)
        if pattern.to_lowercase().ends_with(".gif") {
            return read_animated_gif(&pattern);
        }

        vec![Path::new(&pattern).to_path_buf()]
    };

    // Ensure we found some files
    if file_paths.is_empty() {
        return Err(IoError::FileError(format!(
            "No files found matching pattern: {}",
            pattern
        )));
    }

    // Initialize metadata
    let mut metadata = AnimationMetadata {
        frame_count: file_paths.len(),
        loop_forever: false,
        loop_count: 1,
        ..Default::default()
    };

    // Default frame delay (33ms ≈ 30fps)
    metadata.frame_delays = vec![33; file_paths.len()];

    // Read the first image to determine dimensions and format
    let (first_image, first_meta) = read_image(&file_paths[0], format)?;

    metadata.width = first_meta.width;
    metadata.height = first_meta.height;
    metadata.color_mode = first_meta.color_mode;
    metadata.format = first_meta.format;

    // Create array to hold all frames
    let channels = first_image.shape()[2];
    let mut frames = Array4::zeros((file_paths.len(), metadata.height, metadata.width, channels));

    // Copy first frame data
    for y in 0..metadata.height {
        for x in 0..metadata.width {
            for c in 0..channels {
                frames[[0, y, x, c]] = first_image[[y, x, c]];
            }
        }
    }

    // Read remaining frames
    for (i, path) in file_paths.iter().enumerate().skip(1) {
        let (img, img_meta) = read_image(path, format)?;

        // Verify dimensions match
        if img_meta.width != metadata.width || img_meta.height != metadata.height {
            return Err(IoError::FormatError(format!(
                "Frame size mismatch: frame {} ({}x{}) doesn't match first frame ({}x{})",
                i, img_meta.width, img_meta.height, metadata.width, metadata.height
            )));
        }

        // Copy frame data
        for y in 0..metadata.height {
            for x in 0..metadata.width {
                for c in 0..channels {
                    frames[[i, y, x, c]] = img[[y, x, c]];
                }
            }
        }
    }

    Ok((frames, metadata))
}

/// Write a sequence of image frames to individual files
///
/// # Arguments
///
/// * `pattern` - Output file pattern (e.g., "frames/frame_{:04d}.png")
/// * `frames` - 4D array of frame data (frames, height, width, channels)
/// * `format` - Optional image format
/// * `metadata` - Optional animation metadata
///
/// # Returns
///
/// * `Result<Vec<PathBuf>>` - Paths to the created files
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array4;
/// use scirs2_io::image::{AnimationMetadata, ImageFormat, write_image_sequence};
///
/// // Create a 2-frame animation
/// let mut frames = Array4::zeros((2, 2, 2, 3));
/// frames[[0, 0, 0, 0]] = 255;  // Red in first frame
/// frames[[1, 0, 0, 2]] = 255;  // Blue in second frame
///
/// // Write as PNG sequence
/// let paths = write_image_sequence(
///     "frames/frame_{:04d}.png",
///     &frames,
///     Some(ImageFormat::PNG),
///     None
/// ).unwrap();
///
/// println!("Wrote {} frame files", paths.len());
/// ```
pub fn write_image_sequence<P: AsRef<Path>>(
    pattern: P,
    frames: &Array4<u8>,
    format: Option<ImageFormat>,
    metadata: Option<&AnimationMetadata>,
) -> Result<Vec<std::path::PathBuf>> {
    let pattern = pattern.as_ref().to_string_lossy().to_string();

    // Check if we should write as animated GIF instead
    if pattern.to_lowercase().ends_with(".gif") {
        // Create metadata if not provided
        let gif_metadata = if let Some(meta) = metadata {
            meta.clone()
        } else {
            // Create default metadata
            let shape = frames.shape();
            let frame_count = shape[0];
            let height = shape[1];
            let width = shape[2];

            AnimationMetadata {
                frame_count,
                width,
                height,
                frame_delays: vec![33; frame_count], // ~30fps
                loop_forever: true,
                ..Default::default()
            }
        };

        create_animated_gif(&pattern, frames, &gif_metadata)?;
        return Ok(vec![Path::new(&pattern).to_path_buf()]);
    }

    // Validate dimensions
    let shape = frames.shape();
    if shape.len() != 4 {
        return Err(IoError::FormatError(
            "Frame data must be a 4D array".to_string(),
        ));
    }

    let num_frames = shape[0];
    let height = shape[1];
    let width = shape[2];
    let channels = shape[3];

    // Validate channel count
    if channels != 1 && channels != 3 && channels != 4 {
        return Err(IoError::FormatError(format!(
            "Unsupported number of channels: {}. Must be 1, 3, or 4.",
            channels
        )));
    }

    // Validate pattern
    if !pattern.contains("{}") && !pattern.contains("{:") {
        return Err(IoError::FormatError(
            "Pattern must contain a placeholder like {} or {:04d}".to_string(),
        ));
    }

    // Ensure directory exists
    if let Some(parent) = Path::new(&pattern).parent() {
        fs::create_dir_all(parent)
            .map_err(|e| IoError::FileError(format!("Failed to create directory: {}", e)))?;
    }

    // Create image metadata
    let img_metadata = if let Some(anim_meta) = metadata {
        ImageMetadata {
            width,
            height,
            color_mode: anim_meta.color_mode,
            format: anim_meta.format.clone(),
            ..Default::default()
        }
    } else {
        ImageMetadata {
            width,
            height,
            ..Default::default()
        }
    };

    // Write each frame
    let mut output_paths = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        // Extract this frame as a 3D array
        let mut frame = Array3::zeros((height, width, channels));

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    frame[[y, x, c]] = frames[[i, y, x, c]];
                }
            }
        }

        // Generate the output path
        let output_path = match pattern.find("{:") {
            Some(pos) => {
                let end_pos = pattern[pos..].find('}').ok_or_else(|| {
                    IoError::FormatError("Invalid format placeholder in pattern".to_string())
                })?;

                let format_spec = &pattern[pos + 2..pos + end_pos];

                // Only handle d and 0d format specifiers
                let formatted = if let Some(width_spec) = format_spec.strip_suffix('d') {
                    if !width_spec.is_empty() {
                        // Handle width specifier like {:04d}
                        if let Ok(width) = width_spec.parse::<usize>() {
                            format!("{:0width$}", i, width = width)
                        } else {
                            return Err(IoError::FormatError(
                                "Invalid format width in pattern".to_string(),
                            ));
                        }
                    } else {
                        // Just {:d}
                        format!("{}", i)
                    }
                } else {
                    // Unrecognized format
                    return Err(IoError::FormatError(format!(
                        "Unsupported format specifier: {{:{}}}",
                        format_spec
                    )));
                };

                pattern.replace(&format!("{{:{}}}", format_spec), &formatted)
            }
            None => {
                // Simple {} replacement
                pattern.replace("{}", &i.to_string())
            }
        };

        // Write the frame
        write_image(&output_path, &frame, format, Some(&img_metadata))?;

        output_paths.push(Path::new(&output_path).to_path_buf());
    }

    Ok(output_paths)
}

/// Extract frames from an animated GIF
///
/// # Arguments
///
/// * `input_path` - Path to the animated GIF file
/// * `output_pattern` - Pattern for output files (e.g., "frames/frame_{:04d}.png")
/// * `output_format` - Optional format for output files
///
/// # Returns
///
/// * `Result<Vec<PathBuf>>` - Paths to the extracted frame files
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::image::{extract_gif_frames, ImageFormat};
///
/// // Extract frames as PNG
/// let paths = extract_gif_frames(
///     "animation.gif",
///     "frames/frame_{:04d}.png",
///     Some(ImageFormat::PNG)
/// ).unwrap();
///
/// println!("Extracted {} frames", paths.len());
/// ```
pub fn extract_gif_frames<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_pattern: P2,
    output_format: Option<ImageFormat>,
) -> Result<Vec<std::path::PathBuf>> {
    // Read the animated GIF
    let (frames, metadata) = read_animated_gif(input_path)?;

    // Write the frames as individual files
    write_image_sequence(output_pattern, &frames, output_format, Some(&metadata))
}
