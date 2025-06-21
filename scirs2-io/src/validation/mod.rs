//! Data validation and integrity checking utilities
//!
//! This module provides functionality for validating data integrity through
//! checksums, hash verification, and other validation methods.
//!
//! ## Features
//!
//! - Checksums (CRC32, MD5, SHA-256, BLAKE3)
//! - File integrity validation
//! - Data format validation
//! - Integrity metadata for scientific data

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use blake3::Hasher as Blake3Hasher;
use chrono::{DateTime, Utc};
use crc32fast::Hasher as CrcHasher;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{IoError, Result};

// Export submodules
pub mod formats;

/// Checksum algorithm types available for data validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// CRC32 - Fast but less secure, good for error detection
    CRC32,
    /// SHA-256 - Secure cryptographic hash
    SHA256,
    /// BLAKE3 - Modern, faster cryptographic hash
    BLAKE3,
}

impl ChecksumAlgorithm {
    /// Get a string representation of the algorithm
    pub fn as_str(&self) -> &'static str {
        match self {
            ChecksumAlgorithm::CRC32 => "CRC32",
            ChecksumAlgorithm::SHA256 => "SHA256",
            ChecksumAlgorithm::BLAKE3 => "BLAKE3",
        }
    }

    /// Parse algorithm name from a string
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "CRC32" => Some(ChecksumAlgorithm::CRC32),
            "SHA256" => Some(ChecksumAlgorithm::SHA256),
            "BLAKE3" => Some(ChecksumAlgorithm::BLAKE3),
            _ => None,
        }
    }
}

/// File integrity metadata used for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityMetadata {
    /// Size of the file in bytes
    pub size: u64,
    /// Checksum algorithm used
    pub algorithm: String,
    /// Checksum value (hex encoded)
    pub checksum: String,
    /// Creation or last modification timestamp
    pub timestamp: u64,
    /// Additional integrity information
    pub additional_info: std::collections::HashMap<String, String>,
}

/// Calculate a checksum for the provided data
///
/// # Arguments
///
/// * `data` - The data to calculate the checksum for
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// The checksum as a hex encoded string
pub fn calculate_checksum(data: &[u8], algorithm: ChecksumAlgorithm) -> String {
    match algorithm {
        ChecksumAlgorithm::CRC32 => {
            let mut hasher = CrcHasher::new();
            hasher.update(data);
            format!("{:08x}", hasher.finalize())
        }
        ChecksumAlgorithm::SHA256 => {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hex::encode(hasher.finalize())
        }
        ChecksumAlgorithm::BLAKE3 => {
            let mut hasher = Blake3Hasher::new();
            hasher.update(data);
            hex::encode(hasher.finalize().as_bytes())
        }
    }
}

/// Calculate a checksum for a file
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// The checksum as a hex encoded string
pub fn calculate_file_checksum<P: AsRef<Path>>(
    path: P,
    algorithm: ChecksumAlgorithm,
) -> Result<String> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    match algorithm {
        ChecksumAlgorithm::CRC32 => {
            let mut hasher = CrcHasher::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            Ok(format!("{:08x}", hasher.finalize()))
        }
        ChecksumAlgorithm::SHA256 => {
            let mut hasher = Sha256::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            Ok(hex::encode(hasher.finalize()))
        }
        ChecksumAlgorithm::BLAKE3 => {
            let mut hasher = Blake3Hasher::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            Ok(hex::encode(hasher.finalize().as_bytes()))
        }
    }
}

/// Verify a checksum against the provided data
///
/// # Arguments
///
/// * `data` - The data to verify
/// * `expected_checksum` - The expected checksum value (hex encoded)
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// `true` if the checksum matches, `false` otherwise
pub fn verify_checksum(data: &[u8], expected_checksum: &str, algorithm: ChecksumAlgorithm) -> bool {
    let calculated = calculate_checksum(data, algorithm);
    calculated.eq_ignore_ascii_case(expected_checksum)
}

/// Verify a file's checksum
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `expected_checksum` - The expected checksum value (hex encoded)
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// `Ok(true)` if the checksum matches, `Ok(false)` otherwise, or an error
pub fn verify_file_checksum<P: AsRef<Path>>(
    path: P,
    expected_checksum: &str,
    algorithm: ChecksumAlgorithm,
) -> Result<bool> {
    let calculated = calculate_file_checksum(path, algorithm)?;
    Ok(calculated.eq_ignore_ascii_case(expected_checksum))
}

/// Generate integrity metadata for a file
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `algorithm` - The checksum algorithm to use
///
/// # Returns
///
/// A struct containing integrity metadata
pub fn generate_file_integrity_metadata<P: AsRef<Path>>(
    path: P,
    algorithm: ChecksumAlgorithm,
) -> Result<IntegrityMetadata> {
    let path = path.as_ref();

    // Get file metadata
    let file_metadata = std::fs::metadata(path)
        .map_err(|e| IoError::FileError(format!("Failed to read file metadata: {}", e)))?;

    let size = file_metadata.len();
    let modified = file_metadata
        .modified()
        .map_err(|e| IoError::FileError(format!("Failed to get modification time: {}", e)))?;

    // Convert to timestamp
    let timestamp = modified
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_err(|e| IoError::FileError(format!("Failed to convert time: {}", e)))?
        .as_secs();

    // Calculate checksum
    let checksum = calculate_file_checksum(path, algorithm)?;

    Ok(IntegrityMetadata {
        size,
        algorithm: algorithm.as_str().to_string(),
        checksum,
        timestamp,
        additional_info: std::collections::HashMap::new(),
    })
}

/// Save integrity metadata to a file
///
/// # Arguments
///
/// * `metadata` - The integrity metadata to save
/// * `path` - Path to save the metadata to
///
/// # Returns
///
/// Result indicating success or failure
pub fn save_integrity_metadata<P: AsRef<Path>>(
    metadata: &IntegrityMetadata,
    path: P,
) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    serde_json::to_writer_pretty(file, metadata)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;
    Ok(())
}

/// Load integrity metadata from a file
///
/// # Arguments
///
/// * `path` - Path to the metadata file
///
/// # Returns
///
/// The loaded integrity metadata
pub fn load_integrity_metadata<P: AsRef<Path>>(path: P) -> Result<IntegrityMetadata> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);
    let metadata: IntegrityMetadata = serde_json::from_reader(reader)
        .map_err(|e| IoError::DeserializationError(e.to_string()))?;
    Ok(metadata)
}

/// Validate a file's integrity using metadata
///
/// # Arguments
///
/// * `file_path` - Path to the file to validate
/// * `metadata` - The integrity metadata to validate against
///
/// # Returns
///
/// Result with validation details:
/// - Ok(true) - Validation passed
/// - Ok(false) - Validation failed
/// - Err(e) - Error during validation
pub fn validate_file_integrity<P: AsRef<Path>>(
    file_path: P,
    metadata: &IntegrityMetadata,
) -> Result<bool> {
    let file_path = file_path.as_ref();

    // Check if file exists
    if !file_path.exists() {
        return Err(IoError::ValidationError(format!(
            "File does not exist: {}",
            file_path.display()
        )));
    }

    // Get file size
    let file_size = std::fs::metadata(file_path)
        .map_err(|e| IoError::FileError(e.to_string()))?
        .len();

    // Check file size
    if file_size != metadata.size {
        return Ok(false);
    }

    // Parse algorithm
    let algorithm = ChecksumAlgorithm::from_str(&metadata.algorithm).ok_or_else(|| {
        IoError::ValidationError(format!(
            "Unknown checksum algorithm: {}",
            metadata.algorithm
        ))
    })?;

    // Verify checksum
    verify_file_checksum(file_path, &metadata.checksum, algorithm)
}

/// Generate a validation report for a file
///
/// # Arguments
///
/// * `file_path` - Path to the file to validate
/// * `metadata` - The integrity metadata to validate against
///
/// # Returns
///
/// A struct containing validation results
pub fn generate_validation_report<P: AsRef<Path>>(
    file_path: P,
    metadata: &IntegrityMetadata,
) -> Result<ValidationReport> {
    let file_path = file_path.as_ref();

    // Check if file exists
    if !file_path.exists() {
        return Err(IoError::ValidationError(format!(
            "File does not exist: {}",
            file_path.display()
        )));
    }

    // Get file metadata
    let file_metadata = std::fs::metadata(file_path)
        .map_err(|e| IoError::FileError(format!("Failed to read file metadata: {}", e)))?;

    let actual_size = file_metadata.len();
    let size_valid = actual_size == metadata.size;

    // Parse algorithm
    let algorithm = match ChecksumAlgorithm::from_str(&metadata.algorithm) {
        Some(algo) => algo,
        None => {
            return Err(IoError::ValidationError(format!(
                "Unknown checksum algorithm: {}",
                metadata.algorithm
            )))
        }
    };

    // Calculate and check checksum
    let actual_checksum = calculate_file_checksum(file_path, algorithm)?;
    let checksum_valid = actual_checksum.eq_ignore_ascii_case(&metadata.checksum);

    // Overall validity
    let valid = size_valid && checksum_valid;

    Ok(ValidationReport {
        file_path: file_path.to_string_lossy().to_string(),
        expected_size: metadata.size,
        actual_size,
        size_valid,
        expected_checksum: metadata.checksum.clone(),
        actual_checksum,
        checksum_valid,
        algorithm: metadata.algorithm.clone(),
        valid,
        validation_time: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

/// Report containing the results of a file validation check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Path to the validated file
    pub file_path: String,
    /// Expected file size in bytes
    pub expected_size: u64,
    /// Actual file size in bytes
    pub actual_size: u64,
    /// Whether the size check passed
    pub size_valid: bool,
    /// Expected checksum value
    pub expected_checksum: String,
    /// Actual calculated checksum value
    pub actual_checksum: String,
    /// Whether the checksum check passed
    pub checksum_valid: bool,
    /// Checksum algorithm used
    pub algorithm: String,
    /// Overall validation result
    pub valid: bool,
    /// Time of validation (Unix timestamp)
    pub validation_time: u64,
}

impl ValidationReport {
    /// Get a formatted validation report as a string
    pub fn format(&self) -> String {
        let status = if self.valid { "PASSED" } else { "FAILED" };

        format!(
            "Validation Report ({})\n\
             -------------------------------\n\
             File: {}\n\
             Algorithm: {}\n\
             Size Check: {} (Expected: {} bytes, Found: {} bytes)\n\
             Checksum Check: {} (Expected: {}, Found: {})\n\
             Validation Time: {}\n",
            status,
            self.file_path,
            self.algorithm,
            if self.size_valid { "PASSED" } else { "FAILED" },
            self.expected_size,
            self.actual_size,
            if self.checksum_valid {
                "PASSED"
            } else {
                "FAILED"
            },
            self.expected_checksum,
            self.actual_checksum,
            DateTime::<Utc>::from_timestamp(self.validation_time as i64, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "Unknown".to_string()),
        )
    }

    /// Save the validation report to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        Ok(())
    }
}

/// Data source used for format validation
pub enum ValidationSource<'a> {
    /// Raw data bytes
    Data(&'a [u8]),
    /// File path
    FilePath(&'a Path),
}

/// Format validation rule with type and validation function
pub struct FormatValidator {
    /// Name of the format
    pub format_name: String,
    /// Function to validate the format
    validator: Box<dyn Fn(&[u8]) -> bool + Send + Sync>, // Type complexity is necessary here
}

impl FormatValidator {
    /// Create a new format validator
    pub fn new<F>(format_name: &str, validator: F) -> Self
    where
        F: Fn(&[u8]) -> bool + Send + Sync + 'static,
    {
        Self {
            format_name: format_name.to_string(),
            validator: Box::new(validator),
        }
    }

    /// Validate data against this format
    pub fn validate(&self, data: &[u8]) -> bool {
        (self.validator)(data)
    }
}

/// Registry of available format validators
pub struct FormatValidatorRegistry {
    validators: Vec<FormatValidator>,
}

impl Default for FormatValidatorRegistry {
    fn default() -> Self {
        let mut registry = Self {
            validators: Vec::new(),
        };

        // Add default validators
        registry.add_default_validators();

        registry
    }
}

impl FormatValidatorRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to the registry
    pub fn add_validator<F>(&mut self, format_name: &str, validator: F)
    where
        F: Fn(&[u8]) -> bool + Send + Sync + 'static,
    {
        self.validators
            .push(FormatValidator::new(format_name, validator));
    }

    /// Check if data matches any registered format
    pub fn validate_format(&self, source: ValidationSource) -> Result<Option<String>> {
        // Get data as bytes
        let data = match source {
            ValidationSource::Data(bytes) => bytes.to_vec(),
            ValidationSource::FilePath(path) => {
                let mut file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;

                // Read first 8192 bytes for format detection
                let mut buffer = Vec::with_capacity(8192);
                file.read_to_end(&mut buffer)
                    .map_err(|e| IoError::FileError(e.to_string()))?;

                buffer
            }
        };

        // Check all validators
        for validator in &self.validators {
            if validator.validate(&data) {
                return Ok(Some(validator.format_name.clone()));
            }
        }

        Ok(None)
    }

    /// Add default format validators
    fn add_default_validators(&mut self) {
        // PNG validator
        self.add_validator("PNG", |data| {
            data.len() >= 8 && data[0..8] == [137, 80, 78, 71, 13, 10, 26, 10]
        });

        // JPEG validator
        self.add_validator("JPEG", |data| {
            data.len() >= 3 && data[0..3] == [0xFF, 0xD8, 0xFF]
        });

        // GZIP validator
        self.add_validator("GZIP", |data| data.len() >= 2 && data[0..2] == [0x1F, 0x8B]);

        // ZIP validator
        self.add_validator("ZIP", |data| {
            data.len() >= 4 && data[0..4] == [0x50, 0x4B, 0x03, 0x04]
        });

        // JSON validator (very basic check)
        self.add_validator("JSON", |data| {
            if data.is_empty() {
                return false;
            }

            // Look for { or [ as first non-whitespace
            for &byte in data {
                if byte == b'{' || byte == b'[' {
                    return true;
                }
                if !byte.is_ascii_whitespace() {
                    return false;
                }
            }

            false
        });

        // CSV validator (very basic check)
        self.add_validator("CSV", |data| {
            if data.is_empty() {
                return false;
            }

            // Check for commas and newlines
            let has_comma = data.contains(&b',');
            let has_newline = data.contains(&b'\n');

            has_comma && has_newline
        });
    }
}

/// Check if a file exists and has the expected size
pub fn validate_file_exists_with_size<P: AsRef<Path>>(
    path: P,
    expected_size: Option<u64>,
) -> Result<bool> {
    let path = path.as_ref();

    if !path.exists() {
        return Ok(false);
    }

    if let Some(size) = expected_size {
        let file_size = std::fs::metadata(path)
            .map_err(|e| IoError::FileError(e.to_string()))?
            .len();

        Ok(file_size == size)
    } else {
        Ok(true)
    }
}

/// Utility to create a checksum file for a data file
///
/// # Arguments
///
/// * `data_path` - Path to the data file
/// * `algorithm` - The checksum algorithm to use
/// * `output_path` - Optional path to save the checksum file (if None, uses data_path + ".checksum")
///
/// # Returns
///
/// Result with the path to the checksum file
pub fn create_checksum_file<P, Q>(
    data_path: P,
    algorithm: ChecksumAlgorithm,
    output_path: Option<Q>,
) -> Result<String>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let data_path = data_path.as_ref();

    // Calculate checksum
    let checksum = calculate_file_checksum(data_path, algorithm)?;

    // Determine output path
    let output_path = match output_path {
        Some(path) => path.as_ref().to_path_buf(),
        None => {
            let mut path = data_path.to_path_buf();
            path.set_extension(format!(
                "{}.checksum",
                path.extension().unwrap_or_default().to_string_lossy()
            ));
            path
        }
    };

    // Generate content
    let content = format!(
        "{} *{}\n",
        checksum,
        data_path.file_name().unwrap_or_default().to_string_lossy()
    );

    // Write checksum file
    let mut file = File::create(&output_path)
        .map_err(|e| IoError::FileError(format!("Failed to create checksum file: {}", e)))?;

    file.write_all(content.as_bytes())
        .map_err(|e| IoError::FileError(format!("Failed to write checksum file: {}", e)))?;

    Ok(output_path.to_string_lossy().to_string())
}

/// Verify a file against a checksum file
///
/// # Arguments
///
/// * `data_path` - Path to the data file
/// * `checksum_path` - Path to the checksum file
///
/// # Returns
///
/// Result indicating if the verification passed
pub fn verify_checksum_file<P, Q>(data_path: P, checksum_path: Q) -> Result<bool>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let data_path = data_path.as_ref();

    // Read checksum file
    let mut checksum_file = File::open(checksum_path)
        .map_err(|e| IoError::FileError(format!("Failed to open checksum file: {}", e)))?;

    let mut content = String::new();
    checksum_file
        .read_to_string(&mut content)
        .map_err(|e| IoError::FileError(format!("Failed to read checksum file: {}", e)))?;

    // Parse checksum file (format: "<checksum> *<filename>")
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(IoError::ValidationError(
            "Invalid checksum file format".to_string(),
        ));
    }

    let expected_checksum = parts[0];

    // Determine algorithm from checksum length
    let algorithm = match expected_checksum.len() {
        8 => ChecksumAlgorithm::CRC32,
        64 => ChecksumAlgorithm::SHA256,
        // BLAKE3 produces 64 hex chars by default
        _ => {
            return Err(IoError::ValidationError(format!(
                "Unable to determine checksum algorithm from length: {}",
                expected_checksum.len()
            )))
        }
    };

    // Calculate actual checksum
    let actual_checksum = calculate_file_checksum(data_path, algorithm)?;

    // Compare checksums
    Ok(actual_checksum.eq_ignore_ascii_case(expected_checksum))
}

/// Add integrity metadata to an array of objects
pub fn add_integrity_metadata<T: Serialize>(
    data: &[T],
    algorithm: ChecksumAlgorithm,
) -> Result<std::collections::HashMap<String, String>> {
    // Serialize the data to calculate checksum
    let serialized =
        serde_json::to_vec(data).map_err(|e| IoError::SerializationError(e.to_string()))?;

    // Calculate checksum
    let checksum = calculate_checksum(&serialized, algorithm);

    // Create metadata
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("algorithm".to_string(), algorithm.as_str().to_string());
    metadata.insert("checksum".to_string(), checksum);
    metadata.insert("length".to_string(), data.len().to_string());
    metadata.insert(
        "timestamp".to_string(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string(),
    );

    Ok(metadata)
}

/// Verify array integrity using metadata
pub fn verify_array_integrity<T: Serialize>(
    data: &[T],
    metadata: &std::collections::HashMap<String, String>,
) -> Result<bool> {
    // Check array length
    if let Some(length) = metadata.get("length") {
        if let Ok(expected_length) = length.parse::<usize>() {
            if data.len() != expected_length {
                return Ok(false);
            }
        } else {
            return Err(IoError::ValidationError(
                "Invalid length in metadata".to_string(),
            ));
        }
    } else {
        return Err(IoError::ValidationError(
            "Missing length in metadata".to_string(),
        ));
    }

    // Get algorithm and checksum
    let algorithm_str = metadata
        .get("algorithm")
        .ok_or_else(|| IoError::ValidationError("Missing algorithm in metadata".to_string()))?;

    let algorithm = ChecksumAlgorithm::from_str(algorithm_str).ok_or_else(|| {
        IoError::ValidationError(format!("Unknown algorithm in metadata: {}", algorithm_str))
    })?;

    let expected_checksum = metadata
        .get("checksum")
        .ok_or_else(|| IoError::ValidationError("Missing checksum in metadata".to_string()))?;

    // Serialize the data to calculate checksum
    let serialized =
        serde_json::to_vec(data).map_err(|e| IoError::SerializationError(e.to_string()))?;

    // Calculate and compare checksums
    let actual_checksum = calculate_checksum(&serialized, algorithm);

    Ok(actual_checksum.eq_ignore_ascii_case(expected_checksum))
}

/// Create a manifest file for a directory with checksums
pub fn create_directory_manifest<P, Q>(
    dir_path: P,
    output_path: Q,
    algorithm: ChecksumAlgorithm,
    recursive: bool,
) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let dir_path = dir_path.as_ref();

    // Check if directory exists
    if !dir_path.is_dir() {
        return Err(IoError::ValidationError(format!(
            "Not a directory: {}",
            dir_path.display()
        )));
    }

    // Collect all files
    let mut files = Vec::new();
    collect_files(dir_path, &mut files, recursive)?;

    // Calculate checksums and create manifest entries
    let mut manifest = Vec::new();

    for file_path in files {
        let relative_path = file_path
            .strip_prefix(dir_path)
            .map_err(|e| IoError::ValidationError(e.to_string()))?;

        let checksum = calculate_file_checksum(&file_path, algorithm)?;
        let size = std::fs::metadata(&file_path)
            .map_err(|e| IoError::FileError(e.to_string()))?
            .len();

        manifest.push(ManifestEntry {
            path: relative_path.to_string_lossy().to_string(),
            size,
            checksum,
        });
    }

    // Create manifest file
    let manifest_file = DirectoryManifest {
        directory: dir_path.to_string_lossy().to_string(),
        algorithm: algorithm.as_str().to_string(),
        creation_time: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        files: manifest,
    };

    // Save to output file
    let file = File::create(output_path).map_err(|e| IoError::FileError(e.to_string()))?;

    serde_json::to_writer_pretty(file, &manifest_file)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    Ok(())
}

/// Helper function to collect files in a directory
fn collect_files(dir: &Path, files: &mut Vec<std::path::PathBuf>, recursive: bool) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .map_err(|e| IoError::FileError(format!("Failed to read directory: {}", e)))?
    {
        let entry = entry.map_err(|e| IoError::FileError(e.to_string()))?;
        let path = entry.path();

        if path.is_file() {
            files.push(path);
        } else if path.is_dir() && recursive {
            collect_files(&path, files, recursive)?;
        }
    }

    Ok(())
}

/// Entry in a directory manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Relative path to the file
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// Checksum value
    pub checksum: String,
}

/// Manifest file for a directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryManifest {
    /// Directory path
    pub directory: String,
    /// Checksum algorithm used
    pub algorithm: String,
    /// Creation time (Unix timestamp)
    pub creation_time: u64,
    /// Files in the directory
    pub files: Vec<ManifestEntry>,
}

impl DirectoryManifest {
    /// Verify a directory against the manifest
    pub fn verify_directory<P: AsRef<Path>>(
        &self,
        dir_path: P,
    ) -> Result<ManifestVerificationReport> {
        let dir_path = dir_path.as_ref();

        // Check if directory exists
        if !dir_path.is_dir() {
            return Err(IoError::ValidationError(format!(
                "Not a directory: {}",
                dir_path.display()
            )));
        }

        // Parse algorithm
        let algorithm = ChecksumAlgorithm::from_str(&self.algorithm).ok_or_else(|| {
            IoError::ValidationError(format!("Unknown algorithm in manifest: {}", self.algorithm))
        })?;

        // Check each file
        let mut missing_files = Vec::new();
        let mut modified_files = Vec::new();
        let mut verified_files = Vec::new();

        for entry in &self.files {
            let file_path = dir_path.join(&entry.path);

            // Check if file exists
            if !file_path.exists() {
                missing_files.push(entry.path.clone());
                continue;
            }

            // Check file size
            let file_size = std::fs::metadata(&file_path)
                .map_err(|e| IoError::FileError(e.to_string()))?
                .len();

            if file_size != entry.size {
                modified_files.push(entry.path.clone());
                continue;
            }

            // Check checksum
            let checksum = calculate_file_checksum(&file_path, algorithm)?;
            if !checksum.eq_ignore_ascii_case(&entry.checksum) {
                modified_files.push(entry.path.clone());
                continue;
            }

            // File verified
            verified_files.push(entry.path.clone());
        }

        Ok(ManifestVerificationReport {
            directory: dir_path.to_string_lossy().to_string(),
            total_files: self.files.len(),
            verified_files,
            missing_files,
            modified_files,
            verification_time: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
}

/// Report from a manifest verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestVerificationReport {
    /// Directory that was verified
    pub directory: String,
    /// Total number of files in the manifest
    pub total_files: usize,
    /// Files that were successfully verified
    pub verified_files: Vec<String>,
    /// Files that were missing
    pub missing_files: Vec<String>,
    /// Files that were modified
    pub modified_files: Vec<String>,
    /// Time of verification (Unix timestamp)
    pub verification_time: u64,
}

impl ManifestVerificationReport {
    /// Check if the verification passed (all files verified)
    pub fn passed(&self) -> bool {
        self.missing_files.is_empty() && self.modified_files.is_empty()
    }

    /// Get a formatted report as a string
    pub fn format(&self) -> String {
        let status = if self.passed() { "PASSED" } else { "FAILED" };

        let mut report = format!(
            "Manifest Verification Report ({})\n\
             -------------------------------------\n\
             Directory: {}\n\
             Total Files: {}\n\
             Verified: {} files\n",
            status,
            self.directory,
            self.total_files,
            self.verified_files.len(),
        );

        if !self.missing_files.is_empty() {
            report.push_str(&format!(
                "\nMissing Files ({}):\n",
                self.missing_files.len()
            ));
            for file in &self.missing_files {
                report.push_str(&format!("  - {}\n", file));
            }
        }

        if !self.modified_files.is_empty() {
            report.push_str(&format!(
                "\nModified Files ({}):\n",
                self.modified_files.len()
            ));
            for file in &self.modified_files {
                report.push_str(&format!("  - {}\n", file));
            }
        }

        report.push_str(&format!(
            "\nVerification Time: {}\n",
            DateTime::<Utc>::from_timestamp(self.verification_time as i64, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "Unknown".to_string()),
        ));

        report
    }

    /// Save the report to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        Ok(())
    }
}

// Convenience functions for common checksum algorithms

/// Convenience function to calculate CRC32 checksum for a file
pub fn calculate_crc32<P: AsRef<Path>>(path: P) -> Result<String> {
    calculate_file_checksum(path, ChecksumAlgorithm::CRC32)
}

/// Convenience function to calculate SHA256 checksum for a file
pub fn calculate_sha256<P: AsRef<Path>>(path: P) -> Result<String> {
    calculate_file_checksum(path, ChecksumAlgorithm::SHA256)
}

//
// Schema-based Validation
//

use std::collections::HashMap;

/// Data type for schema validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaDataType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Number type (floating point)
    Number,
    /// Boolean type
    Boolean,
    /// Array type with element schema
    Array(Box<SchemaDefinition>),
    /// Object type with property schemas
    Object(HashMap<String, SchemaDefinition>),
    /// Union type (any of the specified types)
    Union(Vec<SchemaDataType>),
    /// Null/None type
    Null,
}

/// Schema constraint for validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaConstraint {
    /// Minimum value (for numbers)
    MinValue(f64),
    /// Maximum value (for numbers)
    MaxValue(f64),
    /// Minimum length (for strings and arrays)
    MinLength(usize),
    /// Maximum length (for strings and arrays)
    MaxLength(usize),
    /// Pattern match (regex for strings)
    Pattern(String),
    /// Enumeration of allowed values
    Enum(Vec<serde_json::Value>),
    /// Required (cannot be null/missing)
    Required,
    /// Unique items (for arrays)
    UniqueItems,
    /// Format specification (email, date, etc.)
    Format(String),
    /// Custom validation function name
    Custom(String),
}

/// Schema definition for data validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Type of the data
    pub data_type: SchemaDataType,
    /// Constraints to apply
    pub constraints: Vec<SchemaConstraint>,
    /// Human-readable description
    pub description: Option<String>,
    /// Default value if not provided
    pub default: Option<serde_json::Value>,
    /// Whether this field is optional
    pub optional: bool,
}

impl SchemaDefinition {
    /// Create a new schema definition
    pub fn new(data_type: SchemaDataType) -> Self {
        Self {
            data_type,
            constraints: Vec::new(),
            description: None,
            default: None,
            optional: false,
        }
    }

    /// Add a constraint to the schema
    pub fn with_constraint(mut self, constraint: SchemaConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add multiple constraints to the schema
    pub fn with_constraints(mut self, constraints: Vec<SchemaConstraint>) -> Self {
        self.constraints.extend(constraints);
        self
    }

    /// Set description for the schema
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set default value for the schema
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Mark the schema as optional
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Mark the schema as required
    pub fn required(mut self) -> Self {
        self.optional = false;
        self.constraints.push(SchemaConstraint::Required);
        self
    }
}

/// Schema validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationError {
    /// Path to the field that failed validation
    pub path: String,
    /// Expected type or constraint
    pub expected: String,
    /// Actual value that caused the error
    pub actual: String,
    /// Error message
    pub message: String,
}

/// Result of schema validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// List of validation errors
    pub errors: Vec<SchemaValidationError>,
    /// Total number of fields validated
    pub fields_validated: usize,
    /// Schema validation took (in milliseconds)
    pub validation_time_ms: f64,
}

impl SchemaValidationResult {
    /// Create a successful validation result
    pub fn success(fields_validated: usize, validation_time_ms: f64) -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            fields_validated,
            validation_time_ms,
        }
    }

    /// Create a failed validation result
    pub fn failure(
        errors: Vec<SchemaValidationError>,
        fields_validated: usize,
        validation_time_ms: f64,
    ) -> Self {
        Self {
            valid: false,
            errors,
            fields_validated,
            validation_time_ms,
        }
    }

    /// Get a formatted report of the validation result
    pub fn format_report(&self) -> String {
        let status = if self.valid { "PASSED" } else { "FAILED" };

        let mut report = format!(
            "Schema Validation Report ({})\n\
             --------------------------------\n\
             Fields Validated: {}\n\
             Validation Time: {:.2}ms\n",
            status, self.fields_validated, self.validation_time_ms
        );

        if !self.errors.is_empty() {
            report.push_str(&format!("\nValidation Errors ({}):\n", self.errors.len()));
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. Path: {}\n     Expected: {}\n     Actual: {}\n     Message: {}\n",
                    i + 1,
                    error.path,
                    error.expected,
                    error.actual,
                    error.message
                ));
            }
        }

        report
    }
}

/// Schema validator that can validate data against schemas
pub struct SchemaValidator {
    /// Custom validation functions
    custom_validators: HashMap<String, Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>>,
    /// Format validators (email, date, etc.)
    format_validators: HashMap<String, Box<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl Default for SchemaValidator {
    fn default() -> Self {
        let mut validator = Self {
            custom_validators: HashMap::new(),
            format_validators: HashMap::new(),
        };

        // Add default format validators
        validator.add_default_format_validators();

        validator
    }
}

impl SchemaValidator {
    /// Create a new schema validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom validation function
    pub fn add_custom_validator<F>(&mut self, name: &str, validator: F)
    where
        F: Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    {
        self.custom_validators
            .insert(name.to_string(), Box::new(validator));
    }

    /// Add a format validation function
    pub fn add_format_validator<F>(&mut self, format: &str, validator: F)
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.format_validators
            .insert(format.to_string(), Box::new(validator));
    }

    /// Validate data against a schema
    pub fn validate(
        &self,
        data: &serde_json::Value,
        schema: &SchemaDefinition,
    ) -> SchemaValidationResult {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut fields_validated = 0;

        self.validate_recursive(data, schema, "", &mut errors, &mut fields_validated);

        let validation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        if errors.is_empty() {
            SchemaValidationResult::success(fields_validated, validation_time_ms)
        } else {
            SchemaValidationResult::failure(errors, fields_validated, validation_time_ms)
        }
    }

    /// Validate JSON data against a schema
    pub fn validate_json(
        &self,
        json_str: &str,
        schema: &SchemaDefinition,
    ) -> Result<SchemaValidationResult> {
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| IoError::ValidationError(format!("Invalid JSON: {}", e)))?;

        Ok(self.validate(&data, schema))
    }

    /// Validate a file's content against a schema
    pub fn validate_file<P: AsRef<Path>>(
        &self,
        path: P,
        schema: &SchemaDefinition,
    ) -> Result<SchemaValidationResult> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| IoError::FileError(format!("Failed to read file: {}", e)))?;

        self.validate_json(&content, schema)
    }

    /// Recursive validation function
    fn validate_recursive(
        &self,
        data: &serde_json::Value,
        schema: &SchemaDefinition,
        path: &str,
        errors: &mut Vec<SchemaValidationError>,
        fields_validated: &mut usize,
    ) {
        *fields_validated += 1;

        // Check if value is null and schema is optional
        if data.is_null() {
            if !schema.optional {
                errors.push(SchemaValidationError {
                    path: path.to_string(),
                    expected: "non-null value".to_string(),
                    actual: "null".to_string(),
                    message: "Required field is null".to_string(),
                });
            }
            return;
        }

        // Validate data type
        if !self.validate_type(data, &schema.data_type) {
            errors.push(SchemaValidationError {
                path: path.to_string(),
                expected: format!("{:?}", schema.data_type),
                actual: self.get_value_type_string(data),
                message: "Type mismatch".to_string(),
            });
            return;
        }

        // Validate constraints
        for constraint in &schema.constraints {
            if let Some(error) = self.validate_constraint(data, constraint, path) {
                errors.push(error);
            }
        }

        // Recursively validate nested structures
        match &schema.data_type {
            SchemaDataType::Array(element_schema) => {
                if let Some(array) = data.as_array() {
                    for (i, item) in array.iter().enumerate() {
                        let item_path = if path.is_empty() {
                            format!("[{}]", i)
                        } else {
                            format!("{}[{}]", path, i)
                        };
                        self.validate_recursive(
                            item,
                            element_schema,
                            &item_path,
                            errors,
                            fields_validated,
                        );
                    }
                }
            }
            SchemaDataType::Object(properties) => {
                if let Some(object) = data.as_object() {
                    for (key, prop_schema) in properties {
                        let prop_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };

                        if let Some(prop_value) = object.get(key) {
                            self.validate_recursive(
                                prop_value,
                                prop_schema,
                                &prop_path,
                                errors,
                                fields_validated,
                            );
                        } else if !prop_schema.optional {
                            errors.push(SchemaValidationError {
                                path: prop_path,
                                expected: "required property".to_string(),
                                actual: "missing".to_string(),
                                message: format!("Required property '{}' is missing", key),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Validate data type
    #[allow(clippy::only_used_in_recursion)]
    fn validate_type(&self, data: &serde_json::Value, schema_type: &SchemaDataType) -> bool {
        match schema_type {
            SchemaDataType::String => data.is_string(),
            SchemaDataType::Integer => data.is_i64() || data.is_u64(),
            SchemaDataType::Number => data.is_number(),
            SchemaDataType::Boolean => data.is_boolean(),
            SchemaDataType::Array(_) => data.is_array(),
            SchemaDataType::Object(_) => data.is_object(),
            SchemaDataType::Null => data.is_null(),
            SchemaDataType::Union(types) => types.iter().any(|t| self.validate_type(data, t)),
        }
    }

    /// Validate a single constraint
    fn validate_constraint(
        &self,
        data: &serde_json::Value,
        constraint: &SchemaConstraint,
        path: &str,
    ) -> Option<SchemaValidationError> {
        match constraint {
            SchemaConstraint::MinValue(min) => {
                if let Some(num) = data.as_f64() {
                    if num < *min {
                        return Some(SchemaValidationError {
                            path: path.to_string(),
                            expected: format!("value >= {}", min),
                            actual: num.to_string(),
                            message: format!("Value {} is less than minimum {}", num, min),
                        });
                    }
                }
            }
            SchemaConstraint::MaxValue(max) => {
                if let Some(num) = data.as_f64() {
                    if num > *max {
                        return Some(SchemaValidationError {
                            path: path.to_string(),
                            expected: format!("value <= {}", max),
                            actual: num.to_string(),
                            message: format!("Value {} is greater than maximum {}", num, max),
                        });
                    }
                }
            }
            SchemaConstraint::MinLength(min_len) => {
                let len = if let Some(s) = data.as_str() {
                    s.len()
                } else if let Some(arr) = data.as_array() {
                    arr.len()
                } else {
                    return None;
                };

                if len < *min_len {
                    return Some(SchemaValidationError {
                        path: path.to_string(),
                        expected: format!("length >= {}", min_len),
                        actual: len.to_string(),
                        message: format!("Length {} is less than minimum {}", len, min_len),
                    });
                }
            }
            SchemaConstraint::MaxLength(max_len) => {
                let len = if let Some(s) = data.as_str() {
                    s.len()
                } else if let Some(arr) = data.as_array() {
                    arr.len()
                } else {
                    return None;
                };

                if len > *max_len {
                    return Some(SchemaValidationError {
                        path: path.to_string(),
                        expected: format!("length <= {}", max_len),
                        actual: len.to_string(),
                        message: format!("Length {} is greater than maximum {}", len, max_len),
                    });
                }
            }
            SchemaConstraint::Pattern(pattern) => {
                if let Some(s) = data.as_str() {
                    if let Ok(regex) = regex::Regex::new(pattern) {
                        if !regex.is_match(s) {
                            return Some(SchemaValidationError {
                                path: path.to_string(),
                                expected: format!("pattern: {}", pattern),
                                actual: s.to_string(),
                                message: format!(
                                    "String '{}' does not match pattern '{}'",
                                    s, pattern
                                ),
                            });
                        }
                    }
                }
            }
            SchemaConstraint::Enum(allowed_values) => {
                if !allowed_values.contains(data) {
                    return Some(SchemaValidationError {
                        path: path.to_string(),
                        expected: format!("one of: {:?}", allowed_values),
                        actual: data.to_string(),
                        message: "Value is not in the allowed enumeration".to_string(),
                    });
                }
            }
            SchemaConstraint::Required => {
                if data.is_null() {
                    return Some(SchemaValidationError {
                        path: path.to_string(),
                        expected: "non-null value".to_string(),
                        actual: "null".to_string(),
                        message: "Required field cannot be null".to_string(),
                    });
                }
            }
            SchemaConstraint::UniqueItems => {
                if let Some(arr) = data.as_array() {
                    let mut seen = std::collections::HashSet::new();
                    for item in arr {
                        if !seen.insert(item.to_string()) {
                            return Some(SchemaValidationError {
                                path: path.to_string(),
                                expected: "unique items".to_string(),
                                actual: "duplicate items found".to_string(),
                                message: "Array contains duplicate items".to_string(),
                            });
                        }
                    }
                }
            }
            SchemaConstraint::Format(format) => {
                if let Some(s) = data.as_str() {
                    if let Some(validator) = self.format_validators.get(format) {
                        if !validator(s) {
                            return Some(SchemaValidationError {
                                path: path.to_string(),
                                expected: format!("format: {}", format),
                                actual: s.to_string(),
                                message: format!(
                                    "String '{}' does not match format '{}'",
                                    s, format
                                ),
                            });
                        }
                    }
                }
            }
            SchemaConstraint::Custom(name) => {
                if let Some(validator) = self.custom_validators.get(name) {
                    if !validator(data) {
                        return Some(SchemaValidationError {
                            path: path.to_string(),
                            expected: format!("custom validation: {}", name),
                            actual: data.to_string(),
                            message: format!("Custom validation '{}' failed", name),
                        });
                    }
                }
            }
        }

        None
    }

    /// Get a string representation of a JSON value's type
    fn get_value_type_string(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Null => "null".to_string(),
            serde_json::Value::Bool(_) => "boolean".to_string(),
            serde_json::Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    "integer".to_string()
                } else {
                    "number".to_string()
                }
            }
            serde_json::Value::String(_) => "string".to_string(),
            serde_json::Value::Array(_) => "array".to_string(),
            serde_json::Value::Object(_) => "object".to_string(),
        }
    }

    /// Add default format validators
    fn add_default_format_validators(&mut self) {
        // Email format validator
        self.add_format_validator("email", |s| {
            let email_regex =
                regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
            email_regex.is_match(s)
        });

        // Date format validator (ISO 8601)
        self.add_format_validator("date", |s| {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok()
        });

        // DateTime format validator (ISO 8601)
        self.add_format_validator("date-time", |s| {
            chrono::DateTime::parse_from_rfc3339(s).is_ok()
        });

        // UUID format validator
        self.add_format_validator("uuid", |s| {
            s.len() == 36
                && s.chars().enumerate().all(|(i, c)| match i {
                    8 | 13 | 18 | 23 => c == '-',
                    _ => c.is_ascii_hexdigit(),
                })
        });

        // URI format validator (basic)
        self.add_format_validator("uri", |s| s.contains("://") && !s.is_empty());

        // IPv4 format validator
        self.add_format_validator("ipv4", |s| {
            let parts: Vec<&str> = s.split('.').collect();
            if parts.len() != 4 {
                return false;
            }
            parts.iter().all(|part| {
                if let Ok(num) = part.parse::<u32>() {
                    num <= 255
                } else {
                    false
                }
            })
        });
    }
}

/// Helper function to create common schema types
pub mod schema_helpers {
    use super::*;

    /// Create a string schema
    pub fn string() -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::String)
    }

    /// Create an integer schema
    pub fn integer() -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::Integer)
    }

    /// Create a number schema
    pub fn number() -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::Number)
    }

    /// Create a boolean schema
    pub fn boolean() -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::Boolean)
    }

    /// Create an array schema
    pub fn array(element_schema: SchemaDefinition) -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::Array(Box::new(element_schema)))
    }

    /// Create an object schema
    pub fn object(properties: HashMap<String, SchemaDefinition>) -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::Object(properties))
    }

    /// Create a union schema
    pub fn union(types: Vec<SchemaDataType>) -> SchemaDefinition {
        SchemaDefinition::new(SchemaDataType::Union(types))
    }

    /// Create an email string schema
    pub fn email() -> SchemaDefinition {
        string().with_constraint(SchemaConstraint::Format("email".to_string()))
    }

    /// Create a date string schema
    pub fn date() -> SchemaDefinition {
        string().with_constraint(SchemaConstraint::Format("date".to_string()))
    }

    /// Create a UUID string schema
    pub fn uuid() -> SchemaDefinition {
        string().with_constraint(SchemaConstraint::Format("uuid".to_string()))
    }

    /// Create a positive integer schema
    pub fn positive_integer() -> SchemaDefinition {
        integer().with_constraint(SchemaConstraint::MinValue(1.0))
    }

    /// Create a non-negative number schema
    pub fn non_negative_number() -> SchemaDefinition {
        number().with_constraint(SchemaConstraint::MinValue(0.0))
    }
}

/// Build schemas from JSON Schema format
pub fn schema_from_json_schema(json_schema: &serde_json::Value) -> Result<SchemaDefinition> {
    let object = json_schema
        .as_object()
        .ok_or_else(|| IoError::ValidationError("Schema must be an object".to_string()))?;

    let type_name = object
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| IoError::ValidationError("Schema must have a 'type' field".to_string()))?;

    let data_type = match type_name {
        "string" => SchemaDataType::String,
        "integer" => SchemaDataType::Integer,
        "number" => SchemaDataType::Number,
        "boolean" => SchemaDataType::Boolean,
        "array" => {
            let items = object.get("items").ok_or_else(|| {
                IoError::ValidationError("Array schema must have 'items'".to_string())
            })?;
            let element_schema = schema_from_json_schema(items)?;
            SchemaDataType::Array(Box::new(element_schema))
        }
        "object" => {
            let properties = object
                .get("properties")
                .and_then(|p| p.as_object())
                .ok_or_else(|| {
                    IoError::ValidationError("Object schema must have 'properties'".to_string())
                })?;

            let mut prop_schemas = HashMap::new();
            for (key, value) in properties {
                prop_schemas.insert(key.clone(), schema_from_json_schema(value)?);
            }
            SchemaDataType::Object(prop_schemas)
        }
        "null" => SchemaDataType::Null,
        _ => {
            return Err(IoError::ValidationError(format!(
                "Unknown type: {}",
                type_name
            )))
        }
    };

    let mut schema = SchemaDefinition::new(data_type);

    // Add constraints from JSON Schema
    if let Some(min) = object.get("minimum").and_then(|v| v.as_f64()) {
        schema = schema.with_constraint(SchemaConstraint::MinValue(min));
    }
    if let Some(max) = object.get("maximum").and_then(|v| v.as_f64()) {
        schema = schema.with_constraint(SchemaConstraint::MaxValue(max));
    }
    if let Some(min_len) = object.get("minLength").and_then(|v| v.as_u64()) {
        schema = schema.with_constraint(SchemaConstraint::MinLength(min_len as usize));
    }
    if let Some(max_len) = object.get("maxLength").and_then(|v| v.as_u64()) {
        schema = schema.with_constraint(SchemaConstraint::MaxLength(max_len as usize));
    }
    if let Some(pattern) = object.get("pattern").and_then(|v| v.as_str()) {
        schema = schema.with_constraint(SchemaConstraint::Pattern(pattern.to_string()));
    }
    if let Some(format) = object.get("format").and_then(|v| v.as_str()) {
        schema = schema.with_constraint(SchemaConstraint::Format(format.to_string()));
    }
    if let Some(description) = object.get("description").and_then(|v| v.as_str()) {
        schema = schema.with_description(description);
    }

    Ok(schema)
}
