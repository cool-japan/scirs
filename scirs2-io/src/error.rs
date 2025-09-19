//! Error types for the IO module

use std::error::Error;
use std::fmt;

/// Error type for IO operations
#[derive(Debug)]
pub enum IoError {
    /// File error
    FileError(String),
    /// Format error
    FormatError(String),
    /// Serialization error
    SerializationError(String),
    /// Deserialization error
    DeserializationError(String),
    /// Compression error
    CompressionError(String),
    /// Decompression error
    DecompressionError(String),
    /// Unsupported compression algorithm
    UnsupportedCompressionAlgorithm(String),
    /// Unsupported format
    UnsupportedFormat(String),
    /// Conversion error
    ConversionError(String),
    /// File not found
    FileNotFound(String),
    /// Record/resource not found
    NotFound(String),
    /// Parse error
    ParseError(String),
    /// Standard I/O error
    Io(std::io::Error),
    /// Validation error
    ValidationError(String),
    /// Checksum error
    ChecksumError(String),
    /// Integrity error
    IntegrityError(String),
    /// Configuration error
    ConfigError(String),
    /// Network error
    NetworkError(String),
    /// Database error
    DatabaseError(String),
    /// Timeout error
    TimeoutError(String),
    /// Other error
    Other(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IoError::FileError(msg) => write!(f, "File error: {msg}"),
            IoError::FormatError(msg) => write!(f, "Format error: {msg}"),
            IoError::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
            IoError::DeserializationError(msg) => write!(f, "Deserialization error: {msg}"),
            IoError::CompressionError(msg) => write!(f, "Compression error: {msg}"),
            IoError::DecompressionError(msg) => write!(f, "Decompression error: {msg}"),
            IoError::UnsupportedCompressionAlgorithm(algo) => {
                write!(f, "Unsupported compression algorithm: {algo}")
            }
            IoError::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
            IoError::ConversionError(msg) => write!(f, "Conversion error: {msg}"),
            IoError::FileNotFound(path) => write!(f, "File not found: {path}"),
            IoError::NotFound(msg) => write!(f, "Not found: {msg}"),
            IoError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            IoError::Io(e) => write!(f, "I/O error: {e}"),
            IoError::ValidationError(msg) => write!(f, "Validation error: {msg}"),
            IoError::ChecksumError(msg) => write!(f, "Checksum error: {msg}"),
            IoError::IntegrityError(msg) => write!(f, "Integrity error: {msg}"),
            IoError::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
            IoError::NetworkError(msg) => write!(f, "Network error: {msg}"),
            IoError::DatabaseError(msg) => write!(f, "Database error: {msg}"),
            IoError::TimeoutError(msg) => write!(f, "Timeout error: {msg}"),
            IoError::Other(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl Error for IoError {}

impl Clone for IoError {
    fn clone(&self) -> Self {
        match self {
            IoError::FileError(msg) => IoError::FileError(msg.clone()),
            IoError::FormatError(msg) => IoError::FormatError(msg.clone()),
            IoError::SerializationError(msg) => IoError::SerializationError(msg.clone()),
            IoError::DeserializationError(msg) => IoError::DeserializationError(msg.clone()),
            IoError::CompressionError(msg) => IoError::CompressionError(msg.clone()),
            IoError::DecompressionError(msg) => IoError::DecompressionError(msg.clone()),
            IoError::UnsupportedCompressionAlgorithm(algo) => {
                IoError::UnsupportedCompressionAlgorithm(algo.clone())
            }
            IoError::UnsupportedFormat(fmt) => IoError::UnsupportedFormat(fmt.clone()),
            IoError::ConversionError(msg) => IoError::ConversionError(msg.clone()),
            IoError::FileNotFound(path) => IoError::FileNotFound(path.clone()),
            IoError::NotFound(msg) => IoError::NotFound(msg.clone()),
            IoError::ParseError(msg) => IoError::ParseError(msg.clone()),
            IoError::Io(e) => IoError::Io(std::io::Error::new(e.kind(), e.to_string())),
            IoError::ValidationError(msg) => IoError::ValidationError(msg.clone()),
            IoError::ChecksumError(msg) => IoError::ChecksumError(msg.clone()),
            IoError::IntegrityError(msg) => IoError::IntegrityError(msg.clone()),
            IoError::ConfigError(msg) => IoError::ConfigError(msg.clone()),
            IoError::NetworkError(msg) => IoError::NetworkError(msg.clone()),
            IoError::DatabaseError(msg) => IoError::DatabaseError(msg.clone()),
            IoError::TimeoutError(msg) => IoError::TimeoutError(msg.clone()),
            IoError::Other(msg) => IoError::Other(msg.clone()),
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(err: std::io::Error) -> Self {
        IoError::Io(err)
    }
}

impl PartialEq for IoError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (IoError::FileError(a), IoError::FileError(b)) => a == b,
            (IoError::FormatError(a), IoError::FormatError(b)) => a == b,
            (IoError::SerializationError(a), IoError::SerializationError(b)) => a == b,
            (IoError::DeserializationError(a), IoError::DeserializationError(b)) => a == b,
            (IoError::CompressionError(a), IoError::CompressionError(b)) => a == b,
            (IoError::DecompressionError(a), IoError::DecompressionError(b)) => a == b,
            (
                IoError::UnsupportedCompressionAlgorithm(a),
                IoError::UnsupportedCompressionAlgorithm(b),
            ) => a == b,
            (IoError::UnsupportedFormat(a), IoError::UnsupportedFormat(b)) => a == b,
            (IoError::ConversionError(a), IoError::ConversionError(b)) => a == b,
            (IoError::FileNotFound(a), IoError::FileNotFound(b)) => a == b,
            (IoError::NotFound(a), IoError::NotFound(b)) => a == b,
            (IoError::ParseError(a), IoError::ParseError(b)) => a == b,
            (IoError::Io(a), IoError::Io(b)) => {
                a.kind() == b.kind() && a.to_string() == b.to_string()
            }
            (IoError::ValidationError(a), IoError::ValidationError(b)) => a == b,
            (IoError::ChecksumError(a), IoError::ChecksumError(b)) => a == b,
            (IoError::IntegrityError(a), IoError::IntegrityError(b)) => a == b,
            (IoError::ConfigError(a), IoError::ConfigError(b)) => a == b,
            (IoError::NetworkError(a), IoError::NetworkError(b)) => a == b,
            (IoError::DatabaseError(a), IoError::DatabaseError(b)) => a == b,
            (IoError::TimeoutError(a), IoError::TimeoutError(b)) => a == b,
            (IoError::Other(a), IoError::Other(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for IoError {}

impl PartialOrd for IoError {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IoError {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Compare variants first by their order in the enum
        match (self, other) {
            // Same variants, compare contents
            (IoError::FileError(a), IoError::FileError(b)) => a.cmp(b),
            (IoError::FormatError(a), IoError::FormatError(b)) => a.cmp(b),
            (IoError::SerializationError(a), IoError::SerializationError(b)) => a.cmp(b),
            (IoError::DeserializationError(a), IoError::DeserializationError(b)) => a.cmp(b),
            (IoError::CompressionError(a), IoError::CompressionError(b)) => a.cmp(b),
            (IoError::DecompressionError(a), IoError::DecompressionError(b)) => a.cmp(b),
            (
                IoError::UnsupportedCompressionAlgorithm(a),
                IoError::UnsupportedCompressionAlgorithm(b),
            ) => a.cmp(b),
            (IoError::UnsupportedFormat(a), IoError::UnsupportedFormat(b)) => a.cmp(b),
            (IoError::ConversionError(a), IoError::ConversionError(b)) => a.cmp(b),
            (IoError::FileNotFound(a), IoError::FileNotFound(b)) => a.cmp(b),
            (IoError::NotFound(a), IoError::NotFound(b)) => a.cmp(b),
            (IoError::ParseError(a), IoError::ParseError(b)) => a.cmp(b),
            (IoError::Io(a), IoError::Io(b)) => {
                // Compare by kind first, then by string representation
                match (a.kind() as u8).cmp(&(b.kind() as u8)) {
                    Ordering::Equal => a.to_string().cmp(&b.to_string()),
                    other => other,
                }
            }
            (IoError::ValidationError(a), IoError::ValidationError(b)) => a.cmp(b),
            (IoError::ChecksumError(a), IoError::ChecksumError(b)) => a.cmp(b),
            (IoError::IntegrityError(a), IoError::IntegrityError(b)) => a.cmp(b),
            (IoError::ConfigError(a), IoError::ConfigError(b)) => a.cmp(b),
            (IoError::NetworkError(a), IoError::NetworkError(b)) => a.cmp(b),
            (IoError::DatabaseError(a), IoError::DatabaseError(b)) => a.cmp(b),
            (IoError::TimeoutError(a), IoError::TimeoutError(b)) => a.cmp(b),
            (IoError::Other(a), IoError::Other(b)) => a.cmp(b),

            // Different variants, order by enum variant position
            (IoError::FileError(_), _) => Ordering::Less,
            (_, IoError::FileError(_)) => Ordering::Greater,
            (IoError::FormatError(_), _) => Ordering::Less,
            (_, IoError::FormatError(_)) => Ordering::Greater,
            (IoError::SerializationError(_), _) => Ordering::Less,
            (_, IoError::SerializationError(_)) => Ordering::Greater,
            (IoError::DeserializationError(_), _) => Ordering::Less,
            (_, IoError::DeserializationError(_)) => Ordering::Greater,
            (IoError::CompressionError(_), _) => Ordering::Less,
            (_, IoError::CompressionError(_)) => Ordering::Greater,
            (IoError::DecompressionError(_), _) => Ordering::Less,
            (_, IoError::DecompressionError(_)) => Ordering::Greater,
            (IoError::UnsupportedCompressionAlgorithm(_), _) => Ordering::Less,
            (_, IoError::UnsupportedCompressionAlgorithm(_)) => Ordering::Greater,
            (IoError::UnsupportedFormat(_), _) => Ordering::Less,
            (_, IoError::UnsupportedFormat(_)) => Ordering::Greater,
            (IoError::ConversionError(_), _) => Ordering::Less,
            (_, IoError::ConversionError(_)) => Ordering::Greater,
            (IoError::FileNotFound(_), _) => Ordering::Less,
            (_, IoError::FileNotFound(_)) => Ordering::Greater,
            (IoError::NotFound(_), _) => Ordering::Less,
            (_, IoError::NotFound(_)) => Ordering::Greater,
            (IoError::ParseError(_), _) => Ordering::Less,
            (_, IoError::ParseError(_)) => Ordering::Greater,
            (IoError::Io(_), _) => Ordering::Less,
            (_, IoError::Io(_)) => Ordering::Greater,
            (IoError::ValidationError(_), _) => Ordering::Less,
            (_, IoError::ValidationError(_)) => Ordering::Greater,
            (IoError::ChecksumError(_), _) => Ordering::Less,
            (_, IoError::ChecksumError(_)) => Ordering::Greater,
            (IoError::IntegrityError(_), _) => Ordering::Less,
            (_, IoError::IntegrityError(_)) => Ordering::Greater,
            (IoError::ConfigError(_), _) => Ordering::Less,
            (_, IoError::ConfigError(_)) => Ordering::Greater,
            (IoError::NetworkError(_), _) => Ordering::Less,
            (_, IoError::NetworkError(_)) => Ordering::Greater,
            (IoError::DatabaseError(_), _) => Ordering::Less,
            (_, IoError::DatabaseError(_)) => Ordering::Greater,
            (IoError::TimeoutError(_), _) => Ordering::Less,
            (_, IoError::TimeoutError(_)) => Ordering::Greater,
            // Other is last
        }
    }
}

/// Result type for IO operations
pub type Result<T> = std::result::Result<T, IoError>;
