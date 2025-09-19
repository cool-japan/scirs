//! Error types for the text processing module

use thiserror::Error;

/// Error type for text processing operations
#[derive(Error, Debug, Clone)]
pub enum TextError {
    /// Invalid input text
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Tokenization error
    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    /// Text processing error
    #[error("Text processing error: {0}")]
    ProcessingError(String),

    /// Vocabulary error
    #[error("Vocabulary error: {0}")]
    VocabularyError(String),

    /// Embedding error
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Distance calculation error
    #[error("Distance calculation error: {0}")]
    DistanceError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),

    /// Model not fitted error
    #[error("Model not fitted: {0}")]
    ModelNotFitted(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// Result type for text processing operations
pub type Result<T> = std::result::Result<T, TextError>;

/// Implement From trait for converting std::io::Error to TextError
impl From<std::io::Error> for TextError {
    fn from(err: std::io::Error) -> Self {
        TextError::IoError(err.to_string())
    }
}

/// Implement From trait for converting CoreError to TextError
impl From<scirs2_core::CoreError> for TextError {
    fn from(err: scirs2_core::CoreError) -> Self {
        TextError::RuntimeError(err.to_string())
    }
}

/// Implement From trait for converting serde_json::Error to TextError
#[cfg(feature = "serde-support")]
impl From<serde_json::Error> for TextError {
    fn from(err: serde_json::Error) -> Self {
        TextError::InvalidInput(format!("JSON parsing error: {}", err))
    }
}
