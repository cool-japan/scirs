//! Hugging Face compatible pipelines for various NLP tasks
//!
//! This module provides pipeline implementations that match the
//! Hugging Face Transformers library API for common NLP tasks.

mod classification;
mod extraction;
mod fill_mask;
mod generation;
mod question_answering;
mod summarization;
mod token_classification;
mod translation;

pub use classification::*;
pub use extraction::*;
pub use fill_mask::*;
pub use generation::*;
pub use question_answering::*;
pub use summarization::*;
pub use token_classification::*;
pub use translation::*;

/// HF-compatible pipeline types
#[derive(Debug)]
pub enum HfPipeline {
    /// Text classification pipeline
    TextClassification(TextClassificationPipeline),
    /// Feature extraction pipeline
    FeatureExtraction(FeatureExtractionPipeline),
    /// Fill mask pipeline
    FillMask(FillMaskPipeline),
    /// Zero-shot classification pipeline
    ZeroShotClassification(ZeroShotClassificationPipeline),
    /// Question answering pipeline
    QuestionAnswering(QuestionAnsweringPipeline),
    /// Text generation pipeline
    TextGeneration(TextGenerationPipeline),
    /// Summarization pipeline
    Summarization(SummarizationPipeline),
    /// Translation pipeline
    Translation(TranslationPipeline),
    /// Token classification pipeline
    TokenClassification(TokenClassificationPipeline),
}

/// Classification result for text classification tasks
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted label
    pub label: String,
    /// Confidence score
    pub score: f64,
}

/// Fill mask result for masked language modeling
#[derive(Debug, Clone)]
pub struct FillMaskResult {
    /// Token ID
    pub token: usize,
    /// Token string
    pub token_str: String,
    /// Sequence with token filled in
    pub sequence: String,
    /// Score/probability
    pub score: f64,
}

/// Question answering result
#[derive(Debug, Clone)]
pub struct QuestionAnsweringResult {
    /// Answer text
    pub answer: String,
    /// Start position in context
    pub start: usize,
    /// End position in context
    pub end: usize,
    /// Confidence score
    pub score: f64,
}

/// Text generation result
#[derive(Debug, Clone)]
pub struct TextGenerationResult {
    /// Generated text
    pub generated_text: String,
    /// Generation score/likelihood
    pub score: Option<f64>,
}

/// Summarization result
#[derive(Debug, Clone)]
pub struct SummarizationResult {
    /// Summary text
    pub summary_text: String,
}

/// Translation result
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Translated text
    pub translation_text: String,
}

/// Token classification result
#[derive(Debug, Clone)]
pub struct TokenClassificationResult {
    /// Entity group (e.g., "PERSON", "ORG")
    pub entity_group: String,
    /// Confidence score
    pub score: f64,
    /// Word/token
    pub word: String,
    /// Start character position
    pub start: Option<usize>,
    /// End character position
    pub end: Option<usize>,
}
