//! Text processing module for SciRS2
//!
//! This module provides functionality for text processing, tokenization,
//! vectorization, word embeddings, and other NLP-related operations.

#![warn(missing_docs)]

pub mod cleansing;
pub mod distance;
pub mod embeddings;
pub mod enhanced_vectorize;
pub mod error;
pub mod preprocess;
pub mod stemming;
pub mod string_metrics;
pub mod text_statistics;
pub mod tokenize;
pub mod utils;
pub mod vectorize;
pub mod vocabulary;
pub mod weighted_distance;

// Re-export commonly used items
pub use cleansing::{
    expand_contractions, normalize_unicode, normalize_whitespace, remove_accents, replace_emails,
    replace_urls, strip_html_tags, AdvancedTextCleaner,
};
pub use distance::{cosine_similarity, jaccard_similarity, levenshtein_distance};
pub use embeddings::{Word2Vec, Word2VecAlgorithm, Word2VecConfig};
pub use enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};
pub use error::{Result, TextError};
pub use preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer};
pub use stemming::{PorterStemmer, SimpleLemmatizer, SnowballStemmer, Stemmer};
pub use string_metrics::{
    DamerauLevenshteinMetric, Metaphone, PhoneticAlgorithm, Soundex, StringMetric,
};
pub use text_statistics::{ReadabilityMetrics, TextMetrics, TextStatistics};
pub use tokenize::{
    CharacterTokenizer, NgramTokenizer, RegexTokenizer, SentenceTokenizer, Tokenizer,
    WhitespaceTokenizer, WordTokenizer,
};
pub use vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer};
pub use vocabulary::Vocabulary;
pub use weighted_distance::{
    DamerauLevenshteinWeights, LevenshteinWeights, WeightedDamerauLevenshtein, WeightedLevenshtein,
    WeightedStringMetric,
};
