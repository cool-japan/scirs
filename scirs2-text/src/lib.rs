//! Text processing module for SciRS2
//!
//! This module provides functionality for text processing, tokenization,
//! vectorization, word embeddings, and other NLP-related operations.

#![warn(missing_docs)]

pub mod distance;
pub mod embeddings;
pub mod error;
pub mod preprocess;
pub mod tokenize;
pub mod utils;
pub mod vectorize;
pub mod vocabulary;

// Re-export commonly used items
pub use distance::{cosine_similarity, jaccard_similarity, levenshtein_distance};
pub use embeddings::{Word2Vec, Word2VecAlgorithm, Word2VecConfig};
pub use error::{Result, TextError};
pub use preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer};
pub use tokenize::{CharacterTokenizer, SentenceTokenizer, Tokenizer, WordTokenizer};
pub use vectorize::{CountVectorizer, TfidfVectorizer};
pub use vocabulary::Vocabulary;
