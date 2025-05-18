//! Text processing module for SciRS2
//!
//! This module provides functionality for text processing, tokenization,
//! vectorization, word embeddings, and other NLP-related operations.

#![warn(missing_docs)]

pub mod classification;
pub mod cleansing;
pub mod distance;
pub mod embeddings;
pub mod enhanced_vectorize;
pub mod error;
pub mod ml_integration;
pub mod ml_sentiment;
pub mod multilingual;
pub mod parallel;
pub mod preprocess;
pub mod sentiment;
pub mod spelling;
pub mod stemming;
pub mod string_metrics;
pub mod summarization;
pub mod text_statistics;
pub mod token_filter;
pub mod tokenize;
pub mod topic_coherence;
pub mod topic_modeling;
pub mod utils;
pub mod vectorize;
pub mod vocabulary;
pub mod weighted_distance;

// Re-export commonly used items
pub use classification::{
    TextClassificationMetrics, TextClassificationPipeline, TextDataset, TextFeatureSelector,
};
pub use cleansing::{
    expand_contractions, normalize_unicode, normalize_whitespace, remove_accents, replace_emails,
    replace_urls, strip_html_tags, AdvancedTextCleaner,
};
pub use distance::{cosine_similarity, jaccard_similarity, levenshtein_distance};
pub use embeddings::{Word2Vec, Word2VecAlgorithm, Word2VecConfig};
pub use enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};
pub use error::{Result, TextError};
pub use ml_integration::{
    BatchTextProcessor, FeatureExtractionMode, MLTextPreprocessor, TextFeatures, TextMLPipeline,
};
pub use ml_sentiment::{
    ClassMetrics, EvaluationMetrics, MLSentimentAnalyzer, MLSentimentConfig, TrainingMetrics,
};
pub use multilingual::{
    Language, LanguageDetectionResult, LanguageDetector, MultilingualProcessor, ProcessedText,
    StopWords,
};
pub use parallel::{
    ParallelCorpusProcessor, ParallelTextProcessor, ParallelTokenizer, ParallelVectorizer,
};
pub use preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer};
pub use sentiment::{
    LexiconSentimentAnalyzer, RuleBasedSentimentAnalyzer, Sentiment, SentimentLexicon,
    SentimentResult, SentimentRules, SentimentWordCounts,
};
pub use spelling::{
    DictionaryCorrector, DictionaryCorrectorConfig, EditOp, ErrorModel, NGramModel,
    SpellingCorrector, StatisticalCorrector, StatisticalCorrectorConfig,
};
pub use stemming::{
    LancasterStemmer, LemmatizerConfig, PorterStemmer, PosTag, RuleLemmatizer,
    RuleLemmatizerBuilder, SimpleLemmatizer, SnowballStemmer, Stemmer,
};
pub use string_metrics::{
    DamerauLevenshteinMetric, Metaphone, PhoneticAlgorithm, Soundex, StringMetric,
};
pub use summarization::{CentroidSummarizer, KeywordExtractor, TextRank};
pub use text_statistics::{ReadabilityMetrics, TextMetrics, TextStatistics};
pub use token_filter::{
    CompositeFilter, CustomFilter, FrequencyFilter, LengthFilter, RegexFilter, StopwordsFilter,
    TokenFilter,
};
pub use tokenize::{
    bpe::{BpeConfig, BpeTokenizer, BpeVocabulary},
    CharacterTokenizer, NgramTokenizer, RegexTokenizer, SentenceTokenizer, Tokenizer,
    WhitespaceTokenizer, WordTokenizer,
};
pub use topic_coherence::{TopicCoherence, TopicDiversity};
pub use topic_modeling::{
    LatentDirichletAllocation, LdaBuilder, LdaConfig, LdaLearningMethod, Topic,
};
pub use vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer};
pub use vocabulary::Vocabulary;
pub use weighted_distance::{
    DamerauLevenshteinWeights, LevenshteinWeights, WeightedDamerauLevenshtein, WeightedLevenshtein,
    WeightedStringMetric,
};
