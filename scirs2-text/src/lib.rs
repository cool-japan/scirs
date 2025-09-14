#![allow(deprecated)]
#![allow(dead_code)]
//! # Text Processing Module for SciRS2
//!
//! This module provides comprehensive text processing functionality for scientific computing
//! applications, offering SciPy-compatible APIs while leveraging Rust's performance and safety.
//!
//! ## Features
//!
//! - **Tokenization**: Character, word, sentence, N-gram, regex, and BPE tokenizers
//! - **Text Preprocessing**: Unicode normalization, HTML/XML stripping, contraction expansion
//! - **Stemming & Lemmatization**: Porter, Snowball, Lancaster stemmers with rule-based lemmatization
//! - **Vectorization**: TF-IDF, count vectorizers with N-gram support
//! - **Word Embeddings**: Word2Vec implementation with Skip-gram and CBOW
//! - **Similarity Metrics**: Cosine, Jaccard, Levenshtein distance, phonetic algorithms
//! - **Advanced NLP**: Sentiment analysis, topic modeling, text summarization
//! - **Performance**: SIMD-accelerated operations, parallel processing, memory efficiency
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::{
//!     tokenize::WordTokenizer,
//!     vectorize::TfidfVectorizer,
//!     sentiment::LexiconSentimentAnalyzer,
//!     Tokenizer, Vectorizer
//! };
//!
//! // Basic tokenization
//! let tokenizer = WordTokenizer::default();
//! let tokens = tokenizer.tokenize("Hello, world! This is a test.").unwrap();
//!
//! // TF-IDF vectorization
//! let documents = vec![
//!     "The quick brown fox jumps over the lazy dog",
//!     "A quick brown dog outpaces a quick fox",
//!     "The lazy dog sleeps all day"
//! ];
//! let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
//! let matrix = vectorizer.fit_transform(&documents).unwrap();
//!
//! // Sentiment analysis
//! let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();
//! let sentiment = analyzer.analyze("I love this library!").unwrap();
//! println!("Sentiment: {:?}", sentiment.sentiment);
//! ```
//!
//! ## Architecture
//!
//! The module is organized into focused sub-modules:
//!
//! - [`tokenize`]: Text tokenization utilities
//! - [`vectorize`]: Document vectorization and TF-IDF
//! - [`embeddings`]: Word embedding training and utilities
//! - [`sentiment`]: Sentiment analysis tools
//! - [`topic_modeling`]: Topic modeling with LDA
//! - [`string_metrics`]: String similarity and distance metrics
//! - [`preprocess`]: Text cleaning and normalization
//! - [`stemming`]: Stemming and lemmatization
//! - [`parallel`]: Parallel processing utilities
//! - [`simd_ops`]: SIMD-accelerated operations
//!
//! ## Performance
//!
//! SciRS2 Text is designed for high performance:
//!
//! - SIMD acceleration for string operations
//! - Parallel processing for large document collections
//! - Memory-efficient sparse matrix representations
//! - Zero-copy string processing where possible
//! - Optimized algorithms with complexity guarantees

#![warn(missing_docs)]

pub mod classification;
pub mod cleansing;
pub mod distance;
pub mod domain_processors;
pub mod embeddings;
pub mod enhanced_vectorize;
pub mod error;
pub mod huggingface_compat;
pub mod information_extraction;
pub mod ml_integration;
pub mod ml_sentiment;
pub mod model_registry;
pub mod multilingual;
pub mod neural_architectures;
pub mod parallel;
pub mod performance;
pub mod pos_tagging;
pub mod preprocess;
pub mod semantic_similarity;
pub mod sentiment;
pub mod simd_ops;
pub mod sparse;
pub mod sparse_vectorize;
pub mod spelling;
pub mod stemming;
pub mod streaming;
pub mod string_metrics;
pub mod summarization;
pub mod text_coordinator;
pub mod text_statistics;
pub mod token_filter;
pub mod tokenize;
pub mod topic_coherence;
pub mod topic_modeling;
pub mod transformer;
pub mod utils;
pub mod vectorize;
pub mod visualization;
pub mod vocabulary;
pub mod weighted_distance;

// Re-export commonly used items
pub use classification::{
    TextClassificationMetrics, TextClassificationPipeline, TextDataset, TextFeatureSelector,
};
pub use cleansing::{
    expand_contractions, normalize_currencies, normalize_numbers, normalize_ordinals,
    normalize_percentages, normalize_unicode, normalize_whitespace, remove_accents, replace_emails,
    replace_urls, strip_html_tags, AdvancedTextCleaner,
};
pub use distance::{cosine_similarity, jaccard_similarity, levenshtein_distance};
pub use domain_processors::{
    Domain, DomainProcessorConfig, FinancialTextProcessor, LegalTextProcessor,
    MedicalTextProcessor, NewsTextProcessor, PatentTextProcessor, ProcessedDomainText,
    ScientificTextProcessor, SocialMediaTextProcessor, UnifiedDomainProcessor,
};
pub use embeddings::{Word2Vec, Word2VecAlgorithm, Word2VecConfig};
pub use enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};
pub use error::{Result, TextError};
pub use huggingface_compat::{
    ClassificationResult, FeatureExtractionPipeline, FillMaskPipeline, FillMaskResult,
    FormatConverter, HfConfig, HfEncodedInput, HfHub, HfModelAdapter, HfPipeline, HfTokenizer,
    HfTokenizerConfig, QuestionAnsweringPipeline, QuestionAnsweringResult,
    TextClassificationPipeline as HfTextClassificationPipeline, ZeroShotClassificationPipeline,
};
pub use information_extraction::{
    Entity, EntityType, ExtractedInformation, InformationExtractionPipeline, KeyPhraseExtractor,
    PatternExtractor, Relation, RelationExtractor, RuleBasedNER,
};
pub use ml_integration::{
    BatchTextProcessor, FeatureExtractionMode, MLTextPreprocessor, TextFeatures, TextMLPipeline,
};
pub use ml_sentiment::{
    ClassMetrics, EvaluationMetrics, MLSentimentAnalyzer, MLSentimentConfig, TrainingMetrics,
};
pub use model_registry::{
    ModelMetadata, ModelRegistry, ModelType, PrebuiltModels, RegistrableModel,
    SerializableModelData,
};
pub use multilingual::{
    Language, LanguageDetectionResult, LanguageDetector, MultilingualProcessor, ProcessedText,
    StopWords,
};
pub use neural_architectures::{
    ActivationFunction, AdditiveAttention, BiLSTM, CNNLSTMHybrid, Conv1D, CrossAttention, Dropout,
    GRUCell, LSTMCell, LayerNorm as NeuralLayerNorm, MaxPool1D,
    MultiHeadAttention as NeuralMultiHeadAttention, MultiScaleCNN, PositionwiseFeedForward,
    ResidualBlock1D, SelfAttention, TextCNN,
};
pub use parallel::{
    ParallelCorpusProcessor, ParallelTextProcessor, ParallelTokenizer, ParallelVectorizer,
};
pub use performance::{
    AdvancedPerformanceMonitor, DetailedPerformanceReport, OptimizationRecommendation,
    PerformanceSummary, PerformanceThresholds,
};
pub use pos_tagging::{
    PosAwareLemmatizer, PosTagResult, PosTagger, PosTaggerConfig, PosTaggingResult,
};
pub use preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer};
pub use semantic_similarity::{
    LcsSimilarity, SemanticSimilarityEnsemble, SoftCosineSimilarity, WeightedJaccard,
    WordMoversDistance,
};
pub use sentiment::{
    LexiconSentimentAnalyzer, RuleBasedSentimentAnalyzer, Sentiment, SentimentLexicon,
    SentimentResult, SentimentRules, SentimentWordCounts,
};
pub use simd_ops::{
    AdvancedSIMDTextProcessor, SimdEditDistance, SimdStringOps, SimdTextAnalyzer,
    TextProcessingResult,
};
pub use sparse::{CsrMatrix, DokMatrix, SparseMatrixBuilder, SparseVector};
pub use sparse_vectorize::{
    sparse_cosine_similarity, MemoryStats, SparseCountVectorizer, SparseTfidfVectorizer,
};
pub use spelling::{
    DictionaryCorrector, DictionaryCorrectorConfig, EditOp, ErrorModel, NGramModel,
    SpellingCorrector, StatisticalCorrector, StatisticalCorrectorConfig,
};
pub use stemming::{
    LancasterStemmer, LemmatizerConfig, PorterStemmer, PosTag, RuleLemmatizer,
    RuleLemmatizerBuilder, SimpleLemmatizer, SnowballStemmer, Stemmer,
};
pub use streaming::{
    AdvancedStreamingMetrics, AdvancedStreamingProcessor, ChunkedCorpusReader, MemoryMappedCorpus,
    ProgressTracker, StreamingTextProcessor, StreamingVectorizer,
};
pub use string_metrics::{
    AlignmentResult, DamerauLevenshteinMetric, Metaphone, NeedlemanWunsch, Nysiis,
    PhoneticAlgorithm, SmithWaterman, Soundex, StringMetric,
};
pub use summarization::{CentroidSummarizer, KeywordExtractor, TextRank};
pub use text_coordinator::{
    AdvancedBatchClassificationResult, AdvancedSemanticSimilarityResult, AdvancedTextConfig,
    AdvancedTextCoordinator, AdvancedTextResult, AdvancedTopicModelingResult,
};
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
pub use transformer::{
    FeedForward, LayerNorm, MultiHeadAttention, PositionalEncoding, TokenEmbedding,
    TransformerConfig, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer, TransformerModel,
};
pub use vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer};
pub use visualization::{
    AttentionVisualizer, Color, ColorScheme, EmbeddingVisualizer, SentimentVisualizer,
    TextAnalyticsDashboard, TopicVisualizer, VisualizationConfig, WordCloud,
};
pub use vocabulary::Vocabulary;
pub use weighted_distance::{
    DamerauLevenshteinWeights, LevenshteinWeights, WeightedDamerauLevenshtein, WeightedLevenshtein,
    WeightedStringMetric,
};
