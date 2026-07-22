#![allow(clippy::manual_strip)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::cloned_ref_to_slice_refs)]
#![allow(dead_code)]
//! # SciRS2 Text - Natural Language Processing
//!
//! **scirs2-text** provides comprehensive text processing and NLP capabilities,
//! offering tokenization, TF-IDF vectorization, word embeddings, sentiment analysis,
//! topic modeling, and text classification with SIMD acceleration and parallel processing.
//!
//! ## 🎯 Key Features
//!
//! - **Tokenization**: Word, sentence, N-gram, BPE, regex tokenizers
//! - **Vectorization**: TF-IDF, count vectorizers, word embeddings
//! - **Text Processing**: Stemming, lemmatization, normalization, stopword removal
//! - **Embeddings**: Word2Vec (Skip-gram, CBOW), GloVe loading
//! - **Similarity**: Cosine, Jaccard, Levenshtein, phonetic algorithms
//! - **NLP**: Sentiment analysis, topic modeling (LDA), text classification
//! - **Performance**: SIMD operations, parallel processing, sparse matrices
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | Python Equivalent | Description |
//! |---------------|-------------------|-------------|
//! | `tokenize` | `nltk.tokenize` | Text tokenization utilities |
//! | `vectorize` | `sklearn.feature_extraction.text.TfidfVectorizer` | TF-IDF and count vectorization |
//! | `embeddings` | `gensim.models.Word2Vec` | Word embeddings (Word2Vec) |
//! | `sentiment` | `nltk.sentiment` | Sentiment analysis |
//! | `topic_modeling` | `sklearn.decomposition.LatentDirichletAllocation` | Topic modeling (LDA) |
//! | `stemming` | `nltk.stem` | Stemming and lemmatization |
//!
//! ## 🚀 Quick Start
//!
//! ```toml
//! [dependencies]
//! scirs2-text = "0.6.2"
//! ```
//!
//! ```rust,no_run
//! use scirs2_text::{tokenize::WordTokenizer, vectorize::TfidfVectorizer, Tokenizer, Vectorizer};
//!
//! // Tokenization
//! let tokenizer = WordTokenizer::default();
//! let tokens = tokenizer.tokenize("Hello, world!").unwrap();
//!
//! // TF-IDF vectorization
//! let docs = vec!["Hello world", "Good morning world"];
//! let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
//! let matrix = vectorizer.fit_transform(&docs).unwrap();
//! ```
//!
//! ## 🔒 Version: 0.6.2 (July 16, 2026)
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

/// Batch tokenization with padding and attention masks.
pub mod batch_tokenizer;
pub mod classification;
pub mod cleansing;
pub mod distance;
pub mod domain_processors;
pub mod embeddings;
pub mod enhanced_vectorize;
pub mod error;
pub mod evaluation;
/// GPT-2 byte-level BPE tokenizer.
pub mod gpt_bpe;
pub mod huggingface_compat;
pub mod information_extraction;
pub mod language_model;
/// Statistical language models (Unigram, Bigram, N-gram with Kneser-Ney).
pub mod language_models;
pub mod lemmatization;
pub mod ml_integration;
pub mod ml_sentiment;
pub mod model_registry;
pub mod multilingual;
pub mod neural_architectures;
pub mod parallel;
pub mod paraphrasing;
pub mod performance;
pub mod pipeline;
pub mod pos_tagging;
pub mod preprocess;
pub mod semantic_similarity;
/// SentencePiece Unigram Language Model tokenizer.
pub mod sentencepiece;
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
pub mod tokenizer;
pub mod topic_coherence;
pub mod topic_modeling;
pub mod transformer;
pub mod utils;
pub mod vectorize;
pub mod visualization;
pub mod vocabulary;
pub mod weighted_distance;

// New text processing modules
pub mod keyword_extraction;
pub mod language_detection;
pub mod named_entity_recognition;
pub mod text_similarity;
pub mod text_summarization;

// BERT fine-tuning
pub mod bert_finetune;
// Cross-lingual NER and transliteration
pub mod crosslingual;
// Correlated topic model
pub mod ctm;
// Dynamic topic model
pub mod dtm;
// Hierarchical Dirichlet Process topic model
pub mod hdp;
// Sentence embeddings
pub mod sentence_embeddings;
// Semantic similarity with embedding-based search
pub mod similarity;
// Advanced tokenization algorithms (BPE, WordPiece, language-agnostic Unicode)
pub mod tokenization;
// Tokenizer implementations (HuggingFace, byte-level BPE, Unicode)
pub mod tokenizers;
// Transliteration
pub mod transliteration;
// Topic modelling (HDP automatic topic selection)
pub mod topic;

// Additional text processing modules
pub mod abstractive_summary;
pub mod advanced_classification;
pub mod advanced_distance;
pub mod alignment;
pub mod bpe_tokenizer;
pub mod coreference;
pub mod dialog;
pub mod discourse;
pub mod doc_similarity;
pub mod event_extraction;
pub mod keywords;
pub mod multilingual_ext;
pub mod ner;
pub mod pos_tagging_original;
// The `pos_tagging_original_tests` suite is compiled and run as a child of
// `pos_tagging_original` via the `#[cfg(test)] #[path = "pos_tagging_original_tests.rs"]`
// include at the bottom of that module. It must NOT be declared here as a top-level
// crate module: its `use super::*;` relies on the `pos_tagging_original` items
// (`MorphologicalAnalyzer`, `ContextualDisambiguator`, and the private `PosTagger`
// fields/methods) which are not re-exported at the crate root, and the crate root
// instead re-exports the refactored `pos_tagging` versions.
pub mod question_answering;
pub mod regex_lite;
pub mod segmentation;
pub mod summarize_advanced;
pub mod text_classification;
pub mod text_preprocess;
pub mod topic_model;

// Re-export commonly used items
pub use classification::{
    cross_validate_nb, BernoulliNaiveBayes, CrossValidationResult, FeatureHasher, FoldResult,
    MultiLabelClassifier, MultiLabelPrediction, MultinomialNaiveBayes, TextClassificationMetrics,
    TextClassificationPipeline, TextDataset, TextFeatureSelector, TfidfCosineClassifier,
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
pub use embeddings::{
    embedding_cosine_similarity,
    fasttext::{FastText, FastTextConfig},
    glove::{CooccurrenceMatrix, GloVe, GloVeTrainer, GloVeTrainerConfig},
    pairwise_similarity, Word2Vec, Word2VecAlgorithm, Word2VecConfig, WordEmbedding,
};
pub use enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};
pub use error::{Result, TextError};
pub use huggingface_compat::{
    ClassificationResult, FeatureExtractionPipeline, FillMaskPipeline, FillMaskResult,
    FormatConverter, HfConfig, HfEncodedInput, HfHub, HfModelAdapter, HfPipeline, HfTokenizer,
    HfTokenizerConfig, QuestionAnsweringPipeline, QuestionAnsweringResult,
    TextClassificationPipeline as HfTextClassificationPipeline, ZeroShotClassificationPipeline,
};
pub use information_extraction::{
    AdvancedExtractedInformation, AdvancedExtractionPipeline, ConfidenceScorer, CoreferenceChain,
    CoreferenceMention, CoreferenceResolver, DocumentInformationExtractor, DocumentSummary, Entity,
    EntityCluster, EntityLinker, EntityType, Event, ExtractedInformation,
    InformationExtractionPipeline, KeyPhraseExtractor, KnowledgeBaseEntry, LinkedEntity,
    MentionType, PatternExtractor, Relation, RelationExtractor, RuleBasedNER,
    StructuredDocumentInformation, TemporalExtractor, Topic,
};
pub use language_model::{NgramModel, SmoothingMethod};
pub use lemmatization::{Lemmatizer, RuleBasedLemmatizer, WordNetLemmatizer};
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
    is_cjk_char, is_combining_mark, is_cyrillic, Language, LanguageDetectionResult,
    LanguageDetector, MultilingualProcessor, ProcessedText, ScriptFamily, StopWords,
    Transliterator as MultilingualTransliterator, UnicodeTokenizer, UnicodeTokenizerConfig,
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
pub use paraphrasing::{ParaphraseConfig, ParaphraseResult, ParaphraseStrategy, Paraphraser};
pub use performance::{
    AdvancedPerformanceMonitor, DetailedPerformanceReport, OptimizationRecommendation,
    PerformanceSummary, PerformanceThresholds,
};
pub use pipeline::{
    basic_pipeline, lemmatization_pipeline, ngram_pipeline, stemming_pipeline, BatchProcessor,
    NlpPipeline, PipelineBuilder, PipelineStep,
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
    aggregate_sentiment, analyze_and_aggregate, AggregatedSentiment, AspectSentiment,
    AspectSentimentAnalyzer, LexiconSentimentAnalyzer, NaiveBayesSentiment,
    RuleBasedSentimentAnalyzer, Sentiment, SentimentLexicon, SentimentResult, SentimentRules,
    SentimentWordCounts, VaderResult, VaderSentimentAnalyzer,
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
pub use tokenizer::{
    BPETokenizer, SimpleCharTokenizer, SimpleWhitespaceTokenizer, TransformerTokenizer,
    WordPieceTokenizer,
};
pub use topic_coherence::{TopicCoherence, TopicDiversity};
pub use topic_modeling::{
    LatentDirichletAllocation, LdaBuilder, LdaConfig, LdaLearningMethod, Topic as LdaTopic,
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

// Re-exports for new modules
pub use keyword_extraction::{
    extract_keywords, Keyword, KeywordMethod, RakeKeywordExtractor, TextRankKeywordExtractor,
    TfIdfKeywordExtractor,
};
pub use language_detection::{
    detect_language, detect_language_with_strategy, DetectedLanguage, DetectionStrategy,
    LanguageDetectionOutput,
};
pub use named_entity_recognition::{extract_entities, NerEntity, NerEntityType, NerPatternConfig};
pub use text_similarity::{
    bm25_score, char_ngram_jaccard_similarity, edit_distance_similarity, jaccard_token_similarity,
    text_similarity, tfidf_cosine_similarity, Bm25Config, Bm25Scorer, SimilarityMethod,
    SimilarityResult, TfIdfCosineSimilarity,
};
pub use text_summarization::{
    score_position, score_textrank, score_tfidf, summarize, ScoredSentence, SummarizationMethod,
};

// HDP topic model
pub use hdp::{HdpConfig, HdpModel, HdpResult};

// New sentence encoder (USE-style, word-level)
pub use sentence_embeddings::{
    SentenceEncoder, SentenceEncoderConfig, SentenceEncoderPooling, SimCSELoss, SimCSETrainer,
};

// topic module re-exports (HDP with automatic topic selection)
pub use topic::hdp::{
    Hdp, HdpConfig as HdpAutoConfig, HdpState, HdpTopicConfig, HdpTopicModel, TopicError,
};

// tokenizers module (Unicode tokenizer from tokenizers crate kept accessible via path)
pub use embeddings::sentence_encoder::{
    PoolingStrategy as SentencePoolingStrategy, SemanticSimilarity as EmbeddingSearch,
    SentenceEncoder as ProjSentenceEncoder, SimCseConfig, SimCseTrainer as ProjSimCseTrainer,
};

// Transliteration: trait + concrete transliterators
pub use transliteration::{
    CyrillicScheme, CyrillicTransliterator, HepburnTransliterator, PinyinStyle,
    PinyinTransliterator, Transliterator,
};

// Neural NLP: transformer encoder, attention viz, BERT classifier, neural NER
pub mod neural_nlp;
pub use neural_nlp::{
    AttentionHeatmap, AttentionVisualization, BertClassifier, BertClassifierConfig, NerTag,
    NeuralNer, NeuralNerConfig, TransformerEncoderConfig as NeuralEncoderConfig,
    TransformerTextEncoder,
};
