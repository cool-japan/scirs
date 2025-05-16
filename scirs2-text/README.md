# scirs2-text

[![crates.io](https://img.shields.io/crates/v/scirs2-text.svg)](https://crates.io/crates/scirs2-text)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]
[![Documentation](https://img.shields.io/docsrs/scirs2-text)](https://docs.rs/scirs2-text)

Text processing module for SciRS2 (Scientific Computing in Rust - Next Generation), providing comprehensive text processing, natural language processing, and machine learning text utilities.

## Features

### Text Preprocessing
- **Normalization**: Unicode normalization, case folding
- **Cleaning**: 
  - Remove special characters, normalize whitespace, stop word removal
  - HTML/XML stripping, URL/email handling
  - Unicode normalization and accent removal
  - Contraction expansion
- **Tokenization**:
  - Word tokenization with customizable patterns
  - Sentence tokenization
  - Character/grapheme tokenization
  - N-gram tokenization (with range support)
  - Regex-based tokenization
  - Whitespace tokenization

### Stemming and Lemmatization
- **Porter Stemmer**: Classic algorithm for word stemming
- **Snowball Stemmer**: Enhanced Porter stemmer (English)
- **Simple Lemmatizer**: Dictionary-based lemmatization

### Text Vectorization
- **Count Vectorizer**: Bag-of-words representation
- **TF-IDF Vectorizer**: Term frequency-inverse document frequency with normalization
- **Binary Vectorizer**: Binary occurrence vectors
- **Enhanced Vectorizers**: 
  - N-gram support (unigrams, bigrams, trigrams, etc.)
  - Document frequency filtering (min_df, max_df)
  - Maximum features limitation
  - IDF smoothing and sublinear TF scaling

### Word Embeddings
- **Word2Vec**: Skip-gram and CBOW models with negative sampling
- **Embedding utilities**: Loading, saving, and manipulation
- **Similarity computation**: Cosine similarity between word vectors

### Distance and String Metrics
- **Vector Similarity**:
  - **Cosine similarity**: Between vectors and documents
  - **Jaccard similarity**: Set-based similarity
- **String Distances**:
  - **Levenshtein distance**: Basic edit distance
  - **Jaro-Winkler similarity**: String similarity
  - **Damerau-Levenshtein distance**: Edit distance with transpositions
  - **Optimal String Alignment**: Restricted Damerau-Levenshtein
  - **Weighted Levenshtein**: Edit distance with custom operation costs
  - **Weighted Damerau-Levenshtein**: Flexible weights for all edit operations
- **Phonetic Algorithms**:
  - **Soundex**: Phonetic encoding for similar-sounding words
  - **Metaphone**: Improved phonetic algorithm

### Vocabulary Management
- Dynamic vocabulary building
- Vocabulary pruning and filtering
- Persistence (save/load)
- Frequency-based filtering

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-text = "0.1.0-alpha.3"
```

## Quick Start

```rust
use scirs2_text::{
    preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer},
    tokenize::{NgramTokenizer, RegexTokenizer, Tokenizer, WordTokenizer},
    vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer},
    stemming::{PorterStemmer, Stemmer},
};

// Text normalization
let normalizer = BasicNormalizer::default();
let normalized = normalizer.normalize("Hello, World!")?;

// Tokenization
let tokenizer = WordTokenizer::new(true);
let tokens = tokenizer.tokenize("The quick brown fox")?;

// N-gram tokenization
let ngram_tokenizer = NgramTokenizer::new(2)?;
let ngrams = ngram_tokenizer.tokenize("hello world test")?;

// Stemming
let stemmer = PorterStemmer::new();
let stemmed = stemmer.stem("running")?;

// Vectorization
let mut vectorizer = CountVectorizer::new(false);
let documents = vec!["Hello world", "World of Rust"];
let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();
vectorizer.fit(&doc_refs)?;
let vector = vectorizer.transform("Hello Rust")?;
```

## Examples

See the `examples/` directory for comprehensive demonstrations:
- `text_processing_demo.rs`: Complete text processing pipeline
- `word2vec_example.rs`: Word embedding training and usage
- `enhanced_vectorization_demo.rs`: Advanced vectorization with n-grams and filtering

### Text Statistics and Readability

```rust
use scirs2_text::text_statistics::{TextStatistics, ReadabilityMetrics};

// Create text statistics analyzer
let stats = TextStatistics::new();

// Calculate readability metrics
let text = "The quick brown fox jumps over the lazy dog. This is a simple text passage used for demonstration purposes.";
let metrics = stats.get_all_metrics(text)?;

println!("Flesch Reading Ease: {}", metrics.flesch_reading_ease);
println!("Flesch-Kincaid Grade Level: {}", metrics.flesch_kincaid_grade_level);
println!("Gunning Fog Index: {}", metrics.gunning_fog);
println!("Lexical Diversity: {}", metrics.lexical_diversity);
println!("Word Count: {}", metrics.text_statistics.word_count);
println!("Average Sentence Length: {}", metrics.text_statistics.avg_sentence_length);
```

Run examples with:
```bash
cargo run --example text_processing_demo
cargo run --example word2vec_example
cargo run --example enhanced_vectorization_demo
```

## Advanced Usage

### Custom Tokenizers

```rust
use scirs2_text::tokenize::{RegexTokenizer, Tokenizer};

// Custom regex tokenizer
let tokenizer = RegexTokenizer::new(r"\b\w+\b", false)?;
let tokens = tokenizer.tokenize("Hello, world!")?;

// Tokenize with gaps (pattern matches separators)
let gap_tokenizer = RegexTokenizer::new(r"\s*,\s*", true)?;
let tokens = gap_tokenizer.tokenize("apple, banana, cherry")?;
```

### N-gram Extraction

```rust
use scirs2_text::tokenize::{NgramTokenizer, Tokenizer};

// Bigrams
let bigram_tokenizer = NgramTokenizer::new(2)?;
let bigrams = bigram_tokenizer.tokenize("Hello world test")?;

// Range of n-grams (2-3)
let range_tokenizer = NgramTokenizer::with_range(2, 3)?;
let ngrams = range_tokenizer.tokenize("Hello world test")?;

// Alphanumeric only
let alpha_tokenizer = NgramTokenizer::new(2)?.only_alphanumeric(true);
```

### TF-IDF Vectorization

```rust
use scirs2_text::vectorize::{TfidfVectorizer, Vectorizer};

let mut tfidf = TfidfVectorizer::new(false, true, Some("l2".to_string()));
tfidf.fit(&documents)?;
let tfidf_matrix = tfidf.transform_batch(&documents)?;
```

### Enhanced Vectorization with N-grams

```rust
use scirs2_text::enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};

// Count vectorizer with bigrams
let mut count_vec = EnhancedCountVectorizer::new()
    .set_ngram_range((1, 2))?
    .set_max_features(Some(100));
count_vec.fit(&documents)?;

// TF-IDF with document frequency filtering
let mut tfidf = EnhancedTfidfVectorizer::new()
    .set_ngram_range((1, 3))?
    .set_min_df(0.1)?  // Minimum 10% document frequency
    .set_smooth_idf(true)
    .set_sublinear_tf(true);
tfidf.fit(&documents)?;
```

### String Metrics and Phonetic Algorithms

```rust
use scirs2_text::string_metrics::{
    DamerauLevenshteinMetric, StringMetric, Soundex, Metaphone, PhoneticAlgorithm
};
use scirs2_text::weighted_distance::{
    WeightedLevenshtein, WeightedDamerauLevenshtein, WeightedStringMetric,
    LevenshteinWeights, DamerauLevenshteinWeights
};
use std::collections::HashMap;

// Damerau-Levenshtein distance with transpositions
let dl_metric = DamerauLevenshteinMetric::new();
let distance = dl_metric.distance("kitten", "sitting")?;
let similarity = dl_metric.similarity("kitten", "sitting")?;

// Restricted Damerau-Levenshtein (Optimal String Alignment)
let osa_metric = DamerauLevenshteinMetric::restricted();
let osa_distance = osa_metric.distance("kitten", "sitting")?;

// Weighted Levenshtein with custom operation costs
let weights = LevenshteinWeights::new(2.0, 1.0, 0.5);  // insertions=2, deletions=1, substitutions=0.5
let weighted = WeightedLevenshtein::with_weights(weights);
let weighted_distance = weighted.distance("kitten", "sitting")?;

// Weighted Levenshtein with character-specific costs
let mut costs = HashMap::new();
costs.insert(('k', 's'), 0.1); // Make k->s substitution very cheap
let char_weights = LevenshteinWeights::default().with_substitution_costs(costs);
let custom_metric = WeightedLevenshtein::with_weights(char_weights);

// Weighted Damerau-Levenshtein with custom transposition cost
let dl_weights = DamerauLevenshteinWeights::new(1.0, 1.0, 1.0, 0.5); // transpositions cost 0.5
let weighted_dl = WeightedDamerauLevenshtein::with_weights(dl_weights);
let trans_distance = weighted_dl.distance("abc", "acb")?;  // Returns 0.5 (one transposition)

// Soundex phonetic encoding
let soundex = Soundex::new();
let code = soundex.encode("Robert")?;  // Returns "R163"
let sounds_like = soundex.sounds_like("Smith", "Smythe")?;  // Returns true

// Metaphone phonetic algorithm
let metaphone = Metaphone::new();
let code = metaphone.encode("programming")?;  // Returns "PRKRMN"
```

### Text Preprocessing Pipeline

```rust
use scirs2_text::preprocess::{BasicNormalizer, BasicTextCleaner, TextPreprocessor};

// Create a complete preprocessing pipeline
let normalizer = BasicNormalizer::new(true, true);
let cleaner = BasicTextCleaner::new(true, true, true);
let preprocessor = TextPreprocessor::new(normalizer, cleaner);

let processed = preprocessor.process("Hello, WORLD! This is a TEST.")?;
// Output: "hello world test"
```

### Word Embeddings

```rust
use scirs2_text::embeddings::{Word2Vec, Word2VecConfig, Word2VecAlgorithm};

// Configure Word2Vec
let config = Word2VecConfig {
    vector_size: 100,
    window: 5,
    min_count: 2,
    algorithm: Word2VecAlgorithm::SkipGram,
    iterations: 15,
    negative_samples: 5,
    ..Default::default()
};

// Train embeddings
let mut word2vec = Word2Vec::builder()
    .config(config)
    .build()?;

word2vec.train(&documents)?;

// Get word vectors
if let Some(vector) = word2vec.get_vector("hello") {
    println!("Vector for 'hello': {:?}", vector);
}

// Find similar words
let similar = word2vec.most_similar("hello", 5)?;
```

## Performance Considerations

- Uses parallel processing where applicable (via Rayon)
- Efficient sparse matrix representations for vectorizers
- Optimized string operations and pattern matching
- Memory-efficient vocabulary management
- SIMD acceleration for distance calculations (when available)

## Dependencies

- `ndarray`: N-dimensional arrays
- `regex`: Regular expressions
- `unicode-segmentation`: Unicode text segmentation
- `unicode-normalization`: Unicode normalization
- `scirs2-core`: Core utilities and parallel processing
- `lazy_static`: Lazy static initialization

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.