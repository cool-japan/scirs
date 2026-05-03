# scirs2-text

[![crates.io](https://img.shields.io/crates/v/scirs2-text.svg)](https://crates.io/crates/scirs2-text)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-text)](https://docs.rs/scirs2-text)
[![Version](https://img.shields.io/badge/version-0.4.3-green)]()

Comprehensive natural language processing and text processing library for the SciRS2 scientific computing ecosystem. Provides tokenization, vectorization, word embeddings, NER, CRF sequence labelling, BPE, dependency parsing, topic modeling, summarization, question answering, knowledge graph extraction, and much more — all in pure, safe Rust with parallel processing. v0.4.3 features include the `TransformerTextEncoder` (multi-head self-attention + sinusoidal positional encoding), `BertClassifier` (frozen-encoder fine-tuning with SGD head), `NeuralNer` (BIO-9 tagging with token-level F1), the Hierarchical Dirichlet Process Chinese Restaurant Franchise Gibbs sampler, the `UniversalSentenceEncoder` with six pooling strategies, and a language-agnostic Unicode tokenizer.

## Features

### Tokenization
- **Word tokenizer**: Unicode-aware, configurable punctuation handling
- **Sentence tokenizer**: Rule-based sentence boundary detection
- **Character/grapheme tokenizer**: Unicode grapheme cluster segmentation
- **N-gram tokenizer**: Unigrams through arbitrary n-grams, range support
- **Regex tokenizer**: Pattern-based tokenization and gap tokenization
- **BPE tokenizer** (Byte Pair Encoding): Subword tokenization with vocabulary learning
- **WordPiece tokenizer**: BERT-style subword tokenization

### Text Preprocessing
- Unicode normalization (NFC, NFD, NFKC, NFKD)
- Case folding, accent removal
- HTML/XML stripping, URL and email normalization
- Contraction expansion
- Number normalization (dates, currencies, percentages)
- Stopword removal, whitespace normalization
- `TextPreprocessor` pipeline: chain normalizers and cleaners

### Stemming and Lemmatization
- Porter Stemmer (English)
- Snowball Stemmer (English, extensible)
- Lancaster Stemmer
- Rule-based lemmatizer with morphological analysis

### Text Vectorization
- **Count Vectorizer**: Bag-of-words, N-gram support, min/max document frequency filtering
- **TF-IDF Vectorizer**: IDF smoothing, sublinear TF scaling, L1/L2 normalization
- **Binary Vectorizer**: Occurrence-only representation
- **Enhanced vectorizers**: `EnhancedCountVectorizer`, `EnhancedTfidfVectorizer` with max_features, vocabulary pruning
- Sparse matrix representation for memory efficiency

### Word Embeddings
- **Word2Vec** (Skip-gram and CBOW): negative sampling, configurable window size, hierarchical softmax
- **GloVe loading**: Load pre-trained GloVe vectors
- **FastText** (pure Rust): Subword embeddings with character n-gram support
- Embedding similarity: cosine similarity, most-similar-words
- Save/load in binary and text formats

### Sequence Labelling
- **CRF** (Conditional Random Fields): Viterbi decoding, feature engineering, sequence-to-label
- **HMM** (Hidden Markov Model): Forward-backward, Viterbi, for POS tagging
- Custom feature extractors for NER, POS, chunking tasks

### Named Entity Recognition (NER)
- Rule-based NER with regex patterns
- Dictionary-based NER with gazetteer support
- CRF-based NER with feature engineering
- Standard entity types: PER, ORG, LOC, DATE, TIME, MONEY, PERCENT

### Advanced NLP
- **Coreference resolution**: mention detection and clustering
- **Dependency parsing**: arc-factored dependency graph construction
- **Discourse analysis**: rhetorical structure theory primitives
- **Event extraction**: event trigger and argument extraction
- **Question answering**: extractive span detection
- **Knowledge graph extraction**: entity-relation-entity triples from text
- **Semantic parsing**: logical form generation from natural language
- **Temporal extraction**: date and time expression normalization
- **Grammar checking**: rule-based error detection

### Topic Modeling
- **LDA** (Latent Dirichlet Allocation): variational inference, coherence metrics (CV, UMass, UCI)
- **NMF-based topic modeling**: non-negative matrix factorization topics
- `TopicModel` trait for interchangeable backends

### Text Summarization
- **Extractive**: TextRank, centroid-based, keyword-based extraction
- **Abstractive**: sequence-to-sequence summarization primitives (`abstractive_summary.rs`)

### Sentiment Analysis
- **Lexicon-based**: VADER-style sentiment scoring with basic lexicon
- **Rule-based**: negation handling, intensifiers, modifiers
- **ML-based** adapter: integrate trained classifiers

### Text Classification
- Feature extraction pipeline for classification
- Multinomial Naive Bayes (text-optimized)
- Logistic regression adapter
- Dataset handling utilities

### String Metrics and Phonetics
- Levenshtein distance (basic and Damerau-Levenshtein)
- Optimal String Alignment (restricted Damerau-Levenshtein)
- Weighted Levenshtein / Weighted Damerau-Levenshtein (custom operation costs)
- Jaro-Winkler similarity
- Cosine similarity, Jaccard similarity
- **Soundex** phonetic encoding
- **Metaphone** phonetic algorithm
- **NYSIIS** phonetic algorithm

### Language Model Primitives
- N-gram language model with Kneser-Ney smoothing
- Character-level language model
- Perplexity computation

### Multilingual Support
- Unicode-first throughout
- Language detection via N-gram character models
- Multilingual utilities (`multilingual_ext.rs`)

### Performance
- **Parallel processing**: Rayon-based multi-threaded tokenization and corpus vectorization
- **Batch processing**: Efficient large document collection handling
- **Sparse matrices**: Memory-efficient vectorizer outputs
- **SIMD operations**: `simd_ops.rs` for accelerated string comparisons and distance computation
- **Memory-mapped corpus**: Streaming large corpus without full RAM loading

## Installation

```toml
[dependencies]
scirs2-text = "0.4.3"
```

## Quick Start

### Tokenization and Vectorization

```rust
use scirs2_text::{
    tokenize::{WordTokenizer, NgramTokenizer, Tokenizer},
    vectorize::{TfidfVectorizer, CountVectorizer, Vectorizer},
    stemming::{PorterStemmer, Stemmer},
    preprocess::{BasicNormalizer, BasicTextCleaner, TextNormalizer, TextCleaner},
};

// Normalization
let normalizer = BasicNormalizer::default();
let normalized = normalizer.normalize("Hello, World! THIS is a TEST.")?;

// Word tokenization
let tokenizer = WordTokenizer::new(true);  // lowercase=true
let tokens = tokenizer.tokenize("The quick brown fox jumps")?;

// N-gram tokenization
let bigrams = NgramTokenizer::new(2)?.tokenize("hello world test")?;
let range   = NgramTokenizer::with_range(1, 3)?.tokenize("hello world test")?;

// Stemming
let stemmer = PorterStemmer::new();
let stem    = stemmer.stem("running")?;    // "run"

// TF-IDF vectorization
let documents = vec![
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "The lazy dog sleeps all day",
];
let mut tfidf = TfidfVectorizer::new(false, true, Some("l2".to_string()));
let matrix    = tfidf.fit_transform(&documents)?;
```

### BPE Tokenizer

```rust
use scirs2_text::bpe_tokenizer::BpeTokenizer;

let corpus = vec![
    "the quick brown fox",
    "the lazy dog",
    "quick brown dog",
];

// Learn BPE vocabulary from corpus
let mut bpe = BpeTokenizer::new(1000);  // vocab_size=1000
bpe.fit(&corpus)?;

let tokens = bpe.tokenize("the quick fox")?;
println!("{:?}", tokens);

// Save/load vocabulary
bpe.save_vocab("bpe_vocab.json")?;
let loaded = BpeTokenizer::load_vocab("bpe_vocab.json")?;
```

### CRF Sequence Labelling (NER)

```rust
use scirs2_text::sequence_labeling::CrfTagger;

// Build CRF tagger for NER
let mut crf = CrfTagger::new();
crf.add_feature_fn(|token, _context| {
    vec![
        format!("word={}", token.to_lowercase()),
        format!("is_upper={}", token.chars().next().map_or(false, |c| c.is_uppercase())),
    ]
});

// Train on labelled data
crf.fit(&train_sequences, &train_labels, 100, 0.01)?;

// Predict labels for new sequence
let tokens = vec!["John", "Smith", "visited", "London"];
let labels = crf.predict(&tokens)?;
// e.g. ["B-PER", "I-PER", "O", "B-LOC"]
```

### Named Entity Recognition

```rust
use scirs2_text::ner::NerTagger;

let mut ner = NerTagger::new();
ner.add_gazetteer("PER", &["John Smith", "Jane Doe"])?;
ner.add_pattern("DATE", r"\d{4}-\d{2}-\d{2}")?;

let text     = "John Smith visited London on 2024-01-15.";
let entities = ner.extract(text)?;

for entity in &entities {
    println!("{}: {} ({}..{})", entity.label, entity.text, entity.start, entity.end);
}
```

### Topic Modeling (LDA)

```rust
use scirs2_text::topic_model::LatentDirichletAllocation;

let documents = load_corpus()?;
let mut lda = LatentDirichletAllocation::new(10, 0.1, 0.01, 1000, Some(42));
lda.fit(&documents)?;

// Get top words per topic
for (topic_id, words) in lda.top_words(10).iter().enumerate() {
    println!("Topic {}: {}", topic_id, words.join(", "));
}

// Document-topic distributions
let dist = lda.transform(&["new document text"])?;

// Coherence score
let coherence = lda.coherence_cv(&documents, 10)?;
println!("CV coherence: {:.4}", coherence);
```

### Sentiment Analysis

```rust
use scirs2_text::sentiment::LexiconSentimentAnalyzer;

let analyzer  = LexiconSentimentAnalyzer::with_basiclexicon();
let result    = analyzer.analyze("I love this library! It's fantastic.")?;

println!("Sentiment: {:?}", result.sentiment);   // Positive
println!("Score: {:.4}", result.compound_score);
```

### Knowledge Graph Extraction

```rust
use scirs2_text::knowledge_graph::KnowledgeGraphExtractor;

let extractor = KnowledgeGraphExtractor::new();
let text      = "Albert Einstein developed the theory of relativity.";
let triples   = extractor.extract(text)?;

for triple in &triples {
    println!("({}, {}, {})", triple.subject, triple.relation, triple.object);
}
// -> ("Albert Einstein", "developed", "theory of relativity")
```

### Word Embeddings (Word2Vec)

```rust
use scirs2_text::embeddings::{Word2Vec, Word2VecConfig, Word2VecAlgorithm};

let config = Word2VecConfig {
    vector_size: 100,
    window: 5,
    min_count: 2,
    algorithm: Word2VecAlgorithm::SkipGram,
    iterations: 15,
    negative_samples: 5,
    ..Default::default()
};

let mut model = Word2Vec::builder().config(config).build()?;
model.train(&documents)?;

let similar = model.most_similar("computer", 5)?;
for (word, similarity) in &similar {
    println!("{}: {:.4}", word, similarity);
}
```

### Advanced String Metrics

```rust
use scirs2_text::string_metrics::{DamerauLevenshteinMetric, StringMetric, Soundex};
use scirs2_text::weighted_distance::{WeightedLevenshtein, LevenshteinWeights};
use std::collections::HashMap;

// Damerau-Levenshtein with transpositions
let dl = DamerauLevenshteinMetric::new();
let d  = dl.distance("kitten", "sitting")?;

// Weighted Levenshtein with custom operation costs
let weights = LevenshteinWeights::new(2.0, 1.0, 0.5);  // ins=2, del=1, sub=0.5
let wl      = WeightedLevenshtein::with_weights(weights);
let wd      = wl.distance("kitten", "sitting")?;

// Character-pair specific substitution costs
let mut costs = HashMap::new();
costs.insert(('k', 's'), 0.1);   // Make k→s substitution cheap
let char_weights = LevenshteinWeights::default().with_substitution_costs(costs);
let cwl = WeightedLevenshtein::with_weights(char_weights);

// Phonetic encoding
let soundex = Soundex::new();
println!("{}", soundex.encode("Robert")?);      // R163
println!("{}", soundex.sounds_like("Smith", "Smythe")?);  // true
```

### Text Statistics and Readability

```rust
use scirs2_text::text_statistics::{TextStatistics, ReadabilityMetrics};

let stats   = TextStatistics::new();
let text    = "The quick brown fox jumps over the lazy dog. A simple sentence.";
let metrics = stats.get_all_metrics(text)?;

println!("Flesch Reading Ease: {:.2}",          metrics.flesch_reading_ease);
println!("Flesch-Kincaid Grade: {:.2}",         metrics.flesch_kincaid_grade_level);
println!("Gunning Fog Index: {:.2}",            metrics.gunning_fog);
println!("Lexical Diversity: {:.4}",            metrics.lexical_diversity);
println!("Word count: {}",                      metrics.text_statistics.word_count);
println!("Avg sentence length: {:.2}",          metrics.text_statistics.avg_sentence_length);
```

## Module Map

| Module | Contents |
|--------|----------|
| `tokenize` | Word, sentence, char, n-gram, regex tokenizers |
| `bpe_tokenizer` | Byte Pair Encoding tokenizer with vocabulary learning |
| `preprocess` | Normalizers, cleaners, preprocessing pipeline |
| `stemming` | Porter, Snowball, Lancaster stemmers; lemmatizer |
| `vectorize` | Count, TF-IDF, binary vectorizers |
| `enhanced_vectorize` | Enhanced vectorizers with n-gram and filtering |
| `embeddings` | Word2Vec (Skip-gram/CBOW), GloVe loader, FastText |
| `ner` | Named entity recognition (rule, dictionary, CRF-based) |
| `sequence_labeling` | CRF and HMM for sequence labelling (POS, NER, chunking) |
| `sentiment` | Lexicon-based and rule-based sentiment analysis |
| `topic_model` | LDA, NMF-based topic modeling |
| `text_classification` | Feature extraction, Naive Bayes, evaluation |
| `string_metrics` | Levenshtein, Damerau-Levenshtein, Jaro-Winkler, phonetics |
| `weighted_distance` | Weighted edit distances with custom operation costs |
| `text_statistics` | Readability metrics (Flesch, Gunning Fog, SMOG, etc.) |
| `knowledge_graph` | Entity-relation triple extraction |
| `coreference` | Mention detection and coreference clustering |
| `dependency` | Dependency parsing |
| `discourse` | Discourse analysis and RST primitives |
| `event_extraction` | Event trigger and argument extraction |
| `question_answering` | Extractive QA |
| `abstractive_summary` | Abstractive summarization primitives |
| `temporal` | Date/time expression extraction and normalization |
| `grammar` | Rule-based grammar checking |
| `semantic_parsing` | Logical form generation |
| `multilingual_ext` | Multilingual utilities and language detection |
| `language_models` | N-gram language model, character LM, perplexity |
| `information_theory` | Entropy, mutual information, KL divergence for text |
| `simd_ops` | SIMD-accelerated string operations |
| `parallel` | Parallel corpus processing utilities |

## Performance

- Tokenization: ~1M tokens/second (parallel mode)
- TF-IDF vectorization: ~10K documents/second
- String similarity: ~100K comparisons/second
- Topic modeling: scales to 100K+ documents
- Zero-copy sparse matrix output from vectorizers
- Memory-mapped corpus support for corpora larger than RAM

## Dependencies

- `scirs2-core` — RNG, parallel utilities, SIMD primitives
- `regex` — Regular expression matching
- `unicode-segmentation` — Unicode grapheme cluster segmentation
- `unicode-normalization` — Unicode normalization forms (NFC/NFD/NFKC/NFKD)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
