# SciRS2 Text

[![crates.io](https://img.shields.io/crates/v/scirs2-text.svg)](https://crates.io/crates/scirs2-text)
[![License](https://img.shields.io/crates/l/scirs2-text.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-text)](https://docs.rs/scirs2-text)

Text analysis and natural language processing module for the SciRS2 scientific computing library. This module provides tools for text processing, vectorization, and comparison.

## Features

- **Text Preprocessing**: Tokenization, normalization, and cleaning utilities
- **Text Vectorization**: Methods for converting text to numerical representations
- **Text Distance Metrics**: Various string and text distance measures
- **Vocabulary Management**: Tools for building and managing vocabularies
- **Utility Functions**: Helper functions for text manipulation

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-text = "0.1.0-alpha.1"
```

To enable optional features for integration with popular NLP libraries:

```toml
[dependencies]
scirs2-text = { version = "0.1.0-alpha.1", features = ["tokenizers", "wordpiece"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_text::{tokenize, preprocess, vectorize, distance, vocabulary};
use scirs2_core::error::CoreResult;

// Text preprocessing and tokenization
fn preprocessing_example() -> CoreResult<()> {
    // Example text
    let text = "Hello world! This is an example text for SciRS2 NLP module.";
    
    // Preprocessing: lowercase, remove punctuation, etc.
    let clean_text = preprocess::clean_text(text, true, true, true)?;
    println!("Cleaned text: '{}'", clean_text);
    
    // Tokenization
    let tokens = tokenize::word_tokenize(&clean_text)?;
    println!("Tokens: {:?}", tokens);
    
    // Stemming (Porter stemmer)
    let stemmed = tokenize::stem_tokens(&tokens, "porter")?;
    println!("Stemmed tokens: {:?}", stemmed);
    
    // N-grams
    let bigrams = tokenize::ngrams(&tokens, 2)?;
    println!("Bigrams: {:?}", bigrams);
    
    Ok(())
}

// Text vectorization
fn vectorization_example() -> CoreResult<()> {
    // Sample documents
    let documents = vec![
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ];
    
    // Create vocabulary
    let (vocab, word_counts) = vocabulary::build_vocabulary(
        &documents, 1, None, None, false)?;
    
    println!("Vocabulary: {:?}", vocab);
    println!("Word counts: {:?}", word_counts);
    
    // Count vectorizer
    let count_vectors = vectorize::count_vectorize(&documents, &vocab)?;
    println!("Count vectors:");
    for (i, vec) in count_vectors.iter().enumerate() {
        println!("  Document {}: {:?}", i, vec);
    }
    
    // TF-IDF vectorizer
    let tfidf_vectors = vectorize::tfidf_vectorize(&documents, &vocab, None, None)?;
    println!("TF-IDF vectors:");
    for (i, vec) in tfidf_vectors.iter().enumerate() {
        println!("  Document {}: {:?}", i, vec);
    }
    
    Ok(())
}

// Text distance measures
fn distance_example() -> CoreResult<()> {
    // Sample strings
    let s1 = "kitten";
    let s2 = "sitting";
    
    // Calculate Levenshtein distance
    let lev_dist = distance::levenshtein(s1, s2)?;
    println!("Levenshtein distance between '{}' and '{}': {}", s1, s2, lev_dist);
    
    // Calculate Jaro-Winkler similarity
    let jaro_sim = distance::jaro_winkler(s1, s2)?;
    println!("Jaro-Winkler similarity between '{}' and '{}': {}", s1, s2, jaro_sim);
    
    // Calculate Cosine similarity between documents
    let doc1 = "This is a test document about NLP";
    let doc2 = "This document is about natural language processing";
    
    let cos_sim = distance::cosine_similarity(doc1, doc2, None)?;
    println!("Cosine similarity between documents: {}", cos_sim);
    
    Ok(())
}
```

## Components

### Tokenization

Functions for text tokenization:

```rust
use scirs2_text::tokenize::{
    word_tokenize,          // Split text into words
    sent_tokenize,          // Split text into sentences
    regex_tokenize,         // Tokenize using regular expressions
    stem_tokens,            // Apply stemming to tokens
    lemmatize_tokens,       // Apply lemmatization to tokens
    ngrams,                 // Generate n-grams from tokens
    stopwords,              // Get stopwords for a given language
    remove_stopwords,       // Remove stopwords from token list
};
```

### Preprocessing

Text preprocessing utilities:

```rust
use scirs2_text::preprocess::{
    clean_text,             // Clean text (lowercase, remove punctuation, etc.)
    normalize_text,         // Normalize text (unicode normalization)
    expand_contractions,    // Expand contractions (e.g., "don't" -> "do not")
    remove_accents,         // Remove accents from text
    remove_html_tags,       // Remove HTML tags from text
    remove_special_chars,   // Remove special characters
    remove_numbers,         // Remove numbers from text
    remove_whitespace,      // Normalize whitespace
    replace_urls,           // Replace URLs with placeholder
    replace_emails,         // Replace email addresses with placeholder
};
```

### Text Vectorization

Methods for text vectorization:

```rust
use scirs2_text::vectorize::{
    count_vectorize,        // Convert text to count vectors
    tfidf_vectorize,        // Convert text to TF-IDF vectors
    hashing_vectorize,      // Use feature hashing for vectorization
    binary_vectorize,       // Convert text to binary vectors
    bm25_vectorize,         // BM25 vectorization for information retrieval
    cooccurrence_matrix,    // Build word co-occurrence matrix
};
```

### Distance Metrics

Text distance and similarity measures:

```rust
use scirs2_text::distance::{
    // Edit distances
    levenshtein,            // Levenshtein edit distance
    hamming,                // Hamming distance
    damerau_levenshtein,    // Damerau-Levenshtein distance
    
    // String similarities
    jaro_winkler,           // Jaro-Winkler similarity
    jaccard,                // Jaccard similarity coefficient
    sorensen_dice,          // Sørensen-Dice coefficient
    
    // Document similarities
    cosine_similarity,      // Cosine similarity
    euclidean_distance,     // Euclidean distance
    manhattan_distance,     // Manhattan distance
};
```

### Vocabulary Management

Tools for building and managing vocabularies:

```rust
use scirs2_text::vocabulary::{
    build_vocabulary,       // Build vocabulary from text corpus
    filter_vocabulary,      // Filter vocabulary by frequency/count
    save_vocabulary,        // Save vocabulary to file
    load_vocabulary,        // Load vocabulary from file
    map_tokens_to_ids,      // Convert tokens to vocabulary IDs
    map_ids_to_tokens,      // Convert vocabulary IDs to tokens
    Vocabulary,             // Vocabulary struct
};
```

### Utilities

Helper functions for text processing:

```rust
use scirs2_text::utils::{
    split_text,             // Split text by delimiter
    join_tokens,            // Join tokens with delimiter
    is_digit,               // Check if string is a digit
    is_punctuation,         // Check if character is punctuation
    is_stopword,            // Check if word is a stopword
    detect_language,        // Detect text language
    count_words,            // Count words in text
    count_sentences,        // Count sentences in text
};
```

## Integration with Other Libraries

This module provides easy integration with popular NLP libraries through optional features:

- `tokenizers`: Integration with HuggingFace tokenizers
- `wordpiece`: WordPiece tokenization for transformer models
- `sentencepiece`: SentencePiece tokenization

Example using feature-gated functionality:

```rust
// With the 'wordpiece' feature enabled
use scirs2_text::tokenize::wordpiece_tokenize;

let text = "Hello world, this is WordPiece tokenization.";
let vocab_file = "path/to/wordpiece/vocab.txt";

let tokens = wordpiece_tokenize(text, vocab_file, true, 100).unwrap();
println!("WordPiece tokens: {:?}", tokens);
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.