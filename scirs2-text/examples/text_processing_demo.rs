//! Comprehensive text processing demonstration
//!
//! This example shows how to use various text processing capabilities
//! including tokenization, stemming, and vectorization.

use scirs2_text::{
    preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer},
    stemming::{PorterStemmer, SimpleLemmatizer, Stemmer},
    tokenize::{NgramTokenizer, RegexTokenizer, Tokenizer, WordTokenizer},
    vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Text Processing Demo ===\n");

    let documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast red fox leaped over the sleeping canine.",
        "Machine learning algorithms process textual data efficiently.",
        "Text processing and natural language understanding are important.",
    ];

    // 1. Text Normalization
    println!("1. Text Normalization");
    let normalizer = BasicNormalizer::new(true, true);
    for (i, doc) in documents.iter().enumerate() {
        let normalized = normalizer.normalize(doc)?;
        println!("Doc {}: {}", i + 1, normalized);
    }
    println!();

    // 2. Text Cleaning
    println!("2. Text Cleaning");
    let cleaner = BasicTextCleaner::new(true, true, true);
    for (i, doc) in documents.iter().enumerate() {
        let cleaned = cleaner.clean(doc)?;
        println!("Doc {}: {}", i + 1, cleaned);
    }
    println!();

    // 3. Tokenization Examples
    println!("3. Tokenization Examples");

    // Word tokenization
    let word_tokenizer = WordTokenizer::new(true);
    let tokens = word_tokenizer.tokenize(documents[0])?;
    println!("Word tokens: {:?}", tokens);

    // N-gram tokenization
    let ngram_tokenizer = NgramTokenizer::new(2)?;
    let ngrams = ngram_tokenizer.tokenize(documents[0])?;
    println!("2-grams: {:?}", ngrams);

    // Regex tokenization
    let regex_tokenizer = RegexTokenizer::new(r"\b\w+\b", false)?;
    let regex_tokens = regex_tokenizer.tokenize(documents[0])?;
    println!("Regex tokens: {:?}", regex_tokens);
    println!();

    // 4. Stemming and Lemmatization
    println!("4. Stemming and Lemmatization");
    let porter_stemmer = PorterStemmer::new();
    let lemmatizer = SimpleLemmatizer::new();

    let test_words = vec!["running", "jumped", "better", "processing"];
    for word in test_words {
        let stemmed = porter_stemmer.stem(word)?;
        let lemmatized = lemmatizer.stem(word)?;
        println!("{}: stemmed={}, lemmatized={}", word, stemmed, lemmatized);
    }
    println!();

    // 5. Count Vectorization
    println!("5. Count Vectorization");
    let mut count_vectorizer = CountVectorizer::new(false);

    let doc_refs = documents.to_vec();
    count_vectorizer.fit(&doc_refs)?;

    // Transform individual documents
    let count_matrix = count_vectorizer.transform_batch(&doc_refs)?;
    println!("Count vector shape: {:?}", count_matrix.shape());
    println!("Vocabulary size: {}", count_vectorizer.vocabulary().len());

    println!();

    // 6. TF-IDF Vectorization
    println!("6. TF-IDF Vectorization");
    let mut tfidf_vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));

    tfidf_vectorizer.fit(&doc_refs)?;
    let tfidf_matrix = tfidf_vectorizer.transform_batch(&doc_refs)?;

    println!("TF-IDF vector shape: {:?}", tfidf_matrix.shape());
    println!("Sample TF-IDF values:");
    for i in 0..3.min(tfidf_matrix.nrows()) {
        for j in 0..5.min(tfidf_matrix.ncols()) {
            print!("{:.3} ", tfidf_matrix[[i, j]]);
        }
        println!();
    }
    println!();

    // 7. Complete Pipeline Example
    println!("7. Complete Text Processing Pipeline");
    let test_text = "The cats were running quickly through the gardens.";

    // Normalize
    let normalized = normalizer.normalize(test_text)?;
    println!("Normalized: {}", normalized);

    // Clean
    let cleaned = cleaner.clean(&normalized)?;
    println!("Cleaned: {}", cleaned);

    // Tokenize
    let tokens = word_tokenizer.tokenize(&cleaned)?;
    println!("Tokens: {:?}", tokens);

    // Stem
    let stemmed_tokens: Result<Vec<_>, _> = tokens
        .iter()
        .map(|token| porter_stemmer.stem(token))
        .collect();
    let stemmed_tokens = stemmed_tokens?;
    println!("Stemmed: {:?}", stemmed_tokens);

    Ok(())
}
