//! Enhanced text vectorization demonstration
//!
//! This example shows how to use the enhanced vectorizers with n-gram support,
//! document frequency filtering, and advanced TF-IDF options.

use scirs2_text::{
    enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer},
    preprocess::BasicTextCleaner,
    preprocess::TextCleaner,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Enhanced Text Vectorization Demo ===\n");

    let documents = vec![
        "The quick brown fox jumps over the lazy dog.",
        "A fast red fox leaped over the sleeping canine.",
        "Machine learning algorithms process textual data efficiently.",
        "Text processing and natural language understanding are important.",
        "Natural language processing is a field of artificial intelligence.",
        "Deep learning models can understand complex text patterns.",
    ];

    // 1. Enhanced Count Vectorizer with Unigrams
    println!("1. Enhanced Count Vectorizer (Unigrams only)");
    let mut count_vec_unigram = EnhancedCountVectorizer::new()
        .set_binary(false)
        .set_max_features(Some(20));

    count_vec_unigram.fit(&documents)?;
    let count_matrix = count_vec_unigram.transform_batch(&documents)?;

    println!("Vocabulary size: {}", count_vec_unigram.vocabulary().len());
    println!("Count matrix shape: {:?}", count_matrix.shape());
    println!();

    // 2. Enhanced Count Vectorizer with N-grams
    println!("2. Enhanced Count Vectorizer (Unigrams + Bigrams)");
    let mut count_vec_ngram = EnhancedCountVectorizer::new()
        .set_ngram_range((1, 2))?
        .set_max_features(Some(30));

    count_vec_ngram.fit(&documents)?;
    let ngram_matrix = count_vec_ngram.transform_batch(&documents)?;

    println!(
        "Vocabulary size with n-grams: {}",
        count_vec_ngram.vocabulary().len()
    );
    println!("N-gram count matrix shape: {:?}", ngram_matrix.shape());

    // Show some n-gram tokens
    let vocab = count_vec_ngram.vocabulary();
    let mut ngram_tokens: Vec<String> = Vec::new();
    for i in 0..vocab.len().min(10) {
        if let Some(token) = vocab.get_token(i) {
            if token.contains(' ') {
                // This is a bigram
                ngram_tokens.push(token.to_string());
            }
        }
    }
    println!("Sample bigrams: {:?}", ngram_tokens);
    println!();

    // 3. Enhanced Count Vectorizer with Document Frequency Filtering
    println!("3. Count Vectorizer with Document Frequency Filtering");
    let mut count_vec_filtered = EnhancedCountVectorizer::new()
        .set_min_df(0.3)?  // Token must appear in at least 30% of documents
        .set_max_df(0.8)?; // Token must appear in at most 80% of documents

    count_vec_filtered.fit(&documents)?;

    println!(
        "Vocabulary size after DF filtering: {}",
        count_vec_filtered.vocabulary().len()
    );
    println!();

    // 4. Enhanced TF-IDF Vectorizer with N-grams
    println!("4. Enhanced TF-IDF Vectorizer with N-grams");
    let mut tfidf_vec = EnhancedTfidfVectorizer::new()
        .set_ngram_range((1, 2))?
        .set_max_features(Some(50))
        .set_smooth_idf(true)
        .set_sublinear_tf(true)
        .set_norm(Some("l2".to_string()))?;

    tfidf_vec.fit(&documents)?;
    let tfidf_matrix = tfidf_vec.transform_batch(&documents)?;

    println!("TF-IDF matrix shape: {:?}", tfidf_matrix.shape());
    println!("TF-IDF with smoothing and sublinear TF applied");

    // Show TF-IDF values for first document
    let first_doc_tfidf = tfidf_matrix.row(0);
    let mut top_features: Vec<(String, f64)> = Vec::new();

    for (idx, &value) in first_doc_tfidf.iter().enumerate() {
        if value > 0.0 {
            if let Some(token) = tfidf_vec.vocabulary().get_token(idx) {
                top_features.push((token.to_string(), value));
            }
        }
    }

    top_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop TF-IDF features for first document:");
    for (token, score) in top_features.iter().take(5) {
        println!("  {}: {:.3}", token, score);
    }
    println!();

    // 5. Processing with Text Cleaning
    println!("5. Vectorization with Text Preprocessing");
    let cleaner = BasicTextCleaner::new(true, true, true);

    // Clean documents first
    let cleaned_docs: Result<Vec<_>, _> = documents.iter().map(|doc| cleaner.clean(doc)).collect();
    let cleaned_docs = cleaned_docs?;
    let cleaned_refs: Vec<&str> = cleaned_docs.iter().map(|s| s.as_str()).collect();

    let mut tfidf_cleaned = EnhancedTfidfVectorizer::new()
        .set_ngram_range((1, 2))?
        .set_max_features(Some(30));

    tfidf_cleaned.fit(&cleaned_refs)?;
    let cleaned_matrix = tfidf_cleaned.transform_batch(&cleaned_refs)?;

    println!("TF-IDF shape after cleaning: {:?}", cleaned_matrix.shape());
    println!("Processing pipeline: Clean -> Tokenize -> Vectorize");

    Ok(())
}
