use scirs2_text::{
    FrequencyFilter, LengthFilter, RegexFilter, Result, StopwordsFilter, TokenFilter, Tokenizer,
    WordTokenizer,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("Token Filtering Demo");
    println!("===================\n");

    // Create a sample text
    let text = "The quick brown fox jumps over the lazy dog. The fox is quick and brown.";
    println!("Original text: {}\n", text);

    // Create a tokenizer
    let tokenizer = WordTokenizer::default();
    let tokens = tokenizer.tokenize(text)?;
    println!("Tokenized: {:?}\n", tokens);

    // 1. Filter by length
    println!("1. Length Filtering");
    println!("------------------");
    let length_filter = LengthFilter::new(4, 6);
    let filtered = length_filter.apply(&tokens);
    println!("Tokens with length 4-6: {:?}", filtered);

    let filtered_text = length_filter.filter_text(text, &tokenizer)?;
    println!("Filtered text: {}\n", filtered_text);

    // 2. Filter by frequency
    println!("2. Frequency Filtering");
    println!("---------------------");

    // Create token counts
    let mut counts = HashMap::new();
    for token in &tokens {
        *counts.entry(token.clone()).or_insert(0) += 1;
    }

    // Print counts
    println!("Token counts:");
    for (token, count) in &counts {
        println!("  {} : {}", token, count);
    }

    // Filter tokens that appear more than once
    let freq_filter = FrequencyFilter::from_counts(counts.clone(), 2);
    let filtered = freq_filter.apply(&tokens);
    println!("\nTokens that appear 2+ times: {:?}", filtered);

    let filtered_text = freq_filter.filter_text(text, &tokenizer)?;
    println!("Filtered text: {}\n", filtered_text);

    // 3. Filter by regex pattern
    println!("3. Regex Filtering");
    println!("-----------------");

    // Keep only tokens that contain a vowel followed by 'w' or 'r'
    let regex_filter = RegexFilter::new("[aeiou][wr]", true)?;
    let filtered = regex_filter.apply(&tokens);
    println!(
        "Tokens containing a vowel followed by 'w' or 'r': {:?}",
        filtered
    );

    let filtered_text = regex_filter.filter_text(text, &tokenizer)?;
    println!("Filtered text: {}\n", filtered_text);

    // 4. Stopwords filtering
    println!("4. Stopwords Filtering");
    println!("---------------------");

    // Define common stopwords
    let stopwords = vec![
        "the".to_string(),
        "is".to_string(),
        "and".to_string(),
        "over".to_string(),
        "a".to_string(),
        "an".to_string(),
    ];

    let stopwords_filter = StopwordsFilter::new(stopwords, true);
    let filtered = stopwords_filter.apply(&tokens);
    println!("Tokens with stopwords removed: {:?}", filtered);

    let filtered_text = stopwords_filter.filter_text(text, &tokenizer)?;
    println!("Filtered text: {}\n", filtered_text);

    // 5. Composite filtering
    println!("5. Composite Filtering");
    println!("---------------------");

    // Create separate filters
    let length_filter = LengthFilter::new(3, 5);
    let regex_filter = RegexFilter::new("^[a-z]", true)?;

    // Apply filters sequentially
    let filtered_by_length = length_filter.apply(&tokens);
    let filtered = regex_filter.apply(&filtered_by_length);
    println!(
        "Tokens with length 3-5 AND starting with lowercase letter: {:?}",
        filtered
    );

    // First filter by length
    let text_with_length = length_filter.filter_text(text, &tokenizer)?;

    // Then apply regex filter to the already filtered text
    let filtered_text = regex_filter.filter_text(&text_with_length, &tokenizer)?;

    // We should see only words that are 3-5 chars AND start with lowercase
    println!("Filtered text: {}\n", filtered_text);

    Ok(())
}
