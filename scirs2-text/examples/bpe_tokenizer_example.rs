use scirs2_text::{BpeConfig, BpeTokenizer, Result, Tokenizer};
use std::path::Path;

fn main() -> Result<()> {
    // Example corpus for training the tokenizer
    let corpus = [
        "this is a test sentence for bpe tokenization",
        "another test sentence with some overlapping words",
        "bpe works by merging common character pairs",
        "the algorithm builds a vocabulary of subword units",
        "these subword tokens can handle out-of-vocabulary words",
    ];

    println!("Training BPE tokenizer...");

    // Create a BPE tokenizer with custom configuration
    let mut config = BpeConfig::default();
    config.vocab_size = 100;
    config.min_frequency = 1;
    config.special_tokens = vec!["<pad>".to_string(), "<unk>".to_string()];

    let mut tokenizer = BpeTokenizer::new(config);

    // Train the tokenizer on the corpus
    tokenizer.train(&corpus)?;

    println!("Vocabulary size: {}", tokenizer.vocab_size());

    // Test the tokenizer on a new sentence
    let test_text = "this is an unseen sentence with some new words";
    let tokens = tokenizer.tokenize(test_text)?;

    println!("\nInput text: {}", test_text);
    println!("Tokenized: {:?}", tokens);

    // Save the vocabulary for later use
    let vocab_path = Path::new("bpe_vocab.json");
    tokenizer.save_vocabulary(vocab_path)?;
    println!("\nVocabulary saved to: {:?}", vocab_path);

    // Create a new tokenizer and load the saved vocabulary
    let mut new_tokenizer = BpeTokenizer::with_defaults();
    new_tokenizer.load_vocabulary(vocab_path)?;

    // Test that the loaded tokenizer produces the same tokens
    let new_tokens = new_tokenizer.tokenize(test_text)?;
    println!("\nTokenized with loaded vocabulary: {:?}", new_tokens);
    assert_eq!(tokens, new_tokens);

    // Clean up the vocabulary file
    std::fs::remove_file(vocab_path)?;

    Ok(())
}
