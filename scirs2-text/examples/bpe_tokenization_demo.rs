use scirs2_text::{BpeConfig, BpeTokenizer, Result, Tokenizer};

fn main() -> Result<()> {
    println!("Byte Pair Encoding (BPE) Tokenization Demo");
    println!("===========================================\n");

    // Create a simple corpus for training
    let corpus = [
        "Hello, this is a demonstration of BPE tokenization.",
        "BPE learns subword units by iteratively merging the most frequent pairs.",
        "It is particularly useful for languages with rich morphology.",
        "Words like 'uncommonness' can be broken into 'un', 'common', 'ness'.",
        "This improves handling of rare and out-of-vocabulary words.",
    ];

    // Configure and train the BPE tokenizer
    let mut tokenizer = BpeTokenizer::new(BpeConfig {
        vocab_size: 100,  // Small vocabulary for demonstration
        min_frequency: 2, // Only merge pairs that appear at least twice
        special_tokens: vec![
            // Add special tokens
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
        ],
        character_level: true, // Start with characters (not words)
        lowercase: true,       // Convert all text to lowercase
    });

    println!("Training BPE tokenizer on a small corpus...");
    tokenizer.train(&corpus)?;

    // Display vocabulary information
    let vocab_size = tokenizer.vocab_size();
    println!("Learned vocabulary size: {}\n", vocab_size);

    // Tokenize some examples
    let examples = [
        "Hello world!",
        "uncommonness",
        "tokenization demonstration",
        "Out-of-vocabulary handling",
    ];

    for example in &examples {
        let tokens = tokenizer.tokenize(example)?;
        println!("Original: \"{}\"", example);
        println!("Tokenized: {:?}", tokens);
        println!("Token count: {}\n", tokens.len());
    }

    // Save the tokenizer's vocabulary to a file
    let vocab_path = "bpe_vocab.txt";
    tokenizer.save_vocabulary(vocab_path)?;
    println!("Saved vocabulary to {}", vocab_path);

    // Load the vocabulary and tokenize again
    let mut new_tokenizer = BpeTokenizer::with_defaults();
    new_tokenizer.load_vocabulary(vocab_path)?;

    let test_text = "Hello, demonstrating vocabulary loading!";
    let tokens = new_tokenizer.tokenize(test_text)?;
    println!("\nTokenization after loading vocabulary:");
    println!("Original: \"{}\"", test_text);
    println!("Tokenized: {:?}", tokens);

    Ok(())
}
