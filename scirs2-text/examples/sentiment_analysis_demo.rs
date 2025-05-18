//! Sentiment analysis example

use scirs2_text::{LexiconSentimentAnalyzer, RuleBasedSentimentAnalyzer, SentimentLexicon};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sentiment Analysis Demo");
    println!("======================\n");

    // Create sentiment analyzers
    let basic_analyzer = LexiconSentimentAnalyzer::with_basic_lexicon();
    let rule_based_analyzer = RuleBasedSentimentAnalyzer::with_basic_lexicon();

    // Example texts for analysis
    let texts = vec![
        "I absolutely love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "I'm not happy with this purchase.",
        "Extremely disappointed with the quality.",
        "This is very good, I'm really satisfied.",
        "The worst experience ever!",
        "Somewhat decent, but could be better.",
    ];

    println!("Basic Lexicon-based Sentiment Analysis:");
    println!("======================================");

    for text in &texts {
        let result = basic_analyzer.analyze(text)?;
        println!("\nText: \"{}\"", text);
        println!("  Sentiment: {:?}", result.sentiment);
        println!("  Score: {:.2}", result.score);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!(
            "  Word counts: +{} -{} ={} (total: {})",
            result.word_counts.positive_words,
            result.word_counts.negative_words,
            result.word_counts.neutral_words,
            result.word_counts.total_words
        );
    }

    println!("\n\nRule-based Sentiment Analysis:");
    println!("=============================");

    // Examples with intensifiers
    let intensified_texts = vec![
        "This is good",
        "This is very good",
        "This is extremely good",
        "This is somewhat good",
        "This is really bad",
        "This is slightly bad",
    ];

    for text in &intensified_texts {
        let basic_result = basic_analyzer.analyze(text)?;
        let rule_result = rule_based_analyzer.analyze(text)?;

        println!("\nText: \"{}\"", text);
        println!("  Basic score: {:.2}", basic_result.score);
        println!("  Rule-based score: {:.2}", rule_result.score);
        println!(
            "  Difference: {:.2}",
            rule_result.score - basic_result.score
        );
    }

    // Batch analysis example
    println!("\n\nBatch Analysis:");
    println!("==============");

    let batch_texts = vec![
        "Great product!",
        "Terrible service.",
        "Average quality.",
        "Highly recommended!",
        "Would not buy again.",
    ];

    let batch_results = basic_analyzer.analyze_batch(&batch_texts)?;

    for (text, result) in batch_texts.iter().zip(batch_results.iter()) {
        println!(
            "{}: {:?} (score: {:.2})",
            text, result.sentiment, result.score
        );
    }

    // Creating a custom lexicon
    println!("\n\nCustom Lexicon Example:");
    println!("======================");

    let mut custom_lexicon = SentimentLexicon::new();
    custom_lexicon.add_word("awesome".to_string(), 3.0);
    custom_lexicon.add_word("terrible".to_string(), -3.0);
    custom_lexicon.add_word("meh".to_string(), -0.5);

    let custom_analyzer = LexiconSentimentAnalyzer::new(custom_lexicon);

    let custom_texts = vec![
        "This is awesome!",
        "Meh, not impressed",
        "Terrible experience",
    ];

    for text in &custom_texts {
        let result = custom_analyzer.analyze(text)?;
        println!(
            "\n\"{}\" -> {:?} (score: {:.2})",
            text, result.sentiment, result.score
        );
    }

    Ok(())
}
