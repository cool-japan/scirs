// Rule-based lemmatizer example
//
// This example demonstrates the usage of the RuleLemmatizer for lemmatizing text,
// comparing it with the SimpleLemmatizer and traditional stemmers.

use scirs2_text::{
    PorterStemmer, PosTag, RuleLemmatizer, RuleLemmatizerBuilder, SimpleLemmatizer, Stemmer,
};
use std::collections::HashMap;
use std::time::Instant;

// Sample text for testing (excerpts from different domains)
const TEXTS: &[&str] = &[
    // Medical text
    "The patients were diagnosed with chronic obstructive pulmonary disease. Doctors prescribed bronchodilators and corticosteroids. Treatments improved their breathing significantly.",
    // News article
    "The government announced new policies yesterday. Officials said the changes would affect businesses and consumers. Critics argued that these measures weren't addressing the underlying economic issues.",

    // Technical documentation
    "The application uses caching to improve performance. Developers implemented indexing strategies to optimize database queries. Users reported faster loading times after the updates were deployed.",
    // Literature
    "He walked slowly through the forest, listening to the birds singing. The leaves rustled beneath his feet as he went deeper into the woods. Shadows grew longer as the sun began setting beyond the hills.",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Rule-based Lemmatization Demo\n");

    // Create lemmatizers and stemmers
    let simple_lemmatizer = SimpleLemmatizer::new();
    let rule_lemmatizer = RuleLemmatizer::new();
    let porter_stemmer = PorterStemmer::new();

    // Create a POS-aware lemmatizer using the builder pattern
    let pos_aware_lemmatizer = RuleLemmatizerBuilder::new()
        .use_pos_tagging(true)
        .apply_case_restoration(true)
        .check_vowels(true)
        .build();

    // Simple demo comparing lemmatization results
    println!("\n=== Lemmatization Comparison ===\n");
    let test_words = vec![
        ("running", Some(PosTag::Verb)),
        ("ran", Some(PosTag::Verb)),
        ("better", Some(PosTag::Adjective)),
        ("best", Some(PosTag::Adjective)),
        ("feet", Some(PosTag::Noun)),
        ("children", Some(PosTag::Noun)),
        ("went", Some(PosTag::Verb)),
        ("mice", Some(PosTag::Noun)),
        ("quickly", Some(PosTag::Adverb)),
        ("universities", Some(PosTag::Noun)),
        ("studying", Some(PosTag::Verb)),
        ("studied", Some(PosTag::Verb)),
        ("studies", Some(PosTag::Verb)),
    ];

    println!(
        "{:<15} {:<15} {:<15} {:<15}",
        "Word", "Simple", "Rule-based", "Porter"
    );
    println!("{:-<60}", "");

    for (word, pos) in &test_words {
        let simple_result = simple_lemmatizer.stem(word)?;
        let rule_result = if let Some(pos_tag) = pos {
            rule_lemmatizer.lemmatize(word, Some(pos_tag.clone()))
        } else {
            rule_lemmatizer.stem(word)?
        };
        let porter_result = porter_stemmer.stem(word)?;

        println!(
            "{:<15} {:<15} {:<15} {:<15}",
            word, simple_result, rule_result, porter_result
        );
    }

    // POS tagging demonstration
    println!("\n=== Part-of-Speech Aware Lemmatization ===\n");
    println!("Demonstrating how the same word can lemmatize differently based on POS tag:\n");

    let ambiguous_words = vec![
        ("left", vec![PosTag::Verb, PosTag::Adjective, PosTag::Noun]),
        ("close", vec![PosTag::Verb, PosTag::Adjective, PosTag::Noun]),
        ("flies", vec![PosTag::Verb, PosTag::Noun]),
        ("saw", vec![PosTag::Verb, PosTag::Noun]),
        ("light", vec![PosTag::Noun, PosTag::Verb, PosTag::Adjective]),
    ];

    for (word, pos_tags) in &ambiguous_words {
        println!("Word: \"{}\"", word);

        for pos in pos_tags {
            let pos_name = match pos {
                PosTag::Verb => "Verb",
                PosTag::Noun => "Noun",
                PosTag::Adjective => "Adjective",
                PosTag::Adverb => "Adverb",
                PosTag::Other => "Other",
            };
            println!(
                "  as {:<10}: {}",
                pos_name,
                pos_aware_lemmatizer.lemmatize(word, Some(pos.clone()))
            );
        }
        println!();
    }

    // Performance comparison
    println!("\n=== Performance Comparison ===\n");

    // Preprocess text into tokens for benchmarking
    let mut all_tokens = Vec::new();
    for text in TEXTS {
        all_tokens.extend(
            text.split_whitespace()
                .map(|s| s.trim_matches(|c: char| !c.is_alphabetic()))
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>(),
        );
    }

    // Assign random POS tags for the benchmark
    let mut tokens_with_pos = Vec::new();
    let pos_tags = [
        PosTag::Verb,
        PosTag::Noun,
        PosTag::Adjective,
        PosTag::Adverb,
        PosTag::Other,
    ];
    let mut i = 0;
    for token in &all_tokens {
        tokens_with_pos.push((token.to_string(), pos_tags[i % pos_tags.len()].clone()));
        i += 1;
    }

    // Benchmark simple lemmatizer
    let start = Instant::now();
    for token in &all_tokens {
        let _ = simple_lemmatizer.stem(token)?;
    }
    let simple_time = start.elapsed();

    // Benchmark rule-based lemmatizer without POS
    let start = Instant::now();
    for token in &all_tokens {
        let _ = rule_lemmatizer.stem(token)?;
    }
    let rule_time = start.elapsed();

    // Benchmark rule-based lemmatizer with POS
    let start = Instant::now();
    for (token, pos) in &tokens_with_pos {
        let _ = pos_aware_lemmatizer.lemmatize(&token, Some(pos.clone()));
    }
    let pos_rule_time = start.elapsed();

    // Benchmark Porter stemmer
    let start = Instant::now();
    for token in &all_tokens {
        let _ = porter_stemmer.stem(token)?;
    }
    let porter_time = start.elapsed();

    println!("Processing {} tokens:\n", all_tokens.len());
    println!("- SimpleLemmatizer: {:.2?}", simple_time);
    println!("- RuleLemmatizer (without POS): {:.2?}", rule_time);
    println!("- RuleLemmatizer (with POS): {:.2?}", pos_rule_time);
    println!("- PorterStemmer: {:.2?}", porter_time);

    // Example using RuleLemmatizer on real text
    println!("\n=== Text Processing Example ===\n");
    let text = "The scientists were running experiments to test their hypotheses. \
                The children went to the museum, where they saw the fossils of prehistoric animals. \
                Universities are studying better methods to address these issues quickly.";

    println!("Original text:\n{}\n", text);

    // Simple tokenization and lemmatization with POS tags
    let tokens: Vec<&str> = text
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphabetic()))
        .filter(|s| !s.is_empty())
        .collect();

    // Simulate a very basic POS tagger (in a real application, you would use a proper tagger)
    let mut pos_map: HashMap<&str, PosTag> = HashMap::new();
    pos_map.insert("scientists", PosTag::Noun);
    pos_map.insert("were", PosTag::Verb);
    pos_map.insert("running", PosTag::Verb);
    pos_map.insert("experiments", PosTag::Noun);
    pos_map.insert("test", PosTag::Verb);
    pos_map.insert("their", PosTag::Other);
    pos_map.insert("hypotheses", PosTag::Noun);
    pos_map.insert("children", PosTag::Noun);
    pos_map.insert("went", PosTag::Verb);
    pos_map.insert("museum", PosTag::Noun);
    pos_map.insert("saw", PosTag::Verb);
    pos_map.insert("fossils", PosTag::Noun);
    pos_map.insert("prehistoric", PosTag::Adjective);
    pos_map.insert("animals", PosTag::Noun);
    pos_map.insert("universities", PosTag::Noun);
    pos_map.insert("studying", PosTag::Verb);
    pos_map.insert("better", PosTag::Adjective);
    pos_map.insert("methods", PosTag::Noun);
    pos_map.insert("address", PosTag::Verb);
    pos_map.insert("issues", PosTag::Noun);
    pos_map.insert("quickly", PosTag::Adverb);

    // Process text
    println!("Word-by-word lemmatization results:");
    println!(
        "{:<15} {:<15} {:<15} {:<15}",
        "Word", "RuleLemmatizer", "With POS", "Porter"
    );
    println!("{:-<60}", "");

    for token in tokens {
        let pos = pos_map.get(token.to_lowercase().as_str()).cloned();

        let rule_result = rule_lemmatizer.stem(token)?;
        let pos_result = if let Some(pos_tag) = &pos {
            pos_aware_lemmatizer.lemmatize(token, Some(pos_tag.clone()))
        } else {
            pos_aware_lemmatizer.stem(token)?
        };
        let porter_result = porter_stemmer.stem(token)?;

        println!(
            "{:<15} {:<15} {:<15} {:<15}",
            token, rule_result, pos_result, porter_result
        );
    }

    // Custom rules example
    println!("\n=== Custom Rules and Exceptions ===\n");

    // Create a custom lemmatizer with additional rules and exceptions
    let custom_lemmatizer = RuleLemmatizerBuilder::new()
        .add_exception("dataset", "data")
        .add_exception("corpora", "corpus")
        .add_dict_entry("nlp", "natural language processing")
        .build();

    let custom_words = vec!["dataset", "corpora", "nlp", "datasets"];

    println!("Custom lemmatizer results:");
    for word in custom_words {
        println!(
            "{:<15} -> {}",
            word,
            custom_lemmatizer.lemmatize(word, None)
        );
    }

    Ok(())
}
