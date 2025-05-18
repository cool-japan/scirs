use scirs2_text::{LancasterStemmer, PorterStemmer, SimpleLemmatizer, SnowballStemmer, Stemmer};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Stemming Algorithms Comparison Demo");
    println!("-----------------------------------");

    // Create instances of different stemmers
    let porter_stemmer = PorterStemmer::new();
    let snowball_stemmer = SnowballStemmer::new("english")?;
    let lancaster_stemmer = LancasterStemmer::new();
    let lemmatizer = SimpleLemmatizer::new();

    // Test words to compare stemming results
    let test_words = vec![
        "running",
        "ran",
        "runs",
        "easily",
        "fishing",
        "fished",
        "troubled",
        "troubling",
        "troubles",
        "production",
        "productive",
        "argument",
        "arguing",
        "university",
        "universities",
        "maximizing",
        "maximum",
        "presumably",
        "multiply",
        "opposition",
        "computational",
    ];

    // Print results in a table format
    println!(
        "{:<15} {:<15} {:<15} {:<15} {:<15}",
        "Original", "Porter", "Snowball", "Lancaster", "Lemmatizer"
    );
    println!("{}", "-".repeat(75));

    for word in test_words {
        let porter_result = porter_stemmer.stem(word)?;
        let snowball_result = snowball_stemmer.stem(word)?;
        let lancaster_result = lancaster_stemmer.stem(word)?;
        let lemma_result = lemmatizer.stem(word)?;

        println!(
            "{:<15} {:<15} {:<15} {:<15} {:<15}",
            word, porter_result, snowball_result, lancaster_result, lemma_result
        );
    }

    // Demonstrate configurability of the Lancaster stemmer
    println!("\nLancaster Stemmer Configuration Options");
    println!("------------------------------------");

    let default_lancaster = LancasterStemmer::new();
    let custom_lancaster = LancasterStemmer::new()
        .with_min_stemmed_length(3)
        .with_acceptable_check(false);

    println!(
        "{:<15} {:<20} {:<20}",
        "Original", "Default Lancaster", "Custom Lancaster"
    );
    println!("{}", "-".repeat(55));

    let custom_test_words = vec!["provision", "ear", "me", "fishing", "multiply"];

    for word in custom_test_words {
        let default_result = default_lancaster.stem(word)?;
        let custom_result = custom_lancaster.stem(word)?;

        println!("{:<15} {:<20} {:<20}", word, default_result, custom_result);
    }

    println!("\nNotes:");
    println!("- Porter stemmer: Established algorithm, medium aggressiveness");
    println!("- Snowball stemmer: Improved Porter algorithm with language-specific rules");
    println!("- Lancaster stemmer: Most aggressive stemming, can be configured");
    println!("- Lemmatizer: Dictionary-based approach, produces actual words");

    Ok(())
}
