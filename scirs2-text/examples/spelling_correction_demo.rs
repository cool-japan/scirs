// Spelling correction demo
//
// This example demonstrates the dictionary-based spelling correction
// functionality in the scirs2-text crate.

use scirs2_text::{DictionaryCorrector, DictionaryCorrectorConfig, SpellingCorrector};
use std::collections::HashMap;
use std::time::Instant;

// Sample text containing various misspellings
const TEXT_WITH_MISSPELLINGS: &str =
    "Speling erors are commmon in writen text. People often misspel words becuase \
    of typoes, or becase they dont know the corect speling. A good spellimg \
    corection algoritm can idntify and fix these erors automaticaly.";

// Expected corrected text
const EXPECTED_CORRECTED_TEXT: &str =
    "Spelling errors are common in written text. People often misspell words because \
    of typos, or because they dont know the correct spelling. A good spelling \
    correction algorithm can identify and fix these errors automatically.";

// A list of common misspellings and their corrections for testing
const COMMON_MISSPELLINGS: &[(&str, &str)] = &[
    ("speling", "spelling"),
    ("erors", "errors"),
    ("commmon", "common"),
    ("writen", "written"),
    ("misspel", "misspell"),
    ("becuase", "because"),
    ("typoes", "typos"),
    ("becase", "because"),
    ("corect", "correct"),
    ("spellimg", "spelling"),
    ("corection", "correction"),
    ("algoritm", "algorithm"),
    ("idntify", "identify"),
    ("automaticaly", "automatically"),
    // Some more challenging examples
    ("recieve", "receive"),
    ("beleive", "believe"),
    ("freind", "friend"),
    ("wierd", "weird"),
    ("acheive", "achieve"),
    ("accomodate", "accommodate"),
    ("apparant", "apparent"),
    ("begining", "beginning"),
    ("bizzare", "bizarre"),
    ("calender", "calendar"),
    ("commitee", "committee"),
    ("concious", "conscious"),
    ("definate", "definite"),
    ("dissapear", "disappear"),
    ("embarras", "embarrass"),
    ("existance", "existence"),
    ("grammer", "grammar"),
    ("harrass", "harass"),
    ("independant", "independent"),
    ("liason", "liaison"),
    ("maintainance", "maintenance"),
    ("millenium", "millennium"),
    ("neccessary", "necessary"),
    ("occassion", "occasion"),
    ("occurance", "occurrence"),
    ("persistant", "persistent"),
    ("playwrite", "playwright"),
    ("preceeding", "preceding"),
    ("publically", "publicly"),
    ("recomend", "recommend"),
    ("refering", "referring"),
    ("relevent", "relevant"),
    ("rythm", "rhythm"),
    ("seperate", "separate"),
    ("seige", "siege"),
    ("succesful", "successful"),
    ("truely", "truly"),
    ("untill", "until"),
    ("whereever", "wherever"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Dictionary-based Spelling Correction Demo\n");

    // Create a default spelling corrector
    let corrector = DictionaryCorrector::default();

    // Basic correction demo
    basic_correction_demo(&corrector)?;

    // Create a custom spelling corrector with specialized dictionary
    let specialized_corrector = create_specialized_corrector();

    // Compare basic and specialized correctors
    compare_correctors(&corrector, &specialized_corrector)?;

    // Performance test
    performance_test(&corrector, &specialized_corrector)?;

    // Text correction demo
    text_correction_demo(&corrector)?;

    // Custom configuration demo
    custom_config_demo()?;

    Ok(())
}

// Basic correction of common misspellings
fn basic_correction_demo(
    corrector: &DictionaryCorrector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Basic Spelling Correction ===\n");
    println!(
        "{:<20} {:<20} {:<20}",
        "Misspelled", "Corrected", "Expected"
    );
    println!("{:-<60}", "");

    for (misspelled, expected) in COMMON_MISSPELLINGS {
        let corrected = corrector.correct(misspelled)?;
        let success = if &corrected == expected { "✓" } else { "✗" };

        println!(
            "{:<20} {:<20} {:<20} {}",
            misspelled, corrected, expected, success
        );
    }

    // Count successful corrections
    let mut success_count = 0;
    for (misspelled, expected) in COMMON_MISSPELLINGS {
        let corrected = corrector.correct(misspelled)?;
        if &corrected == expected {
            success_count += 1;
        }
    }

    let success_rate = (success_count as f64 / COMMON_MISSPELLINGS.len() as f64) * 100.0;
    println!(
        "\nCorrectly fixed {}/{} misspellings ({:.1}%)",
        success_count,
        COMMON_MISSPELLINGS.len(),
        success_rate
    );

    Ok(())
}

// Create a specialized spelling corrector for programming-related terms
fn create_specialized_corrector() -> DictionaryCorrector {
    let mut dictionary = HashMap::new();

    // Add programming language terms
    let programming_terms = [
        ("rust", 100),
        ("python", 95),
        ("javascript", 90),
        ("typescript", 85),
        ("java", 80),
        ("kotlin", 75),
        ("ruby", 70),
        ("golang", 65),
        ("scala", 60),
        ("swift", 55),
        ("algorithm", 100),
        ("function", 95),
        ("variable", 90),
        ("constant", 85),
        ("class", 80),
        ("object", 75),
        ("method", 70),
        ("structure", 65),
        ("inheritance", 60),
        ("polymorphism", 55),
        ("recursion", 50),
        ("iteration", 45),
        ("compiler", 40),
        ("interpreter", 35),
        ("virtual", 30),
        ("machine", 25),
        ("memory", 20),
        ("pointer", 15),
        ("reference", 10),
        ("array", 5),
    ];

    for (term, freq) in &programming_terms {
        dictionary.insert(term.to_string(), *freq);
    }

    // Create the corrector with our specialized dictionary
    DictionaryCorrector::with_dictionary(dictionary)
}

// Compare the basic and specialized correctors
fn compare_correctors(
    basic: &DictionaryCorrector,
    specialized: &DictionaryCorrector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Comparison of Correctors ===\n");
    println!("Standard vs. Programming-specialized Dictionary\n");
    println!(
        "{:<15} {:<15} {:<15}",
        "Misspelled", "Standard", "Specialized"
    );
    println!("{:-<45}", "");

    // Some programming-related misspellings
    let programming_misspellings = [
        "funcion",  // function
        "algoritm", // algorithm
        "poiner",   // pointer
        "virable",  // variable
        "consant",  // constant
        "memmory",  // memory
        "refrence", // reference
        "aray",     // array
        "virtal",   // virtual
        "objetc",   // object
    ];

    for misspelled in &programming_misspellings {
        let basic_correction = basic.correct(misspelled)?;
        let specialized_correction = specialized.correct(misspelled)?;

        println!(
            "{:<15} {:<15} {:<15}",
            misspelled, basic_correction, specialized_correction
        );
    }

    println!("\nDictionary sizes:");
    println!("  - Standard dictionary: {} words", basic.dictionary_size());
    println!(
        "  - Specialized dictionary: {} words",
        specialized.dictionary_size()
    );

    Ok(())
}

// Test performance of different correctors and configurations
fn performance_test(
    basic: &DictionaryCorrector,
    specialized: &DictionaryCorrector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Performance Test ===\n");

    // Create a list of misspelled words to test
    let test_words: Vec<&str> = COMMON_MISSPELLINGS
        .iter()
        .map(|(misspelled, _)| *misspelled)
        .collect();

    // Measure performance of basic corrector
    let start = Instant::now();
    for word in &test_words {
        let _ = basic.correct(word)?;
    }
    let basic_time = start.elapsed();

    // Measure performance of specialized corrector
    let start = Instant::now();
    for word in &test_words {
        let _ = specialized.correct(word)?;
    }
    let specialized_time = start.elapsed();

    // Create a more restrictive configuration for testing
    let config = DictionaryCorrectorConfig {
        max_edit_distance: 1, // More strict
        case_sensitive: false,
        max_suggestions: 3,
        min_frequency: 10,
        prioritize_by_frequency: true,
    };

    let strict_corrector = DictionaryCorrector::new(config);

    // Measure performance of strict corrector
    let start = Instant::now();
    for word in &test_words {
        let _ = strict_corrector.correct(word)?;
    }
    let strict_time = start.elapsed();

    println!("Time to correct {} words:", test_words.len());
    println!("  - Standard dictionary: {:?}", basic_time);
    println!("  - Specialized dictionary: {:?}", specialized_time);
    println!(
        "  - Strict configuration (max_edit_distance=1): {:?}",
        strict_time
    );

    Ok(())
}

// Demonstrate text-wide correction
fn text_correction_demo(corrector: &DictionaryCorrector) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Text Correction Demo ===\n");

    println!("Original text with misspellings:");
    println!("{}\n", TEXT_WITH_MISSPELLINGS);

    // Correct the text
    let corrected_text = corrector.correct_text(TEXT_WITH_MISSPELLINGS)?;

    println!("Corrected text:");
    println!("{}\n", corrected_text);

    println!("Expected text:");
    println!("{}\n", EXPECTED_CORRECTED_TEXT);

    // Calculate how many misspellings were corrected
    let original_words: Vec<&str> = TEXT_WITH_MISSPELLINGS.split_whitespace().collect();

    let corrected_words: Vec<&str> = corrected_text.split_whitespace().collect();

    let expected_words: Vec<&str> = EXPECTED_CORRECTED_TEXT.split_whitespace().collect();

    let mut corrected_count = 0;
    let mut expected_corrections = 0;

    for i in 0..original_words.len() {
        if i < corrected_words.len()
            && i < expected_words.len()
            && original_words[i] != expected_words[i]
        {
            expected_corrections += 1;

            if corrected_words[i] == expected_words[i] {
                corrected_count += 1;
            }
        }
    }

    println!(
        "Correctly fixed {}/{} words ({:.1}%)",
        corrected_count,
        expected_corrections,
        (corrected_count as f64 / expected_corrections as f64) * 100.0
    );

    Ok(())
}

// Demonstrate custom configurations
fn custom_config_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Custom Configuration Demo ===\n");

    // Create different configurations
    let configs = [
        ("Default", DictionaryCorrectorConfig::default()),
        (
            "Conservative (max_edit_distance=1)",
            DictionaryCorrectorConfig {
                max_edit_distance: 1,
                case_sensitive: false,
                max_suggestions: 5,
                min_frequency: 1,
                prioritize_by_frequency: true,
            },
        ),
        (
            "Aggressive (max_edit_distance=3)",
            DictionaryCorrectorConfig {
                max_edit_distance: 3,
                case_sensitive: false,
                max_suggestions: 5,
                min_frequency: 1,
                prioritize_by_frequency: true,
            },
        ),
        (
            "Frequency-prioritized",
            DictionaryCorrectorConfig {
                max_edit_distance: 2,
                case_sensitive: false,
                max_suggestions: 5,
                min_frequency: 10,
                prioritize_by_frequency: true,
            },
        ),
        (
            "Non-prioritized",
            DictionaryCorrectorConfig {
                max_edit_distance: 2,
                case_sensitive: false,
                max_suggestions: 5,
                min_frequency: 1,
                prioritize_by_frequency: false,
            },
        ),
    ];

    // Test words that are sensitive to configuration differences
    let test_cases = [
        "recieve",    // Should be "receive"
        "accidant",   // Could be "accident" or "accidental" depending on max_edit_distance
        "programing", // Single-m, should be "programming"
        "languge",    // Should be "language"
        "freind",     // Should be "friend"
    ];

    // Print table header
    print!("{:<30}", "Configuration");
    for word in &test_cases {
        print!("{:<15}", word);
    }
    println!();
    println!("{:-<90}", "");

    // Test each configuration
    for (name, config) in &configs {
        let corrector = DictionaryCorrector::new(config.clone());

        print!("{:<30}", name);

        for word in &test_cases {
            let corrected = corrector.correct(word)?;
            print!("{:<15}", corrected);
        }

        println!();
    }

    println!("\nMultiple suggestions demo (using default config):\n");

    let corrector = DictionaryCorrector::default();

    for word in &test_cases {
        let suggestions = corrector.get_suggestions(word, 3)?;

        println!("Suggestions for '{}': ", word);
        for (i, suggestion) in suggestions.iter().enumerate() {
            println!("  {}. {}", i + 1, suggestion);
        }
        println!();
    }

    Ok(())
}
