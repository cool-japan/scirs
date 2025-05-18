//! Multilingual text processing and language detection example

use scirs2_text::{Language, LanguageDetector, MultilingualProcessor, StopWords};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multilingual Text Processing Demo");
    println!("================================\n");

    // Create language detector
    let detector = LanguageDetector::new();

    // Example texts in different languages
    let texts = vec![
        (
            "This is a sample text in English to test language detection.",
            "English",
        ),
        (
            "Este es un texto de ejemplo en español para probar la detección de idioma.",
            "Spanish",
        ),
        (
            "Ceci est un exemple de texte en français pour tester la détection de langue.",
            "French",
        ),
        (
            "Dies ist ein Beispieltext auf Deutsch zum Testen der Spracherkennung.",
            "German",
        ),
        ("The quick brown fox jumps over the lazy dog.", "English"),
        (
            "El rápido zorro marrón salta sobre el perro perezoso.",
            "Spanish",
        ),
        (
            "Le rapide renard brun saute par-dessus le chien paresseux.",
            "French",
        ),
        (
            "Der schnelle braune Fuchs springt über den faulen Hund.",
            "German",
        ),
    ];

    println!("Language Detection:");
    println!("==================");

    for (text, expected) in &texts {
        let result = detector.detect(text)?;
        println!("\nText: \"{}\"", text);
        println!("Expected: {}", expected);
        println!(
            "Detected: {} (confidence: {:.2}%)",
            result.language.name(),
            result.confidence * 100.0
        );

        if !result.alternatives.is_empty() {
            println!("Alternatives:");
            for (lang, score) in &result.alternatives {
                println!("  - {}: {:.2}%", lang.name(), score * 100.0);
            }
        }
    }

    // Demonstrate stop words functionality
    println!("\n\nStop Words Processing:");
    println!("=====================");

    let stop_words = StopWords::new();

    let test_sentences = vec![
        ("The cat is on the mat", Language::English),
        ("Le chat est sur le tapis", Language::French),
        ("El gato está en la alfombra", Language::Spanish),
    ];

    for (sentence, language) in &test_sentences {
        let tokens: Vec<String> = sentence.split_whitespace().map(|s| s.to_string()).collect();

        let filtered = stop_words.remove_stop_words(&tokens, *language);

        println!("\nLanguage: {}", language.name());
        println!("Original: {}", sentence);
        println!("Tokens: {:?}", tokens);
        println!("Without stop words: {:?}", filtered);
    }

    // Demonstrate multilingual processor
    println!("\n\nMultilingual Processor:");
    println!("======================");

    let processor = MultilingualProcessor::new();

    let mixed_texts = vec![
        "Machine learning algorithms are transforming artificial intelligence",
        "Los algoritmos de aprendizaje automático están transformando la inteligencia artificial",
        "Les algorithmes d'apprentissage automatique transforment l'intelligence artificielle",
    ];

    for text in &mixed_texts {
        let result = processor.process(text)?;

        println!("\nOriginal: \"{}\"", text);
        println!(
            "Detected Language: {} (confidence: {:.2}%)",
            result.language.name(),
            result.confidence * 100.0
        );
        println!("Tokens: {} total", result.tokens.len());
        println!(
            "Filtered Tokens: {} after stop word removal",
            result.filtered_tokens.len()
        );

        if result.filtered_tokens.len() <= 5 {
            println!("Filtered: {:?}", result.filtered_tokens);
        } else {
            println!("First 5 filtered: {:?}...", &result.filtered_tokens[..5]);
        }
    }

    // Language code conversions
    println!("\n\nLanguage Code Conversions:");
    println!("=========================");

    let languages = vec![
        Language::English,
        Language::Spanish,
        Language::French,
        Language::German,
        Language::Chinese,
        Language::Japanese,
    ];

    for lang in &languages {
        println!("{}: ISO code = {}", lang.name(), lang.iso_code());
    }

    println!("\nReverse lookup:");
    let codes = vec!["en", "es", "fr", "de", "zh", "ja", "xx"];
    for code in &codes {
        let lang = Language::from_iso_code(code);
        println!("{} -> {}", code, lang.name());
    }

    Ok(())
}
