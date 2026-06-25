use super::*;
use crate::stemming::Stemmer;
use crate::tokenize::WordTokenizer;

#[test]
fn test_pos_tagger_lexicon() {
    let tagger = PosTagger::new();

    // Test known words from lexicon
    let result = tagger.tag_word("running");
    assert_eq!(result.tag, PosTag::Verb);
    assert!(result.confidence > 0.8);

    let result = tagger.tag_word("house");
    assert_eq!(result.tag, PosTag::Noun);
    assert!(result.confidence > 0.8);

    let result = tagger.tag_word("beautiful");
    assert_eq!(result.tag, PosTag::Adjective);
    assert!(result.confidence > 0.8);

    let result = tagger.tag_word("quickly");
    assert_eq!(result.tag, PosTag::Adverb);
    assert!(result.confidence > 0.8);
}

#[test]
fn test_pos_tagger_morphology() {
    let tagger = PosTagger::new();

    // Test morphological patterns for unknown words
    let result = tagger.tag_word("walking");
    assert_eq!(result.tag, PosTag::Verb); // -ing suffix

    let result = tagger.tag_word("happiness");
    assert_eq!(result.tag, PosTag::Noun); // -ness suffix

    let result = tagger.tag_word("colorful");
    assert_eq!(result.tag, PosTag::Adjective); // -ful suffix

    let result = tagger.tag_word("carefully");
    assert_eq!(result.tag, PosTag::Adverb); // -ly suffix
}

#[test]
fn test_pos_tagger_sequence() {
    let tagger = PosTagger::new();
    let tokens = vec![
        "The".to_string(),
        "quick".to_string(),
        "brown".to_string(),
        "fox".to_string(),
        "jumps".to_string(),
    ];

    let result = tagger.tag_sequence(&tokens);

    assert_eq!(result.tokens.len(), 5);
    assert_eq!(result.tags.len(), 5);
    assert_eq!(result.confidences.len(), 5);

    // Check that we got reasonable tags
    assert_eq!(result.tags[0], PosTag::Other); // "The" (determiner)
    assert_eq!(result.tags[1], PosTag::Adjective); // "quick"
    assert_eq!(result.tags[2], PosTag::Adjective); // "brown"
    assert_eq!(result.tags[3], PosTag::Noun); // "fox"
    assert_eq!(result.tags[4], PosTag::Verb); // "jumps"
}

#[test]
fn test_pos_taggertext() {
    let tagger = PosTagger::new();
    let tokenizer = WordTokenizer::default();

    let result = tagger
        .tagtext("The cat runs quickly", &tokenizer)
        .expect("Operation failed");

    assert_eq!(result.tokens.len(), 4);
    assert_eq!(result.tags[0], PosTag::Other); // "The"
    assert_eq!(result.tags[1], PosTag::Noun); // "cat"
    assert_eq!(result.tags[2], PosTag::Verb); // "runs"
    assert_eq!(result.tags[3], PosTag::Adverb); // "quickly"
}

#[test]
fn test_pos_tagger_custom_words() {
    let mut tagger = PosTagger::new();

    // Add custom word
    tagger.add_word("scirs", PosTag::Noun);

    let result = tagger.tag_word("scirs");
    assert_eq!(result.tag, PosTag::Noun);
    assert!(result.confidence > 0.8);

    // Remove word
    tagger.remove_word("scirs");
    let result = tagger.tag_word("scirs");
    assert_ne!(result.confidence, 0.9); // Should use morphological pattern now
}

#[test]
fn test_pos_tagger_transition_probs() {
    let tagger = PosTagger::new();

    // Test transition probabilities
    let prob = tagger.get_transition_probability(&PosTag::Adjective, &PosTag::Noun);
    assert!(prob > 0.0);

    let prob = tagger.get_transition_probability(&PosTag::Noun, &PosTag::Verb);
    assert!(prob > 0.0);
}

#[test]
fn test_pos_aware_lemmatizer() {
    let lemmatizer = PosAwareLemmatizer::new();

    // Test automatic POS detection and lemmatization
    assert_eq!(lemmatizer.lemmatize("running"), "run");
    assert_eq!(lemmatizer.lemmatize("cats"), "cat");
    assert_eq!(lemmatizer.lemmatize("better"), "good");
    assert_eq!(lemmatizer.lemmatize("quickly"), "quick");

    // Test explicit POS tagging
    assert_eq!(lemmatizer.lemmatize_with_pos("flies", PosTag::Verb), "fly");
    assert_eq!(lemmatizer.lemmatize_with_pos("flies", PosTag::Noun), "fly");

    // Test irregular forms
    assert_eq!(lemmatizer.lemmatize("went"), "go");
    assert_eq!(lemmatizer.lemmatize("children"), "child");
    assert_eq!(lemmatizer.lemmatize("feet"), "foot");
}

#[test]
fn test_pos_aware_lemmatizertext() {
    let lemmatizer = PosAwareLemmatizer::new();
    let tokenizer = WordTokenizer::default();

    let result = lemmatizer
        .lemmatizetext("The cats are running quickly", &tokenizer)
        .expect("Operation failed");

    assert_eq!(result, vec!["the", "cat", "be", "run", "quick"]);
}

#[test]
fn test_pos_aware_lemmatizer_stemmer_trait() {
    let lemmatizer = PosAwareLemmatizer::new();

    // Test Stemmer trait implementation
    assert_eq!(lemmatizer.stem("running").expect("Operation failed"), "run");
    assert_eq!(
        lemmatizer.stem("children").expect("Operation failed"),
        "child"
    );

    // Test batch processing
    let words = vec!["running", "cats", "better", "quickly"];
    let expected = vec!["run", "cat", "good", "quick"];
    assert_eq!(
        lemmatizer.stem_batch(&words).expect("Operation failed"),
        expected
    );
}

#[test]
fn test_pos_aware_lemmatizer_custom_additions() {
    let mut lemmatizer = PosAwareLemmatizer::new();

    // Add custom POS word
    lemmatizer.add_pos_word("tensorflow", PosTag::Noun);

    // Add custom lemma
    lemmatizer.add_lemma("tensorflow", "tf");

    assert_eq!(lemmatizer.lemmatize("tensorflow"), "tf");

    // Add custom exception
    lemmatizer.add_exception("pytorch", "torch");
    assert_eq!(lemmatizer.lemmatize("pytorch"), "torch");
}

#[test]
fn test_pos_tagger_config() {
    let config = PosTaggerConfig {
        use_context: false,
        smoothing_factor: 0.01,
        use_morphology: true,
        use_capitalization: true,
    };

    let tagger = PosTagger::with_config(config);
    assert!(!tagger.use_context);
    assert_eq!(tagger.smoothing_factor, 0.01);
}

#[test]
fn test_pos_tagging_capitalization() {
    let tagger = PosTagger::new();

    // Test proper noun detection
    let result = tagger.tag_word("John");
    assert_eq!(result.tag, PosTag::Noun); // Should be detected as proper noun

    let result = tagger.tag_word("USA");
    assert_eq!(result.tag, PosTag::Noun); // All caps should be noun
}

#[test]
fn test_pos_tagger_confidence_scores() {
    let tagger = PosTagger::new();

    // Known word should have high confidence
    let result = tagger.tag_word("running");
    assert!(result.confidence > 0.8);

    // Unknown word should have lower confidence
    let result = tagger.tag_word("xyzunknown");
    assert!(result.confidence < 0.8);
}

#[test]
fn test_pos_aware_lemmatizer_configurations() {
    let pos_config = PosTaggerConfig {
        use_context: false,
        smoothing_factor: 0.01,
        use_morphology: true,
        use_capitalization: true,
    };

    let lemma_config = crate::stemming::LemmatizerConfig {
        use_pos_tagging: true,
        default_pos: PosTag::Noun,
        apply_case_restoration: false,
        check_vowels: true,
    };

    let lemmatizer = PosAwareLemmatizer::with_configs(pos_config, lemma_config);

    // Test that it works with custom configs
    assert_eq!(lemmatizer.lemmatize("Running"), "run"); // Should not restore case
}

#[test]
fn test_morphological_analyzer() {
    let analyzer = MorphologicalAnalyzer::new();

    // Test suffix analysis
    let predictions = analyzer.analyze("quickly");
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Adverb); // -ly suffix

    let predictions = analyzer.analyze("happiness");
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Noun); // -ness suffix

    let predictions = analyzer.analyze("beautiful");
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Adjective); // -ful suffix

    // Test prefix analysis
    let predictions = analyzer.analyze("unhappy");
    assert!(!predictions.is_empty());
    assert!(predictions.iter().any(|(tag, _)| *tag == PosTag::Adjective)); // un- prefix

    let predictions = analyzer.analyze("rebuild");
    assert!(!predictions.is_empty());
    assert!(predictions.iter().any(|(tag, _)| *tag == PosTag::Verb)); // re- prefix

    // Test word shape analysis
    let predictions = analyzer.analyze("JavaScript");
    assert!(!predictions.is_empty());
    assert!(predictions.iter().any(|(tag, _)| *tag == PosTag::Noun)); // CamelCase shape

    let predictions = analyzer.analyze("well-known");
    assert!(!predictions.is_empty());
    assert!(predictions.iter().any(|(tag, _)| *tag == PosTag::Adjective)); // hyphenated shape

    // Test prediction method
    let prediction = analyzer.predict_pos("running");
    assert!(prediction.is_some());
    let (tag, score) = prediction.expect("Operation failed");
    assert_eq!(tag, PosTag::Verb); // -ing suffix
    assert!(score > 0.0);
}

#[test]
fn test_wordshape_detection() {
    let analyzer = MorphologicalAnalyzer::new();

    assert_eq!(analyzer.get_wordshape("Hello"), "Title");
    assert_eq!(analyzer.get_wordshape("WORLD"), "UPPER");
    assert_eq!(analyzer.get_wordshape("hello"), "lower");
    assert_eq!(analyzer.get_wordshape("CamelCase"), "CamelCase");
    assert_eq!(analyzer.get_wordshape("well-known"), "with-dash");
    assert_eq!(analyzer.get_wordshape("file_name"), "has_underscore");
    assert_eq!(analyzer.get_wordshape("Dr."), "has.period");
    assert_eq!(analyzer.get_wordshape("item123"), "123number");
    assert_eq!(analyzer.get_wordshape(""), "empty");
}

#[test]
fn test_contextual_disambiguator() {
    let disambiguator = ContextualDisambiguator::new();

    // Test simple disambiguation
    let tokens = vec!["the".to_string(), "quick".to_string(), "fox".to_string()];
    let mut tags = vec![PosTag::Other, PosTag::Verb, PosTag::Noun]; // "quick" mistagged as verb
    let mut confidences = vec![0.9, 0.6, 0.9]; // low confidence for "quick"

    disambiguator.disambiguate(&tokens, &mut tags, &mut confidences);

    // "quick" should be disambiguated to adjective (determiner + ? + noun pattern)
    assert_eq!(tags[1], PosTag::Adjective);
    assert!(confidences[1] > 0.6); // confidence should increase

    // Test another pattern
    let tokens = vec!["he".to_string(), "run".to_string(), "fast".to_string()];
    let mut tags = vec![PosTag::Other, PosTag::Verb, PosTag::Adjective]; // "fast" might be adjective
    let mut confidences = vec![0.9, 0.8, 0.6];

    disambiguator.disambiguate(&tokens, &mut tags, &mut confidences);

    // "fast" after verb should be adverb
    assert_eq!(tags[2], PosTag::Adverb);
}

#[test]
fn test_pattern_matching() {
    let disambiguator = ContextualDisambiguator::new();

    // Test exact pattern matching
    let pattern = (Some(PosTag::Other), PosTag::Adjective, Some(PosTag::Noun));
    assert!(disambiguator.matches_pattern(
        &pattern,
        Some(&PosTag::Other),
        &PosTag::Adjective,
        Some(&PosTag::Noun)
    ));

    // Test partial pattern matching (no left context required)
    let pattern = (None, PosTag::Adjective, Some(PosTag::Noun));
    assert!(disambiguator.matches_pattern(&pattern, None, &PosTag::Adjective, Some(&PosTag::Noun)));

    // Test mismatch
    let pattern = (Some(PosTag::Verb), PosTag::Adjective, Some(PosTag::Noun));
    assert!(!disambiguator.matches_pattern(
        &pattern,
        Some(&PosTag::Other), // Wrong left context
        &PosTag::Adjective,
        Some(&PosTag::Noun)
    ));
}

#[test]
fn test_morphological_edge_cases() {
    let analyzer = MorphologicalAnalyzer::new();

    // Test short words (should not match prefixes/suffixes)
    let predictions = analyzer.analyze("be");
    assert!(predictions.is_empty() || predictions[0].1 < 0.5); // Low confidence for short words

    // Test words that don't match any patterns
    let predictions = analyzer.analyze("xyz");
    assert!(predictions.is_empty() || predictions[0].1 < 0.5);

    // Test mixed patterns
    let predictions = analyzer.analyze("preprocessing"); // pre- prefix + -ing suffix
    assert!(!predictions.is_empty());
    // Should combine evidence from both prefix and suffix

    // Test capitalized technical terms
    let predictions = analyzer.analyze("PostgreSQL");
    assert!(!predictions.is_empty());
    assert!(predictions.iter().any(|(tag, _)| *tag == PosTag::Noun));
}

#[test]
fn test_advanced_morphological_patterns() {
    let analyzer = MorphologicalAnalyzer::new();

    // Test compound patterns
    let predictions = analyzer.analyze("unhappiness"); // un- + -ness
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Noun); // -ness dominates

    let predictions = analyzer.analyze("reusable"); // re- + -able
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Adjective); // -able dominates

    // Test scientific/technical terms
    let predictions = analyzer.analyze("biodegradable");
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Adjective);

    let predictions = analyzer.analyze("programmer");
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Noun);

    // Test comparative/superlative
    let predictions = analyzer.analyze("fastest");
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0].0, PosTag::Adjective);
}
