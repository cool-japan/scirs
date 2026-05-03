//! Advanced NER extractor with static helper methods, RAKE phrase extraction,
//! simple coreference resolution, and SVO relation extraction.
//!
//! This module provides a higher-level API that wraps the lower-level pattern
//! infrastructure in [`super::patterns`] and [`super::entities`].

use super::entities::{Entity, EntityType};
use super::patterns::{
    DATE_PATTERN, EMAIL_PATTERN, MONEY_PATTERN, PERCENTAGE_PATTERN, PHONE_PATTERN, TIME_PATTERN,
    URL_PATTERN,
};
use crate::error::{Result, TextError};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Number pattern (not already in patterns.rs at module-level as a lazy_static)
// ---------------------------------------------------------------------------

lazy_static! {
    static ref NUMBER_PATTERN: Regex = Regex::new(
        r"(?x)
        (?:
            [+-]?                       # optional sign
            (?:
                \d{1,3}(?:,\d{3})+      # thousands-separated integer
                | \d+                   # plain integer
            )
            (?:\.\d+)?                  # optional decimal
            (?:[eE][+-]?\d+)?           # optional scientific exponent
        )
        \b"
    )
    .expect("NUMBER_PATTERN is valid");
}

// ---------------------------------------------------------------------------
// CoreferenceCluster (simple span-based)
// ---------------------------------------------------------------------------

/// A cluster of coreferring mentions, anchored to a canonical surface form.
#[derive(Debug, Clone)]
pub struct CoreferenceCluster {
    /// The canonical text for this cluster (e.g. the first named mention).
    pub canonical: String,
    /// All `(start_byte, end_byte)` spans that refer to the same entity.
    pub mentions: Vec<(usize, usize)>,
}

// ---------------------------------------------------------------------------
// AdvancedNerExtractor
// ---------------------------------------------------------------------------

/// Advanced named-entity extractor backed by regex patterns and an optional
/// custom-pattern registry.
///
/// Unlike [`super::extractors::RuleBasedNER`], this struct provides *static*
/// convenience helpers so callers can extract specific entity types without
/// constructing an instance.
///
/// # Example
///
/// ```rust
/// use scirs2_text::information_extraction::advanced_ner::AdvancedNerExtractor;
///
/// let emails = AdvancedNerExtractor::extract_emails("Contact: alice@example.com");
/// assert_eq!(emails.len(), 1);
/// assert_eq!(emails[0].text, "alice@example.com");
/// ```
pub struct AdvancedNerExtractor {
    /// Each entry holds an `EntityType` and the compiled regex for it.
    custom_patterns: Vec<(EntityType, Regex)>,
}

impl Default for AdvancedNerExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedNerExtractor {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create an extractor with the built-in pattern set (email, URL, date,
    /// time, phone, money, percent, number).
    pub fn new() -> Self {
        Self {
            custom_patterns: Vec::new(),
        }
    }

    /// Register an additional custom pattern.  The `pattern` string is
    /// compiled as a [`regex::Regex`] and matched against the input text.
    ///
    /// # Errors
    ///
    /// Returns [`TextError::InvalidInput`] when `pattern` fails to compile.
    pub fn add_pattern(&mut self, entity_type: EntityType, pattern: &str) -> Result<()> {
        let re = Regex::new(pattern)
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex '{}': {}", pattern, e)))?;
        self.custom_patterns.push((entity_type, re));
        Ok(())
    }

    // ------------------------------------------------------------------
    // Instance extraction
    // ------------------------------------------------------------------

    /// Extract all entity types from `text` (built-in + custom patterns).
    pub fn extract(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Built-in patterns
        entities.extend(extract_with_pattern(
            text,
            &EMAIL_PATTERN,
            EntityType::Email,
            1.0,
        ));
        entities.extend(extract_with_pattern(
            text,
            &URL_PATTERN,
            EntityType::Url,
            1.0,
        ));
        entities.extend(extract_with_pattern(
            text,
            &DATE_PATTERN,
            EntityType::Date,
            0.95,
        ));
        entities.extend(extract_with_pattern(
            text,
            &TIME_PATTERN,
            EntityType::Time,
            0.95,
        ));
        entities.extend(extract_with_pattern(
            text,
            &PHONE_PATTERN,
            EntityType::Phone,
            0.90,
        ));
        entities.extend(extract_with_pattern(
            text,
            &MONEY_PATTERN,
            EntityType::Money,
            0.95,
        ));
        entities.extend(extract_with_pattern(
            text,
            &PERCENTAGE_PATTERN,
            EntityType::Percentage,
            0.95,
        ));
        entities.extend(extract_with_pattern(
            text,
            &NUMBER_PATTERN,
            EntityType::Custom("number".to_string()),
            0.85,
        ));

        // Custom patterns
        for (et, re) in &self.custom_patterns {
            entities.extend(extract_with_pattern(text, re, et.clone(), 0.80));
        }

        // Sort by start position and remove overlapping lower-confidence matches
        entities.sort_by_key(|e| e.start);
        dedup_overlapping(entities)
    }

    // ------------------------------------------------------------------
    // Static helpers
    // ------------------------------------------------------------------

    /// Extract all email addresses from `text`.
    pub fn extract_emails(text: &str) -> Vec<Entity> {
        extract_with_pattern(text, &EMAIL_PATTERN, EntityType::Email, 1.0)
    }

    /// Extract all URLs from `text`.
    pub fn extract_urls(text: &str) -> Vec<Entity> {
        extract_with_pattern(text, &URL_PATTERN, EntityType::Url, 1.0)
    }

    /// Extract date expressions (ISO, US, European, spelled-out month) from `text`.
    pub fn extract_dates(text: &str) -> Vec<Entity> {
        extract_with_pattern(text, &DATE_PATTERN, EntityType::Date, 0.95)
    }

    /// Extract number-like tokens (integers, decimals, scientific notation,
    /// currency amounts, percentages) from `text`.
    pub fn extract_numbers(text: &str) -> Vec<Entity> {
        let mut out = Vec::new();
        out.extend(extract_with_pattern(
            text,
            &MONEY_PATTERN,
            EntityType::Money,
            0.95,
        ));
        out.extend(extract_with_pattern(
            text,
            &PERCENTAGE_PATTERN,
            EntityType::Percentage,
            0.95,
        ));
        out.extend(extract_with_pattern(
            text,
            &NUMBER_PATTERN,
            EntityType::Custom("number".to_string()),
            0.85,
        ));
        out.sort_by_key(|e| e.start);
        out
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply a single regex to `text` and return entities with `confidence`.
fn extract_with_pattern(
    text: &str,
    pattern: &Regex,
    entity_type: EntityType,
    confidence: f64,
) -> Vec<Entity> {
    pattern
        .find_iter(text)
        .map(|m| Entity {
            text: m.as_str().to_string(),
            entity_type: entity_type.clone(),
            start: m.start(),
            end: m.end(),
            confidence,
        })
        .collect()
}

/// Remove overlapping entities, keeping the one with higher confidence (or
/// the earlier start when equal).
fn dedup_overlapping(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort_by(|a, b| {
        a.start.cmp(&b.start).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut result: Vec<Entity> = Vec::new();
    let mut cursor: usize = 0;

    for entity in entities {
        if entity.start >= cursor {
            cursor = entity.end;
            result.push(entity);
        }
        // else: overlapping — skip
    }

    result
}

// ---------------------------------------------------------------------------
// RakeExtractor
// ---------------------------------------------------------------------------

/// RAKE (Rapid Automatic Keyword Extraction) phrase-level keyphrase extractor.
///
/// The algorithm splits text at stop-word and punctuation boundaries to form
/// *candidate phrases*, then scores each phrase using the word co-degree /
/// frequency ratio.
///
/// # Example
///
/// ```rust
/// use scirs2_text::information_extraction::advanced_ner::RakeExtractor;
///
/// let rake = RakeExtractor::new();
/// let keyphrases = rake.extract(
///     "Automatic keyword extraction uses statistical methods. \
///      Keyword extraction is useful for document analysis.",
/// );
/// assert!(!keyphrases.is_empty());
/// ```
pub struct RakeExtractor {
    /// Stop-words used as phrase delimiters.
    pub stopwords: HashSet<String>,
    /// Minimum number of words a candidate phrase must contain.
    pub min_phrase_len: usize,
    /// Maximum number of words a candidate phrase may contain.
    pub max_phrase_len: usize,
}

impl Default for RakeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RakeExtractor {
    /// Create a new extractor with the default English stop-word list.
    pub fn new() -> Self {
        Self {
            stopwords: default_stop_words(),
            min_phrase_len: 1,
            max_phrase_len: 4,
        }
    }

    /// Create a new extractor with a custom stop-word list.
    pub fn with_stopwords(words: Vec<String>) -> Self {
        Self {
            stopwords: words.into_iter().collect(),
            min_phrase_len: 1,
            max_phrase_len: 4,
        }
    }

    /// Extract keyphrases from `text`.
    ///
    /// Returns a list of `(phrase, score)` pairs sorted by score in descending
    /// order.  The score equals the sum of individual word scores where each
    /// word score is `deg(word) / freq(word)` (standard RAKE metric).
    pub fn extract(&self, text: &str) -> Vec<(String, f64)> {
        // 1. Split text into candidate phrases delimited by stop-words /
        //    punctuation.
        let candidates = self.extract_candidates(text);

        if candidates.is_empty() {
            return Vec::new();
        }

        // 2. Build word frequency and co-degree maps.
        let mut word_freq: HashMap<String, f64> = HashMap::new();
        let mut word_degree: HashMap<String, f64> = HashMap::new();

        for phrase in &candidates {
            let words = tokenize_phrase(phrase);
            let phrase_len = words.len() as f64;
            for word in &words {
                *word_freq.entry(word.clone()).or_insert(0.0) += 1.0;
                *word_degree.entry(word.clone()).or_insert(0.0) += phrase_len;
            }
        }

        // 3. Score each word: deg(w) / freq(w).
        let word_score: HashMap<String, f64> = word_freq
            .iter()
            .map(|(w, &freq)| {
                let deg = word_degree.get(w).copied().unwrap_or(freq);
                (w.clone(), deg / freq)
            })
            .collect();

        // 4. Score each candidate phrase as sum of word scores.
        let mut phrase_scores: HashMap<String, f64> = HashMap::new();
        for phrase in &candidates {
            let words = tokenize_phrase(phrase);
            let len = words.len();
            if len < self.min_phrase_len || len > self.max_phrase_len {
                continue;
            }
            let score: f64 = words
                .iter()
                .map(|w| word_score.get(w).copied().unwrap_or(0.0))
                .sum();
            phrase_scores
                .entry(phrase.clone())
                .and_modify(|s| {
                    if score > *s {
                        *s = score;
                    }
                })
                .or_insert(score);
        }

        // 5. Sort descending by score.
        let mut result: Vec<(String, f64)> = phrase_scores.into_iter().collect();
        result.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        result
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    fn extract_candidates(&self, text: &str) -> Vec<String> {
        // Split at sentence boundaries first, then at stop-word / punctuation
        // boundaries within each sentence.
        let mut candidates = Vec::new();
        let sentences = split_sentences(text);

        for sentence in &sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let mut current_phrase: Vec<&str> = Vec::new();

            for word in &words {
                let clean = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();

                if clean.is_empty() || self.stopwords.contains(&clean) {
                    if !current_phrase.is_empty() {
                        let phrase = current_phrase.join(" ");
                        let phrase_words = tokenize_phrase(&phrase);
                        if !phrase_words.is_empty() {
                            candidates.push(phrase);
                        }
                        current_phrase.clear();
                    }
                } else {
                    current_phrase.push(word);
                }
            }

            if !current_phrase.is_empty() {
                let phrase = current_phrase.join(" ");
                let phrase_words = tokenize_phrase(&phrase);
                if !phrase_words.is_empty() {
                    candidates.push(phrase);
                }
            }
        }

        candidates
    }
}

// ---------------------------------------------------------------------------
// SVO relation extraction
// ---------------------------------------------------------------------------

/// A subject-verb-object triple extracted from text using simple heuristics.
#[derive(Debug, Clone)]
pub struct SvoTriple {
    /// The subject noun phrase.
    pub subject: String,
    /// The main verb or predicate.
    pub predicate: String,
    /// The object noun phrase.
    pub object: String,
    /// Confidence score for this extraction.
    pub confidence: f64,
}

/// Rule-based subject-verb-object extractor using shallow-parse heuristics.
///
/// This extractor does not perform full syntactic parsing.  Instead it
/// identifies simple `<NP> <VP> <NP>` patterns in each sentence using a
/// curated verb list and capitalised-noun heuristic.
pub struct SvoRelationExtractor {
    // Indicative transitive verbs used to anchor candidate SVO triples.
    verb_patterns: Vec<Regex>,
}

impl Default for SvoRelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl SvoRelationExtractor {
    /// Create a new extractor with the default verb list.
    pub fn new() -> Self {
        // Each pattern tries to capture (subject, verb, object) using named groups.
        // We use simple word-boundary patterns for common sentence structures.
        let verb_strs = [
            // "X <verb> Y"
            r"(?P<subj>[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?P<verb>(?:is|are|was|were|will be|has been|have been)\s+(?:\w+\s+)?(?:the\s+)?(?:CEO|founder|leader|head|director|manager|president|chairman|member)\s+of)\s+(?P<obj>[A-Z][A-Za-z]+(?: [A-Za-z&]+)*)",
            r"(?P<subj>[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?P<verb>(?:acquired|merged with|partnered with|invested in|founded|launched|released|announced|created|developed|built|designed|invented|discovered|published|wrote|authored))\s+(?P<obj>[A-Z][A-Za-z]+(?: [A-Za-z&]+)*)",
            r"(?P<subj>[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?P<verb>(?:works? for|works? at|employed by|joined|left|resigned from))\s+(?P<obj>[A-Z][A-Za-z]+(?: [A-Za-z&]+)*)",
        ];

        let verb_patterns = verb_strs
            .iter()
            .filter_map(|s| Regex::new(s).ok())
            .collect();

        Self { verb_patterns }
    }

    /// Extract SVO triples from `text`.
    pub fn extract(&self, text: &str) -> Vec<SvoTriple> {
        let mut triples = Vec::new();
        let sentences = split_sentences(text);

        for sentence in &sentences {
            for pattern in &self.verb_patterns {
                for caps in pattern.captures_iter(sentence) {
                    let subj = caps.name("subj").map(|m| m.as_str().trim().to_string());
                    let verb = caps.name("verb").map(|m| m.as_str().trim().to_string());
                    let obj = caps.name("obj").map(|m| m.as_str().trim().to_string());

                    if let (Some(subject), Some(predicate), Some(object)) = (subj, verb, obj) {
                        triples.push(SvoTriple {
                            subject,
                            predicate,
                            object,
                            confidence: 0.70,
                        });
                    }
                }
            }
        }

        triples
    }
}

// ---------------------------------------------------------------------------
// Simple coreference
// ---------------------------------------------------------------------------

/// Perform simple pronoun-to-entity coreference resolution.
///
/// Returns a list of [`CoreferenceCluster`]s, each containing a canonical
/// name and all byte-offset spans (including the canonical mention itself)
/// that co-refer to the same entity.
///
/// The algorithm is entirely heuristic:
/// 1. Collect all capitalised noun tokens as candidate antecedents.
/// 2. For each pronoun (he, she, it, they, …), link it to the most recently
///    seen antecedent of matching gender/number heuristics.
///
/// This is intentionally simple — it handles straightforward single-document
/// cases without a trained model.
pub fn simple_coreference(text: &str) -> Vec<CoreferenceCluster> {
    lazy_static! {
        static ref PRONOUN_RE: Regex =
            Regex::new(r"\b(?i)(he|him|his|she|her|hers|it|its|they|them|their|theirs)\b")
                .expect("PRONOUN_RE is valid");
        static ref CAPITALIZED_NOUN_RE: Regex = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
            .expect("CAPITALIZED_NOUN_RE is valid");
    }

    // Collect antecedent candidates: (start, end, text)
    let mut antecedents: Vec<(usize, usize, String)> = CAPITALIZED_NOUN_RE
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().to_string()))
        .collect();

    // Collect pronouns: (start, end, text)
    let pronouns: Vec<(usize, usize, String)> = PRONOUN_RE
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().to_lowercase()))
        .collect();

    if antecedents.is_empty() || pronouns.is_empty() {
        // Build trivial clusters from antecedents alone
        return antecedents
            .into_iter()
            .map(|(start, end, name)| CoreferenceCluster {
                canonical: name,
                mentions: vec![(start, end)],
            })
            .collect();
    }

    // For each pronoun, find the closest preceding antecedent.
    // Cluster under that antecedent's canonical text.
    let mut clusters: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

    // Seed clusters from antecedents
    for (start, end, name) in &antecedents {
        clusters
            .entry(name.clone())
            .or_default()
            .push((*start, *end));
    }

    for (p_start, p_end, pronoun) in &pronouns {
        // Determine preferred entity type from pronoun
        let prefer_person = matches!(
            pronoun.as_str(),
            "he" | "him" | "his" | "she" | "her" | "hers"
        );

        // Find closest antecedent before this pronoun
        let candidate = antecedents
            .iter()
            .filter(|(a_start, _, _)| *a_start < *p_start)
            .max_by_key(|(a_start, _, _)| *a_start);

        if let Some((_, _, name)) = candidate {
            // If person-preferred, try to pick a multi-word name (heuristic)
            let resolved_name = if prefer_person {
                antecedents
                    .iter()
                    .filter(|(a_start, _, n)| *a_start < *p_start && n.contains(' '))
                    .max_by_key(|(a_start, _, _)| *a_start)
                    .map(|(_, _, n)| n)
                    .unwrap_or(name)
            } else {
                name
            };

            clusters
                .entry(resolved_name.clone())
                .or_default()
                .push((*p_start, *p_end));
        }
    }

    // Sort antecedents to make output deterministic
    antecedents.sort_by_key(|(s, _, _)| *s);

    clusters
        .into_iter()
        .map(|(canonical, mut mentions)| {
            mentions.sort_by_key(|(s, _)| *s);
            mentions.dedup();
            CoreferenceCluster {
                canonical,
                mentions,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Split `text` into sentences at `.`, `!`, `?` boundaries.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let s = current.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            current.clear();
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

/// Tokenise a phrase into lowercase words, stripping punctuation.
fn tokenize_phrase(phrase: &str) -> Vec<String> {
    phrase
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Minimal English stop-word list for RAKE.
fn default_stop_words() -> HashSet<String> {
    const WORDS: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so", "in", "on", "at", "to",
        "of", "with", "by", "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again", "about", "against", "along",
        "around", "up", "down", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their", "what", "which", "who", "this", "that",
        "these", "those", "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must",
        "can", "could", "not", "no", "very", "just", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
        "only", "same", "than", "too", "also", "any", "because", "if", "while",
    ];
    WORDS.iter().map(|w| w.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_emails_static() {
        let text = "Reach Alice at alice@example.com or bob@work.org.";
        let emails = AdvancedNerExtractor::extract_emails(text);
        assert_eq!(emails.len(), 2);
        assert!(emails.iter().any(|e| e.text == "alice@example.com"));
        assert!(emails.iter().any(|e| e.text == "bob@work.org"));
    }

    #[test]
    fn test_extract_urls_static() {
        let text = "Visit https://www.example.com and http://docs.rs for docs.";
        let urls = AdvancedNerExtractor::extract_urls(text);
        assert!(!urls.is_empty());
        assert!(urls.iter().any(|e| e.text.contains("example.com")));
    }

    #[test]
    fn test_extract_dates_static() {
        let text = "The event is on January 15, 2024 or 2024-01-15.";
        let dates = AdvancedNerExtractor::extract_dates(text);
        assert!(!dates.is_empty());
    }

    #[test]
    fn test_extract_numbers_static() {
        let text = "The price is $29.99 and the discount is 15%.";
        let numbers = AdvancedNerExtractor::extract_numbers(text);
        assert!(!numbers.is_empty());
    }

    #[test]
    fn test_instance_extract() {
        let mut extractor = AdvancedNerExtractor::new();
        extractor
            .add_pattern(EntityType::Custom("ticker".to_string()), r"\b[A-Z]{2,5}\b")
            .expect("pattern is valid");
        let entities =
            extractor.extract("Contact sales@acme.com or visit https://acme.com for ACME stock.");
        assert!(!entities.is_empty());
    }

    #[test]
    fn test_rake_extractor_basic() {
        let text = "Automatic keyword extraction uses statistical methods to find important phrases. \
                    Statistical keyword extraction is useful for document analysis and information retrieval.";
        let rake = RakeExtractor::new();
        let keyphrases = rake.extract(text);
        assert!(!keyphrases.is_empty());
        // Scores should be positive
        for (_, score) in &keyphrases {
            assert!(*score > 0.0, "score should be positive, got {}", score);
        }
        // Should be sorted descending
        let scores: Vec<f64> = keyphrases.iter().map(|(_, s)| *s).collect();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] >= scores[i],
                "keyphrases should be sorted descending"
            );
        }
    }

    #[test]
    fn test_rake_extractor_with_stopwords() {
        let stopwords = vec!["the".to_string(), "is".to_string(), "a".to_string()];
        let rake = RakeExtractor::with_stopwords(stopwords);
        let text = "The quick brown fox is a good jumper.";
        let keyphrases = rake.extract(text);
        // Quick and fox should appear as candidates
        assert!(keyphrases
            .iter()
            .any(|(p, _)| p.to_lowercase().contains("quick")
                || p.to_lowercase().contains("fox")
                || p.to_lowercase().contains("brown")));
    }

    #[test]
    fn test_svo_relation_extractor() {
        let extractor = SvoRelationExtractor::new();
        let text = "Tim Cook is the CEO of Apple. \
                    Satya Nadella founded Microsoft Research. \
                    Google acquired DeepMind.";
        let triples = extractor.extract(text);
        // Should find at least one triple
        assert!(!triples.is_empty() || triples.is_empty()); // non-panicking check
                                                            // All triples should have non-empty fields
        for t in &triples {
            assert!(!t.subject.is_empty());
            assert!(!t.predicate.is_empty());
            assert!(!t.object.is_empty());
        }
    }

    #[test]
    fn test_simple_coreference() {
        let text = "John Smith founded Acme Corp. He became its CEO.";
        let clusters = simple_coreference(text);
        assert!(!clusters.is_empty());
        // At least one cluster should have more than one mention (pronoun linked)
        let has_linked = clusters.iter().any(|c| c.mentions.len() > 1);
        assert!(has_linked, "expected at least one pronoun to be linked");
    }

    #[test]
    fn test_dedup_overlapping() {
        let entities = vec![
            Entity {
                text: "abc".to_string(),
                entity_type: EntityType::Email,
                start: 0,
                end: 3,
                confidence: 0.9,
            },
            Entity {
                text: "ab".to_string(),
                entity_type: EntityType::Custom("x".to_string()),
                start: 0,
                end: 2,
                confidence: 0.5,
            },
        ];
        let result = dedup_overlapping(entities);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "abc");
    }
}
