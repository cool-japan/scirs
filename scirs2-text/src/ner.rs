//! Advanced Named Entity Recognition (`ner.rs`)
//!
//! Provides rule-based and pattern-matching NER with a rich public API:
//!
//! - [`EntityType`] -- discriminated union for entity categories
//! - [`Entity`] -- extracted entity with byte offsets and confidence score
//! - [`NerConfig`] -- configuration (case sensitivity, custom patterns, gazetteer)
//! - [`NerExtractor`] -- main extractor struct
//!
//! All detection is purely rule-based (no trained models) and 100% Pure Rust.

use crate::error::{Result, TextError};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Discriminated union of recognised entity kinds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// A person's name.
    Person,
    /// An organisation name.
    Organization,
    /// A geographic location.
    Location,
    /// A date expression.
    Date,
    /// A time expression.
    Time,
    /// A cardinal number (integer, float, scientific notation).
    Number,
    /// An email address.
    Email,
    /// A URL or URI.
    Url,
    /// A telephone number.
    PhoneNumber,
    /// A monetary amount.
    Currency,
    /// A percentage value.
    Percentage,
    /// User-defined entity type, identified by label.
    Custom(String),
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Person => write!(f, "PERSON"),
            Self::Organization => write!(f, "ORGANIZATION"),
            Self::Location => write!(f, "LOCATION"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Number => write!(f, "NUMBER"),
            Self::Email => write!(f, "EMAIL"),
            Self::Url => write!(f, "URL"),
            Self::PhoneNumber => write!(f, "PHONE_NUMBER"),
            Self::Currency => write!(f, "CURRENCY"),
            Self::Percentage => write!(f, "PERCENTAGE"),
            Self::Custom(label) => write!(f, "CUSTOM({})", label),
        }
    }
}

/// An entity extracted from text.
#[derive(Debug, Clone)]
pub struct Entity {
    /// The matched text slice (owned copy).
    pub text: String,
    /// The entity category.
    pub entity_type: EntityType,
    /// Byte offset of the entity start in the original text.
    pub start: usize,
    /// Byte offset of the entity end (exclusive) in the original text.
    pub end: usize,
    /// Confidence score: 1.0 for deterministic patterns, lower for heuristics.
    pub score: f32,
}

/// Configuration for [`NerExtractor`].
#[derive(Default)]
pub struct NerConfig {
    /// Whether pattern matching is case-sensitive.
    pub case_sensitive: bool,
    /// Custom patterns: `(regex_string, entity_type)`.
    pub custom_patterns: Vec<(String, EntityType)>,
    /// Gazetteer: maps exact tokens (words) to entity types.
    pub gazetteer: HashMap<String, EntityType>,
}

impl NerConfig {
    /// Convenience constructor with all defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Built-in patterns (compiled once, stored in lazy_static)
// ---------------------------------------------------------------------------

lazy_static! {
    // Email (RFC-5321 simplified)
    static ref RE_EMAIL: Regex = Regex::new(
        r"(?i)\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ).expect("email regex");

    // URL
    static ref RE_URL: Regex = Regex::new(
        r"(?i)https?://[^\s<>\x22\{\}\|\\\^\[\]`]+"
    ).expect("url regex");

    // Phone: (NNN) NNN-NNNN, NNN-NNN-NNNN, +1-NNN-NNN-NNNN, etc.
    static ref RE_PHONE: Regex = Regex::new(
        r"(?:(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})\b"
    ).expect("phone regex");

    // Date: ISO (YYYY-MM-DD), US (MM/DD/YYYY), written
    static ref RE_DATE: Regex = Regex::new(
        r"(?i)(?:\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b)"
    ).expect("date regex");

    // Time: HH:MM or HH:MM:SS with optional AM/PM
    static ref RE_TIME: Regex = Regex::new(
        r"(?i)\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b"
    ).expect("time regex");

    // Currency: $NNN, EUR NNN, etc.
    static ref RE_CURRENCY: Regex = Regex::new(
        r"(?:[\$\x{20AC}\x{00A3}\x{00A5}])\s*\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY)\b"
    ).expect("currency regex");

    // Percentage
    static ref RE_PERCENTAGE: Regex = Regex::new(
        r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent\b)"
    ).expect("percentage regex");

    // Number: integers, floats, scientific notation, ordinals
    static ref RE_NUMBER: Regex = Regex::new(
        r"\b(?:\d+(?:\.\d+)?[eE][+\-]?\d+|\d+(?:\.\d+)?|\d+(?:st|nd|rd|th))\b"
    ).expect("number regex");

    // Person-title prefixes
    static ref RE_PERSON_PREFIX: Regex = Regex::new(
        r"\b(?:Dr|Prof|Mr|Mrs|Ms|Miss|Rev|Gen|Col|Capt|Lt|Sgt|Cpl|Pte|Sir|Lord|Lady|dr|prof|mr|mrs|ms|miss|rev|gen|col|capt|lt|sgt|cpl|pte|sir|lord|lady)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    ).expect("person prefix regex");

    // Organization suffixes
    static ref RE_ORG_SUFFIX: Regex = Regex::new(
        r"\b([A-Z][A-Za-z&\s]+(?:Inc|LLC|Ltd|Corp|Co|GmbH|AG|SA|PLC|LLP|LP|NV|BV|AB|AS|Pty)\.?)\b"
    ).expect("org suffix regex");
}

// ---------------------------------------------------------------------------
// Known location/org gazetteers (minimal but useful defaults)
// ---------------------------------------------------------------------------

fn default_location_gazetteer() -> &'static [&'static str] {
    &[
        "Africa",
        "America",
        "Antarctica",
        "Arctic",
        "Asia",
        "Australia",
        "Europe",
        "China",
        "France",
        "Germany",
        "India",
        "Italy",
        "Japan",
        "Russia",
        "Spain",
        "United States",
        "United Kingdom",
        "Canada",
        "Brazil",
        "Mexico",
        "Argentina",
        "South Korea",
        "North Korea",
        "Saudi Arabia",
        "South Africa",
        "New York",
        "London",
        "Paris",
        "Tokyo",
        "Beijing",
        "Shanghai",
        "Sydney",
        "Moscow",
        "Berlin",
        "Madrid",
        "Rome",
        "Seoul",
        "Mumbai",
        "Dubai",
        "Los Angeles",
        "Chicago",
        "San Francisco",
        "Houston",
        "Phoenix",
        "California",
        "Texas",
        "Florida",
        "Illinois",
        "Pennsylvania",
        "Ohio",
        "Georgia",
        "Michigan",
        "New Jersey",
        "Virginia",
        "Washington",
        "Arizona",
        "Massachusetts",
        "Tennessee",
        "Indiana",
    ]
}

fn default_org_gazetteer() -> &'static [&'static str] {
    &[
        "Google",
        "Apple",
        "Microsoft",
        "Amazon",
        "Meta",
        "Netflix",
        "Tesla",
        "IBM",
        "Intel",
        "Oracle",
        "SAP",
        "Adobe",
        "Salesforce",
        "Twitter",
        "LinkedIn",
        "Facebook",
        "WhatsApp",
        "Instagram",
        "YouTube",
        "TikTok",
        "Snapchat",
        "Uber",
        "Lyft",
        "Airbnb",
        "Spotify",
        "Slack",
        "Zoom",
        "Dropbox",
        "NASA",
        "CIA",
        "FBI",
        "NSA",
        "UN",
        "NATO",
        "WHO",
        "IMF",
        "WTO",
        "Harvard",
        "MIT",
        "Stanford",
        "Oxford",
        "Cambridge",
    ]
}

// ---------------------------------------------------------------------------
// NerExtractor
// ---------------------------------------------------------------------------

/// Rule-based and pattern-matching NER extractor.
///
/// # Example
///
/// ```rust
/// use scirs2_text::ner::{NerExtractor, NerConfig, EntityType};
///
/// let config = NerConfig::default();
/// let extractor = NerExtractor::new(config);
/// let entities = extractor.extract("Contact us at support@example.com").unwrap();
/// assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
/// ```
pub struct NerExtractor {
    config: NerConfig,
    /// Compiled custom-pattern regexes and their associated entity types.
    compiled_custom: Vec<(Regex, EntityType)>,
    /// Effective gazetteer (merged with built-in defaults).
    effective_gazetteer: HashMap<String, EntityType>,
}

impl NerExtractor {
    /// Create a new extractor from the given configuration.
    pub fn new(config: NerConfig) -> Self {
        let compiled_custom: Vec<(Regex, EntityType)> = config
            .custom_patterns
            .iter()
            .filter_map(|(pattern, etype)| Regex::new(pattern).ok().map(|re| (re, etype.clone())))
            .collect();

        let mut effective_gazetteer: HashMap<String, EntityType> = HashMap::new();

        for loc in default_location_gazetteer() {
            let key = if config.case_sensitive {
                loc.to_string()
            } else {
                loc.to_lowercase()
            };
            effective_gazetteer.insert(key, EntityType::Location);
        }
        for org in default_org_gazetteer() {
            let key = if config.case_sensitive {
                org.to_string()
            } else {
                org.to_lowercase()
            };
            effective_gazetteer.insert(key, EntityType::Organization);
        }
        for (word, etype) in &config.gazetteer {
            let key = if config.case_sensitive {
                word.clone()
            } else {
                word.to_lowercase()
            };
            effective_gazetteer.insert(key, etype.clone());
        }

        Self {
            config,
            compiled_custom,
            effective_gazetteer,
        }
    }

    /// Attempt to create a new extractor, returning an error if any custom
    /// pattern fails to compile.
    pub fn try_new(config: NerConfig) -> Result<Self> {
        for (pattern, _) in &config.custom_patterns {
            Regex::new(pattern).map_err(|e| {
                TextError::InvalidInput(format!(
                    "Custom NER pattern '{}' is invalid: {}",
                    pattern, e
                ))
            })?;
        }
        Ok(Self::new(config))
    }

    /// Add a single entry to the gazetteer.
    pub fn add_gazetteer_entry(&mut self, word: &str, entity_type: EntityType) {
        let key = if self.config.case_sensitive {
            word.to_string()
        } else {
            word.to_lowercase()
        };
        self.effective_gazetteer.insert(key, entity_type.clone());
        self.config.gazetteer.insert(word.to_string(), entity_type);
    }

    /// Extract all entities from `text`.
    ///
    /// Entities are returned sorted by byte `start` offset. Overlapping
    /// entities are resolved by keeping the longer match.
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut candidates: Vec<Entity> = Vec::new();

        // 1. Built-in regex patterns (score = 1.0)
        self.apply_pattern(text, &RE_EMAIL, EntityType::Email, 1.0, &mut candidates);
        self.apply_pattern(text, &RE_URL, EntityType::Url, 1.0, &mut candidates);
        self.apply_pattern(
            text,
            &RE_PHONE,
            EntityType::PhoneNumber,
            1.0,
            &mut candidates,
        );
        self.apply_pattern(text, &RE_DATE, EntityType::Date, 1.0, &mut candidates);
        self.apply_pattern(text, &RE_TIME, EntityType::Time, 1.0, &mut candidates);
        self.apply_pattern(
            text,
            &RE_CURRENCY,
            EntityType::Currency,
            1.0,
            &mut candidates,
        );
        self.apply_pattern(
            text,
            &RE_PERCENTAGE,
            EntityType::Percentage,
            1.0,
            &mut candidates,
        );
        self.apply_pattern(text, &RE_NUMBER, EntityType::Number, 1.0, &mut candidates);

        // 2. Heuristic person detection via title prefix
        self.extract_persons(text, &mut candidates);

        // 3. Heuristic organisation detection via suffix
        self.extract_organizations(text, &mut candidates);

        // 4. Gazetteer lookup
        self.extract_gazetteer(text, &mut candidates);

        // 5. Custom user-supplied patterns (score = 1.0)
        for (re, etype) in &self.compiled_custom {
            self.apply_pattern(text, re, etype.clone(), 1.0, &mut candidates);
        }

        // 6. Resolve overlaps and sort
        let resolved = resolve_overlaps(candidates);
        Ok(resolved)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn apply_pattern(
        &self,
        text: &str,
        re: &Regex,
        etype: EntityType,
        score: f32,
        out: &mut Vec<Entity>,
    ) {
        for m in re.find_iter(text) {
            out.push(Entity {
                text: m.as_str().to_string(),
                entity_type: etype.clone(),
                start: m.start(),
                end: m.end(),
                score,
            });
        }
    }

    fn extract_persons(&self, text: &str, out: &mut Vec<Entity>) {
        for cap in RE_PERSON_PREFIX.captures_iter(text) {
            if let Some(full) = cap.get(0) {
                out.push(Entity {
                    text: full.as_str().to_string(),
                    entity_type: EntityType::Person,
                    start: full.start(),
                    end: full.end(),
                    score: 0.9,
                });
            }
        }
    }

    fn extract_organizations(&self, text: &str, out: &mut Vec<Entity>) {
        for cap in RE_ORG_SUFFIX.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                out.push(Entity {
                    text: m.as_str().to_string(),
                    entity_type: EntityType::Organization,
                    start: m.start(),
                    end: m.end(),
                    score: 0.85,
                });
            }
        }
    }

    fn extract_gazetteer(&self, text: &str, out: &mut Vec<Entity>) {
        let lookup_text = if self.config.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        let mut entries: Vec<(&String, &EntityType)> = self.effective_gazetteer.iter().collect();
        entries.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));

        for (entry, etype) in entries {
            let escaped = regex::escape(entry);
            let pattern = format!(r"(?i)\b{}\b", escaped);
            if let Ok(re) = Regex::new(&pattern) {
                for m in re.find_iter(&lookup_text) {
                    let original_text = &text[m.start()..m.end()];
                    out.push(Entity {
                        text: original_text.to_string(),
                        entity_type: etype.clone(),
                        start: m.start(),
                        end: m.end(),
                        score: 1.0,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Overlap resolution
// ---------------------------------------------------------------------------

fn resolve_overlaps(mut entities: Vec<Entity>) -> Vec<Entity> {
    if entities.is_empty() {
        return entities;
    }

    entities.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
            .then_with(|| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut result: Vec<Entity> = Vec::new();
    let mut last_end: usize = 0;

    for entity in entities {
        if entity.start >= last_end {
            last_end = entity.end;
            result.push(entity);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_extractor() -> NerExtractor {
        NerExtractor::new(NerConfig::default())
    }

    #[test]
    fn test_email_extraction() {
        let extractor = default_extractor();
        let entities = extractor
            .extract("Please contact support@example.com for help.")
            .expect("should succeed");
        let emails: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Email)
            .collect();
        assert!(!emails.is_empty(), "Should detect at least one email");
        assert_eq!(emails[0].text, "support@example.com");
    }

    #[test]
    fn test_multiple_emails() {
        let extractor = default_extractor();
        let text = "Send to alice@foo.com and bob@bar.org please.";
        let entities = extractor.extract(text).expect("ok");
        let emails: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Email)
            .collect();
        assert_eq!(emails.len(), 2);
    }

    #[test]
    fn test_url_extraction() {
        let extractor = default_extractor();
        let entities = extractor
            .extract("Visit https://www.rust-lang.org for docs.")
            .expect("ok");
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Url));
    }

    #[test]
    fn test_phone_extraction() {
        let extractor = default_extractor();
        let entities = extractor.extract("Call us at (800) 555-1234.").expect("ok");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::PhoneNumber),
            "Should detect phone number"
        );
    }

    #[test]
    fn test_iso_date() {
        let extractor = default_extractor();
        let entities = extractor.extract("Event on 2025-06-15.").expect("ok");
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_written_date() {
        let extractor = default_extractor();
        let entities = extractor
            .extract("He was born on March 5, 1990.")
            .expect("ok");
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_currency_dollar() {
        let extractor = default_extractor();
        let entities = extractor.extract("The price is $42.99.").expect("ok");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::Currency),
            "Should detect currency"
        );
    }

    #[test]
    fn test_percentage() {
        let extractor = default_extractor();
        let entities = extractor.extract("Growth rate is 15.3%.").expect("ok");
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::Percentage));
    }

    #[test]
    fn test_integer_number() {
        let extractor = default_extractor();
        let entities = extractor.extract("There are 42 items.").expect("ok");
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Number));
    }

    #[test]
    fn test_person_with_title() {
        let extractor = default_extractor();
        let entities = extractor
            .extract("We met Dr. Jane Smith yesterday.")
            .expect("ok");
        assert!(
            entities.iter().any(|e| e.entity_type == EntityType::Person),
            "Should detect person with title"
        );
    }

    #[test]
    fn test_org_with_suffix() {
        let extractor = default_extractor();
        let entities = extractor.extract("She works at Acme Corp.").expect("ok");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::Organization),
            "Should detect organization"
        );
    }

    #[test]
    fn test_gazetteer_location() {
        let extractor = default_extractor();
        let entities = extractor
            .extract("The summit was held in Paris.")
            .expect("ok");
        assert!(
            entities.iter().any(|e| {
                e.entity_type == EntityType::Location && e.text.to_lowercase() == "paris"
            }),
            "Should detect Paris as location via gazetteer"
        );
    }

    #[test]
    fn test_gazetteer_organization() {
        let extractor = default_extractor();
        let entities = extractor
            .extract("Google announced new products.")
            .expect("ok");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::Organization),
            "Should detect Google as organization"
        );
    }

    #[test]
    fn test_custom_pattern() {
        let config = NerConfig {
            custom_patterns: vec![(
                r"\b[A-Z]{3,5}-\d{4}\b".to_string(),
                EntityType::Custom("TICKET_ID".to_string()),
            )],
            ..NerConfig::default()
        };
        let extractor = NerExtractor::new(config);
        let entities = extractor
            .extract("Issue JIRA-1234 is resolved.")
            .expect("ok");
        assert!(entities.iter().any(|e| matches!(
            &e.entity_type,
            EntityType::Custom(label) if label == "TICKET_ID"
        )));
    }

    #[test]
    fn test_invalid_custom_pattern_returns_error() {
        let config = NerConfig {
            custom_patterns: vec![(r"[invalid".to_string(), EntityType::Custom("X".to_string()))],
            ..NerConfig::default()
        };
        assert!(NerExtractor::try_new(config).is_err());
    }

    #[test]
    fn test_add_gazetteer_entry() {
        let mut extractor = NerExtractor::new(NerConfig::default());
        extractor.add_gazetteer_entry("Rustacean", EntityType::Custom("COMMUNITY".to_string()));
        let entities = extractor
            .extract("The Rustacean organized an event.")
            .expect("ok");
        assert!(entities.iter().any(|e| matches!(
            &e.entity_type,
            EntityType::Custom(label) if label == "COMMUNITY"
        )));
    }

    #[test]
    fn test_entities_non_overlapping() {
        let extractor = default_extractor();
        let text = "Email info@test.com, call (555) 123-4567.";
        let entities = extractor.extract(text).expect("ok");
        for i in 1..entities.len() {
            assert!(
                entities[i].start >= entities[i - 1].end,
                "Entities should not overlap"
            );
        }
    }

    #[test]
    fn test_empty_text() {
        let extractor = default_extractor();
        let entities = extractor.extract("").expect("ok");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_email_score_is_one() {
        let extractor = default_extractor();
        let entities = extractor.extract("user@domain.com").expect("ok");
        let emails: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Email)
            .collect();
        assert!(!emails.is_empty());
        assert!((emails[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(EntityType::Email.to_string(), "EMAIL");
        assert_eq!(EntityType::Person.to_string(), "PERSON");
        assert_eq!(
            EntityType::Custom("FOO".to_string()).to_string(),
            "CUSTOM(FOO)"
        );
    }

    #[test]
    fn test_mixed_entities() {
        let text = "On 2025-01-15 at 10:30, Dr. John Smith emailed john@example.com.";
        let entities = extractor_all().extract(text).expect("ok");
        let types: std::collections::HashSet<String> =
            entities.iter().map(|e| e.entity_type.to_string()).collect();
        assert!(types.contains("DATE"), "missing DATE in {:?}", types);
        assert!(types.contains("EMAIL"), "missing EMAIL in {:?}", types);
    }

    fn extractor_all() -> NerExtractor {
        NerExtractor::new(NerConfig::default())
    }
}
