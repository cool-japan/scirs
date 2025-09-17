//! Core information extractors for named entities, key phrases, and patterns

use super::entities::{Entity, EntityType};
use super::patterns::*;
use crate::error::Result;
use crate::tokenize::Tokenizer;
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Simple rule-based named entity recognizer
pub struct RuleBasedNER {
    person_names: HashSet<String>,
    organizations: HashSet<String>,
    locations: HashSet<String>,
    custom_patterns: HashMap<String, Regex>,
}

impl RuleBasedNER {
    /// Create a new rule-based NER
    pub fn new() -> Self {
        Self {
            person_names: HashSet::new(),
            organizations: HashSet::new(),
            locations: HashSet::new(),
            custom_patterns: HashMap::new(),
        }
    }

    /// Create a new rule-based NER with basic knowledge
    pub fn with_basic_knowledge() -> Self {
        let mut ner = Self::new();

        // Add common person names and titles
        ner.add_person_names(vec![
            "Tim Cook".to_string(),
            "Satya Nadella".to_string(),
            "Elon Musk".to_string(),
            "Jeff Bezos".to_string(),
            "Mark Zuckerberg".to_string(),
            "Bill Gates".to_string(),
            "Sundar Pichai".to_string(),
            "Andy Jassy".to_string(),
            "Susan Wojcicki".to_string(),
            "Reed Hastings".to_string(),
            "Jensen Huang".to_string(),
            "Lisa Su".to_string(),
        ]);

        // Add common organizations
        ner.add_organizations(vec![
            "Apple Inc.".to_string(),
            "Apple".to_string(),
            "Microsoft Corporation".to_string(),
            "Microsoft".to_string(),
            "Google".to_string(),
            "Alphabet Inc.".to_string(),
            "Amazon".to_string(),
            "Meta".to_string(),
            "Facebook".to_string(),
            "Tesla".to_string(),
            "Netflix".to_string(),
            "NVIDIA".to_string(),
            "AMD".to_string(),
            "Intel".to_string(),
            "IBM".to_string(),
            "Oracle".to_string(),
            "Salesforce".to_string(),
        ]);

        // Add common locations
        ner.add_locations(vec![
            "San Francisco".to_string(),
            "New York".to_string(),
            "London".to_string(),
            "Tokyo".to_string(),
            "Paris".to_string(),
            "Berlin".to_string(),
            "Sydney".to_string(),
            "Toronto".to_string(),
            "Singapore".to_string(),
            "Hong Kong".to_string(),
            "Los Angeles".to_string(),
            "Chicago".to_string(),
            "Boston".to_string(),
            "Seattle".to_string(),
            "Austin".to_string(),
            "Denver".to_string(),
            "California".to_string(),
            "New York".to_string(),
            "Texas".to_string(),
            "Washington".to_string(),
            "Florida".to_string(),
        ]);

        ner
    }

    /// Add person names to the recognizer
    pub fn add_person_names<I: IntoIterator<Item = String>>(&mut self, names: I) {
        self.person_names.extend(names);
    }

    /// Add organization names
    pub fn add_organizations<I: IntoIterator<Item = String>>(&mut self, orgs: I) {
        self.organizations.extend(orgs);
    }

    /// Add location names
    pub fn add_locations<I: IntoIterator<Item = String>>(&mut self, locations: I) {
        self.locations.extend(locations);
    }

    /// Add custom pattern for entity extraction
    pub fn add_custom_pattern(&mut self, name: String, pattern: Regex) {
        self.custom_patterns.insert(name, pattern);
    }

    /// Extract entities from text
    pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Extract regex-based entities
        entities.extend(self.extract_pattern_entities(text, &EMAIL_PATTERN, EntityType::Email)?);
        entities.extend(self.extract_pattern_entities(text, &URL_PATTERN, EntityType::Url)?);
        entities.extend(self.extract_pattern_entities(text, &PHONE_PATTERN, EntityType::Phone)?);
        entities.extend(self.extract_pattern_entities(text, &DATE_PATTERN, EntityType::Date)?);
        entities.extend(self.extract_pattern_entities(text, &TIME_PATTERN, EntityType::Time)?);
        entities.extend(self.extract_pattern_entities(text, &MONEY_PATTERN, EntityType::Money)?);
        entities.extend(self.extract_pattern_entities(
            text,
            &PERCENTAGE_PATTERN,
            EntityType::Percentage,
        )?);

        // Extract custom patterns
        for (name, pattern) in &self.custom_patterns {
            entities.extend(self.extract_pattern_entities(
                text,
                pattern,
                EntityType::Custom(name.clone()),
            )?);
        }

        // Extract dictionary-based entities
        entities.extend(self.extract_dictionary_entities(text)?);

        // Sort by start position
        entities.sort_by_key(|e| e.start);

        Ok(entities)
    }

    /// Extract entities using regex patterns
    fn extract_pattern_entities(
        &self,
        text: &str,
        pattern: &Regex,
        entity_type: EntityType,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for mat in pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: entity_type.clone(),
                start: mat.start(),
                end: mat.end(),
                confidence: 1.0, // High confidence for pattern matches
            });
        }

        Ok(entities)
    }

    /// Extract dictionary-based entities
    fn extract_dictionary_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();

        // Check for multi-word entities first (e.g., "Apple Inc.", "Tim Cook")
        for entity_name in &self.person_names {
            let entity_lower = entity_name.to_lowercase();
            if let Some(start) = text_lower.find(&entity_lower) {
                // Verify word boundaries
                let at_word_start =
                    start == 0 || !text.chars().nth(start - 1).unwrap_or(' ').is_alphanumeric();
                let at_word_end = start + entity_name.len() >= text.len()
                    || !text
                        .chars()
                        .nth(start + entity_name.len())
                        .unwrap_or(' ')
                        .is_alphanumeric();

                if at_word_start && at_word_end {
                    entities.push(Entity {
                        text: text[start..start + entity_name.len()].to_string(),
                        entity_type: EntityType::Person,
                        start,
                        end: start + entity_name.len(),
                        confidence: 0.9,
                    });
                }
            }
        }

        for entity_name in &self.organizations {
            let entity_lower = entity_name.to_lowercase();
            if let Some(start) = text_lower.find(&entity_lower) {
                // Verify word boundaries
                let at_word_start =
                    start == 0 || !text.chars().nth(start - 1).unwrap_or(' ').is_alphanumeric();
                let at_word_end = start + entity_name.len() >= text.len()
                    || !text
                        .chars()
                        .nth(start + entity_name.len())
                        .unwrap_or(' ')
                        .is_alphanumeric();

                if at_word_start && at_word_end {
                    entities.push(Entity {
                        text: text[start..start + entity_name.len()].to_string(),
                        entity_type: EntityType::Organization,
                        start,
                        end: start + entity_name.len(),
                        confidence: 0.9,
                    });
                }
            }
        }

        for entity_name in &self.locations {
            let entity_lower = entity_name.to_lowercase();
            if let Some(start) = text_lower.find(&entity_lower) {
                // Verify word boundaries
                let at_word_start =
                    start == 0 || !text.chars().nth(start - 1).unwrap_or(' ').is_alphanumeric();
                let at_word_end = start + entity_name.len() >= text.len()
                    || !text
                        .chars()
                        .nth(start + entity_name.len())
                        .unwrap_or(' ')
                        .is_alphanumeric();

                if at_word_start && at_word_end {
                    entities.push(Entity {
                        text: text[start..start + entity_name.len()].to_string(),
                        entity_type: EntityType::Location,
                        start,
                        end: start + entity_name.len(),
                        confidence: 0.9,
                    });
                }
            }
        }

        Ok(entities)
    }
}

impl Default for RuleBasedNER {
    fn default() -> Self {
        Self::new()
    }
}

/// Key phrase extractor using statistical methods
pub struct KeyPhraseExtractor {
    min_phrase_length: usize,
    max_phrase_length: usize,
    min_frequency: usize,
}

impl KeyPhraseExtractor {
    /// Create a new key phrase extractor
    pub fn new() -> Self {
        Self {
            min_phrase_length: 1,
            max_phrase_length: 3,
            min_frequency: 2,
        }
    }

    /// Set minimum phrase length
    pub fn with_min_length(mut self, length: usize) -> Self {
        self.min_phrase_length = length;
        self
    }

    /// Set maximum phrase length
    pub fn with_max_length(mut self, length: usize) -> Self {
        self.max_phrase_length = length;
        self
    }

    /// Set minimum frequency threshold
    pub fn with_min_frequency(mut self, freq: usize) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Extract key phrases from text
    pub fn extract(&self, text: &str, tokenizer: &dyn Tokenizer) -> Result<Vec<(String, f64)>> {
        let tokens = tokenizer.tokenize(text)?;
        let mut phrase_counts: HashMap<String, usize> = HashMap::new();

        // Generate n-grams
        for n in self.min_phrase_length..=self.max_phrase_length {
            if tokens.len() >= n {
                for i in 0..=tokens.len() - n {
                    let phrase = tokens[i..i + n].join(" ");
                    *phrase_counts.entry(phrase).or_insert(0) += 1;
                }
            }
        }

        // Filter by frequency and calculate scores
        let mut phrases: Vec<(String, f64)> = phrase_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_frequency)
            .map(|(phrase, count)| {
                // Simple scoring: frequency * length
                let score = count as f64 * (phrase.split_whitespace().count() as f64).sqrt();
                (phrase, score)
            })
            .collect();

        // Sort by score descending
        phrases.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(phrases)
    }
}

impl Default for KeyPhraseExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern-based information extractor
pub struct PatternExtractor {
    patterns: Vec<(String, Regex)>,
}

impl PatternExtractor {
    /// Create a new pattern extractor
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a named pattern
    pub fn add_pattern(&mut self, name: String, pattern: Regex) {
        self.patterns.push((name, pattern));
    }

    /// Extract information matching patterns
    pub fn extract(&self, text: &str) -> Result<HashMap<String, Vec<String>>> {
        let mut results: HashMap<String, Vec<String>> = HashMap::new();

        for (name, pattern) in &self.patterns {
            let mut matches = Vec::new();

            for mat in pattern.find_iter(text) {
                matches.push(mat.as_str().to_string());
            }

            if !matches.is_empty() {
                results.insert(name.clone(), matches);
            }
        }

        Ok(results)
    }

    /// Extract with capture groups
    pub fn extract_with_groups(
        &self,
        text: &str,
    ) -> Result<HashMap<String, Vec<HashMap<String, String>>>> {
        let mut results: HashMap<String, Vec<HashMap<String, String>>> = HashMap::new();

        for (name, pattern) in &self.patterns {
            let mut matches = Vec::new();

            for caps in pattern.captures_iter(text) {
                let mut groups = HashMap::new();

                // Add full match
                if let Some(full_match) = caps.get(0) {
                    groups.insert("full".to_string(), full_match.as_str().to_string());
                }

                // Add numbered groups
                for i in 1..caps.len() {
                    if let Some(group) = caps.get(i) {
                        groups.insert(format!("group{i}"), group.as_str().to_string());
                    }
                }

                // Add named groups if any
                for name in pattern.capture_names().flatten() {
                    if let Some(group) = caps.name(name) {
                        groups.insert(name.to_string(), group.as_str().to_string());
                    }
                }

                matches.push(groups);
            }

            if !matches.is_empty() {
                results.insert(name.clone(), matches);
            }
        }

        Ok(results)
    }
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self::new()
    }
}
