//! Coreference resolution for pronoun and entity linking

use super::entities::{Entity, EntityType};
use crate::error::Result;
use regex::Regex;

/// Type of coreference mention
#[derive(Debug, Clone, PartialEq)]
pub enum MentionType {
    /// Named entity mention
    Entity,
    /// Pronoun mention
    Pronoun,
    /// Descriptive mention
    Description,
}

/// Individual mention in a coreference chain
#[derive(Debug, Clone)]
pub struct CoreferenceMention {
    /// Text content of the mention
    pub text: String,
    /// Start position in the document
    pub start: usize,
    /// End position in the document
    pub end: usize,
    /// Type of mention
    pub mention_type: MentionType,
}

/// Coreference chain representing linked mentions
#[derive(Debug, Clone)]
pub struct CoreferenceChain {
    /// List of mentions in this chain
    pub mentions: Vec<CoreferenceMention>,
    /// Confidence score for the coreference chain
    pub confidence: f64,
}

/// Coreference resolver for basic pronoun resolution
pub struct CoreferenceResolver {
    pronoun_patterns: Vec<Regex>,
}

impl Default for CoreferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreferenceResolver {
    /// Create new coreference resolver
    pub fn new() -> Self {
        let pronoun_patterns = vec![
            Regex::new(r"\b(?i)(?:he|she|it|they|him|her|them|his|hers|its|their)\b").unwrap(),
            Regex::new(r"\b(?i)(?:this|that|these|those)\b").unwrap(),
            Regex::new(r"\b(?i)(?:the (?:company|organization|person|individual|entity))\b")
                .unwrap(),
        ];

        Self { pronoun_patterns }
    }

    /// Resolve coreferences in text with entities
    pub fn resolve(&self, text: &str, entities: &[Entity]) -> Result<Vec<CoreferenceChain>> {
        let mut chains = Vec::new();
        let sentences = self.split_into_sentences(text);

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            // Find entities in this sentence
            let _sentence_entities: Vec<&Entity> = entities
                .iter()
                .filter(|e| {
                    text[e.start..e.end].trim() == sentence.trim() || sentence.contains(&e.text)
                })
                .collect();

            // Find pronouns in this sentence
            for pattern in &self.pronoun_patterns {
                for mat in pattern.find_iter(sentence) {
                    // Try to resolve to nearest appropriate entity in previous sentences
                    if let Some(antecedent) = self.find_antecedent(
                        &mat.as_str().to_lowercase(),
                        &sentences[..sent_idx],
                        entities,
                    ) {
                        chains.push(CoreferenceChain {
                            mentions: vec![
                                CoreferenceMention {
                                    text: antecedent.text.clone(),
                                    start: antecedent.start,
                                    end: antecedent.end,
                                    mention_type: MentionType::Entity,
                                },
                                CoreferenceMention {
                                    text: mat.as_str().to_string(),
                                    start: mat.start(),
                                    end: mat.end(),
                                    mention_type: MentionType::Pronoun,
                                },
                            ],
                            confidence: 0.6,
                        });
                    }
                }
            }
        }

        Ok(chains)
    }

    /// Split text into sentences (simple implementation)
    pub fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Find antecedent for a pronoun
    fn find_antecedent<'a>(
        &self,
        pronoun: &str,
        previous_sentences: &[String],
        entities: &'a [Entity],
    ) -> Option<&'a Entity> {
        // Simple heuristic: find the closest person/organization entity
        let target_type = match pronoun {
            "he" | "him" | "his" => Some(EntityType::Person),
            "she" | "her" | "hers" => Some(EntityType::Person),
            "it" | "its" => Some(EntityType::Organization),
            "they" | "them" | "their" => None, // Could be either
            _ => None,
        };

        // Look for entities in reverse order (most recent first)
        for sentence in previous_sentences.iter().rev() {
            for entity in entities.iter().rev() {
                if sentence.contains(&entity.text) {
                    if let Some(expected_type) = &target_type {
                        if entity.entity_type == *expected_type {
                            return Some(entity);
                        }
                    } else {
                        // For ambiguous pronouns, return any person or organization
                        if matches!(
                            entity.entity_type,
                            EntityType::Person | EntityType::Organization
                        ) {
                            return Some(entity);
                        }
                    }
                }
            }
        }

        None
    }
}
