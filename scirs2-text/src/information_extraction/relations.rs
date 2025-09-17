//! Relation extraction and relationship modeling

use super::entities::Entity;
use crate::error::Result;
use regex::Regex;

/// Extracted relation between entities
#[derive(Debug, Clone)]
pub struct Relation {
    /// Type of relation (e.g., "works_for", "located_in")
    pub relation_type: String,
    /// Subject entity in the relation
    pub subject: Entity,
    /// Object entity in the relation
    pub object: Entity,
    /// Context text where the relation was found
    pub context: String,
    /// Confidence score for the relation extraction (0.0 to 1.0)
    pub confidence: f64,
}

/// Relation extractor for finding relationships between entities
pub struct RelationExtractor {
    relation_patterns: Vec<(String, Regex)>,
}

impl Default for RelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationExtractor {
    /// Create a new relation extractor
    pub fn new() -> Self {
        Self {
            relation_patterns: Vec::new(),
        }
    }

    /// Add a relation pattern
    pub fn add_relation(&mut self, relationtype: String, pattern: Regex) {
        self.relation_patterns.push((relationtype, pattern));
    }

    /// Extract relations from text
    pub fn extract_relations(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relation>> {
        let mut relations = Vec::new();

        for (relation_type, pattern) in &self.relation_patterns {
            for caps in pattern.captures_iter(text) {
                if let Some(full_match) = caps.get(0) {
                    // Find entities that might be involved in this relation
                    let match_start = full_match.start();
                    let match_end = full_match.end();

                    let involved_entities: Vec<&Entity> = entities
                        .iter()
                        .filter(|e| e.start >= match_start && e.end <= match_end)
                        .collect();

                    if involved_entities.len() >= 2 {
                        relations.push(Relation {
                            relation_type: relation_type.clone(),
                            subject: involved_entities[0].clone(),
                            object: involved_entities[1].clone(),
                            context: full_match.as_str().to_string(),
                            confidence: 0.7,
                        });
                    }
                }
            }
        }

        Ok(relations)
    }
}

/// Extracted event from text
#[derive(Debug, Clone)]
pub struct Event {
    /// Type or category of the event
    pub event_type: String,
    /// Entities participating in the event
    pub participants: Vec<Entity>,
    /// Location where the event occurred
    pub location: Option<Entity>,
    /// Time when the event occurred
    pub time: Option<Entity>,
    /// Description of the event
    pub description: String,
    /// Confidence score for the event extraction
    pub confidence: f64,
}
