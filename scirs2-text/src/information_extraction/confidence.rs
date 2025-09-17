//! Confidence scoring for extracted entities and information

use super::entities::{Entity, EntityType};
use std::collections::HashMap;

/// Advanced confidence scorer for entities
pub struct ConfidenceScorer {
    featureweights: HashMap<String, f64>,
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceScorer {
    /// Create new confidence scorer
    pub fn new() -> Self {
        let mut feature_weights = HashMap::new();
        feature_weights.insert("pattern_match".to_string(), 0.3);
        feature_weights.insert("dictionary_match".to_string(), 0.2);
        feature_weights.insert("context_score".to_string(), 0.3);
        feature_weights.insert("length_score".to_string(), 0.1);
        feature_weights.insert("position_score".to_string(), 0.1);

        Self {
            featureweights: feature_weights,
        }
    }

    /// Calculate confidence score for an entity
    pub fn score_entity(&self, entity: &Entity, text: &str, contextwindow: usize) -> f64 {
        let mut features = HashMap::new();

        // Pattern match confidence (based on entity type)
        let pattern_score = match entity.entity_type {
            EntityType::Email | EntityType::Url | EntityType::Phone => 1.0,
            EntityType::Date | EntityType::Time | EntityType::Money | EntityType::Percentage => 0.9,
            _ => 0.7,
        };
        features.insert("pattern_match".to_string(), pattern_score);

        // Context score (surrounding words)
        let context_score = self.calculate_context_score(entity, text, contextwindow);
        features.insert("context_score".to_string(), context_score);

        // Length score (longer entities tend to be more reliable)
        let length_score = (entity.text.len() as f64 / 20.0).min(1.0);
        features.insert("length_score".to_string(), length_score);

        // Position score (entities at beginning/end might be more important)
        let position_score = if entity.start < text.len() / 4 || entity.end > 3 * text.len() / 4 {
            0.8
        } else {
            0.6
        };
        features.insert("position_score".to_string(), position_score);

        // Calculate weighted sum
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for (feature, score) in features {
            if let Some(weight) = self.featureweights.get(&feature) {
                total_score += score * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.5
        }
    }

    /// Calculate context score based on surrounding words
    fn calculate_context_score(&self, entity: &Entity, text: &str, window: usize) -> f64 {
        let start = entity.start.saturating_sub(window);
        let end = (entity.end + window).min(text.len());
        let context = &text[start..end];

        // Simple scoring based on presence of relevant keywords
        let keywords = match entity.entity_type {
            EntityType::Person => vec!["Mr.", "Ms.", "Dr.", "CEO", "President", "Manager"],
            EntityType::Organization => {
                vec!["Inc.", "Corp.", "LLC", "Ltd.", "Company", "Foundation"]
            }
            EntityType::Location => vec!["in", "at", "from", "to", "near", "City", "State"],
            EntityType::Money => vec!["cost", "price", "pay", "budget", "revenue", "profit"],
            EntityType::Date => vec!["on", "in", "during", "until", "since", "when"],
            _ => vec![],
        };

        let matches = keywords
            .iter()
            .filter(|&keyword| context.contains(keyword))
            .count();

        if keywords.is_empty() {
            0.5
        } else {
            (matches as f64 / keywords.len() as f64).min(1.0)
        }
    }
}
