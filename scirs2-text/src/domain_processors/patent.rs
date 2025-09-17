//! Patent text processing for patent documents and applications

use super::types::{Domain, DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::HashMap;

/// Patent text processor
pub struct PatentTextProcessor {
    config: DomainProcessorConfig,
    patent_number_regex: Regex,
    claim_regex: Regex,
}

impl PatentTextProcessor {
    /// Create new patent text processor
    pub fn new(mut config: DomainProcessorConfig) -> Self {
        // Ensure domain is set to Patent
        config.domain = Domain::Patent;

        // Patent number patterns - handle regex creation errors gracefully
        let patent_number_regex = Regex::new(r"\b(?:US|EP|WO)\s*\d{6,8}\s*[A-Z]?\d?\b")
            .unwrap_or_else(|_| Regex::new(r"\d+").unwrap());

        // Patent claim patterns
        let claim_regex =
            Regex::new(r"(?i)\bclaim\s+\d+").unwrap_or_else(|_| Regex::new(r"claim").unwrap());

        Self {
            config,
            patent_number_regex,
            claim_regex,
        }
    }

    /// Process patent text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract patent entities
        let patent_entities = self.extract_patents_with_positions(&processedtext)?;
        entities.extend(patent_entities);

        // Clean patent text
        processedtext = self.clean_patenttext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract patents with position information
    fn extract_patents_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .patent_number_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("patent_number".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean patent text
    fn clean_patenttext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
