//! News text processing for journalism and news articles

use super::types::{DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::HashMap;

/// News text processor
pub struct NewsTextProcessor {
    config: DomainProcessorConfig,
    headline_regex: Regex,
    dateline_regex: Regex,
}

impl NewsTextProcessor {
    /// Create new news text processor
    pub fn new(config: DomainProcessorConfig) -> Self {
        // Headline patterns
        let headline_regex =
            Regex::new(r"^[A-Z][^.!?]*[.!?]?$").unwrap_or_else(|_| Regex::new(r"[A-Z]").unwrap());

        // Dateline patterns
        let dateline_regex = Regex::new(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\s*-")
            .unwrap_or_else(|_| Regex::new(r"-").unwrap());

        Self {
            config,
            headline_regex,
            dateline_regex,
        }
    }

    /// Process news text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract news entities
        let dateline_entities = self.extract_datelines_with_positions(&processedtext)?;
        entities.extend(dateline_entities);

        // Clean news text
        processedtext = self.clean_newstext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract datelines with position information
    fn extract_datelines_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .dateline_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("dateline".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean news text
    fn clean_newstext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
