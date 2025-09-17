//! Financial text processing for financial documents and reports

use super::types::{DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::HashMap;

/// Financial text processor
pub struct FinancialTextProcessor {
    config: DomainProcessorConfig,
    currency_regex: Regex,
    percentage_regex: Regex,
    financial_term_regex: Regex,
}

impl FinancialTextProcessor {
    /// Create new financial text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Currency patterns
        let currency_regex = Regex::new(
            r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|dollars?|euros?|pounds?|yen)"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Percentage patterns
        let percentage_regex = Regex::new(r"\b\d+(?:\.\d+)?%\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Financial terms
        let financial_term_regex = Regex::new(
            r"\b(?:revenue|profit|loss|EBITDA|dividend|equity|debt|assets|liabilities|ROI|ROE|market cap)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        Ok(Self {
            config,
            currency_regex,
            percentage_regex,
            financial_term_regex,
        })
    }

    /// Process financial text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract financial entities
        let currency_entities = self.extract_currencies_with_positions(&processedtext)?;
        entities.extend(currency_entities);

        let percentage_entities = self.extract_percentages_with_positions(&processedtext)?;
        entities.extend(percentage_entities);

        // Clean financial text
        processedtext = self.clean_financialtext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract currencies with position information
    fn extract_currencies_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .currency_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Money,
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract percentages with position information
    fn extract_percentages_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .percentage_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Percentage,
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean financial text
    fn clean_financialtext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
