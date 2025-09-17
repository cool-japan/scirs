//! Legal text processing for contracts and legal documents

use super::types::{Domain, DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::HashMap;

/// Legal text processor
pub struct LegalTextProcessor {
    config: DomainProcessorConfig,
    case_citation_regex: Regex,
    statute_regex: Regex,
    contract_clause_regex: Regex,
}

impl LegalTextProcessor {
    /// Create new legal text processor
    pub fn new(mut config: DomainProcessorConfig) -> Result<Self> {
        // Ensure domain is set to Legal
        config.domain = Domain::Legal;
        // Legal case citations
        let case_citation_regex = Regex::new(
            r"\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+(?:\s*,\s*\d+\s+[A-Z][a-z]*\.?\s*\d*(?:\s*\(\d{4}\))?)?|\d+\s+[A-Z][a-z]*\.?\s*\d+(?:\s*\(\d{4}\))?"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Statute references
        let statute_regex =
            Regex::new(r"\b(?:\d+\s+)?[A-Z]\.?[A-Z]\.?[A-Z]?\.?\s*§\s*\d+(?:\.\d+)*(?:\([a-z]\))?")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Contract clauses
        let contract_clause_regex = Regex::new(
            r"(?i)\b(?:whereas|therefore|party|agreement|contract|shall|liability|damages|breach)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        Ok(Self {
            config,
            case_citation_regex,
            statute_regex,
            contract_clause_regex,
        })
    }

    /// Process legal text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract case citations
        let case_citation_entities = self.extract_case_citations_with_positions(&processedtext)?;
        entities.extend(case_citation_entities);

        // Extract statute references
        let statute_entities = self.extract_statutes_with_positions(&processedtext)?;
        entities.extend(statute_entities);

        // Identify contract clauses
        let clauses = self.identify_contract_clauses(&processedtext)?;
        metadata.insert("contract_clauses".to_string(), clauses.join("|"));

        // Normalize legal formatting
        processedtext = self.normalize_legaltext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract case citations with position information
    fn extract_case_citations_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .case_citation_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("case_citation".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract statutes with position information
    fn extract_statutes_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .statute_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("statute".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Identify contract clauses
    fn identify_contract_clauses(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .contract_clause_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Normalize legal text formatting
    fn normalize_legaltext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
