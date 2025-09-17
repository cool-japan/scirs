//! Scientific text processing for academic and research documents

use super::types::{DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Scientific text processor
pub struct ScientificTextProcessor {
    config: DomainProcessorConfig,
    citation_regex: Regex,
    formula_regex: Regex,
    chemical_regex: Regex,
    measurement_regex: Regex,
    abbreviation_map: HashMap<String, String>,
    #[allow(dead_code)]
    technical_terms: HashSet<String>,
}

impl ScientificTextProcessor {
    /// Create new scientific text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Scientific citation patterns
        let citation_regex = Regex::new(
            r"\(([A-Za-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?(?:;\s*[A-Za-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?)*)\)"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Mathematical formulas and equations
        let formula_regex =
            Regex::new(r"\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]|\\begin\{[^}]+\}.*?\\end\{[^}]+\}")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Chemical formulas
        let chemical_regex = Regex::new(r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Scientific measurements
        let measurement_regex = Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:nm|μm|mm|cm|m|km|mg|g|kg|ml|l|°C|°F|K|Pa|kPa|MPa|Hz|kHz|MHz|GHz|V|mV|A|mA|Ω|W|kW|MW)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Common scientific abbreviations
        let mut abbreviation_map = HashMap::new();
        abbreviation_map.insert("e.g.".to_string(), "for example".to_string());
        abbreviation_map.insert("i.e.".to_string(), "that is".to_string());
        abbreviation_map.insert("et al.".to_string(), "and others".to_string());
        abbreviation_map.insert("cf.".to_string(), "compare".to_string());
        abbreviation_map.insert("viz.".to_string(), "namely".to_string());

        // Technical terms to preserve
        let technical_terms = [
            "algorithm",
            "hypothesis",
            "methodology",
            "quantitative",
            "qualitative",
            "statistical",
            "correlation",
            "regression",
            "significance",
            "p-value",
            "standard deviation",
            "confidence interval",
            "meta-analysis",
            "systematic review",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Ok(Self {
            config,
            citation_regex,
            formula_regex,
            chemical_regex,
            measurement_regex,
            abbreviation_map,
            technical_terms,
        })
    }

    /// Process scientific text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract citations
        if self.config.handle_citations {
            let citation_entities = self.extract_citations_with_positions(&processedtext)?;
            entities.extend(citation_entities);
        }

        // Extract and preserve formulas
        let formulas = self.extract_formulas(&processedtext)?;
        for (i, formula) in formulas.iter().enumerate() {
            let placeholder = format!("[FORMULA_{i}]");
            processedtext = processedtext.replace(formula, &placeholder);
        }
        metadata.insert("formulas".to_string(), formulas.join("|"));

        // Extract measurements
        let measurement_entities = self.extract_measurements_with_positions(&processedtext)?;
        entities.extend(measurement_entities);

        // Extract chemical formulas
        let chemical_entities = self.extract_chemicals_with_positions(&processedtext)?;
        entities.extend(chemical_entities);

        // Normalize abbreviations
        if self.config.normalize_abbreviations {
            for (abbrev, expansion) in &self.abbreviation_map {
                processedtext = processedtext.replace(abbrev, expansion);
            }
        }

        // Clean text while preserving technical terms
        processedtext = self.clean_scientifictext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract citations from text
    #[allow(dead_code)]
    fn extract_citations(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .citation_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract citations with position information
    fn extract_citations_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .citation_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("citation".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract mathematical formulas
    fn extract_formulas(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .formula_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract chemical formulas
    #[allow(dead_code)]
    fn extract_chemicals(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .chemical_regex
            .find_iter(text)
            .filter(|m| {
                let formula = m.as_str();
                // Basic heuristic to filter out non-chemical words
                formula.chars().any(|c| c.is_ascii_uppercase())
                    && formula.chars().any(|c| c.is_ascii_digit())
            })
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract scientific measurements
    #[allow(dead_code)]
    fn extract_measurements(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .measurement_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract measurements with position information
    fn extract_measurements_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .measurement_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("measurement".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract chemicals with position information
    fn extract_chemicals_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .chemical_regex
            .find_iter(text)
            .filter(|m| {
                let formula = m.as_str();
                // Basic heuristic to filter out non-chemical words
                formula.chars().any(|c| c.is_ascii_uppercase())
                    && formula.chars().any(|c| c.is_ascii_digit())
            })
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("chemical".to_string()),
                confidence: 0.7,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean scientific text while preserving important elements
    fn clean_scientifictext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        // Normalize section headers
        cleaned = Regex::new(r"(?i)\b(abstract|introduction|methods?|results?|discussion|conclusion|references?)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[SECTION_{}] ", caps[1].to_uppercase())
            })
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
