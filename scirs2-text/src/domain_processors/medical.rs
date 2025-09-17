//! Medical text processing for clinical and healthcare documents

use super::types::{DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::HashMap;

/// Medical text processor
pub struct MedicalTextProcessor {
    config: DomainProcessorConfig,
    medication_regex: Regex,
    diagnosis_regex: Regex,
    measurement_regex: Regex,
}

impl MedicalTextProcessor {
    /// Create new medical text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Medical measurements
        let measurement_regex = Regex::new(r"\b\d+(?:\.\d+)?\s*(?:mg|μg|g|ml|l|mmHg|bpm|°C|°F)\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Medication patterns
        let medication_regex = Regex::new(r"\b(?:[A-Z][a-z]+(?:ine|ol|ium|ide|ate|ose))\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        // Basic diagnosis patterns
        let diagnosis_regex =
            Regex::new(r"\b(?:diagnosis|diagnosed|condition|syndrome|disease|disorder)\b")
                .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?;

        Ok(Self {
            config,
            medication_regex,
            diagnosis_regex,
            measurement_regex,
        })
    }

    /// Process medical text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract medical entities
        let medication_entities = self.extract_medications_with_positions(&processedtext)?;
        entities.extend(medication_entities);

        let measurement_entities = self.extract_measurements_with_positions(&processedtext)?;
        entities.extend(measurement_entities);

        // Clean medical text
        processedtext = self.clean_medicaltext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract medications with position information
    fn extract_medications_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .medication_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("medication".to_string()),
                confidence: 0.7,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract measurements with position information
    fn extract_measurements_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .measurement_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("medical_measurement".to_string()),
                confidence: 0.8,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean medical text
    fn clean_medicaltext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
