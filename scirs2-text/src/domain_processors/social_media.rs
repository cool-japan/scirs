//! Social media text processing for social media content

use super::types::{Domain, DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};
use crate::information_extraction::{Entity, EntityType};
use regex::Regex;
use std::collections::HashMap;

/// Social media text processor
pub struct SocialMediaTextProcessor {
    config: DomainProcessorConfig,
    hashtag_regex: Regex,
    mention_regex: Regex,
    url_regex: Regex,
}

impl SocialMediaTextProcessor {
    /// Create new social media text processor
    pub fn new(mut config: DomainProcessorConfig) -> Self {
        // Ensure domain is set to SocialMedia
        config.domain = Domain::SocialMedia;
        // Hashtag patterns
        let hashtag_regex = Regex::new(r"#\w+").unwrap_or_else(|_| Regex::new(r"#").unwrap());

        // Mention patterns
        let mention_regex = Regex::new(r"@\w+").unwrap_or_else(|_| Regex::new(r"@").unwrap());

        // URL patterns
        let url_regex =
            Regex::new(r"https?://\S+").unwrap_or_else(|_| Regex::new(r"http").unwrap());

        Self {
            config,
            hashtag_regex,
            mention_regex,
            url_regex,
        }
    }

    /// Process social media text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processedtext = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();

        // Extract social media entities
        let hashtag_entities = self.extract_hashtags_with_positions(&processedtext)?;
        entities.extend(hashtag_entities);

        let mention_entities = self.extract_mentions_with_positions(&processedtext)?;
        entities.extend(mention_entities);

        let url_entities = self.extract_urls_with_positions(&processedtext)?;
        entities.extend(url_entities);

        // Clean social media text
        processedtext = self.clean_socialmediatext(&processedtext)?;

        Ok(ProcessedDomainText {
            originaltext: text.to_string(),
            processedtext,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }

    /// Extract hashtags with position information
    fn extract_hashtags_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .hashtag_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("hashtag".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract mentions with position information
    fn extract_mentions_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .mention_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Custom("mention".to_string()),
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Extract URLs with position information
    fn extract_urls_with_positions(&self, text: &str) -> Result<Vec<Entity>> {
        Ok(self
            .url_regex
            .find_iter(text)
            .map(|m| Entity {
                text: m.as_str().to_string(),
                entity_type: EntityType::Url,
                confidence: 0.9,
                start: m.start(),
                end: m.end(),
            })
            .collect())
    }

    /// Clean social media text
    fn clean_socialmediatext(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {e}")))?
            .replace_all(&cleaned, " ")
            .to_string();

        Ok(cleaned.trim().to_string())
    }
}
