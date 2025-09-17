//! Unified domain processor that can handle multiple domains

use super::financial::FinancialTextProcessor;
use super::legal::LegalTextProcessor;
use super::medical::MedicalTextProcessor;
use super::news::NewsTextProcessor;
use super::patent::PatentTextProcessor;
use super::scientific::ScientificTextProcessor;
use super::social_media::SocialMediaTextProcessor;
use super::types::{Domain, DomainProcessorConfig, ProcessedDomainText};
use crate::error::{Result, TextError};

/// Unified domain processor that can handle multiple domains
pub struct UnifiedDomainProcessor {
    scientific_processor: Option<ScientificTextProcessor>,
    legal_processor: Option<LegalTextProcessor>,
    medical_processor: Option<MedicalTextProcessor>,
    financial_processor: Option<FinancialTextProcessor>,
    patent_processor: Option<PatentTextProcessor>,
    news_processor: Option<NewsTextProcessor>,
    social_media_processor: Option<SocialMediaTextProcessor>,
}

impl Default for UnifiedDomainProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedDomainProcessor {
    /// Create new unified domain processor
    pub fn new() -> Self {
        Self {
            scientific_processor: None,
            legal_processor: None,
            medical_processor: None,
            financial_processor: None,
            patent_processor: None,
            news_processor: None,
            social_media_processor: None,
        }
    }

    /// Add scientific text processing capability
    pub fn with_scientific(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.scientific_processor = Some(ScientificTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add legal text processing capability
    pub fn with_legal(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.legal_processor = Some(LegalTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add medical text processing capability
    pub fn with_medical(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.medical_processor = Some(MedicalTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add financial text processing capability
    pub fn with_financial(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.financial_processor = Some(FinancialTextProcessor::new(config)?);
        Ok(self)
    }

    /// Add patent text processing capability
    pub fn with_patent(mut self, config: DomainProcessorConfig) -> Self {
        self.patent_processor = Some(PatentTextProcessor::new(config));
        self
    }

    /// Add news text processing capability
    pub fn with_news(mut self, config: DomainProcessorConfig) -> Self {
        self.news_processor = Some(NewsTextProcessor::new(config));
        self
    }

    /// Add social media text processing capability
    pub fn with_social_media(mut self, config: DomainProcessorConfig) -> Self {
        self.social_media_processor = Some(SocialMediaTextProcessor::new(config));
        self
    }

    /// Process text for specified domain
    pub fn process(&self, text: &str, domain: Domain) -> Result<ProcessedDomainText> {
        match domain {
            Domain::Scientific => {
                if let Some(processor) = &self.scientific_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Scientific processor not configured".to_string(),
                    ))
                }
            }
            Domain::Legal => {
                if let Some(processor) = &self.legal_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Legal processor not configured".to_string(),
                    ))
                }
            }
            Domain::Medical => {
                if let Some(processor) = &self.medical_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Medical processor not configured".to_string(),
                    ))
                }
            }
            Domain::Financial => {
                if let Some(processor) = &self.financial_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Financial processor not configured".to_string(),
                    ))
                }
            }
            Domain::Patent => {
                if let Some(processor) = &self.patent_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Patent processor not configured".to_string(),
                    ))
                }
            }
            Domain::News => {
                if let Some(processor) = &self.news_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "News processor not configured".to_string(),
                    ))
                }
            }
            Domain::SocialMedia => {
                if let Some(processor) = &self.social_media_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput(
                        "Social media processor not configured".to_string(),
                    ))
                }
            }
        }
    }

    /// Process text with auto-detection of domain
    pub fn process_auto(&self, text: &str) -> Result<ProcessedDomainText> {
        // Simple domain detection heuristics
        let domain = self.detect_domain(text);
        self.process(text, domain)
    }

    /// Simple domain detection based on content
    pub fn detect_domain(&self, text: &str) -> Domain {
        let text_lower = text.to_lowercase();

        // Legal indicators
        if text_lower.contains("whereas")
            || text_lower.contains("liability")
            || text_lower.contains("contract")
        {
            return Domain::Legal;
        }

        // Medical indicators
        if text_lower.contains("patient")
            || text_lower.contains("diagnosis")
            || text_lower.contains("medication")
        {
            return Domain::Medical;
        }

        // Financial indicators
        if text_lower.contains("revenue")
            || text_lower.contains("profit")
            || text_lower.contains("investment")
        {
            return Domain::Financial;
        }

        // Patent indicators
        if text_lower.contains("claim")
            || text_lower.contains("invention")
            || text_lower.contains("patent")
        {
            return Domain::Patent;
        }

        // News indicators (check before social media due to length)
        if text_lower.contains("breaking")
            || text_lower.contains("according to")
            || text_lower.contains("officials")
        {
            return Domain::News;
        }

        // Social media indicators
        if text_lower.contains("#") || text_lower.contains("@") {
            return Domain::SocialMedia;
        }

        // Default to scientific
        Domain::Scientific
    }
}
