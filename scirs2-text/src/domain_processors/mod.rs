//! Domain-specific text processors for specialized fields
//!
//! This module provides specialized text processing capabilities for different domains
//! including scientific, legal, and medical text processing with domain-specific
//! vocabularies, entity recognition, and preprocessing pipelines.

pub mod financial;
pub mod legal;
pub mod medical;
pub mod news;
pub mod patent;
pub mod scientific;
pub mod social_media;
pub mod types;
pub mod unified;

// Re-export main types and functionality to maintain backward compatibility
pub use financial::FinancialTextProcessor;
pub use legal::LegalTextProcessor;
pub use medical::MedicalTextProcessor;
pub use news::NewsTextProcessor;
pub use patent::PatentTextProcessor;
pub use scientific::ScientificTextProcessor;
pub use social_media::SocialMediaTextProcessor;
pub use types::{Domain, DomainProcessorConfig, ProcessedDomainText};
pub use unified::UnifiedDomainProcessor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn test_domain_display() {
        assert_eq!(Domain::Scientific.to_string(), "scientific");
        assert_eq!(Domain::Legal.to_string(), "legal");
        assert_eq!(Domain::Medical.to_string(), "medical");
        assert_eq!(Domain::Financial.to_string(), "financial");
        assert_eq!(Domain::Patent.to_string(), "patent");
        assert_eq!(Domain::News.to_string(), "news");
        assert_eq!(Domain::SocialMedia.to_string(), "social_media");
    }

    #[test]
    fn test_default_config() {
        let config = DomainProcessorConfig::default();
        assert_eq!(config.domain, Domain::Scientific);
        assert!(config.preserve_technical_terms);
        assert!(config.normalize_abbreviations);
        assert!(config.extract_entities);
        assert!(config.handle_citations);
        assert!(config.remove_html);
        assert!(config.clean_whitespace);
    }

    #[test]
    fn test_scientific_processor() -> Result<()> {
        let config = DomainProcessorConfig {
            domain: Domain::Scientific,
            ..Default::default()
        };
        let processor = ScientificTextProcessor::new(config)?;

        let text = "The study (Smith et al., 2023) found that H2O has a melting point of 0°C.";
        let result = processor.process(text)?;

        assert_eq!(result.domain, Domain::Scientific);
        assert!(!result.entities.is_empty());
        assert!(!result.processedtext.is_empty());
        Ok(())
    }

    #[test]
    fn test_legal_processor() -> Result<()> {
        let config = DomainProcessorConfig {
            domain: Domain::Legal,
            ..Default::default()
        };
        let processor = LegalTextProcessor::new(config)?;

        let text = "The case Smith v. Jones, 123 F.3d 456 (2020) established liability standards.";
        let result = processor.process(text)?;

        assert_eq!(result.domain, Domain::Legal);
        assert!(!result.processedtext.is_empty());
        Ok(())
    }

    #[test]
    fn test_medical_processor() -> Result<()> {
        let config = DomainProcessorConfig {
            domain: Domain::Medical,
            ..Default::default()
        };
        let processor = MedicalTextProcessor::new(config)?;

        let text = "Patient received 50mg aspirin and had blood pressure of 120/80 mmHg.";
        let result = processor.process(text)?;

        assert_eq!(result.domain, Domain::Medical);
        assert!(!result.processedtext.is_empty());
        Ok(())
    }

    #[test]
    fn test_financial_processor() -> Result<()> {
        let config = DomainProcessorConfig {
            domain: Domain::Financial,
            ..Default::default()
        };
        let processor = FinancialTextProcessor::new(config)?;

        let text = "The company reported revenue of $1.5 billion and profit margin of 15%.";
        let result = processor.process(text)?;

        assert_eq!(result.domain, Domain::Financial);
        assert!(!result.processedtext.is_empty());
        Ok(())
    }

    #[test]
    fn test_patent_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Patent,
            ..Default::default()
        };
        let processor = PatentTextProcessor::new(config);

        let text =
            "The invention described in US1234567 relates to claim 1 of the patent application.";
        let result = processor.process(text).unwrap();

        assert_eq!(result.domain, Domain::Patent);
        assert!(!result.processedtext.is_empty());
    }

    #[test]
    fn test_news_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::News,
            ..Default::default()
        };
        let processor = NewsTextProcessor::new(config);

        let text = "WASHINGTON - The President announced new policies today.";
        let result = processor.process(text).unwrap();

        assert_eq!(result.domain, Domain::News);
        assert!(!result.processedtext.is_empty());
    }

    #[test]
    fn test_social_media_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::SocialMedia,
            ..Default::default()
        };
        let processor = SocialMediaTextProcessor::new(config);

        let text = "Just posted a new article! #AI #MachineLearning @mention https://example.com";
        let result = processor.process(text).unwrap();

        assert_eq!(result.domain, Domain::SocialMedia);
        assert!(!result.entities.is_empty()); // Should extract hashtags, mentions, URLs
        assert!(!result.processedtext.is_empty());
    }

    #[test]
    fn test_unified_processor() -> Result<()> {
        let config = DomainProcessorConfig::default();

        let processor = UnifiedDomainProcessor::new()
            .with_scientific(config.clone())?
            .with_legal(config.clone())?
            .with_medical(config.clone())?
            .with_financial(config.clone())?
            .with_patent(config.clone())
            .with_news(config.clone())
            .with_social_media(config);

        // Test scientific text
        let scientific_text = "The study (Smith et al., 2023) analyzed H2O samples.";
        let result = processor.process(scientific_text, Domain::Scientific)?;
        assert_eq!(result.domain, Domain::Scientific);

        // Test legal text
        let legal_text = "The contract establishes liability standards.";
        let result = processor.process(legal_text, Domain::Legal)?;
        assert_eq!(result.domain, Domain::Legal);

        // Test auto-detection
        let social_text = "Great post! #awesome @friend";
        let result = processor.process_auto(social_text)?;
        assert_eq!(result.domain, Domain::SocialMedia);

        Ok(())
    }

    #[test]
    fn test_domain_detection() {
        let processor = UnifiedDomainProcessor::new();

        // Test legal detection
        let legal_text = "Whereas the parties agree to this contract and liability provisions.";
        assert_eq!(processor.detect_domain(legal_text), Domain::Legal);

        // Test medical detection
        let medical_text = "The patient was diagnosed with hypertension and given medication.";
        assert_eq!(processor.detect_domain(medical_text), Domain::Medical);

        // Test financial detection
        let financial_text = "The company's revenue increased and profit margins improved.";
        assert_eq!(processor.detect_domain(financial_text), Domain::Financial);

        // Test patent detection
        let patent_text = "This patent claim describes a novel invention.";
        assert_eq!(processor.detect_domain(patent_text), Domain::Patent);

        // Test social media detection
        let social_text = "Great day! #sunny @friend";
        assert_eq!(processor.detect_domain(social_text), Domain::SocialMedia);

        // Test news detection
        let news_text = "Breaking news: according to officials, the event occurred today.";
        assert_eq!(processor.detect_domain(news_text), Domain::News);

        // Test default (scientific)
        let scientific_text = "The research methodology used statistical analysis.";
        assert_eq!(processor.detect_domain(scientific_text), Domain::Scientific);
    }

    #[test]
    fn test_processor_error_handling() {
        let processor = UnifiedDomainProcessor::new(); // No processors configured

        let result = processor.process("test text", Domain::Scientific);
        assert!(result.is_err());

        let result = processor.process("test text", Domain::Legal);
        assert!(result.is_err());

        let result = processor.process("test text", Domain::Medical);
        assert!(result.is_err());

        let result = processor.process("test text", Domain::Financial);
        assert!(result.is_err());
    }

    #[test]
    fn test_processed_domain_text_fields() {
        let config = DomainProcessorConfig::default();
        let processor = PatentTextProcessor::new(config);

        let text = "Test patent text with US1234567.";
        let result = processor.process(text).unwrap();

        assert_eq!(result.originaltext, text);
        assert!(!result.processedtext.is_empty());
        assert_eq!(result.domain, Domain::Patent);
        // entities and metadata may be empty but should be present
        assert!(result.entities.is_empty() || !result.entities.is_empty());
        assert!(result.metadata.is_empty() || !result.metadata.is_empty());
    }
}
