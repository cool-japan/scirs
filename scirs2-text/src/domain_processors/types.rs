//! Core types for domain-specific text processing

use crate::information_extraction::Entity;
use std::collections::{HashMap, HashSet};

/// Domain-specific text processing domains
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Scientific and academic text
    Scientific,
    /// Legal documents and contracts
    Legal,
    /// Medical and clinical text
    Medical,
    /// Financial documents
    Financial,
    /// Patent documents
    Patent,
    /// News and journalism
    News,
    /// Social media content
    SocialMedia,
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Scientific => write!(f, "scientific"),
            Domain::Legal => write!(f, "legal"),
            Domain::Medical => write!(f, "medical"),
            Domain::Financial => write!(f, "financial"),
            Domain::Patent => write!(f, "patent"),
            Domain::News => write!(f, "news"),
            Domain::SocialMedia => write!(f, "social_media"),
        }
    }
}

/// Configuration for domain-specific processing
#[derive(Debug, Clone)]
pub struct DomainProcessorConfig {
    /// Target domain
    pub domain: Domain,
    /// Whether to preserve technical terms
    pub preserve_technical_terms: bool,
    /// Whether to normalize abbreviations
    pub normalize_abbreviations: bool,
    /// Whether to extract domain-specific entities
    pub extract_entities: bool,
    /// Whether to handle citations and references
    pub handle_citations: bool,
    /// Whether to remove HTML/XML tags
    pub remove_html: bool,
    /// Whether to clean whitespace
    pub clean_whitespace: bool,
    /// Custom stop words for the domain
    pub custom_stop_words: HashSet<String>,
    /// Domain-specific regex patterns
    pub custom_patterns: HashMap<String, String>,
}

impl Default for DomainProcessorConfig {
    fn default() -> Self {
        Self {
            domain: Domain::Scientific,
            preserve_technical_terms: true,
            normalize_abbreviations: true,
            extract_entities: true,
            handle_citations: true,
            remove_html: true,
            clean_whitespace: true,
            custom_stop_words: HashSet::new(),
            custom_patterns: HashMap::new(),
        }
    }
}

/// Result of domain-specific text processing
#[derive(Debug, Clone)]
pub struct ProcessedDomainText {
    /// Original input text
    pub originaltext: String,
    /// Processed text
    pub processedtext: String,
    /// Domain type
    pub domain: Domain,
    /// Extracted domain-specific entities
    pub entities: Vec<Entity>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}
