//! Information extraction pipelines and result containers

use super::confidence::ConfidenceScorer;
use super::coreference::{CoreferenceChain, CoreferenceResolver};
use super::entities::Entity;
use super::extractors::{KeyPhraseExtractor, PatternExtractor, RuleBasedNER};
use super::linking::{EntityLinker, LinkedEntity};
use super::relations::{Relation, RelationExtractor};
use super::temporal::TemporalExtractor;
use crate::error::Result;
use crate::tokenize::WordTokenizer;
use std::collections::HashMap;

/// Container for all extracted information
#[derive(Debug)]
pub struct ExtractedInformation {
    /// All entities extracted from the text
    pub entities: Vec<Entity>,
    /// Key phrases with importance scores
    pub key_phrases: Vec<(String, f64)>,
    /// Patterns found in the text organized by pattern type
    pub patterns: HashMap<String, Vec<String>>,
    /// Relations found between entities
    pub relations: Vec<Relation>,
}

/// Enhanced container for all extracted information
#[derive(Debug)]
pub struct AdvancedExtractedInformation {
    /// All entities extracted from the text
    pub entities: Vec<Entity>,
    /// Entities linked to knowledge base
    pub linked_entities: Vec<LinkedEntity>,
    /// Key phrases with importance scores
    pub key_phrases: Vec<(String, f64)>,
    /// Patterns found in the text organized by pattern type
    pub patterns: HashMap<String, Vec<String>>,
    /// Relations found between entities
    pub relations: Vec<Relation>,
    /// Coreference chains
    pub coreference_chains: Vec<CoreferenceChain>,
}

/// Comprehensive information extraction pipeline
pub struct InformationExtractionPipeline {
    ner: RuleBasedNER,
    key_phrase_extractor: KeyPhraseExtractor,
    pattern_extractor: PatternExtractor,
    relation_extractor: RelationExtractor,
}

impl Default for InformationExtractionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl InformationExtractionPipeline {
    /// Create a new extraction pipeline
    pub fn new() -> Self {
        Self {
            ner: RuleBasedNER::new(),
            key_phrase_extractor: KeyPhraseExtractor::new(),
            pattern_extractor: PatternExtractor::new(),
            relation_extractor: RelationExtractor::new(),
        }
    }

    /// Set the NER component
    pub fn with_ner(mut self, ner: RuleBasedNER) -> Self {
        self.ner = ner;
        self
    }

    /// Set the key phrase extractor
    pub fn with_key_phrase_extractor(mut self, extractor: KeyPhraseExtractor) -> Self {
        self.key_phrase_extractor = extractor;
        self
    }

    /// Set the pattern extractor
    pub fn with_pattern_extractor(mut self, extractor: PatternExtractor) -> Self {
        self.pattern_extractor = extractor;
        self
    }

    /// Set the relation extractor
    pub fn with_relation_extractor(mut self, extractor: RelationExtractor) -> Self {
        self.relation_extractor = extractor;
        self
    }

    /// Extract all information from text
    pub fn extract(&self, text: &str) -> Result<ExtractedInformation> {
        let tokenizer = WordTokenizer::default();

        let entities = self.ner.extract_entities(text)?;
        let key_phrases = self.key_phrase_extractor.extract(text, &tokenizer)?;
        let patterns = self.pattern_extractor.extract(text)?;
        let relations = self.relation_extractor.extract_relations(text, &entities)?;

        Ok(ExtractedInformation {
            entities,
            key_phrases,
            patterns,
            relations,
        })
    }
}

/// Enhanced information extraction pipeline with advanced features
pub struct AdvancedExtractionPipeline {
    ner: RuleBasedNER,
    key_phrase_extractor: KeyPhraseExtractor,
    pattern_extractor: PatternExtractor,
    relation_extractor: RelationExtractor,
    temporal_extractor: TemporalExtractor,
    entity_linker: EntityLinker,
    coreference_resolver: CoreferenceResolver,
    confidence_scorer: ConfidenceScorer,
}

impl Default for AdvancedExtractionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedExtractionPipeline {
    /// Create new advanced extraction pipeline
    pub fn new() -> Self {
        Self {
            ner: RuleBasedNER::new(),
            key_phrase_extractor: KeyPhraseExtractor::new(),
            pattern_extractor: PatternExtractor::new(),
            relation_extractor: RelationExtractor::new(),
            temporal_extractor: TemporalExtractor::new(),
            entity_linker: EntityLinker::new(),
            coreference_resolver: CoreferenceResolver::new(),
            confidence_scorer: ConfidenceScorer::new(),
        }
    }

    /// Configure components
    pub fn with_ner(mut self, ner: RuleBasedNER) -> Self {
        self.ner = ner;
        self
    }

    /// Configure the entity linker component
    pub fn with_entity_linker(mut self, linker: EntityLinker) -> Self {
        self.entity_linker = linker;
        self
    }

    /// Extract comprehensive information with advanced features
    pub fn extract_advanced(&self, text: &str) -> Result<AdvancedExtractedInformation> {
        let tokenizer = WordTokenizer::default();

        // Basic extractions
        let mut entities = self.ner.extract_entities(text)?;
        let temporal_entities = self.temporal_extractor.extract(text)?;
        entities.extend(temporal_entities);

        // Enhance confidence scores
        for entity in &mut entities {
            entity.confidence = self.confidence_scorer.score_entity(entity, text, 50);
        }

        let key_phrases = self.key_phrase_extractor.extract(text, &tokenizer)?;
        let patterns = self.pattern_extractor.extract(text)?;
        let relations = self.relation_extractor.extract_relations(text, &entities)?;

        // Advanced extractions
        let linked_entities = self.entity_linker.link_entities(&mut entities)?;
        let coreference_chains = self.coreference_resolver.resolve(text, &entities)?;

        Ok(AdvancedExtractedInformation {
            entities,
            linked_entities,
            key_phrases,
            patterns,
            relations,
            coreference_chains,
        })
    }
}
