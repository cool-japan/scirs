//! Entity linking and knowledge base integration

use super::entities::{Entity, EntityType};
use crate::error::Result;
use std::collections::HashMap;

/// Knowledge base entry for entity linking
#[derive(Debug, Clone)]
pub struct KnowledgeBaseEntry {
    /// Canonical name of the entity
    pub canonical_name: String,
    /// Type of the entity
    pub entity_type: EntityType,
    /// Alternative names for the entity
    pub aliases: Vec<String>,
    /// Confidence score for this entry
    pub confidence: f64,
    /// Additional metadata about the entity
    pub metadata: HashMap<String, String>,
}

/// Entity with knowledge base linking
#[derive(Debug, Clone)]
pub struct LinkedEntity {
    /// The original entity
    pub entity: Entity,
    /// Canonical name from knowledge base
    pub canonical_name: String,
    /// Confidence score for the linking
    pub linked_confidence: f64,
    /// Additional metadata from knowledge base
    pub metadata: HashMap<String, String>,
}

/// Entity linker for connecting entities to knowledge bases
pub struct EntityLinker {
    knowledge_base: HashMap<String, KnowledgeBaseEntry>,
    alias_map: HashMap<String, String>,
}

impl Default for EntityLinker {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityLinker {
    /// Create new entity linker
    pub fn new() -> Self {
        Self {
            knowledge_base: HashMap::new(),
            alias_map: HashMap::new(),
        }
    }

    /// Add entity to knowledge base
    pub fn add_entity(&mut self, entry: KnowledgeBaseEntry) {
        let canonical = entry.canonical_name.clone();

        // Add aliases to alias map (store in lowercase for case-insensitive lookup)
        for alias in &entry.aliases {
            self.alias_map
                .insert(alias.to_lowercase(), canonical.clone());
        }
        self.alias_map
            .insert(canonical.to_lowercase(), canonical.clone());

        self.knowledge_base.insert(canonical, entry);
    }

    /// Link extracted entities to knowledge base
    pub fn link_entities(&self, entities: &mut [Entity]) -> Result<Vec<LinkedEntity>> {
        let mut linked_entities = Vec::new();

        for entity in entities {
            if let Some(canonical_name) = self.alias_map.get(&entity.text.to_lowercase()) {
                if let Some(kb_entry) = self.knowledge_base.get(canonical_name) {
                    let confidence = entity.confidence * kb_entry.confidence;
                    linked_entities.push(LinkedEntity {
                        entity: entity.clone(),
                        canonical_name: kb_entry.canonical_name.clone(),
                        linked_confidence: confidence,
                        metadata: kb_entry.metadata.clone(),
                    });
                }
            }
        }

        Ok(linked_entities)
    }
}
