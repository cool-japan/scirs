//! Document-level information extraction and organization

use super::entities::{Entity, EntityCluster, EntityType};
use super::pipeline::AdvancedExtractionPipeline;
use super::relations::{Event, Relation};
use crate::error::Result;
use std::collections::HashMap;

/// Summary information for a single document
#[derive(Debug, Clone)]
pub struct DocumentSummary {
    /// Index of the document in the corpus
    pub document_index: usize,
    /// Number of entities found in the document
    pub entity_count: usize,
    /// Number of relations found in the document
    pub relation_count: usize,
    /// Key phrases with confidence scores
    pub key_phrases: Vec<(String, f64)>,
    /// Overall confidence score for the summary
    pub confidence_score: f64,
}

/// Identified topic across documents
#[derive(Debug, Clone)]
pub struct Topic {
    /// Name of the topic
    pub name: String,
    /// Key phrases that define the topic
    pub key_phrases: Vec<String>,
    /// Indices of documents containing this topic
    pub document_indices: Vec<usize>,
    /// Confidence score for the topic identification
    pub confidence: f64,
}

/// Structured information extracted from multiple documents
#[derive(Debug)]
pub struct StructuredDocumentInformation {
    /// Summaries of individual documents
    pub documents: Vec<DocumentSummary>,
    /// Clusters of similar entities across documents
    pub entity_clusters: Vec<EntityCluster>,
    /// Relations found across documents
    pub relations: Vec<Relation>,
    /// Events extracted from documents
    pub events: Vec<Event>,
    /// Topics identified across documents
    pub topics: Vec<Topic>,
    /// Total number of entities across all documents
    pub total_entities: usize,
    /// Total number of relations across all documents
    pub total_relations: usize,
}

/// Document-level information extraction and organization
pub struct DocumentInformationExtractor {
    topic_threshold: f64,
    similarity_threshold: f64,
    max_topics: usize,
}

impl Default for DocumentInformationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentInformationExtractor {
    /// Create new document information extractor
    pub fn new() -> Self {
        Self {
            topic_threshold: 0.3,
            similarity_threshold: 0.7,
            max_topics: 10,
        }
    }

    /// Extract information organized by topics and themes
    pub fn extract_structured_information(
        &self,
        documents: &[String],
        pipeline: &AdvancedExtractionPipeline,
    ) -> Result<StructuredDocumentInformation> {
        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();
        let mut document_summaries = Vec::new();

        // Extract information from each document
        for (doc_idx, document) in documents.iter().enumerate() {
            let info = pipeline.extract_advanced(document)?;

            // Add document index to entities
            let mut doc_entities = info.entities;
            for entity in &mut doc_entities {
                entity.confidence *= 0.9; // Slight confidence reduction for batch processing
            }

            let doc_summary = DocumentSummary {
                document_index: doc_idx,
                entity_count: doc_entities.len(),
                relation_count: info.relations.len(),
                key_phrases: info.key_phrases.clone(),
                confidence_score: self.calculate_document_confidence(&doc_entities),
            };

            all_entities.extend(doc_entities);
            all_relations.extend(info.relations);
            document_summaries.push(doc_summary);
        }

        // Cluster similar entities
        let entity_clusters = self.cluster_entities(&all_entities)?;

        // Extract events from relations
        let events = self.extract_events(&all_relations, &all_entities)?;

        // Identify document topics
        let topics = self.identify_topics(&document_summaries)?;

        let total_relations = all_relations.len();
        Ok(StructuredDocumentInformation {
            documents: document_summaries,
            entity_clusters,
            relations: all_relations,
            events,
            topics,
            total_entities: all_entities.len(),
            total_relations,
        })
    }

    /// Calculate overall confidence for a document
    fn calculate_document_confidence(&self, entities: &[Entity]) -> f64 {
        if entities.is_empty() {
            return 0.0;
        }

        let sum: f64 = entities.iter().map(|e| e.confidence).sum();
        sum / entities.len() as f64
    }

    /// Cluster similar entities across documents
    pub fn cluster_entities(&self, entities: &[Entity]) -> Result<Vec<EntityCluster>> {
        let mut clusters = Vec::new();
        let mut used = vec![false; entities.len()];

        for (i, entity) in entities.iter().enumerate() {
            if used[i] {
                continue;
            }

            let mut cluster = EntityCluster {
                representative: entity.clone(),
                members: vec![entity.clone()],
                entity_type: entity.entity_type.clone(),
                confidence: entity.confidence,
            };

            used[i] = true;

            // Find similar entities
            for (j, other) in entities.iter().enumerate().skip(i + 1) {
                if used[j] || other.entity_type != entity.entity_type {
                    continue;
                }

                let similarity = self.calculate_entity_similarity(entity, other);
                if similarity > self.similarity_threshold {
                    cluster.members.push(other.clone());
                    cluster.confidence = (cluster.confidence + other.confidence) / 2.0;
                    used[j] = true;
                }
            }

            clusters.push(cluster);
        }

        clusters.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        Ok(clusters)
    }

    /// Calculate similarity between two entities
    pub fn calculate_entity_similarity(&self, entity1: &Entity, entity2: &Entity) -> f64 {
        if entity1.entity_type != entity2.entity_type {
            return 0.0;
        }

        // Simple Levenshtein-based similarity
        let text1 = entity1.text.to_lowercase();
        let text2 = entity2.text.to_lowercase();

        if text1 == text2 {
            return 1.0;
        }

        // Calculate character-level similarity
        let max_len = text1.len().max(text2.len());
        if max_len == 0 {
            return 1.0;
        }

        let distance = self.levenshtein_distance(&text1, &text2);
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Simple Levenshtein distance implementation
    pub fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        #[allow(clippy::needless_range_loop)]
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for (i, &c1) in s1_chars.iter().enumerate() {
            for (j, &c2) in s2_chars.iter().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = std::cmp::min(
                    std::cmp::min(matrix[i][j + 1] + 1, matrix[i + 1][j] + 1),
                    matrix[i][j] + cost,
                );
            }
        }

        matrix[len1][len2]
    }

    /// Extract events from relations and entities
    pub fn extract_events(
        &self,
        relations: &[Relation],
        entities: &[Entity],
    ) -> Result<Vec<Event>> {
        let mut events = Vec::new();

        // Group relations by context to identify events
        let mut relation_groups: std::collections::HashMap<String, Vec<&Relation>> =
            std::collections::HashMap::new();

        for relation in relations {
            let context_key = format!(
                "{}_{}",
                relation.subject.start / 100, // Group by approximate position
                relation.object.start / 100
            );
            relation_groups
                .entry(context_key)
                .or_default()
                .push(relation);
        }

        // Convert relation groups to events
        for (_, group_relations) in relation_groups {
            if group_relations.len() >= 2 {
                let event = Event {
                    event_type: self.infer_event_type(&group_relations),
                    participants: self.extract_participants(&group_relations),
                    location: self.extract_location(&group_relations, entities),
                    time: self.extract_time(&group_relations, entities),
                    description: self.generate_event_description(&group_relations),
                    confidence: self.calculate_event_confidence(&group_relations),
                };
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Infer event type from relations
    fn infer_event_type(&self, relations: &[&Relation]) -> String {
        let relation_types: std::collections::HashMap<String, usize> =
            relations
                .iter()
                .fold(std::collections::HashMap::new(), |mut acc, rel| {
                    *acc.entry(rel.relation_type.clone()).or_insert(0) += 1;
                    acc
                });

        relation_types
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(rel_type_, _)| rel_type_)
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Extract participants from relations
    fn extract_participants(&self, relations: &[&Relation]) -> Vec<Entity> {
        let mut participants = Vec::new();
        for relation in relations {
            participants.push(relation.subject.clone());
            participants.push(relation.object.clone());
        }

        // Deduplicate participants
        participants.sort_by_key(|e| e.text.clone());
        participants.dedup_by_key(|e| e.text.clone());
        participants
    }

    /// Extract location entities near the relations
    fn extract_location(&self, relations: &[&Relation], entities: &[Entity]) -> Option<Entity> {
        for relation in relations {
            for entity in entities {
                if matches!(entity.entity_type, EntityType::Location) {
                    let relation_span = relation.subject.start..relation.object.end;
                    let entity_span = entity.start..entity.end;

                    // Check if location entity is near the relation
                    if relation_span.contains(&entity.start)
                        || entity_span.contains(&relation.subject.start)
                        || (entity.start as i32 - relation.subject.start as i32).abs() < 100
                    {
                        return Some(entity.clone());
                    }
                }
            }
        }
        None
    }

    /// Extract temporal entities near the relations
    fn extract_time(&self, relations: &[&Relation], entities: &[Entity]) -> Option<Entity> {
        for relation in relations {
            for entity in entities {
                if matches!(entity.entity_type, EntityType::Date | EntityType::Time) {
                    let relation_span = relation.subject.start..relation.object.end;
                    let entity_span = entity.start..entity.end;

                    // Check if temporal entity is near the relation
                    if relation_span.contains(&entity.start)
                        || entity_span.contains(&relation.subject.start)
                        || (entity.start as i32 - relation.subject.start as i32).abs() < 100
                    {
                        return Some(entity.clone());
                    }
                }
            }
        }
        None
    }

    /// Generate description for an event
    fn generate_event_description(&self, relations: &[&Relation]) -> String {
        if relations.is_empty() {
            return "Unknown event".to_string();
        }

        let contexts: Vec<String> = relations.iter().map(|r| r.context.clone()).collect();

        // Find the longest context as primary description
        contexts
            .into_iter()
            .max_by_key(|s| s.len())
            .unwrap_or_else(|| "Event description unavailable".to_string())
    }

    /// Calculate confidence for an event
    fn calculate_event_confidence(&self, relations: &[&Relation]) -> f64 {
        if relations.is_empty() {
            return 0.0;
        }

        let sum: f64 = relations.iter().map(|r| r.confidence).sum();
        (sum / relations.len() as f64) * 0.8 // Reduce confidence for inferred events
    }

    /// Identify topics from document summaries
    pub fn identify_topics(&self, summaries: &[DocumentSummary]) -> Result<Vec<Topic>> {
        let mut topics = Vec::new();
        let mut topic_phrases: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        // Collect all key phrases with document indices
        for summary in summaries {
            for (phrase, score) in &summary.key_phrases {
                if *score > self.topic_threshold {
                    topic_phrases
                        .entry(phrase.clone())
                        .or_default()
                        .push(summary.document_index);
                }
            }
        }

        // Create topics from frequent phrases
        for (phrase, doc_indices) in topic_phrases {
            if doc_indices.len() >= 2 {
                // Phrase appears in multiple documents
                let topic = Topic {
                    name: phrase.clone(),
                    key_phrases: vec![phrase],
                    document_indices: doc_indices,
                    confidence: 0.8,
                };
                topics.push(topic);
            }
        }

        // Limit to max topics
        topics.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        topics.truncate(self.max_topics);

        Ok(topics)
    }
}
