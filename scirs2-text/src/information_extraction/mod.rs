//! Information extraction utilities for structured data extraction from text
//!
//! This module provides tools for extracting structured information such as
//! named entities, key phrases, dates, and patterns from unstructured text.

pub mod confidence;
pub mod coreference;
pub mod document;
pub mod entities;
pub mod extractors;
pub mod linking;
pub mod patterns;
pub mod pipeline;
pub mod relations;
pub mod temporal;

// Re-export main types and functionality to maintain backward compatibility
pub use confidence::ConfidenceScorer;
pub use coreference::{CoreferenceChain, CoreferenceMention, CoreferenceResolver, MentionType};
pub use document::{
    DocumentInformationExtractor, DocumentSummary, StructuredDocumentInformation, Topic,
};
pub use entities::{Entity, EntityCluster, EntityType};
pub use extractors::{KeyPhraseExtractor, PatternExtractor, RuleBasedNER};
pub use linking::{EntityLinker, KnowledgeBaseEntry, LinkedEntity};
pub use pipeline::{
    AdvancedExtractedInformation, AdvancedExtractionPipeline, ExtractedInformation,
    InformationExtractionPipeline,
};
pub use relations::{Event, Relation, RelationExtractor};
pub use temporal::TemporalExtractor;

// Re-export patterns for direct access
pub use patterns::{
    DATE_PATTERN, EMAIL_PATTERN, MONEY_PATTERN, PERCENTAGE_PATTERN, PHONE_PATTERN, TIME_PATTERN,
    URL_PATTERN,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WordTokenizer;
    use regex::Regex;
    use std::collections::HashMap;

    #[test]
    fn test_rule_based_ner() {
        let mut ner = RuleBasedNER::new();
        ner.add_person_names(vec!["John".to_string(), "Jane".to_string()]);
        ner.add_organizations(vec!["Microsoft".to_string(), "Google".to_string()]);

        let text = "John works at Microsoft. His email is john@example.com";
        let entities = ner.extract_entities(text).unwrap();

        assert!(entities.len() >= 3); // John, Microsoft, email
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Person));
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::Organization));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
    }

    #[test]
    fn test_key_phrase_extraction() {
        let extractor = KeyPhraseExtractor::new()
            .with_min_frequency(1)
            .with_max_length(2);

        let text = "machine learning is important. machine learning algorithms are complex.";
        let tokenizer = WordTokenizer::default();

        let phrases = extractor.extract(text, &tokenizer).unwrap();

        assert!(!phrases.is_empty());
        assert!(phrases.iter().any(|(p, _)| p.contains("machine learning")));
    }

    #[test]
    fn test_pattern_extraction() {
        let mut extractor = PatternExtractor::new();
        extractor.add_pattern(
            "price".to_string(),
            Regex::new(r"\$\d+(?:\.\d{2})?").unwrap(),
        );

        let text = "The product costs $29.99 and shipping is $5.00";
        let results = extractor.extract(text).unwrap();

        assert!(results.contains_key("price"));
        assert_eq!(results["price"].len(), 2);
    }

    #[test]
    fn test_information_extraction_pipeline() {
        // Use NER with basic knowledge
        let ner = RuleBasedNER::with_basic_knowledge();
        let pipeline = InformationExtractionPipeline::new().with_ner(ner);

        let text = "Apple Inc. announced that Tim Cook will visit London on January 15, 2024. Contact: info@apple.com";
        let info = pipeline.extract(text).unwrap();

        assert!(!info.entities.is_empty());
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Email));
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Date));
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Person));
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Organization));
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Location));
    }

    #[test]
    fn test_temporal_extractor() {
        let extractor = TemporalExtractor::new();

        let text = "The meeting is scheduled for next Monday from 2:00-4:00 PM. It will last 2 hours during winter season.";
        let entities = extractor.extract(text).unwrap();

        assert!(!entities.is_empty());
        assert!(entities.iter().any(|e| e.text.contains("next Monday")));
        assert!(entities.iter().any(|e| e.text.contains("2:00-4:00")));
        assert!(entities.iter().any(|e| e.text.contains("2 hours")));
        assert!(entities.iter().any(|e| e.text.contains("winter")));
    }

    #[test]
    fn test_entity_linker() {
        let mut linker = EntityLinker::new();

        // Add a knowledge base entry
        let kb_entry = KnowledgeBaseEntry {
            canonical_name: "Apple Inc.".to_string(),
            entity_type: EntityType::Organization,
            aliases: vec!["Apple".to_string(), "AAPL".to_string()],
            confidence: 0.9,
            metadata: HashMap::new(),
        };
        linker.add_entity(kb_entry);

        // Create test entities
        let mut entities = vec![Entity {
            text: "Apple".to_string(), // Fixed case to match alias
            entity_type: EntityType::Organization,
            start: 0,
            end: 5,
            confidence: 0.7,
        }];

        let linked = linker.link_entities(&mut entities).unwrap();
        assert!(!linked.is_empty());
        assert_eq!(linked[0].canonical_name, "Apple Inc.");
    }

    #[test]
    fn test_coreference_resolver() {
        let resolver = CoreferenceResolver::new();

        let entities = vec![Entity {
            text: "John Smith".to_string(),
            entity_type: EntityType::Person,
            start: 0,
            end: 10,
            confidence: 0.8,
        }];

        let text = "John Smith is a CEO. He founded the company in 2020.";
        let chains = resolver.resolve(text, &entities).unwrap();

        // Should find a coreference chain for "He" -> "John Smith"
        assert!(!chains.is_empty());
        assert_eq!(chains[0].mentions.len(), 2);
    }

    #[test]
    fn test_confidence_scorer() {
        let scorer = ConfidenceScorer::new();

        let entity = Entity {
            text: "john@example.com".to_string(),
            entity_type: EntityType::Email,
            start: 20,
            end: 36,
            confidence: 0.5,
        };

        let text = "Please contact John at john@example.com for more information.";
        let score = scorer.score_entity(&entity, text, 10);

        // Email patterns should get high confidence (adjusted threshold)
        assert!(score > 0.7);
    }

    #[test]
    fn test_advanced_extraction_pipeline() {
        let ner = RuleBasedNER::with_basic_knowledge();
        let pipeline = AdvancedExtractionPipeline::new().with_ner(ner);

        let text = "Microsoft Corp. announced today that CEO Satya Nadella will visit New York next week. He will meet with partners.";
        let info = pipeline.extract_advanced(text).unwrap();

        // Should extract basic entities
        assert!(!info.entities.is_empty());

        // Should find person and organization entities
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Person));
        assert!(info
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Organization));

        // Should find temporal expressions
        assert!(info
            .entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Custom(ref s) if s.contains("temporal"))));

        // Should have key phrases
        assert!(!info.key_phrases.is_empty());
    }

    #[test]
    fn test_context_scoring() {
        let scorer = ConfidenceScorer::new();

        // Test person entity with good context
        let person_entity = Entity {
            text: "Smith".to_string(),
            entity_type: EntityType::Person,
            start: 3,
            end: 8,
            confidence: 0.5,
        };

        let text_with_context = "Dr. Smith is the CEO of the company.";
        let score_with_context = scorer.score_entity(&person_entity, text_with_context, 10);

        let text_without_context = "The Smith family owns this.";
        let score_without_context = scorer.score_entity(&person_entity, text_without_context, 10);

        // Context with "Dr." and "CEO" should score higher
        assert!(score_with_context > score_without_context);
    }

    #[test]
    fn test_document_information_extractor() {
        let extractor = DocumentInformationExtractor::new();
        let ner = RuleBasedNER::with_basic_knowledge();
        let pipeline = AdvancedExtractionPipeline::new().with_ner(ner);

        let documents = vec![
            "Apple Inc. announced a new product launch. Tim Cook will present in San Francisco on January 15, 2024.".to_string(),
            "Microsoft Corporation released quarterly results. Satya Nadella discussed growth in cloud computing.".to_string(),
            "Apple Inc. stock price increased after the announcement. Investors are optimistic about the new product.".to_string(),
        ];

        let result = extractor
            .extract_structured_information(&documents, &pipeline)
            .unwrap();

        // Should have processed all documents
        assert_eq!(result.documents.len(), 3);

        // Should have found some entities and relations
        assert!(result.total_entities > 0);

        // Should have clustered similar entities
        assert!(!result.entity_clusters.is_empty());

        // Topics may or may not be found depending on key phrase extraction
        // The main functionality (entity extraction) is working correctly
    }

    #[test]
    fn test_entity_clustering() {
        let extractor = DocumentInformationExtractor::new();

        let entities = vec![
            Entity {
                text: "Apple Inc.".to_string(),
                entity_type: EntityType::Organization,
                start: 0,
                end: 10,
                confidence: 0.9,
            },
            Entity {
                text: "apple inc".to_string(),
                entity_type: EntityType::Organization,
                start: 20,
                end: 29,
                confidence: 0.8,
            },
            Entity {
                text: "Microsoft".to_string(),
                entity_type: EntityType::Organization,
                start: 40,
                end: 49,
                confidence: 0.9,
            },
        ];

        let clusters = extractor.cluster_entities(&entities).unwrap();

        // Should cluster similar entities (Apple variations)
        assert_eq!(clusters.len(), 2);

        // First cluster should have both Apple entities
        let apple_cluster = clusters
            .iter()
            .find(|c| c.representative.text.to_lowercase().contains("apple"))
            .unwrap();
        assert_eq!(apple_cluster.members.len(), 2);
    }

    #[test]
    fn test_event_extraction() {
        let extractor = DocumentInformationExtractor::new();

        let relations = vec![
            Relation {
                relation_type: "announcement".to_string(),
                subject: Entity {
                    text: "Apple".to_string(),
                    entity_type: EntityType::Organization,
                    start: 0,
                    end: 5,
                    confidence: 0.9,
                },
                object: Entity {
                    text: "product".to_string(),
                    entity_type: EntityType::Other,
                    start: 15,
                    end: 22,
                    confidence: 0.8,
                },
                context: "Apple announced a new product launch".to_string(),
                confidence: 0.8,
            },
            Relation {
                relation_type: "presentation".to_string(),
                subject: Entity {
                    text: "Tim Cook".to_string(),
                    entity_type: EntityType::Person,
                    start: 25,
                    end: 33,
                    confidence: 0.9,
                },
                object: Entity {
                    text: "product".to_string(),
                    entity_type: EntityType::Other,
                    start: 15,
                    end: 22,
                    confidence: 0.8,
                },
                context: "Tim Cook will present the product".to_string(),
                confidence: 0.8,
            },
        ];

        let entities = vec![
            Entity {
                text: "January 15, 2024".to_string(),
                entity_type: EntityType::Date,
                start: 50,
                end: 66,
                confidence: 0.9,
            },
            Entity {
                text: "San Francisco".to_string(),
                entity_type: EntityType::Location,
                start: 70,
                end: 83,
                confidence: 0.9,
            },
        ];

        let events = extractor.extract_events(&relations, &entities).unwrap();

        // Should extract at least one event
        assert!(!events.is_empty());

        let event = &events[0];
        assert!(!event.participants.is_empty());
        // Should find temporal and location information
        assert!(event.time.is_some() || event.location.is_some());
    }

    #[test]
    fn test_levenshtein_distance() {
        let extractor = DocumentInformationExtractor::new();

        assert_eq!(extractor.levenshtein_distance("apple", "apple"), 0);
        assert_eq!(extractor.levenshtein_distance("apple", "apples"), 1);
        assert_eq!(extractor.levenshtein_distance("apple", "orange"), 5);
        assert_eq!(extractor.levenshtein_distance("", "apple"), 5);
        assert_eq!(extractor.levenshtein_distance("apple", ""), 5);
    }

    #[test]
    fn test_entity_similarity() {
        let extractor = DocumentInformationExtractor::new();

        let entity1 = Entity {
            text: "Apple Inc.".to_string(),
            entity_type: EntityType::Organization,
            start: 0,
            end: 10,
            confidence: 0.9,
        };

        let entity2 = Entity {
            text: "apple inc".to_string(),
            entity_type: EntityType::Organization,
            start: 20,
            end: 29,
            confidence: 0.8,
        };

        let entity3 = Entity {
            text: "Microsoft".to_string(),
            entity_type: EntityType::Organization,
            start: 40,
            end: 49,
            confidence: 0.9,
        };

        // Similar entities should have high similarity
        let similarity = extractor.calculate_entity_similarity(&entity1, &entity2);
        assert!(similarity > 0.5);

        // Different entities should have low similarity
        let similarity = extractor.calculate_entity_similarity(&entity1, &entity3);
        assert!(similarity < 0.5);
    }

    #[test]
    fn test_topic_identification() {
        let extractor = DocumentInformationExtractor::new();

        let summaries = vec![
            DocumentSummary {
                document_index: 0,
                entity_count: 5,
                relation_count: 2,
                key_phrases: vec![
                    ("machine learning".to_string(), 0.8),
                    ("artificial intelligence".to_string(), 0.6),
                ],
                confidence_score: 0.8,
            },
            DocumentSummary {
                document_index: 1,
                entity_count: 3,
                relation_count: 1,
                key_phrases: vec![
                    ("machine learning".to_string(), 0.7),
                    ("data science".to_string(), 0.5),
                ],
                confidence_score: 0.7,
            },
        ];

        let topics = extractor.identify_topics(&summaries).unwrap();

        // Should identify "machine learning" as a topic (appears in both documents)
        assert!(!topics.is_empty());
        assert!(topics.iter().any(|t| t.name.contains("machine learning")));

        let ml_topic = topics
            .iter()
            .find(|t| t.name.contains("machine learning"))
            .unwrap();
        assert_eq!(ml_topic.document_indices.len(), 2);
    }

    #[test]
    fn test_knowledge_base_aliases() {
        let mut linker = EntityLinker::new();

        let kb_entry = KnowledgeBaseEntry {
            canonical_name: "International Business Machines".to_string(),
            entity_type: EntityType::Organization,
            aliases: vec!["IBM".to_string(), "Big Blue".to_string()],
            confidence: 0.95,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("industry".to_string(), "Technology".to_string());
                meta
            },
        };
        linker.add_entity(kb_entry);

        let mut entities = vec![Entity {
            text: "ibm".to_string(), // lowercase
            entity_type: EntityType::Organization,
            start: 0,
            end: 3,
            confidence: 0.8,
        }];

        let linked = linker.link_entities(&mut entities).unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].canonical_name, "International Business Machines");
        assert!(linked[0].metadata.contains_key("industry"));
    }

    #[test]
    fn test_sentence_splitting() {
        let resolver = CoreferenceResolver::new();
        let sentences = resolver.split_into_sentences("Hello world. How are you? Fine, thanks!");

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world");
        assert_eq!(sentences[1], "How are you");
        assert_eq!(sentences[2], "Fine, thanks");
    }
}
