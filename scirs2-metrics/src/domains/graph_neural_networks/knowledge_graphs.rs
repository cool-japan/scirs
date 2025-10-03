//! Knowledge graph evaluation metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating knowledge graph completion,
//! entity alignment, and relation extraction tasks.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{RankingMetrics, Triple};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Knowledge graph completion metrics
#[derive(Debug, Clone)]
pub struct KnowledgeGraphMetrics {
    /// Triple classification metrics
    pub triple_classification: TripleClassificationMetrics,
    /// Link prediction metrics for KG
    pub kg_link_prediction: KgLinkPredictionMetrics,
    /// Entity alignment metrics
    pub entity_alignment: EntityAlignmentMetrics,
    /// Relation extraction metrics
    pub relation_extraction: RelationExtractionMetrics,
}

/// Triple classification evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleClassificationMetrics {
    /// Accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// AUC score
    pub auc: f64,
}

impl Default for TripleClassificationMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            auc: 0.0,
        }
    }
}

impl TripleClassificationMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Knowledge graph link prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KgLinkPredictionMetrics {
    /// Head entity prediction metrics
    pub head_prediction: RankingMetrics,
    /// Tail entity prediction metrics
    pub tail_prediction: RankingMetrics,
    /// Relation prediction metrics
    pub relation_prediction: RankingMetrics,
}

impl KgLinkPredictionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Entity alignment evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAlignmentMetrics {
    /// Hits@1 for entity alignment
    pub hits_at_1: f64,
    /// Hits@5 for entity alignment
    pub hits_at_5: f64,
    /// Hits@10 for entity alignment
    pub hits_at_10: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Mean Rank
    pub mean_rank: f64,
}

impl Default for EntityAlignmentMetrics {
    fn default() -> Self {
        Self {
            hits_at_1: 0.0,
            hits_at_5: 0.0,
            hits_at_10: 0.0,
            mrr: 0.0,
            mean_rank: 0.0,
        }
    }
}

impl EntityAlignmentMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Relation extraction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationExtractionMetrics {
    /// Precision for relation extraction
    pub precision: f64,
    /// Recall for relation extraction
    pub recall: f64,
    /// F1 score for relation extraction
    pub f1_score: f64,
    /// Per-relation performance
    pub per_relation_metrics: HashMap<String, (f64, f64, f64)>, // (precision, recall, f1)
}

impl Default for RelationExtractionMetrics {
    fn default() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            per_relation_metrics: HashMap::new(),
        }
    }
}

impl RelationExtractionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KnowledgeGraphMetrics {
    pub fn new() -> Self {
        Self {
            triple_classification: TripleClassificationMetrics::new(),
            kg_link_prediction: KgLinkPredictionMetrics::new(),
            entity_alignment: EntityAlignmentMetrics::new(),
            relation_extraction: RelationExtractionMetrics::new(),
        }
    }

    /// Evaluate triple classification performance
    pub fn evaluate_triple_classification(
        &mut self,
        predicted_scores: &[(Triple, f64)],
        positive_triples: &[Triple],
        negative_triples: &[Triple],
    ) -> Result<()> {
        if predicted_scores.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Predicted scores cannot be empty".to_string(),
            ));
        }

        let positive_set: HashSet<_> = positive_triples.iter().collect();
        let negative_set: HashSet<_> = negative_triples.iter().collect();

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for (triple, score) in predicted_scores {
            let predicted_positive = *score > 0.5;

            if positive_set.contains(triple) {
                if predicted_positive {
                    tp += 1;
                } else {
                    fn_count += 1;
                }
            } else if negative_set.contains(triple) {
                if predicted_positive {
                    fp += 1;
                } else {
                    tn += 1;
                }
            }
        }

        // Calculate metrics
        self.triple_classification.accuracy = if tp + fp + tn + fn_count > 0 {
            (tp + tn) as f64 / (tp + fp + tn + fn_count) as f64
        } else {
            0.0
        };

        self.triple_classification.precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        self.triple_classification.recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        self.triple_classification.f1_score =
            if self.triple_classification.precision + self.triple_classification.recall > 0.0 {
                2.0 * self.triple_classification.precision * self.triple_classification.recall
                    / (self.triple_classification.precision + self.triple_classification.recall)
            } else {
                0.0
            };

        Ok(())
    }

    /// Evaluate entity alignment performance
    pub fn evaluate_entity_alignment(
        &mut self,
        entity_similarities: &[(String, String, f64)], // (entity1, entity2, similarity)
        true_alignments: &[(String, String)],
    ) -> Result<()> {
        if entity_similarities.is_empty() || true_alignments.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let alignment_map: HashMap<String, String> = true_alignments
            .iter()
            .map(|(e1, e2)| (e1.clone(), e2.clone()))
            .collect();

        // Group similarities by first entity
        let mut entity_rankings: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (e1, e2, sim) in entity_similarities {
            entity_rankings
                .entry(e1.clone())
                .or_default()
                .push((e2.clone(), *sim));
        }

        // Sort by similarity descending
        for rankings in entity_rankings.values_mut() {
            rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut hits_1 = 0;
        let mut hits_5 = 0;
        let mut hits_10 = 0;
        let mut reciprocal_ranks = 0.0;
        let mut total_ranks = 0.0;
        let mut valid_entities = 0;

        for (source_entity, true_target) in &alignment_map {
            if let Some(rankings) = entity_rankings.get(source_entity) {
                if let Some(rank) = rankings
                    .iter()
                    .position(|(target, _)| target == true_target)
                {
                    let rank_1_based = rank + 1;

                    if rank_1_based <= 1 {
                        hits_1 += 1;
                    }
                    if rank_1_based <= 5 {
                        hits_5 += 1;
                    }
                    if rank_1_based <= 10 {
                        hits_10 += 1;
                    }

                    reciprocal_ranks += 1.0 / rank_1_based as f64;
                    total_ranks += rank_1_based as f64;
                    valid_entities += 1;
                }
            }
        }

        if valid_entities > 0 {
            self.entity_alignment.hits_at_1 = hits_1 as f64 / valid_entities as f64;
            self.entity_alignment.hits_at_5 = hits_5 as f64 / valid_entities as f64;
            self.entity_alignment.hits_at_10 = hits_10 as f64 / valid_entities as f64;
            self.entity_alignment.mrr = reciprocal_ranks / valid_entities as f64;
            self.entity_alignment.mean_rank = total_ranks / valid_entities as f64;
        }

        Ok(())
    }

    /// Evaluate relation extraction performance
    pub fn evaluate_relation_extraction(
        &mut self,
        predicted_relations: &[(String, String, String)], // (head, relation, tail)
        true_relations: &[(String, String, String)],
    ) -> Result<()> {
        if predicted_relations.is_empty() && true_relations.is_empty() {
            return Ok(());
        }

        let predicted_set: HashSet<_> = predicted_relations.iter().collect();
        let true_set: HashSet<_> = true_relations.iter().collect();

        let tp = predicted_set.intersection(&true_set).count();
        let fp = predicted_set.len() - tp;
        let fn_count = true_set.len() - tp;

        // Overall metrics
        self.relation_extraction.precision = if predicted_set.len() > 0 {
            tp as f64 / predicted_set.len() as f64
        } else {
            0.0
        };

        self.relation_extraction.recall = if true_set.len() > 0 {
            tp as f64 / true_set.len() as f64
        } else {
            0.0
        };

        self.relation_extraction.f1_score =
            if self.relation_extraction.precision + self.relation_extraction.recall > 0.0 {
                2.0 * self.relation_extraction.precision * self.relation_extraction.recall
                    / (self.relation_extraction.precision + self.relation_extraction.recall)
            } else {
                0.0
            };

        // Per-relation metrics
        let all_relations: HashSet<String> = predicted_relations
            .iter()
            .chain(true_relations.iter())
            .map(|(_, rel, _)| rel.clone())
            .collect();

        for relation in all_relations {
            let pred_for_rel: HashSet<_> = predicted_relations
                .iter()
                .filter(|(_, r, _)| r == &relation)
                .collect();
            let true_for_rel: HashSet<_> = true_relations
                .iter()
                .filter(|(_, r, _)| r == &relation)
                .collect();

            let tp_rel = pred_for_rel.intersection(&true_for_rel).count();
            let fp_rel = pred_for_rel.len() - tp_rel;
            let fn_rel = true_for_rel.len() - tp_rel;

            let precision_rel = if pred_for_rel.len() > 0 {
                tp_rel as f64 / pred_for_rel.len() as f64
            } else {
                0.0
            };

            let recall_rel = if true_for_rel.len() > 0 {
                tp_rel as f64 / true_for_rel.len() as f64
            } else {
                0.0
            };

            let f1_rel = if precision_rel + recall_rel > 0.0 {
                2.0 * precision_rel * recall_rel / (precision_rel + recall_rel)
            } else {
                0.0
            };

            self.relation_extraction
                .per_relation_metrics
                .insert(relation, (precision_rel, recall_rel, f1_rel));
        }

        Ok(())
    }
}

impl Default for KnowledgeGraphMetrics {
    fn default() -> Self {
        Self::new()
    }
}
