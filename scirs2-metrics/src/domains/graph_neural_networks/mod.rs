//! Graph Neural Network evaluation metrics
//!
//! This module provides specialized metrics for evaluating Graph Neural Networks (GNNs)
//! across various graph learning tasks including:
//! - Node classification and regression
//! - Edge prediction and link prediction
//! - Graph classification and regression
//! - Community detection and clustering
//! - Graph generation and reconstruction
//! - Knowledge graph completion
//! - Social network analysis
//! - Molecular property prediction

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

// Module declarations
pub mod core;
pub mod node_level;
pub mod edge_level;
pub mod graph_level;
pub mod community_detection;
pub mod graph_generation;
pub mod knowledge_graphs;
pub mod social_networks;
pub mod molecular_graphs;

// Re-export core types
pub use core::*;

// Re-export from node level module
pub use node_level::{
    NodeLevelMetrics, NodeClassificationMetrics, NodeEmbeddingMetrics,
    HomophilyAwareMetrics, NodeFairnessMetrics,
};

// Re-export from edge level module
pub use edge_level::{
    EdgeLevelMetrics, LinkPredictionMetrics, EdgeClassificationMetrics,
    EdgeRegressionMetrics, TemporalEdgeMetrics,
};

// Re-export from graph level module
pub use graph_level::{
    GraphLevelMetrics, GraphClassificationMetrics, GraphRegressionMetrics,
    GraphPropertyMetrics, GraphSimilarityMetrics,
};

// Re-export from community detection module
pub use community_detection::{
    CommunityDetectionMetrics, OverlappingCommunityMetrics,
};

// Re-export from graph generation module
pub use graph_generation::{
    GraphGenerationMetrics, StructuralSimilarityMetrics, StatisticalSimilarityMetrics,
    SpectralSimilarityMetrics, GenerationDiversityMetrics,
};

// Re-export from knowledge graphs module
pub use knowledge_graphs::{
    KnowledgeGraphMetrics, TripleClassificationMetrics, KgLinkPredictionMetrics,
    EntityAlignmentMetrics, RelationExtractionMetrics,
};

// Re-export from social networks module
pub use social_networks::{
    SocialNetworkMetrics, InfluencePredictionMetrics, SocialRoleMetrics,
    SocialRecommendationMetrics, InformationDiffusionMetrics,
};

// Re-export from molecular graphs module
pub use molecular_graphs::{
    MolecularGraphMetrics, MolecularPropertyMetrics, PropertyMetrics,
    DrugDiscoveryMetrics, ToxicityMetrics, DtiPredictionMetrics,
    ChemicalSimilarityMetrics, ReactionPredictionMetrics,
};

/// Comprehensive Graph Neural Network metrics suite
#[derive(Debug)]
pub struct GraphNeuralNetworkMetrics {
    /// Node-level task metrics
    pub node_metrics: NodeLevelMetrics,
    /// Edge-level task metrics
    pub edge_metrics: EdgeLevelMetrics,
    /// Graph-level task metrics
    pub graph_metrics: GraphLevelMetrics,
    /// Community detection metrics
    pub community_metrics: CommunityDetectionMetrics,
    /// Graph generation metrics
    pub generation_metrics: GraphGenerationMetrics,
    /// Knowledge graph metrics
    pub knowledge_graph_metrics: KnowledgeGraphMetrics,
    /// Social network metrics
    pub social_network_metrics: SocialNetworkMetrics,
    /// Molecular graph metrics
    pub molecular_metrics: MolecularGraphMetrics,
}

impl GraphNeuralNetworkMetrics {
    /// Create new GNN metrics
    pub fn new() -> Self {
        Self {
            node_metrics: NodeLevelMetrics::new(),
            edge_metrics: EdgeLevelMetrics::new(),
            graph_metrics: GraphLevelMetrics::new(),
            community_metrics: CommunityDetectionMetrics::new(),
            generation_metrics: GraphGenerationMetrics::new(),
            knowledge_graph_metrics: KnowledgeGraphMetrics::new(),
            social_network_metrics: SocialNetworkMetrics::new(),
            molecular_metrics: MolecularGraphMetrics::new(),
        }
    }
}

impl Default for GraphNeuralNetworkMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph Neural Network evaluation computer
pub struct GraphNeuralNetworkMetricsComputer {
    config: GnnEvaluationConfig,
}

/// Configuration for GNN evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnEvaluationConfig {
    /// Enable node-level evaluation
    pub enable_node_tasks: bool,
    /// Enable edge-level evaluation
    pub enable_edge_tasks: bool,
    /// Enable graph-level evaluation
    pub enable_graph_tasks: bool,
    /// Enable community detection evaluation
    pub enable_community_detection: bool,
    /// Enable graph generation evaluation
    pub enable_graph_generation: bool,
    /// Enable knowledge graph evaluation
    pub enable_knowledge_graphs: bool,
    /// Enable social network evaluation
    pub enable_social_networks: bool,
    /// Enable molecular graph evaluation
    pub enable_molecular_graphs: bool,
    /// Task-specific parameters
    pub task_parameters: HashMap<String, f64>,
}

impl Default for GnnEvaluationConfig {
    fn default() -> Self {
        Self {
            enable_node_tasks: true,
            enable_edge_tasks: true,
            enable_graph_tasks: true,
            enable_community_detection: false,
            enable_graph_generation: false,
            enable_knowledge_graphs: false,
            enable_social_networks: false,
            enable_molecular_graphs: false,
            task_parameters: HashMap::new(),
        }
    }
}

impl GraphNeuralNetworkMetricsComputer {
    /// Create new GNN metrics computer
    pub fn new(config: GnnEvaluationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(GnnEvaluationConfig::default())
    }

    /// Compute comprehensive GNN evaluation
    pub fn compute_metrics<F: Float + 'static>(
        &mut self,
        predicted: &ArrayView2<F>,
        actual: &ArrayView2<F>,
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<GraphNeuralNetworkMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let mut metrics = GraphNeuralNetworkMetrics::new();

        // Node-level evaluation
        if self.config.enable_node_tasks {
            metrics.node_metrics = self.evaluate_node_tasks(predicted, actual, metadata)?;
        }

        // Edge-level evaluation
        if self.config.enable_edge_tasks {
            metrics.edge_metrics = self.evaluate_edge_tasks(predicted, actual, metadata)?;
        }

        // Graph-level evaluation
        if self.config.enable_graph_tasks {
            metrics.graph_metrics = self.evaluate_graph_tasks(predicted, actual, metadata)?;
        }

        // Community detection evaluation
        if self.config.enable_community_detection {
            metrics.community_metrics = self.evaluate_community_detection(predicted, actual, metadata)?;
        }

        // Graph generation evaluation
        if self.config.enable_graph_generation {
            metrics.generation_metrics = self.evaluate_graph_generation(predicted, actual, metadata)?;
        }

        // Knowledge graph evaluation
        if self.config.enable_knowledge_graphs {
            metrics.knowledge_graph_metrics = self.evaluate_knowledge_graphs(predicted, actual, metadata)?;
        }

        // Social network evaluation
        if self.config.enable_social_networks {
            metrics.social_network_metrics = self.evaluate_social_networks(predicted, actual, metadata)?;
        }

        // Molecular graph evaluation
        if self.config.enable_molecular_graphs {
            metrics.molecular_metrics = self.evaluate_molecular_graphs(predicted, actual, metadata)?;
        }

        Ok(metrics)
    }

    fn evaluate_node_tasks<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<NodeLevelMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(NodeLevelMetrics::new())
    }

    fn evaluate_edge_tasks<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<EdgeLevelMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(EdgeLevelMetrics::new())
    }

    fn evaluate_graph_tasks<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<GraphLevelMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(GraphLevelMetrics::new())
    }

    fn evaluate_community_detection<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<CommunityDetectionMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(CommunityDetectionMetrics::new())
    }

    fn evaluate_graph_generation<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<GraphGenerationMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(GraphGenerationMetrics::new())
    }

    fn evaluate_knowledge_graphs<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<KnowledgeGraphMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(KnowledgeGraphMetrics::new())
    }

    fn evaluate_social_networks<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<SocialNetworkMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(SocialNetworkMetrics::new())
    }

    fn evaluate_molecular_graphs<F: Float + 'static>(
        &self,
        _predicted: &ArrayView2<F>,
        _actual: &ArrayView2<F>,
        _metadata: Option<&HashMap<String, String>>,
    ) -> Result<MolecularGraphMetrics>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Implementation would go here
        Ok(MolecularGraphMetrics::new())
    }
}

impl DomainMetrics for GraphNeuralNetworkMetrics {
    fn domain_name(&self) -> &'static str {
        "graph_neural_networks"
    }

    fn primary_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Node-level metrics
        metrics.insert("node_classification_f1".to_string(), self.node_metrics.classification_metrics.macro_f1);
        metrics.insert("node_embedding_quality".to_string(), self.node_metrics.embedding_metrics.silhouette_score);

        // Edge-level metrics
        metrics.insert("link_prediction_auc".to_string(), self.edge_metrics.link_prediction.auc_roc);
        metrics.insert("edge_classification_f1".to_string(), self.edge_metrics.edge_classification.macro_f1);

        // Graph-level metrics
        metrics.insert("graph_classification_accuracy".to_string(), self.graph_metrics.classification.accuracy);
        metrics.insert("graph_regression_r2".to_string(), self.graph_metrics.regression.r2_score);

        // Community detection metrics
        metrics.insert("community_modularity".to_string(), self.community_metrics.modularity);
        metrics.insert("community_nmi".to_string(), self.community_metrics.nmi);

        // Knowledge graph metrics
        metrics.insert("kg_triple_classification_f1".to_string(), self.knowledge_graph_metrics.triple_classification.f1_score);
        metrics.insert("kg_link_prediction_mrr".to_string(), self.knowledge_graph_metrics.kg_link_prediction.head_prediction.mrr);

        // Molecular metrics
        metrics.insert("molecular_property_r2".to_string(), self.molecular_metrics.property_prediction.overall_r2);
        metrics.insert("drug_discovery_auc".to_string(), self.molecular_metrics.drug_discovery.bioactivity_auc);

        metrics
    }

    fn secondary_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Additional node-level metrics
        metrics.insert("node_homophily_ratio".to_string(), self.node_metrics.homophily_metrics.homophily_ratio);
        metrics.insert("node_fairness_demographic_parity".to_string(), self.node_metrics.fairness_metrics.demographic_parity);

        // Additional edge-level metrics
        metrics.insert("link_prediction_precision".to_string(),
                      self.edge_metrics.link_prediction.precision_at_k.get(&10).unwrap_or(&0.0).clone());
        metrics.insert("temporal_edge_accuracy".to_string(), self.edge_metrics.temporal_metrics.persistence_accuracy);

        // Additional graph-level metrics
        metrics.insert("graph_similarity_ged".to_string(), self.graph_metrics.similarity_metrics.ged_correlation);

        // Additional community metrics
        metrics.insert("community_ari".to_string(), self.community_metrics.ari);
        metrics.insert("community_conductance".to_string(), self.community_metrics.conductance);

        // Additional generation metrics
        metrics.insert("generation_structural_similarity".to_string(),
                      self.generation_metrics.structural_similarity.degree_distribution_kl);
        metrics.insert("generation_diversity".to_string(), self.generation_metrics.diversity_metrics.structural_diversity);

        // Additional social network metrics
        metrics.insert("social_influence_correlation".to_string(), self.social_network_metrics.influence_prediction.ranking_correlation);
        metrics.insert("social_role_accuracy".to_string(), self.social_network_metrics.social_role.role_accuracy);

        // Additional molecular metrics
        metrics.insert("molecular_toxicity_auc".to_string(), self.molecular_metrics.toxicity_metrics.acute_toxicity_accuracy);
        metrics.insert("dti_prediction_auc".to_string(), self.molecular_metrics.dti_prediction.dti_auc);

        metrics
    }

    fn evaluation_summary(&self) -> DomainEvaluationResult {
        let primary = self.primary_metrics();
        let secondary = self.secondary_metrics();

        // Calculate overall performance score
        let node_score = (primary.get("node_classification_f1").unwrap_or(&0.0) +
                         primary.get("node_embedding_quality").unwrap_or(&0.0)) / 2.0;

        let edge_score = (primary.get("link_prediction_auc").unwrap_or(&0.0) +
                         primary.get("edge_classification_f1").unwrap_or(&0.0)) / 2.0;

        let graph_score = (primary.get("graph_classification_accuracy").unwrap_or(&0.0) +
                          primary.get("graph_regression_r2").unwrap_or(&0.0)) / 2.0;

        let community_score = (primary.get("community_modularity").unwrap_or(&0.0) +
                              primary.get("community_nmi").unwrap_or(&0.0)) / 2.0;

        let kg_score = (primary.get("kg_triple_classification_f1").unwrap_or(&0.0) +
                       primary.get("kg_link_prediction_mrr").unwrap_or(&0.0)) / 2.0;

        let molecular_score = (primary.get("molecular_property_r2").unwrap_or(&0.0) +
                              primary.get("drug_discovery_auc").unwrap_or(&0.0)) / 2.0;

        let overall_score = (node_score + edge_score + graph_score + community_score + kg_score + molecular_score) / 6.0;

        // Determine performance level
        let performance_level = if overall_score >= 0.9 {
            "Excellent"
        } else if overall_score >= 0.8 {
            "Good"
        } else if overall_score >= 0.7 {
            "Fair"
        } else if overall_score >= 0.6 {
            "Poor"
        } else {
            "Critical"
        };

        let mut recommendations = Vec::new();

        if node_score < 0.8 {
            recommendations.push("Improve node classification accuracy and embedding quality".to_string());
        }
        if edge_score < 0.8 {
            recommendations.push("Enhance link prediction and edge classification performance".to_string());
        }
        if graph_score < 0.8 {
            recommendations.push("Optimize graph-level task performance".to_string());
        }
        if community_score < 0.7 {
            recommendations.push("Improve community detection algorithms".to_string());
        }
        if kg_score < 0.7 {
            recommendations.push("Enhance knowledge graph reasoning capabilities".to_string());
        }
        if molecular_score < 0.7 {
            recommendations.push("Optimize molecular property prediction models".to_string());
        }

        DomainEvaluationResult {
            domain: "graph_neural_networks".to_string(),
            overall_score,
            performance_level: performance_level.to_string(),
            primary_metrics: primary,
            secondary_metrics: secondary,
            recommendations,
            confidence_interval: (overall_score - 0.05, overall_score + 0.05),
        }
    }
}