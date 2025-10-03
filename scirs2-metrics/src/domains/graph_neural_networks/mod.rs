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
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

// Module declarations
pub mod community_detection;
pub mod core;
pub mod edge_level;
pub mod graph_generation;
pub mod graph_level;
pub mod knowledge_graphs;
pub mod molecular_graphs;
pub mod node_level;
pub mod social_networks;

// Re-export core types
pub use core::*;

// Re-export from node level module
pub use node_level::{
    HomophilyAwareMetrics, NodeClassificationMetrics, NodeEmbeddingMetrics, NodeFairnessMetrics,
    NodeLevelMetrics,
};

// Re-export from edge level module
pub use edge_level::{
    EdgeClassificationMetrics, EdgeLevelMetrics, EdgeRegressionMetrics, LinkPredictionMetrics,
    TemporalEdgeMetrics,
};

// Re-export from graph level module
pub use graph_level::{
    GraphClassificationMetrics, GraphLevelMetrics, GraphPropertyMetrics, GraphRegressionMetrics,
    GraphSimilarityMetrics,
};

// Re-export from community detection module
pub use community_detection::{CommunityDetectionMetrics, OverlappingCommunityMetrics};

// Re-export from graph generation module
pub use graph_generation::{
    GenerationDiversityMetrics, GraphGenerationMetrics, SpectralSimilarityMetrics,
    StatisticalSimilarityMetrics, StructuralSimilarityMetrics,
};

// Re-export from knowledge graphs module
pub use knowledge_graphs::{
    EntityAlignmentMetrics, KgLinkPredictionMetrics, KnowledgeGraphMetrics,
    RelationExtractionMetrics, TripleClassificationMetrics,
};

// Re-export from social networks module
pub use social_networks::{
    InfluencePredictionMetrics, InformationDiffusionMetrics, SocialNetworkMetrics,
    SocialRecommendationMetrics, SocialRoleMetrics,
};

// Re-export from molecular graphs module
pub use molecular_graphs::{
    ChemicalSimilarityMetrics, DrugDiscoveryMetrics, DtiPredictionMetrics, MolecularGraphMetrics,
    MolecularPropertyMetrics, PropertyMetrics, ReactionPredictionMetrics, ToxicityMetrics,
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
            metrics.community_metrics =
                self.evaluate_community_detection(predicted, actual, metadata)?;
        }

        // Graph generation evaluation
        if self.config.enable_graph_generation {
            metrics.generation_metrics =
                self.evaluate_graph_generation(predicted, actual, metadata)?;
        }

        // Knowledge graph evaluation
        if self.config.enable_knowledge_graphs {
            metrics.knowledge_graph_metrics =
                self.evaluate_knowledge_graphs(predicted, actual, metadata)?;
        }

        // Social network evaluation
        if self.config.enable_social_networks {
            metrics.social_network_metrics =
                self.evaluate_social_networks(predicted, actual, metadata)?;
        }

        // Molecular graph evaluation
        if self.config.enable_molecular_graphs {
            metrics.molecular_metrics =
                self.evaluate_molecular_graphs(predicted, actual, metadata)?;
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
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Graph Neural Networks"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "node_classification_accuracy",
            "node_classification_f1",
            "node_embedding_quality",
            "link_prediction_auc",
            "edge_classification_f1",
            "graph_classification_accuracy",
            "graph_regression_r2",
            "community_modularity",
            "community_nmi",
            "kg_triple_classification_f1",
            "kg_link_prediction_mrr",
            "molecular_property_r2",
            "drug_discovery_auc",
            "node_homophily_ratio",
            "node_fairness_demographic_parity",
            "link_prediction_precision",
            "temporal_edge_accuracy",
            "graph_similarity_ged",
            "community_ari",
            "community_conductance",
            "generation_structural_similarity",
            "generation_diversity",
            "social_influence_correlation",
            "social_role_accuracy",
            "molecular_toxicity_auc",
            "dti_prediction_auc",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "node_classification_accuracy",
            "Node classification accuracy",
        );
        descriptions.insert("node_classification_f1", "Node classification F1 score");
        descriptions.insert(
            "node_embedding_quality",
            "Node embedding quality (silhouette score)",
        );
        descriptions.insert(
            "link_prediction_auc",
            "Link prediction area under ROC curve",
        );
        descriptions.insert("edge_classification_f1", "Edge classification F1 score");
        descriptions.insert(
            "graph_classification_accuracy",
            "Graph classification accuracy",
        );
        descriptions.insert("graph_regression_r2", "Graph regression R² score");
        descriptions.insert("community_modularity", "Community detection modularity");
        descriptions.insert(
            "community_nmi",
            "Community detection normalized mutual information",
        );
        descriptions.insert(
            "kg_triple_classification_f1",
            "Knowledge graph triple classification F1",
        );
        descriptions.insert(
            "kg_link_prediction_mrr",
            "Knowledge graph link prediction mean reciprocal rank",
        );
        descriptions.insert("molecular_property_r2", "Molecular property prediction R²");
        descriptions.insert(
            "drug_discovery_auc",
            "Drug discovery bioactivity prediction AUC",
        );
        descriptions.insert("node_homophily_ratio", "Node homophily ratio");
        descriptions.insert(
            "node_fairness_demographic_parity",
            "Node fairness demographic parity",
        );
        descriptions.insert(
            "link_prediction_precision",
            "Link prediction precision at K",
        );
        descriptions.insert(
            "temporal_edge_accuracy",
            "Temporal edge persistence accuracy",
        );
        descriptions.insert(
            "graph_similarity_ged",
            "Graph similarity (graph edit distance) correlation",
        );
        descriptions.insert("community_ari", "Community detection adjusted rand index");
        descriptions.insert("community_conductance", "Community detection conductance");
        descriptions.insert(
            "generation_structural_similarity",
            "Graph generation structural similarity",
        );
        descriptions.insert("generation_diversity", "Graph generation diversity");
        descriptions.insert(
            "social_influence_correlation",
            "Social influence prediction correlation",
        );
        descriptions.insert("social_role_accuracy", "Social role prediction accuracy");
        descriptions.insert(
            "molecular_toxicity_auc",
            "Molecular toxicity prediction AUC",
        );
        descriptions.insert(
            "dti_prediction_auc",
            "Drug-target interaction prediction AUC",
        );
        descriptions
    }
}
