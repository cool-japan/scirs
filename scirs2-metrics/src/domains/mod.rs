//! Domain-specific metric collections
//!
//! This module provides pre-configured metric suites tailored for specific
//! machine learning domains, making it easy to evaluate models with the most
//! appropriate metrics for each application area.
//!
//! # Features
//!
//! - **Computer Vision**: Object detection, image classification, segmentation metrics
//! - **Natural Language Processing**: Text generation, classification, similarity metrics
//! - **Time Series**: Forecasting, anomaly detection, trend analysis metrics
//! - **Recommendation Systems**: Ranking, relevance, diversity metrics
//! - **Anomaly Detection**: Detection accuracy, false alarm rates, distribution metrics
//! - **Audio Processing**: Speech recognition, music analysis, sound event detection metrics
//! - **Graph Neural Networks**: Node/edge/graph classification, community detection, graph generation metrics
//! - **Biomedical & Healthcare**: Clinical trials, drug discovery, medical imaging, genomics, epidemiology metrics
//! - **Financial Modeling**: Risk management, portfolio optimization, credit risk, trading strategy evaluation metrics
//! - **Quantum Machine Learning**: Quantum fidelity, entanglement, circuit complexity, quantum advantage metrics
//! - **Edge AI & Federated Learning**: Edge performance, privacy preservation, federated learning convergence metrics
//!
//! # Examples
//!
//! ## Computer Vision Metrics
//!
//! ```
//! use scirs2_metrics::domains::computer_vision::ObjectDetectionMetrics;
//! use scirs2_core::ndarray::array;
//!
//! let mut cv_metrics = ObjectDetectionMetrics::new();
//!
//! // Example bounding box predictions (x1, y1, x2, y2, confidence, class)
//! let predictions = vec![
//!     (10.0, 10.0, 50.0, 50.0, 0.9, 1),  // High confidence detection
//!     (60.0, 60.0, 100.0, 100.0, 0.7, 2), // Medium confidence detection
//! ];
//!
//! let ground_truth = vec![
//!     (12.0, 12.0, 48.0, 48.0, 1),  // Close to first prediction
//!     (70.0, 70.0, 110.0, 110.0, 2), // Close to second prediction
//! ];
//!
//! // Compute comprehensive object detection metrics
//! let results = cv_metrics.evaluate_object_detection(&predictions, &ground_truth, 0.5).unwrap();
//! println!("mAP@0.5: {:.4}", results.map);
//! println!("Precision: {:.4}", results.precision);
//! println!("Recall: {:.4}", results.recall);
//! ```
//!
//! ## NLP Metrics
//!
//! ```
//! use scirs2_metrics::domains::nlp::TextGenerationMetrics;
//!
//! let mut nlp_metrics = TextGenerationMetrics::new();
//!
//! let references = vec![
//!     "The cat sat on the mat".to_string(),
//!     "A quick brown fox jumps".to_string(),
//! ];
//!
//! let candidates = vec![
//!     "The cat sits on the mat".to_string(),
//!     "A quick brown fox jumped".to_string(),
//! ];
//!
//! let results = nlp_metrics.evaluate_generation(&references, &candidates).unwrap();
//! println!("BLEU-4: {:.4}", results.bleu_4);
//! println!("ROUGE-L: {:.4}", results.rouge_l);
//! ```

use crate::error::{MetricsError, Result};
use std::collections::HashMap;

// Import domain-specific metric suites
use crate::domains::anomaly_detection::AnomalyDetectionSuite;
use crate::domains::audio_processing::AudioProcessingMetrics;
use crate::domains::computer_vision::ComputerVisionSuite;
use crate::domains::edge_ai::EdgeAISuite;
use crate::domains::generative_ai::GenerativeAISuite;
use crate::domains::graph_neural_networks::GraphNeuralNetworkMetrics;
use crate::domains::quantum_ml::QuantumMLSuite;
use crate::domains::time_series::TimeSeriesSuite;

pub mod anomaly_detection;
pub mod audio_processing;
pub mod biomedical;
pub mod computer_vision;
pub mod edge_ai;
pub mod financial;
pub mod generative_ai;
pub mod graph_neural_networks;
pub mod neuromorphic;
pub mod nlp;
pub mod quantum_ml;
pub mod recommender;
pub mod time_series;

/// Common trait for domain-specific metric collections
pub trait DomainMetrics {
    /// Type of the evaluation result
    type Result;

    /// Name of the domain
    fn domain_name(&self) -> &'static str;

    /// List of available metrics in this domain
    fn available_metrics(&self) -> Vec<&'static str>;

    /// Get a description of what each metric measures
    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str>;
}

/// Standard evaluation result that can be used across domains
#[derive(Debug, Clone)]
pub struct DomainEvaluationResult {
    /// Primary metrics (most important for the domain)
    pub primary_metrics: HashMap<String, f64>,
    /// Secondary metrics (additional useful metrics)
    pub secondary_metrics: HashMap<String, f64>,
    /// Detailed breakdown (e.g., per-class metrics)
    pub detailed_metrics: Option<HashMap<String, HashMap<String, f64>>>,
    /// Human-readable summary
    pub summary: String,
}

impl DomainEvaluationResult {
    /// Create a new domain evaluation result
    pub fn new() -> Self {
        Self {
            primary_metrics: HashMap::new(),
            secondary_metrics: HashMap::new(),
            detailed_metrics: None,
            summary: String::new(),
        }
    }

    /// Add a primary metric
    pub fn add_primary_metric(&mut self, name: String, value: f64) {
        self.primary_metrics.insert(name, value);
    }

    /// Add a secondary metric
    pub fn add_secondary_metric(&mut self, name: String, value: f64) {
        self.secondary_metrics.insert(name, value);
    }

    /// Add detailed metrics (e.g., per-class results)
    pub fn add_detailed_metrics(&mut self, category: String, metrics: HashMap<String, f64>) {
        if self.detailed_metrics.is_none() {
            self.detailed_metrics = Some(HashMap::new());
        }
        self.detailed_metrics
            .as_mut()
            .unwrap()
            .insert(category, metrics);
    }

    /// Set the summary text
    pub fn set_summary(&mut self, summary: String) {
        self.summary = summary;
    }

    /// Get all metrics as a flat HashMap
    pub fn all_metrics(&self) -> HashMap<String, f64> {
        let mut all = HashMap::new();

        // Add primary metrics
        for (name, value) in &self.primary_metrics {
            all.insert(format!("primary_{}", name), *value);
        }

        // Add secondary metrics
        for (name, value) in &self.secondary_metrics {
            all.insert(format!("secondary_{}", name), *value);
        }

        // Add detailed metrics
        if let Some(detailed) = &self.detailed_metrics {
            for (category, metrics) in detailed {
                for (name, value) in metrics {
                    all.insert(format!("{}_{}", category, name), *value);
                }
            }
        }

        all
    }

    /// Get the most important metric value for this domain
    pub fn primary_score(&self) -> Option<f64> {
        // Return the first primary metric as the main score
        self.primary_metrics.values().next().copied()
    }
}

impl Default for DomainEvaluationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create metric collections for multiple domains
#[allow(dead_code)]
pub fn create_domain_suite() -> DomainSuite {
    DomainSuite::new()
}

/// Collection of domain-specific metric calculators
pub struct DomainSuite {
    cv_metrics: ComputerVisionSuite,
    nlp_metrics: nlp::NLPSuite,
    ts_metrics: TimeSeriesSuite,
    rec_metrics: recommender::RecommenderSuite,
    ad_metrics: AnomalyDetectionSuite,
    gen_ai_metrics: GenerativeAISuite<f64>,
    audio_metrics: AudioProcessingMetrics,
    gnn_metrics: GraphNeuralNetworkMetrics,
    biomedical_metrics: biomedical::BiomedicalSuite,
    financial_metrics: financial::FinancialSuite,
    quantum_ml_metrics: QuantumMLSuite,
    edge_ai_metrics: EdgeAISuite,
}

impl DomainSuite {
    /// Create a new domain suite with all available domain metrics
    pub fn new() -> Self {
        Self {
            cv_metrics: ComputerVisionSuite::new(),
            nlp_metrics: nlp::NLPSuite::new(),
            ts_metrics: TimeSeriesSuite::new(),
            rec_metrics: recommender::RecommenderSuite::new(),
            ad_metrics: AnomalyDetectionSuite::new(),
            gen_ai_metrics: GenerativeAISuite::new(),
            audio_metrics: AudioProcessingMetrics::new(),
            gnn_metrics: GraphNeuralNetworkMetrics::new(),
            biomedical_metrics: biomedical::BiomedicalSuite::new(),
            financial_metrics: financial::FinancialSuite::new(),
            quantum_ml_metrics: QuantumMLSuite::new(),
            edge_ai_metrics: EdgeAISuite::new(),
        }
    }

    /// Get computer vision metrics
    pub fn computer_vision(&self) -> &computer_vision::ComputerVisionSuite {
        &self.cv_metrics
    }

    /// Get NLP metrics
    pub fn nlp(&self) -> &nlp::NLPSuite {
        &self.nlp_metrics
    }

    /// Get time series metrics
    pub fn time_series(&self) -> &time_series::TimeSeriesSuite {
        &self.ts_metrics
    }

    /// Get recommender system metrics
    pub fn recommender(&self) -> &recommender::RecommenderSuite {
        &self.rec_metrics
    }

    /// Get anomaly detection metrics
    pub fn anomaly_detection(&self) -> &anomaly_detection::AnomalyDetectionSuite {
        &self.ad_metrics
    }

    /// Get generative AI metrics
    pub fn generative_ai(&self) -> &generative_ai::GenerativeAISuite<f64> {
        &self.gen_ai_metrics
    }

    /// Get audio processing metrics
    pub fn audio_processing(&self) -> &audio_processing::AudioProcessingMetrics {
        &self.audio_metrics
    }

    /// Get graph neural network metrics
    pub fn graph_neural_networks(&self) -> &graph_neural_networks::GraphNeuralNetworkMetrics {
        &self.gnn_metrics
    }

    /// Get biomedical and healthcare metrics
    pub fn biomedical(&self) -> &biomedical::BiomedicalSuite {
        &self.biomedical_metrics
    }

    /// Get financial modeling metrics
    pub fn financial(&self) -> &financial::FinancialSuite {
        &self.financial_metrics
    }

    /// Get quantum machine learning metrics
    pub fn quantum_ml(&self) -> &quantum_ml::QuantumMLSuite {
        &self.quantum_ml_metrics
    }

    /// Get edge AI and federated learning metrics
    pub fn edge_ai(&self) -> &edge_ai::EdgeAISuite {
        &self.edge_ai_metrics
    }

    /// List all available domains
    pub fn available_domains(&self) -> Vec<&'static str> {
        vec![
            self.cv_metrics.domain_name(),
            self.nlp_metrics.domain_name(),
            self.ts_metrics.domain_name(),
            self.rec_metrics.domain_name(),
            self.ad_metrics.domain_name(),
            self.gen_ai_metrics.domain_name(),
            self.audio_metrics.domain_name(),
            self.gnn_metrics.domain_name(),
            self.biomedical_metrics.domain_name(),
            self.financial_metrics.domain_name(),
            self.quantum_ml_metrics.domain_name(),
            self.edge_ai_metrics.domain_name(),
        ]
    }
}

impl Default for DomainSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_evaluation_result() {
        let mut result = DomainEvaluationResult::new();

        result.add_primary_metric("accuracy".to_string(), 0.85);
        result.add_secondary_metric("precision".to_string(), 0.82);

        let mut class_metrics = HashMap::new();
        class_metrics.insert("f1_score".to_string(), 0.83);
        result.add_detailed_metrics("class_1".to_string(), class_metrics);

        result.set_summary("Good performance overall".to_string());

        assert_eq!(result.primary_score(), Some(0.85));
        assert_eq!(result.summary, "Good performance overall");

        let all_metrics = result.all_metrics();
        assert_eq!(all_metrics.get("primary_accuracy"), Some(&0.85));
        assert_eq!(all_metrics.get("secondary_precision"), Some(&0.82));
        assert_eq!(all_metrics.get("class_1_f1_score"), Some(&0.83));
    }

    #[test]
    fn test_domain_suite_creation() {
        let suite = create_domain_suite();
        let domains = suite.available_domains();

        assert_eq!(domains.len(), 12);
        assert!(domains.contains(&"Computer Vision"));
        assert!(domains.contains(&"Natural Language Processing"));
        assert!(domains.contains(&"Time Series"));
        assert!(domains.contains(&"Recommender Systems"));
        assert!(domains.contains(&"Anomaly Detection"));
        assert!(domains.contains(&"Generative AI & Deep Learning"));
        assert!(domains.contains(&"Audio Processing"));
        assert!(domains.contains(&"Graph Neural Networks"));
        assert!(domains.contains(&"Biomedical & Healthcare"));
        assert!(domains.contains(&"Financial Modeling & Quantitative Finance"));
        assert!(domains.contains(&"Quantum Machine Learning"));
        assert!(domains.contains(&"Edge AI & Federated Learning"));
    }
}
