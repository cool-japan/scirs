//! Edge AI and Federated Learning Metrics Collection
//!
//! This module provides specialized metrics for evaluating edge AI deployments,
//! federated learning systems, and privacy-preserving machine learning applications.
//!
//! # Features
//!
//! - **Edge Performance Metrics**: Latency, throughput, power consumption, memory usage
//! - **Federated Learning Metrics**: Communication efficiency, convergence analysis, client participation
//! - **Privacy Metrics**: Differential privacy guarantees, membership inference resistance
//! - **Resource Efficiency**: Model compression ratios, quantization quality, edge optimization
//! - **Network Metrics**: Bandwidth utilization, communication rounds, data locality
//! - **Robustness**: Byzantine resistance, non-IID data handling, fault tolerance
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::domains::edge_ai::EdgeAISuite;
//! use scirs2_core::ndarray::array;
//!
//! let mut edge_suite = EdgeAISuite::new();
//!
//! // Edge performance evaluation
//! let latencies = vec![10.5, 12.3, 9.8, 11.2, 10.9]; // milliseconds
//! let power_consumption = vec![2.1, 2.3, 2.0, 2.2, 2.1]; // watts
//! let memory_usage = vec![150.0, 160.0, 145.0, 155.0, 148.0]; // MB
//!
//! let edge_metrics = edge_suite.evaluate_edge_performance(
//!     &latencies,
//!     &power_consumption,
//!     &memory_usage,
//!     1000, // target throughput (inferences/sec)
//! ).unwrap();
//!
//! println!("Average Latency: {:.2} ms", edge_metrics.avg_latency);
//! println!("Energy Efficiency: {:.2} inferences/joule", edge_metrics.energy_efficiency);
//! ```

use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::Duration;

/// Edge AI and Federated Learning metrics suite
#[derive(Debug, Clone)]
pub struct EdgeAISuite {
    /// Configuration for edge AI metrics
    config: EdgeAIConfig,
    /// Performance baselines for comparison
    baselines: EdgeBaselines,
}

/// Configuration for edge AI metrics computation
#[derive(Debug, Clone)]
pub struct EdgeAIConfig {
    /// Target latency threshold (milliseconds)
    pub target_latency_ms: f64,
    /// Target power budget (watts)
    pub target_power_budget: f64,
    /// Target memory limit (MB)
    pub target_memory_limit: f64,
    /// Minimum acceptable accuracy
    pub min_accuracy_threshold: f64,
    /// Privacy epsilon for differential privacy
    pub privacy_epsilon: f64,
    /// Network bandwidth limit (Mbps)
    pub bandwidth_limit: f64,
    /// Enable federated learning metrics
    pub enable_federated_metrics: bool,
}

/// Performance baselines for edge devices
#[derive(Debug, Clone)]
pub struct EdgeBaselines {
    /// Cloud baseline performance
    pub cloud_latency: f64,
    /// Cloud accuracy baseline
    pub cloud_accuracy: f64,
    /// Desktop/server baseline
    pub server_latency: f64,
    /// Mobile device baseline
    pub mobile_latency: f64,
}

/// Edge performance metrics
#[derive(Debug, Clone)]
pub struct EdgePerformanceMetrics {
    /// Average inference latency (ms)
    pub avg_latency: f64,
    /// Latency percentiles (P50, P90, P95, P99)
    pub latency_percentiles: [f64; 4],
    /// Throughput (inferences per second)
    pub throughput: f64,
    /// Average power consumption (watts)
    pub avg_power: f64,
    /// Energy efficiency (inferences per joule)
    pub energy_efficiency: f64,
    /// Memory utilization (MB)
    pub memory_usage: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Model size (MB)
    pub model_size: f64,
}

/// Federated learning metrics
#[derive(Debug, Clone)]
pub struct FederatedLearningMetrics {
    /// Communication efficiency (accuracy/communication_cost)
    pub communication_efficiency: f64,
    /// Number of communication rounds to convergence
    pub convergence_rounds: usize,
    /// Total communication cost (MB)
    pub total_communication: f64,
    /// Client participation rate
    pub client_participation_rate: f64,
    /// Convergence rate (accuracy improvement per round)
    pub convergence_rate: f64,
    /// Data heterogeneity measure
    pub data_heterogeneity: f64,
    /// Byzantine robustness score
    pub byzantine_robustness: f64,
}

/// Privacy preservation metrics
#[derive(Debug, Clone)]
pub struct PrivacyMetrics {
    /// Differential privacy epsilon
    pub epsilon: f64,
    /// Privacy-utility tradeoff score
    pub privacy_utility_score: f64,
    /// Membership inference attack resistance
    pub membership_inference_resistance: f64,
    /// Model inversion attack resistance
    pub model_inversion_resistance: f64,
    /// Property inference attack resistance
    pub property_inference_resistance: f64,
    /// Information leakage measure
    pub information_leakage: f64,
}

/// Model compression and optimization metrics
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    /// Compression ratio (original_size / compressed_size)
    pub compression_ratio: f64,
    /// Accuracy retention after compression
    pub accuracy_retention: f64,
    /// Quantization quality score
    pub quantization_quality: f64,
    /// Pruning efficiency
    pub pruning_efficiency: f64,
    /// Knowledge distillation effectiveness
    pub distillation_effectiveness: f64,
    /// Model complexity reduction
    pub complexity_reduction: f64,
}

/// Network and communication metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// Bandwidth utilization efficiency
    pub bandwidth_efficiency: f64,
    /// Data locality score
    pub data_locality: f64,
    /// Network overhead ratio
    pub network_overhead: f64,
    /// Communication pattern efficiency
    pub communication_pattern_score: f64,
    /// Network fault tolerance
    pub fault_tolerance: f64,
}

/// Client diversity and heterogeneity metrics
#[derive(Debug, Clone)]
pub struct ClientDiversityMetrics {
    /// Statistical heterogeneity measure
    pub statistical_heterogeneity: f64,
    /// System heterogeneity (hardware diversity)
    pub system_heterogeneity: f64,
    /// Participation fairness
    pub participation_fairness: f64,
    /// Resource availability variance
    pub resource_variance: f64,
    /// Data distribution skewness
    pub data_skewness: f64,
}

/// Edge AI deployment quality metrics
#[derive(Debug, Clone)]
pub struct DeploymentQualityMetrics {
    /// Deployment success rate
    pub deployment_success_rate: f64,
    /// Model update efficiency
    pub update_efficiency: f64,
    /// Rollback capability score
    pub rollback_capability: f64,
    /// Multi-device compatibility
    pub device_compatibility: f64,
    /// Operating system portability
    pub os_portability: f64,
}

impl Default for EdgeAIConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 100.0,
            target_power_budget: 5.0,
            target_memory_limit: 512.0,
            min_accuracy_threshold: 0.85,
            privacy_epsilon: 1.0,
            bandwidth_limit: 10.0,
            enable_federated_metrics: true,
        }
    }
}

impl Default for EdgeBaselines {
    fn default() -> Self {
        Self {
            cloud_latency: 200.0,
            cloud_accuracy: 0.95,
            server_latency: 50.0,
            mobile_latency: 150.0,
        }
    }
}

impl EdgeAISuite {
    /// Create a new edge AI metrics suite
    pub fn new() -> Self {
        Self {
            config: EdgeAIConfig::default(),
            baselines: EdgeBaselines::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EdgeAIConfig) -> Self {
        Self {
            config,
            baselines: EdgeBaselines::default(),
        }
    }

    /// Set performance baselines
    pub fn with_baselines(mut self, baselines: EdgeBaselines) -> Self {
        self.baselines = baselines;
        self
    }

    /// Evaluate edge device performance
    pub fn evaluate_edge_performance(
        &self,
        latencies: &[f64],
        power_consumption: &[f64],
        memory_usage: &[f64],
        target_throughput: usize,
    ) -> Result<EdgePerformanceMetrics> {
        if latencies.is_empty() || power_consumption.is_empty() || memory_usage.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Performance metrics arrays cannot be empty".to_string(),
            ));
        }

        // Calculate latency statistics
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let latency_percentiles = [
            self.percentile(&sorted_latencies, 50.0),
            self.percentile(&sorted_latencies, 90.0),
            self.percentile(&sorted_latencies, 95.0),
            self.percentile(&sorted_latencies, 99.0),
        ];

        // Calculate _throughput
        let throughput = 1000.0 / avg_latency; // inferences per second

        // Calculate power metrics
        let avg_power = power_consumption.iter().sum::<f64>() / power_consumption.len() as f64;
        let energy_efficiency = throughput / avg_power; // inferences per watt per second

        // Calculate memory _usage
        let memory_usage_avg = memory_usage.iter().sum::<f64>() / memory_usage.len() as f64;

        // Estimate CPU utilization based on _throughput vs target
        let cpu_utilization = (throughput / target_throughput as f64 * 100.0).min(100.0);

        // Model size estimation (simplified)
        let model_size = memory_usage_avg * 0.6; // Assume 60% of memory is model

        Ok(EdgePerformanceMetrics {
            avg_latency,
            latency_percentiles,
            throughput,
            avg_power,
            energy_efficiency,
            memory_usage: memory_usage_avg,
            cpu_utilization,
            model_size,
        })
    }

    /// Evaluate federated learning system performance
    pub fn evaluate_federated_learning(
        &self,
        accuracy_perround: &[f64],
        communication_costs: &[f64],
        client_participation: &[f64],
        datadistributions: &[Vec<f64>],
    ) -> Result<FederatedLearningMetrics> {
        if accuracy_perround.is_empty() || communication_costs.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Federated learning metrics require non-empty data".to_string(),
            ));
        }

        // Find convergence point
        let convergence_rounds = self.find_convergence_round(accuracy_perround)?;

        // Calculate communication efficiency
        let final_accuracy = accuracy_perround.last().copied().unwrap_or(0.0);
        let total_communication = communication_costs.iter().sum::<f64>();
        let communication_efficiency = if total_communication > 0.0 {
            final_accuracy / total_communication
        } else {
            0.0
        };

        // Calculate client _participation rate
        let client_participation_rate =
            client_participation.iter().sum::<f64>() / client_participation.len() as f64;

        // Calculate convergence rate
        let convergence_rate = if convergence_rounds > 0 {
            final_accuracy / convergence_rounds as f64
        } else {
            0.0
        };

        // Calculate data heterogeneity
        let data_heterogeneity = self.calculate_data_heterogeneity(datadistributions)?;

        // Byzantine robustness (simplified metric)
        let byzantine_robustness = self.estimate_byzantine_robustness(accuracy_perround)?;

        Ok(FederatedLearningMetrics {
            communication_efficiency,
            convergence_rounds,
            total_communication,
            client_participation_rate,
            convergence_rate,
            data_heterogeneity,
            byzantine_robustness,
        })
    }

    /// Evaluate privacy preservation effectiveness
    pub fn evaluate_privacy_preservation(
        &self,
        epsilon: f64,
        original_accuracy: f64,
        private_accuracy: f64,
        attack_success_rates: &HashMap<String, f64>,
    ) -> Result<PrivacyMetrics> {
        if epsilon <= 0.0 {
            return Err(MetricsError::InvalidInput(
                "Privacy epsilon must be positive".to_string(),
            ));
        }

        // Privacy-utility tradeoff
        let utility_loss = (original_accuracy - private_accuracy) / original_accuracy;
        let privacy_utility_score = private_accuracy / epsilon; // Higher is better

        // Attack resistance metrics
        let membership_inference_resistance = 1.0
            - attack_success_rates
                .get("membership_inference")
                .unwrap_or(&0.5);
        let model_inversion_resistance =
            1.0 - attack_success_rates.get("model_inversion").unwrap_or(&0.5);
        let property_inference_resistance = 1.0
            - attack_success_rates
                .get("property_inference")
                .unwrap_or(&0.5);

        // Information leakage estimate
        let information_leakage = utility_loss * epsilon; // Simplified measure

        Ok(PrivacyMetrics {
            epsilon,
            privacy_utility_score,
            membership_inference_resistance,
            model_inversion_resistance,
            property_inference_resistance,
            information_leakage,
        })
    }

    /// Evaluate model compression effectiveness
    pub fn evaluate_model_compression(
        &self,
        original_size: f64,
        compressed_size: f64,
        original_accuracy: f64,
        compressed_accuracy: f64,
        compression_technique: &str,
    ) -> Result<CompressionMetrics> {
        if original_size <= 0.0 || compressed_size <= 0.0 {
            return Err(MetricsError::InvalidInput(
                "Model sizes must be positive".to_string(),
            ));
        }

        let compression_ratio = original_size / compressed_size;
        let accuracy_retention = compressed_accuracy / original_accuracy;

        // Technique-specific quality scores
        let (quantization_quality, pruning_efficiency, distillation_effectiveness) = self
            .assess_compression_technique(
                compression_technique,
                compression_ratio,
                accuracy_retention,
            );

        let complexity_reduction = 1.0 - (compressed_size / original_size);

        Ok(CompressionMetrics {
            compression_ratio,
            accuracy_retention,
            quantization_quality,
            pruning_efficiency,
            distillation_effectiveness,
            complexity_reduction,
        })
    }

    /// Evaluate network communication efficiency
    pub fn evaluate_network_efficiency(
        &self,
        bandwidth_used: &[f64],
        data_locality_scores: &[f64],
        network_faults: usize,
        total_operations: usize,
    ) -> Result<NetworkMetrics> {
        if bandwidth_used.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Bandwidth usage data cannot be empty".to_string(),
            ));
        }

        let avg_bandwidth = bandwidth_used.iter().sum::<f64>() / bandwidth_used.len() as f64;
        let bandwidth_efficiency = self.config.bandwidth_limit / avg_bandwidth;

        let data_locality = if data_locality_scores.is_empty() {
            0.5 // Default neutral score
        } else {
            data_locality_scores.iter().sum::<f64>() / data_locality_scores.len() as f64
        };

        let network_overhead = avg_bandwidth / (avg_bandwidth + 1.0); // Simplified calculation
        let communication_pattern_score = data_locality * bandwidth_efficiency;

        let fault_tolerance = if total_operations > 0 {
            1.0 - (network_faults as f64 / total_operations as f64)
        } else {
            1.0
        };

        Ok(NetworkMetrics {
            bandwidth_efficiency,
            data_locality,
            network_overhead,
            communication_pattern_score,
            fault_tolerance,
        })
    }

    /// Comprehensive edge AI system evaluation
    pub fn evaluate_edge_ai_system(
        &self,
        edge_performance: &EdgePerformanceMetrics,
        federated_metrics: Option<&FederatedLearningMetrics>,
        privacy_metrics: Option<&PrivacyMetrics>,
        compression_metrics: &CompressionMetrics,
        network_metrics: &NetworkMetrics,
    ) -> Result<DomainEvaluationResult> {
        let mut result = DomainEvaluationResult::new();

        // Primary _metrics - most important for edge AI
        result.add_primary_metric(
            "latency_efficiency".to_string(),
            self.config.target_latency_ms / edge_performance.avg_latency,
        );
        result.add_primary_metric(
            "energy_efficiency".to_string(),
            edge_performance.energy_efficiency,
        );
        result.add_primary_metric(
            "compression_effectiveness".to_string(),
            compression_metrics.compression_ratio * compression_metrics.accuracy_retention,
        );
        result.add_primary_metric(
            "deployment_readiness".to_string(),
            self.calculate_deployment_readiness(edge_performance, compression_metrics),
        );

        // Secondary _metrics
        result.add_secondary_metric("avg_latency".to_string(), edge_performance.avg_latency);
        result.add_secondary_metric("throughput".to_string(), edge_performance.throughput);
        result.add_secondary_metric(
            "memory_efficiency".to_string(),
            self.config.target_memory_limit / edge_performance.memory_usage,
        );
        result.add_secondary_metric(
            "network_efficiency".to_string(),
            network_metrics.bandwidth_efficiency,
        );

        // Federated learning _metrics (if applicable)
        if let Some(fl_metrics) = federated_metrics {
            result.add_secondary_metric(
                "communication_efficiency".to_string(),
                fl_metrics.communication_efficiency,
            );
            result
                .add_secondary_metric("convergence_rate".to_string(), fl_metrics.convergence_rate);
            result.add_secondary_metric(
                "client_participation".to_string(),
                fl_metrics.client_participation_rate,
            );
        }

        // Privacy _metrics (if applicable)
        if let Some(priv_metrics) = privacy_metrics {
            result.add_secondary_metric(
                "privacy_utility_score".to_string(),
                priv_metrics.privacy_utility_score,
            );
            result.add_secondary_metric(
                "attack_resistance".to_string(),
                (priv_metrics.membership_inference_resistance
                    + priv_metrics.model_inversion_resistance
                    + priv_metrics.property_inference_resistance)
                    / 3.0,
            );
        }

        // Generate summary
        let edge_score = self.calculate_overall_edge_score(
            edge_performance,
            federated_metrics,
            privacy_metrics,
            compression_metrics,
            network_metrics,
        );

        let summary = format!(
            "Edge AI System Score: {:.2}/10. Latency: {:.1}ms, Energy Efficiency: {:.1} inf/J, Compression: {:.1}x",
            edge_score,
            edge_performance.avg_latency,
            edge_performance.energy_efficiency,
            compression_metrics.compression_ratio
        );
        result.set_summary(summary);

        Ok(result)
    }

    // Helper methods

    fn percentile(&self, sorteddata: &[f64], percentile: f64) -> f64 {
        if sorteddata.is_empty() {
            return 0.0;
        }
        let index = (percentile / 100.0 * (sorteddata.len() - 1) as f64) as usize;
        sorteddata[index.min(sorteddata.len() - 1)]
    }

    fn find_convergence_round(&self, accuracy_perround: &[f64]) -> Result<usize> {
        let threshold = 0.01; // 1% improvement threshold

        for i in 1..accuracy_perround.len() {
            let improvement = accuracy_perround[i] - accuracy_perround[i - 1];
            if improvement < threshold {
                return Ok(i);
            }
        }

        Ok(accuracy_perround.len())
    }

    fn calculate_data_heterogeneity(&self, datadistributions: &[Vec<f64>]) -> Result<f64> {
        if datadistributions.is_empty() {
            return Ok(0.0);
        }

        // Calculate KL divergence between client _distributions
        let mut total_divergence = 0.0;
        let mut comparisons = 0;

        for i in 0..datadistributions.len() {
            for j in i + 1..datadistributions.len() {
                let kl_div = self.kl_divergence(&datadistributions[i], &datadistributions[j])?;
                total_divergence += kl_div;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            Ok(total_divergence / comparisons as f64)
        } else {
            Ok(0.0)
        }
    }

    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64> {
        if p.len() != q.len() {
            return Err(MetricsError::InvalidInput(
                "Distributions must have the same length".to_string(),
            ));
        }

        let mut kl = 0.0;
        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > 0.0 && *qi > 0.0 {
                kl += pi * (pi / qi).ln();
            }
        }
        Ok(kl)
    }

    fn estimate_byzantine_robustness(&self, accuracy_perround: &[f64]) -> Result<f64> {
        // Simple robustness estimate based on accuracy stability
        if accuracy_perround.len() < 2 {
            return Ok(1.0);
        }

        let mut variance = 0.0;
        let mean = accuracy_perround.iter().sum::<f64>() / accuracy_perround.len() as f64;

        for acc in accuracy_perround {
            variance += (acc - mean).powi(2);
        }
        variance /= accuracy_perround.len() as f64;

        // Lower variance indicates higher robustness
        Ok(1.0 / (1.0 + variance))
    }

    fn assess_compression_technique(
        &self,
        technique: &str,
        ratio: f64,
        retention: f64,
    ) -> (f64, f64, f64) {
        match technique.to_lowercase().as_str() {
            "quantization" => (ratio * retention, 0.0, 0.0),
            "pruning" => (0.0, ratio * retention, 0.0),
            "distillation" => (0.0, 0.0, ratio * retention),
            "mixed" => {
                let score = ratio * retention;
                (score * 0.4, score * 0.4, score * 0.2)
            }
            _ => (0.0, 0.0, 0.0),
        }
    }

    fn calculate_deployment_readiness(
        &self,
        perf: &EdgePerformanceMetrics,
        comp: &CompressionMetrics,
    ) -> f64 {
        let latency_ok = if perf.avg_latency <= self.config.target_latency_ms {
            1.0
        } else {
            0.5
        };
        let power_ok = if perf.avg_power <= self.config.target_power_budget {
            1.0
        } else {
            0.5
        };
        let memory_ok = if perf.memory_usage <= self.config.target_memory_limit {
            1.0
        } else {
            0.5
        };
        let accuracy_ok = if comp.accuracy_retention >= self.config.min_accuracy_threshold {
            1.0
        } else {
            0.3
        };

        (latency_ok + power_ok + memory_ok + accuracy_ok) / 4.0
    }

    fn calculate_overall_edge_score(
        &self,
        edge_perf: &EdgePerformanceMetrics,
        fl_metrics: Option<&FederatedLearningMetrics>,
        privacy_metrics: Option<&PrivacyMetrics>,
        compression: &CompressionMetrics,
        network: &NetworkMetrics,
    ) -> f64 {
        let mut score = 0.0;
        let mut components = 0;

        // Edge performance (40% weight)
        let edge_score = (self.config.target_latency_ms / edge_perf.avg_latency * 2.0
            + edge_perf.energy_efficiency * 2.0
            + compression.accuracy_retention * 3.0)
            / 7.0;
        score += edge_score * 4.0;
        components += 4;

        // Federated learning (30% weight if available)
        if let Some(fl) = fl_metrics {
            let fl_score = (fl.communication_efficiency
                + fl.convergence_rate * 5.0
                + fl.client_participation_rate)
                / 7.0;
            score += fl_score * 3.0;
            components += 3;
        }

        // Privacy (20% weight if available)
        if let Some(privacy) = privacy_metrics {
            let privacy_score = privacy.privacy_utility_score / 2.0; // Normalize
            score += privacy_score * 2.0;
            components += 2;
        }

        // Network efficiency (10% weight)
        let network_score = (network.bandwidth_efficiency + network.fault_tolerance) / 2.0;
        score += network_score;
        components += 1;

        // Normalize to 0-10 scale
        (score / components as f64 * 10.0).min(10.0)
    }
}

impl DomainMetrics for EdgeAISuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Edge AI & Federated Learning"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "edge_latency",
            "energy_efficiency",
            "memory_efficiency",
            "throughput",
            "compression_ratio",
            "accuracy_retention",
            "communication_efficiency",
            "convergence_rate",
            "client_participation",
            "privacy_utility_score",
            "attack_resistance",
            "data_heterogeneity",
            "bandwidth_efficiency",
            "fault_tolerance",
            "deployment_readiness",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert("edge_latency", "Average inference latency on edge devices");
        descriptions.insert(
            "energy_efficiency",
            "Inferences per unit of energy consumed",
        );
        descriptions.insert("memory_efficiency", "Efficient use of limited edge memory");
        descriptions.insert("throughput", "Number of inferences processed per second");
        descriptions.insert(
            "compression_ratio",
            "Model size reduction through compression",
        );
        descriptions.insert(
            "accuracy_retention",
            "Accuracy preserved after optimization",
        );
        descriptions.insert(
            "communication_efficiency",
            "Learning effectiveness per communication cost",
        );
        descriptions.insert(
            "convergence_rate",
            "Speed of federated learning convergence",
        );
        descriptions.insert("client_participation", "Rate of client participation in FL");
        descriptions.insert(
            "privacy_utility_score",
            "Balance between privacy and utility",
        );
        descriptions.insert("attack_resistance", "Resistance to privacy attacks");
        descriptions.insert(
            "data_heterogeneity",
            "Measure of data distribution differences",
        );
        descriptions.insert("bandwidth_efficiency", "Efficient use of network bandwidth");
        descriptions.insert("fault_tolerance", "System resilience to network failures");
        descriptions.insert(
            "deployment_readiness",
            "Readiness for production deployment",
        );
        descriptions
    }
}

impl Default for EdgeAISuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_performance_evaluation() {
        let suite = EdgeAISuite::new();
        let latencies = vec![10.0, 12.0, 11.0, 13.0, 9.0];
        let power = vec![2.0, 2.1, 2.0, 2.2, 1.9];
        let memory = vec![150.0, 160.0, 155.0, 165.0, 145.0];

        let metrics = suite
            .evaluate_edge_performance(&latencies, &power, &memory, 100)
            .unwrap();

        assert!(metrics.avg_latency > 0.0);
        assert!(metrics.throughput > 0.0);
        assert!(metrics.energy_efficiency > 0.0);
        assert_eq!(metrics.latency_percentiles.len(), 4);
    }

    #[test]
    fn test_federated_learning_evaluation() {
        let suite = EdgeAISuite::new();
        let accuracy = vec![0.7, 0.75, 0.8, 0.82, 0.84, 0.845, 0.847];
        let comm_costs = vec![10.0, 12.0, 11.0, 13.0, 12.5, 11.8, 12.2];
        let participation = vec![0.8, 0.85, 0.82, 0.88, 0.86, 0.84, 0.87];
        let distributions = vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.4, 0.3, 0.3],
        ];

        let metrics = suite
            .evaluate_federated_learning(&accuracy, &comm_costs, &participation, &distributions)
            .unwrap();

        assert!(metrics.communication_efficiency > 0.0);
        assert!(metrics.convergence_rounds > 0);
        assert!(metrics.client_participation_rate > 0.0);
    }

    #[test]
    fn test_privacy_evaluation() {
        let suite = EdgeAISuite::new();
        let mut attack_rates = HashMap::new();
        attack_rates.insert("membership_inference".to_string(), 0.6);
        attack_rates.insert("model_inversion".to_string(), 0.4);
        attack_rates.insert("property_inference".to_string(), 0.3);

        let metrics = suite
            .evaluate_privacy_preservation(1.0, 0.9, 0.85, &attack_rates)
            .unwrap();

        assert_eq!(metrics.epsilon, 1.0);
        assert!(metrics.privacy_utility_score > 0.0);
        assert!(metrics.membership_inference_resistance > 0.0);
    }

    #[test]
    fn test_model_compression_evaluation() {
        let suite = EdgeAISuite::new();

        let metrics = suite
            .evaluate_model_compression(100.0, 25.0, 0.92, 0.89, "quantization")
            .unwrap();

        assert_eq!(metrics.compression_ratio, 4.0);
        assert!((metrics.accuracy_retention - 0.9673913).abs() < 1e-6);
        assert!(metrics.quantization_quality > 0.0);
    }

    #[test]
    fn test_network_efficiency_evaluation() {
        let suite = EdgeAISuite::new();
        let bandwidth = vec![5.0, 6.0, 5.5, 7.0, 6.5];
        let locality = vec![0.8, 0.7, 0.9, 0.75, 0.85];

        let metrics = suite
            .evaluate_network_efficiency(&bandwidth, &locality, 2, 100)
            .unwrap();

        assert!(metrics.bandwidth_efficiency > 0.0);
        assert!(metrics.data_locality > 0.0);
        assert!(metrics.fault_tolerance > 0.0);
    }

    #[test]
    fn test_domain_metrics_trait() {
        let suite = EdgeAISuite::new();

        assert_eq!(suite.domain_name(), "Edge AI & Federated Learning");
        assert!(suite.available_metrics().contains(&"edge_latency"));
        assert!(suite.available_metrics().contains(&"energy_efficiency"));

        let descriptions = suite.metric_descriptions();
        assert!(descriptions.contains_key("edge_latency"));
        assert!(descriptions.contains_key("communication_efficiency"));
    }
}
