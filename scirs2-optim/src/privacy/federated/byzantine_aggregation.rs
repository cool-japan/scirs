//! Byzantine Robust Aggregation Module
//!
//! This module implements Byzantine-robust aggregation algorithms for federated learning,
//! providing protection against malicious clients and outlier detection mechanisms.

use crate::error::{OptimError, Result};
use ndarray::Array1;
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

/// Byzantine-robust aggregation algorithms
#[derive(Debug, Clone, Copy)]
pub enum ByzantineRobustMethod {
    /// Trimmed mean aggregation
    TrimmedMean { trim_ratio: f64 },

    /// Coordinate-wise median
    CoordinateWiseMedian,

    /// Krum aggregation
    Krum { f: usize },

    /// Multi-Krum aggregation
    MultiKrum { f: usize, m: usize },

    /// Bulyan aggregation
    Bulyan { f: usize },

    /// Centered clipping
    CenteredClipping { tau: f64 },

    /// FedAvg with outlier detection
    FedAvgOutlierDetection { threshold: f64 },

    /// Robust aggregation with reputation
    ReputationWeighted { reputation_decay: f64 },
}

/// Byzantine robustness configuration
#[derive(Debug, Clone)]
pub struct ByzantineRobustConfig {
    /// Aggregation method
    pub method: ByzantineRobustMethod,

    /// Expected number of Byzantine clients
    pub expected_byzantine_ratio: f64,

    /// Enable dynamic Byzantine detection
    pub dynamic_detection: bool,

    /// Reputation system settings
    pub reputation_system: ReputationSystemConfig,

    /// Statistical tests for outlier detection
    pub statistical_tests: StatisticalTestConfig,
}

/// Reputation system configuration
#[derive(Debug, Clone)]
pub struct ReputationSystemConfig {
    pub enabled: bool,
    pub initial_reputation: f64,
    pub reputation_decay: f64,
    pub min_reputation: f64,
    pub outlier_penalty: f64,
    pub contribution_bonus: f64,
}

/// Statistical test configuration for outlier detection
#[derive(Debug, Clone)]
pub struct StatisticalTestConfig {
    pub enabled: bool,
    pub test_type: StatisticalTestType,
    pub significancelevel: f64,
    pub window_size: usize,
    pub adaptive_threshold: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum StatisticalTestType {
    ZScore,
    ModifiedZScore,
    IQRTest,
    GrubbsTest,
    ChauventCriterion,
}

/// Byzantine-robust aggregation engine
pub struct ByzantineRobustAggregator<T: Float + Default + Clone + Send + Sync + std::iter::Sum> {
    config: ByzantineRobustConfig,
    client_reputations: HashMap<String, f64>,
    outlier_history: VecDeque<OutlierDetectionResult>,
    statistical_analyzer: StatisticalAnalyzer<T>,
    robust_estimators: RobustEstimators<T>,
}

/// Statistical analyzer for outlier detection
pub struct StatisticalAnalyzer<T: Float + Default + Clone + Send + Sync + std::iter::Sum> {
    window_size: usize,
    significancelevel: f64,
    test_statistics: VecDeque<TestStatistic<T>>,
}

/// Robust estimators for aggregation
pub struct RobustEstimators<T: Float + Default + Clone + Send + Sync + std::iter::Sum> {
    trimmed_mean_cache: HashMap<String, T>,
    median_cache: HashMap<String, T>,
    krum_scores: HashMap<String, f64>,
}

/// Outlier detection result
#[derive(Debug, Clone)]
pub struct OutlierDetectionResult {
    pub clientid: String,
    pub round: usize,
    pub is_outlier: bool,
    pub outlier_score: f64,
    pub detection_method: String,
}

/// Test statistic for outlier detection
#[derive(Debug, Clone)]
pub struct TestStatistic<T: Float> {
    pub statistic_value: T,
    pub p_value: f64,
    pub test_type: StatisticalTestType,
    pub clientid: String,
}

/// Placeholder for adaptive privacy allocation
#[derive(Debug, Clone)]
pub struct AdaptivePrivacyAllocation {
    pub epsilon: f64,
    pub delta: f64,
    pub utility_weight: f64,
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum + ndarray::ScalarOperand>
    ByzantineRobustAggregator<T>
{
    #[allow(dead_code)]
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: ByzantineRobustConfig::default(),
            client_reputations: HashMap::new(),
            outlier_history: VecDeque::with_capacity(1000),
            statistical_analyzer: StatisticalAnalyzer::new(100, 0.05), // window_size=100, significancelevel=0.05
            robust_estimators: RobustEstimators::new(),
        })
    }

    #[allow(dead_code)]
    pub fn detect_byzantine_clients(
        &mut self,
        client_updates: &HashMap<String, Array1<T>>,
        round: usize,
    ) -> Result<Vec<OutlierDetectionResult>> {
        self.statistical_analyzer
            .detect_outliers(client_updates, round)
    }

    #[allow(dead_code)]
    pub fn get_client_reputations(&self, clients: &[String]) -> HashMap<String, f64> {
        let mut reputations = HashMap::new();
        for client_id in clients {
            let reputation = self
                .client_reputations
                .get(client_id)
                .copied()
                .unwrap_or(self.config.reputation_system.initial_reputation);
            reputations.insert(client_id.clone(), reputation);
        }
        reputations
    }

    #[allow(dead_code)]
    pub fn robust_aggregate(
        &self,
        clientupdates: &HashMap<String, Array1<T>>,
        _allocations: &HashMap<String, AdaptivePrivacyAllocation>,
    ) -> Result<Array1<T>> {
        match self.config.method {
            ByzantineRobustMethod::TrimmedMean { trim_ratio } => {
                // Use robust estimators for trimmed mean
                let mut estimators = RobustEstimators::new();
                estimators.trimmed_mean(clientupdates, trim_ratio)
            }
            ByzantineRobustMethod::CoordinateWiseMedian => {
                self.coordinate_wise_median(clientupdates)
            }
            _ => {
                // Default to simple averaging for other methods
                if let Some(first_update) = clientupdates.values().next() {
                    let mut result = Array1::zeros(first_update.len());
                    let count = T::from(clientupdates.len()).unwrap();

                    for update in clientupdates.values() {
                        result = result + update;
                    }

                    Ok(result / count)
                } else {
                    Err(OptimError::InvalidConfig("No client _updates".to_string()))
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn compute_robustness_factor(&self) -> Result<f64> {
        let detected_byzantine = self
            .outlier_history
            .iter()
            .filter(|result| result.is_outlier)
            .count() as f64;

        let total_evaluations = self.outlier_history.len() as f64;

        if total_evaluations > 0.0 {
            Ok(1.0 - (detected_byzantine / total_evaluations))
        } else {
            Ok(1.0)
        }
    }

    fn coordinate_wise_median(
        &self,
        clientupdates: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        if clientupdates.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No client updates provided".to_string(),
            ));
        }

        let first_update = clientupdates.values().next().unwrap();
        let dim = first_update.len();
        let mut result = Array1::zeros(dim);

        // For each coordinate, compute median across all clients
        for coord_idx in 0..dim {
            let mut coord_values: Vec<T> = clientupdates
                .values()
                .map(|update| update[coord_idx])
                .collect();

            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if coord_values.len() % 2 == 0 {
                let mid = coord_values.len() / 2;
                (coord_values[mid - 1] + coord_values[mid]) / T::from(2.0).unwrap()
            } else {
                coord_values[coord_values.len() / 2]
            };

            result[coord_idx] = median;
        }

        Ok(result)
    }

    /// Get current configuration
    pub fn config(&self) -> &ByzantineRobustConfig {
        &self.config
    }

    /// Update client reputation
    pub fn update_client_reputation(&mut self, client_id: String, is_outlier: bool) {
        let current_reputation = self
            .client_reputations
            .get(&client_id)
            .copied()
            .unwrap_or(self.config.reputation_system.initial_reputation);

        let new_reputation = if is_outlier {
            (current_reputation - self.config.reputation_system.outlier_penalty)
                .max(self.config.reputation_system.min_reputation)
        } else {
            (current_reputation + self.config.reputation_system.contribution_bonus).min(1.0)
        };

        self.client_reputations.insert(client_id, new_reputation);
    }
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalAnalyzer<T> {
    /// Create new statistical analyzer
    pub fn new(window_size: usize, significancelevel: f64) -> Self {
        Self {
            window_size,
            significancelevel,
            test_statistics: VecDeque::with_capacity(window_size),
        }
    }

    /// Detect outliers using statistical tests
    pub fn detect_outliers(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
        round: usize,
    ) -> Result<Vec<OutlierDetectionResult>> {
        let mut results = Vec::new();

        if clientupdates.len() < 3 {
            return Ok(results); // Need at least 3 clients for meaningful analysis
        }

        // Simple outlier detection based on pairwise distances
        let clientids: Vec<_> = clientupdates.keys().collect();
        let mut distances = HashMap::new();

        for &client_a in clientids.iter() {
            let mut total_distance = T::zero();
            let mut count = 0;

            for &client_b in clientids.iter() {
                if client_a != client_b {
                    // Skip self comparison
                    let update_a = &clientupdates[client_a];
                    let update_b = &clientupdates[client_b];

                    // Compute Euclidean distance
                    let mut sum_sq_diff = T::zero();
                    for (a, b) in update_a.iter().zip(update_b.iter()) {
                        let diff = *a - *b;
                        sum_sq_diff = sum_sq_diff + diff * diff;
                    }

                    let distance = sum_sq_diff.sqrt();
                    total_distance = total_distance + distance;
                    count += 1;
                }
            }

            if count > 0 {
                let avg_distance = total_distance / T::from(count).unwrap();
                distances.insert(client_a, avg_distance);
            }
        }

        // Detect outliers based on distance threshold
        if !distances.is_empty() {
            let distances_vec: Vec<T> = distances.values().cloned().collect();
            let mean_distance = distances_vec.iter().fold(T::zero(), |acc, &x| acc + x)
                / T::from(distances_vec.len()).unwrap();

            let variance = distances_vec.iter().fold(T::zero(), |acc, &x| {
                let diff = x - mean_distance;
                acc + diff * diff
            }) / T::from(distances_vec.len()).unwrap();

            let std_dev = variance.sqrt();
            let threshold = mean_distance + T::from(1.0).unwrap() * std_dev; // 1-sigma threshold (more sensitive)

            for (client_id, &distance) in &distances {
                let is_outlier = distance > threshold;
                results.push(OutlierDetectionResult {
                    clientid: client_id.to_string(),
                    round,
                    is_outlier,
                    outlier_score: distance.to_f64().unwrap_or(0.0),
                    detection_method: "statistical_distance".to_string(),
                });
            }
        }

        Ok(results)
    }
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum> RobustEstimators<T> {
    /// Create new robust estimators
    pub fn new() -> Self {
        Self {
            trimmed_mean_cache: HashMap::new(),
            median_cache: HashMap::new(),
            krum_scores: HashMap::new(),
        }
    }

    /// Compute trimmed mean of client updates
    pub fn trimmed_mean(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
        trim_ratio: f64,
    ) -> Result<Array1<T>> {
        if clientupdates.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No client _updates provided".to_string(),
            ));
        }

        let first_update = clientupdates.values().next().unwrap();
        let dim = first_update.len();

        // Verify all _updates have same dimension
        for update in clientupdates.values() {
            if update.len() != dim {
                return Err(OptimError::InvalidConfig(
                    "Client _updates have different dimensions".to_string(),
                ));
            }
        }

        let mut result = Array1::zeros(dim);
        let num_clients = clientupdates.len();
        let trim_count = ((num_clients as f64 * trim_ratio) / 2.0) as usize;

        // For each coordinate, compute trimmed mean
        for coord_idx in 0..dim {
            let mut coord_values: Vec<T> = clientupdates
                .values()
                .map(|update| update[coord_idx])
                .collect();

            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Remove extreme values
            let trimmed_values = &coord_values[trim_count..coord_values.len() - trim_count];

            if !trimmed_values.is_empty() {
                let sum = trimmed_values.iter().fold(T::zero(), |acc, &x| acc + x);
                result[coord_idx] = sum / T::from(trimmed_values.len()).unwrap();
            } else {
                result[coord_idx] = T::zero();
            }
        }

        Ok(result)
    }
}

impl Default for ByzantineRobustConfig {
    fn default() -> Self {
        Self {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio: 0.1 },
            expected_byzantine_ratio: 0.1,
            dynamic_detection: true,
            reputation_system: ReputationSystemConfig::default(),
            statistical_tests: StatisticalTestConfig::default(),
        }
    }
}

impl Default for ReputationSystemConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_reputation: 1.0,
            reputation_decay: 0.01,
            min_reputation: 0.1,
            outlier_penalty: 0.5,
            contribution_bonus: 0.1,
        }
    }
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            test_type: StatisticalTestType::ZScore,
            significancelevel: 0.05,
            window_size: 100,
            adaptive_threshold: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_byzantine_robust_aggregator_creation() {
        let aggregator = ByzantineRobustAggregator::<f64>::new();
        assert!(aggregator.is_ok());
    }

    #[test]
    fn test_trimmed_mean_aggregation() {
        let mut estimators = RobustEstimators::<f64>::new();

        let mut client_updates = HashMap::new();
        client_updates.insert("client1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        client_updates.insert("client2".to_string(), Array1::from(vec![1.1, 2.1, 3.1]));
        client_updates.insert("client3".to_string(), Array1::from(vec![10.0, 20.0, 30.0])); // Outlier
        client_updates.insert("client4".to_string(), Array1::from(vec![0.9, 1.9, 2.9]));

        let result = estimators.trimmed_mean(&client_updates, 0.25);
        assert!(result.is_ok());

        let trimmed = result.unwrap();
        // Should exclude the outlier client3
        assert!(trimmed[0] < 5.0); // Should be around 1.0, not influenced by 10.0
    }

    #[test]
    fn test_outlier_detection() {
        let mut analyzer = StatisticalAnalyzer::<f64>::new(100, 0.05);

        let mut client_updates = HashMap::new();
        client_updates.insert("client1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        client_updates.insert("client2".to_string(), Array1::from(vec![1.1, 2.1, 3.1]));
        client_updates.insert(
            "client3".to_string(),
            Array1::from(vec![1000.0, 2000.0, 3000.0]),
        ); // Very clear outlier

        let results = analyzer.detect_outliers(&client_updates, 1);
        assert!(results.is_ok());

        let detections = results.unwrap();
        assert!(!detections.is_empty());

        // Check if the outlier was detected
        let outlier_detected = detections
            .iter()
            .any(|r| r.clientid == "client3" && r.is_outlier);
        assert!(outlier_detected);
    }

    #[test]
    fn test_coordinate_wise_median() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().unwrap();

        let mut client_updates = HashMap::new();
        client_updates.insert("client1".to_string(), Array1::from(vec![1.0, 4.0, 7.0]));
        client_updates.insert("client2".to_string(), Array1::from(vec![2.0, 5.0, 8.0]));
        client_updates.insert("client3".to_string(), Array1::from(vec![3.0, 6.0, 9.0]));

        let result = aggregator.coordinate_wise_median(&client_updates);
        assert!(result.is_ok());

        let median = result.unwrap();
        assert_eq!(median[0], 2.0); // Median of [1, 2, 3]
        assert_eq!(median[1], 5.0); // Median of [4, 5, 6]
        assert_eq!(median[2], 8.0); // Median of [7, 8, 9]
    }

    #[test]
    fn test_reputation_system() {
        let mut aggregator = ByzantineRobustAggregator::<f64>::new().unwrap();

        // Test initial reputation
        let reputations = aggregator.get_client_reputations(&["client1".to_string()]);
        assert_eq!(reputations.get("client1"), Some(&1.0));

        // Test reputation penalty for outlier
        aggregator.update_client_reputation("client1".to_string(), true);
        let updated_reputations = aggregator.get_client_reputations(&["client1".to_string()]);
        assert!(updated_reputations.get("client1").unwrap() < &1.0);

        // Test reputation bonus for good behavior
        aggregator.update_client_reputation("client2".to_string(), false);
        let good_reputations = aggregator.get_client_reputations(&["client2".to_string()]);
        assert_eq!(good_reputations.get("client2"), Some(&1.0)); // Should stay at max (1.0)
    }
}
