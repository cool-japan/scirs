//! Byzantine Fault Tolerance for Federated Learning
//!
//! This module implements Byzantine-robust aggregation algorithms that can
//! tolerate malicious participants in federated learning scenarios.

#![allow(dead_code)]

use crate::error::{OptimError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Byzantine fault tolerant aggregator
pub struct ByzantineTolerantAggregator<T: Float> {
    /// Configuration for Byzantine tolerance
    config: ByzantineConfig,

    /// Participant reputation scores
    reputation_scores: HashMap<String, ReputationScore>,

    /// History of participant behavior
    behavior_history: HashMap<String, BehaviorHistory>,

    /// Anomaly detection engine
    anomaly_detector: AnomalyDetector<T>,

    /// Statistical analysis engine
    statistics_engine: StatisticalAnalysis<T>,

    /// Gradient verification system
    gradient_verifier: GradientVerifier<T>,
}

#[derive(Debug, Clone)]
pub struct ByzantineConfig {
    /// Maximum number of Byzantine participants to tolerate
    pub max_byzantine: usize,

    /// Minimum number of participants required
    pub min_participants: usize,

    /// Aggregation method for Byzantine tolerance
    pub aggregation_method: ByzantineAggregationMethod,

    /// Anomaly detection threshold
    pub anomaly_threshold: f64,

    /// Reputation decay factor
    pub reputation_decay: f64,

    /// Enable gradient verification
    pub gradient_verification: bool,

    /// Statistical outlier detection
    pub outlier_detection: OutlierDetectionMethod,

    /// Consensus threshold for decision making
    pub consensus_threshold: f64,
}

/// Byzantine-robust aggregation methods
#[derive(Debug, Clone, Copy)]
pub enum ByzantineAggregationMethod {
    /// Trimmed mean (remove extreme values)
    TrimmedMean,

    /// Coordinate-wise median
    CoordinateMedian,

    /// Krum algorithm (select most representative gradient)
    Krum,

    /// Multi-Krum (select multiple representative gradients)
    MultiKrum,

    /// Bulyan algorithm (robust mean estimation)
    Bulyan,

    /// FoolsGold (defend against Sybil attacks)
    FoolsGold,

    /// FLAME (Federated Learning with Approximate Model Enhancement)
    FLAME,

    /// Median-based aggregation
    Median,

    /// Geometric median
    GeometricMedian,
}

/// Outlier detection methods
#[derive(Debug, Clone, Copy)]
pub enum OutlierDetectionMethod {
    /// Z-score based detection
    ZScore,

    /// Interquartile range method
    IQR,

    /// Isolation forest
    IsolationForest,

    /// Local outlier factor
    LocalOutlierFactor,

    /// Mahalanobis distance
    MahalanobisDistance,
}

/// Reputation score for participants
#[derive(Debug, Clone)]
pub struct ReputationScore {
    /// Current reputation score (0.0 to 1.0)
    pub score: f64,

    /// Number of successful aggregations
    pub successful_aggregations: usize,

    /// Number of detected anomalies
    pub detected_anomalies: usize,

    /// Average gradient quality score
    pub gradient_quality: f64,

    /// Consistency score across rounds
    pub consistency_score: f64,

    /// Trust level
    pub trust_level: TrustLevel,
}

/// Trust levels for participants
#[derive(Debug, Clone, Copy)]
pub enum TrustLevel {
    /// Highly trusted participant
    High,

    /// Moderately trusted participant
    Medium,

    /// Low trust participant
    Low,

    /// Blacklisted participant
    Blacklisted,
}

/// Behavior history for participants
#[derive(Debug, Clone)]
pub struct BehaviorHistory {
    /// History of gradient norms
    pub gradient_norms: Vec<f64>,

    /// History of gradient directions (cosine similarities)
    pub gradient_similarities: Vec<f64>,

    /// History of participation patterns
    pub participation_pattern: Vec<bool>,

    /// History of anomaly scores
    pub anomaly_scores: Vec<f64>,

    /// Number of rounds participated
    pub rounds_participated: usize,
}

/// Anomaly detection engine
pub struct AnomalyDetector<T: Float> {
    /// Detection threshold
    threshold: f64,

    /// Historical gradient statistics
    gradient_stats: GradientStatistics<T>,

    /// Pattern recognition model
    pattern_model: PatternModel<T>,
}

/// Gradient statistics for anomaly detection
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float> {
    /// Mean gradient
    pub mean: Array1<T>,

    /// Gradient covariance matrix
    pub covariance: Array2<T>,

    /// Historical gradient norms
    pub norm_history: Vec<T>,

    /// Gradient direction patterns
    pub direction_patterns: Array2<T>,
}

/// Pattern recognition model for detecting malicious behavior
pub struct PatternModel<T: Float> {
    /// Reference patterns for normal behavior
    normal_patterns: Vec<Array1<T>>,

    /// Reference patterns for attack behaviors
    attack_patterns: Vec<Array1<T>>,

    /// Pattern matching threshold
    matching_threshold: f64,
}

/// Statistical analysis engine
pub struct StatisticalAnalysis<T: Float> {
    /// Window size for statistical analysis
    window_size: usize,

    /// Statistical measures
    measures: StatisticalMeasures<T>,
}

/// Statistical measures for gradient analysis
#[derive(Debug, Clone)]
pub struct StatisticalMeasures<T: Float> {
    /// Mean of gradients
    pub mean: Array1<T>,

    /// Standard deviation
    pub std_dev: Array1<T>,

    /// Median
    pub median: Array1<T>,

    /// Interquartile range
    pub iqr: Array1<T>,

    /// Skewness
    pub skewness: Array1<T>,

    /// Kurtosis
    pub kurtosis: Array1<T>,
}

/// Gradient verification system
pub struct GradientVerifier<T: Float> {
    /// Expected gradient properties
    expected_properties: GradientProperties<T>,

    /// Verification rules
    verification_rules: Vec<VerificationRule<T>>,
}

/// Expected gradient properties
#[derive(Debug, Clone)]
pub struct GradientProperties<T: Float> {
    /// Expected norm range
    pub norm_range: (T, T),

    /// Expected sparsity
    pub sparsity_threshold: f64,

    /// Expected direction consistency
    pub direction_consistency: f64,
}

/// Verification rule for gradients
pub struct VerificationRule<T: Float> {
    /// Rule name
    pub name: String,

    /// Rule function
    pub rule_fn: Box<dyn Fn(&Array1<T>) -> bool + Send + Sync>,

    /// Rule weight in verification
    pub weight: f64,
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand> ByzantineTolerantAggregator<T> {
    /// Create new Byzantine tolerant aggregator
    pub fn new(config: ByzantineConfig) -> Self {
        let anomaly_threshold = config.anomaly_threshold;
        Self {
            config,
            reputation_scores: HashMap::new(),
            behavior_history: HashMap::new(),
            anomaly_detector: AnomalyDetector::new(anomaly_threshold),
            statistics_engine: StatisticalAnalysis::new(100), // 100-round window
            gradient_verifier: GradientVerifier::new(),
        }
    }

    /// Perform Byzantine-robust aggregation
    pub fn byzantine_robust_aggregate(
        &mut self,
        participant_gradients: &HashMap<String, Array1<T>>,
    ) -> Result<ByzantineAggregationResult<T>> {
        // Step 1: Pre-filtering based on reputation
        let filtered_participants = self.filter_by_reputation(participant_gradients)?;

        // Step 2: Anomaly detection
        let anomaly_results = self.detect_anomalies(&filtered_participants)?;

        // Step 3: Statistical outlier detection
        let outlier_results = self.detect_statistical_outliers(&filtered_participants)?;

        // Step 4: Gradient verification
        let verification_results = if self.config.gradient_verification {
            self.verify_gradients(&filtered_participants)?
        } else {
            HashMap::new()
        };

        // Step 5: Identify Byzantine participants
        let byzantine_participants = self.identify_byzantine_participants(
            &anomaly_results,
            &outlier_results,
            &verification_results,
        )?;

        // Step 6: Select honest participants
        let honest_participants =
            self.select_honest_participants(&filtered_participants, &byzantine_participants)?;

        // Step 7: Perform robust aggregation
        let aggregate = self.perform_robust_aggregation(&honest_participants)?;

        // Step 8: Update participant reputations
        self.update_reputations(&honest_participants, &byzantine_participants)?;

        Ok(ByzantineAggregationResult {
            aggregate,
            honest_participants: honest_participants.keys().cloned().collect(),
            byzantine_participants,
            reputation_updates: self.get_reputation_updates(),
            aggregation_method: self.config.aggregation_method,
            confidence_score: self.calculate_confidence_score(&honest_participants),
        })
    }

    /// Filter participants based on reputation scores
    fn filter_by_reputation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut filtered = HashMap::new();

        for (participant_id, gradient) in gradients {
            if let Some(reputation) = self.reputation_scores.get(participant_id) {
                if !matches!(reputation.trust_level, TrustLevel::Blacklisted) {
                    filtered.insert(participant_id.clone(), gradient.clone());
                }
            } else {
                // New participant - allow with medium trust
                filtered.insert(participant_id.clone(), gradient.clone());
            }
        }

        Ok(filtered)
    }

    /// Detect anomalies in gradients
    fn detect_anomalies(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, AnomalyScore>> {
        let mut anomaly_results = HashMap::new();

        for (participant_id, gradient) in gradients {
            let anomaly_score = self.anomaly_detector.detect_anomaly(gradient)?;
            anomaly_results.insert(participant_id.clone(), anomaly_score);
        }

        Ok(anomaly_results)
    }

    /// Detect statistical outliers
    fn detect_statistical_outliers(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, OutlierScore>> {
        let mut outlier_results = HashMap::new();

        // Collect all gradients for statistical analysis
        let gradient_values: Vec<&Array1<T>> = gradients.values().collect();
        let stats = self
            .statistics_engine
            .compute_statistics(&gradient_values)?;

        for (participant_id, gradient) in gradients {
            let outlier_score = self.compute_outlier_score(gradient, &stats)?;
            outlier_results.insert(participant_id.clone(), outlier_score);
        }

        Ok(outlier_results)
    }

    /// Verify gradients using verification rules
    fn verify_gradients(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, VerificationScore>> {
        let mut verification_results = HashMap::new();

        for (participant_id, gradient) in gradients {
            let verification_score = self.gradient_verifier.verify_gradient(gradient)?;
            verification_results.insert(participant_id.clone(), verification_score);
        }

        Ok(verification_results)
    }

    /// Identify Byzantine participants based on multiple criteria
    fn identify_byzantine_participants(
        &self,
        anomaly_results: &HashMap<String, AnomalyScore>,
        outlier_results: &HashMap<String, OutlierScore>,
        verification_results: &HashMap<String, VerificationScore>,
    ) -> Result<Vec<String>> {
        let mut byzantine_participants = Vec::new();

        for participant_id in anomaly_results.keys() {
            let anomaly_score = anomaly_results.get(participant_id).unwrap();
            let outlier_score = outlier_results.get(participant_id).unwrap();
            let verification_score = verification_results.get(participant_id);

            // Combine scores to determine if participant is Byzantine
            let combined_score =
                self.compute_byzantine_score(anomaly_score, outlier_score, verification_score);

            if combined_score > self.config.anomaly_threshold {
                byzantine_participants.push(participant_id.clone());
            }
        }

        Ok(byzantine_participants)
    }

    /// Select honest participants for aggregation
    fn select_honest_participants(
        &self,
        all_participants: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut honest_participants = HashMap::new();

        for (participant_id, gradient) in all_participants {
            if !byzantine_participants.contains(participant_id) {
                honest_participants.insert(participant_id.clone(), gradient.clone());
            }
        }

        // Ensure we have enough honest _participants
        if honest_participants.len() < self.config.min_participants {
            return Err(OptimError::InvalidConfig(
                "Insufficient honest _participants for aggregation".to_string(),
            ));
        }

        Ok(honest_participants)
    }

    /// Perform robust aggregation using the configured method
    fn perform_robust_aggregation(
        &self,
        honest_gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        match self.config.aggregation_method {
            ByzantineAggregationMethod::TrimmedMean => {
                self.trimmed_mean_aggregation(honest_gradients)
            }
            ByzantineAggregationMethod::CoordinateMedian => {
                self.coordinate_median_aggregation(honest_gradients)
            }
            ByzantineAggregationMethod::Krum => self.krum_aggregation(honest_gradients),
            ByzantineAggregationMethod::MultiKrum => self.multi_krum_aggregation(honest_gradients),
            ByzantineAggregationMethod::Bulyan => self.bulyan_aggregation(honest_gradients),
            ByzantineAggregationMethod::FoolsGold => self.fools_gold_aggregation(honest_gradients),
            ByzantineAggregationMethod::FLAME => self.flame_aggregation(honest_gradients),
            ByzantineAggregationMethod::Median => self.median_aggregation(honest_gradients),
            ByzantineAggregationMethod::GeometricMedian => {
                self.geometric_median_aggregation(honest_gradients)
            }
        }
    }

    /// Trimmed mean aggregation
    fn trimmed_mean_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        let values: Vec<&Array1<T>> = gradients.values().collect();
        let first_gradient = values[0];
        let dim = first_gradient.len();
        let mut result = Array1::zeros(dim);

        // For each coordinate, compute trimmed mean
        for i in 0..dim {
            let mut coord_values: Vec<T> = values.iter().map(|g| g[i]).collect();
            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Remove top and bottom 10% (trim parameter), but ensure at least 1 element is trimmed if we have outliers
            let trim_count = std::cmp::max(1, (coord_values.len() as f64 * 0.1) as usize);
            let start_idx = std::cmp::min(trim_count, coord_values.len() / 2);
            let end_idx = std::cmp::max(coord_values.len() - trim_count, coord_values.len() / 2);
            let trimmed_values = &coord_values[start_idx..end_idx];

            if !trimmed_values.is_empty() {
                let sum: T = trimmed_values
                    .iter()
                    .copied()
                    .fold(T::zero(), |acc, x| acc + x);
                result[i] = sum / T::from(trimmed_values.len()).unwrap();
            }
        }

        Ok(result)
    }

    /// Coordinate-wise median aggregation
    fn coordinate_median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        let values: Vec<&Array1<T>> = gradients.values().collect();
        let first_gradient = values[0];
        let dim = first_gradient.len();
        let mut result = Array1::zeros(dim);

        // For each coordinate, compute median
        for i in 0..dim {
            let mut coord_values: Vec<T> = values.iter().map(|g| g[i]).collect();
            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let median = if coord_values.len() % 2 == 0 {
                let mid = coord_values.len() / 2;
                (coord_values[mid - 1] + coord_values[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
            } else {
                coord_values[coord_values.len() / 2]
            };

            result[i] = median;
        }

        Ok(result)
    }

    /// Krum aggregation (select single most representative gradient)
    fn krum_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        let participants: Vec<&String> = gradients.keys().collect();
        let mut min_score = T::infinity();
        let mut selected_gradient = None;

        // For each gradient, compute Krum score
        for (i, participant) in participants.iter().enumerate() {
            let gradient = &gradients[*participant];
            let mut score = T::zero();
            let mut distances = Vec::new();

            // Compute distances to all other gradients
            for (j, other_participant) in participants.iter().enumerate() {
                if i != j {
                    let other_gradient = &gradients[*other_participant];
                    let distance = self.compute_euclidean_distance(gradient, other_gradient)?;
                    distances.push(distance);
                }
            }

            // Sort distances and take sum of smallest (n - f - 2) distances
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let take_count = (participants.len() - self.config.max_byzantine - 2).max(1);

            for &distance in distances.iter().take(take_count) {
                score = score + distance;
            }

            if score < min_score {
                min_score = score;
                selected_gradient = Some(gradient.clone());
            }
        }

        selected_gradient.ok_or_else(|| {
            OptimError::InvalidConfig("Failed to select gradient with Krum".to_string())
        })
    }

    /// Multi-Krum aggregation (select multiple representative gradients)
    fn multi_krum_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        // Select top-k gradients using Krum scoring
        let k = (gradients.len() - self.config.max_byzantine).max(1);
        let selected_gradients = self.select_top_k_krum(gradients, k)?;

        // Average the selected gradients
        let first_gradient = selected_gradients.values().next().unwrap();
        let mut result = Array1::zeros(first_gradient.len());

        for gradient in selected_gradients.values() {
            result = result + gradient;
        }

        result = result / T::from(selected_gradients.len()).unwrap();
        Ok(result)
    }

    /// Bulyan aggregation
    fn bulyan_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        // Bulyan combines Multi-Krum with trimmed mean
        let selected_gradients =
            self.select_top_k_krum(gradients, gradients.len() - self.config.max_byzantine)?;
        self.trimmed_mean_aggregation(&selected_gradients)
    }

    /// FoolsGold aggregation (defend against Sybil attacks)
    fn fools_gold_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        // Compute learning rates based on historical cosine similarities
        let learning_rates = self.compute_fools_gold_weights(gradients)?;

        // Weighted aggregation
        let first_gradient = gradients.values().next().unwrap();
        let mut result = Array1::zeros(first_gradient.len());
        let mut total_weight = T::zero();

        for (participant_id, gradient) in gradients {
            let weight = learning_rates
                .get(participant_id)
                .copied()
                .unwrap_or(T::one());
            result = result + gradient * weight;
            total_weight = total_weight + weight;
        }

        if total_weight > T::zero() {
            result = result / total_weight;
        }

        Ok(result)
    }

    /// FLAME aggregation
    fn flame_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        // FLAME uses clustering to identify and filter out Byzantine gradients
        let clusters = self.cluster_gradients(gradients)?;
        let largest_cluster = self.find_largest_cluster(&clusters)?;

        // Aggregate gradients from the largest cluster
        self.average_aggregation(&largest_cluster)
    }

    /// Simple median aggregation
    fn median_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        self.coordinate_median_aggregation(gradients)
    }

    /// Geometric median aggregation
    fn geometric_median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        // Geometric median using Weiszfeld's algorithm
        let values: Vec<&Array1<T>> = gradients.values().collect();
        let first_gradient = values[0];
        let mut current = first_gradient.clone();

        // Iterative algorithm
        for _ in 0..100 {
            // Maximum iterations
            let mut numerator = Array1::zeros(current.len());
            let mut denominator = T::zero();

            for &gradient in &values {
                let distance = self
                    .compute_euclidean_distance(&current, gradient)
                    .unwrap_or(T::one());
                if distance > T::zero() {
                    let weight = T::one() / distance;
                    numerator = numerator + gradient * weight;
                    denominator = denominator + weight;
                }
            }

            if denominator > T::zero() {
                let new_estimate = numerator / denominator;

                // Check convergence
                let change = self
                    .compute_euclidean_distance(&current, &new_estimate)
                    .unwrap_or(T::zero());
                if change < num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()) {
                    break;
                }

                current = new_estimate;
            }
        }

        Ok(current)
    }

    /// Simple average aggregation
    fn average_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        let first_gradient = gradients.values().next().unwrap();
        let mut result = Array1::zeros(first_gradient.len());

        for gradient in gradients.values() {
            result = result + gradient;
        }

        result = result / T::from(gradients.len()).unwrap();
        Ok(result)
    }

    /// Update participant reputations based on aggregation results
    fn update_reputations(
        &mut self,
        honest_participants: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<()> {
        // Update honest _participants (increase reputation)
        for participant_id in honest_participants.keys() {
            let reputation = self
                .reputation_scores
                .entry(participant_id.clone())
                .or_insert_with(|| ReputationScore::new());

            reputation.successful_aggregations += 1;
            reputation.score = (reputation.score * 0.9 + 0.1).min(1.0);

            // Update trust level based on score
            reputation.trust_level = match reputation.score {
                s if s >= 0.8 => TrustLevel::High,
                s if s >= 0.5 => TrustLevel::Medium,
                _ => TrustLevel::Low,
            };
        }

        // Update Byzantine _participants (decrease reputation)
        for participant_id in byzantine_participants {
            let reputation = self
                .reputation_scores
                .entry(participant_id.clone())
                .or_insert_with(|| ReputationScore::new());

            reputation.detected_anomalies += 1;
            reputation.score = (reputation.score * 0.5).max(0.0);

            // Update trust level
            reputation.trust_level = if reputation.score < 0.1 {
                TrustLevel::Blacklisted
            } else if reputation.score < 0.3 {
                TrustLevel::Low
            } else {
                TrustLevel::Medium
            };
        }

        Ok(())
    }

    /// Compute Byzantine score combining multiple detection methods
    fn compute_byzantine_score(
        &self,
        anomaly_score: &AnomalyScore,
        outlier_score: &OutlierScore,
        verification_score: Option<&VerificationScore>,
    ) -> f64 {
        let mut combined_score = 0.0;

        // Weight anomaly score (40%)
        combined_score += anomaly_score.score * 0.4;

        // Weight outlier score (30%)
        combined_score += outlier_score.score * 0.3;

        // Weight verification score (30%)
        if let Some(verification) = verification_score {
            combined_score += (1.0 - verification.score) * 0.3;
        }

        combined_score
    }

    /// Calculate confidence score for aggregation
    fn calculate_confidence_score(&self, honest_participants: &HashMap<String, Array1<T>>) -> f64 {
        let honest_count = honest_participants.len() as f64;
        let total_expected = (self.config.min_participants + self.config.max_byzantine) as f64;

        (honest_count / total_expected).min(1.0)
    }

    /// Compute Euclidean distance between two gradients
    fn compute_euclidean_distance(&self, a: &Array1<T>, b: &Array1<T>) -> Result<T> {
        if a.len() != b.len() {
            return Err(OptimError::InvalidConfig(
                "Gradient dimensions don't match".to_string(),
            ));
        }

        let mut sum = T::zero();
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = *x - *y;
            sum = sum + diff * diff;
        }

        Ok(sum.sqrt())
    }

    /// Select top-k gradients using Krum scoring
    fn select_top_k_krum(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        k: usize,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut scores = Vec::new();
        let participants: Vec<&String> = gradients.keys().collect();

        // Compute Krum scores for all participants
        for (i, participant) in participants.iter().enumerate() {
            let gradient = &gradients[*participant];
            let mut score = T::zero();
            let mut distances = Vec::new();

            for (j, other_participant) in participants.iter().enumerate() {
                if i != j {
                    let other_gradient = &gradients[*other_participant];
                    let distance = self.compute_euclidean_distance(gradient, other_gradient)?;
                    distances.push(distance);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let take_count = (participants.len() - self.config.max_byzantine - 2).max(1);

            for &distance in distances.iter().take(take_count) {
                score = score + distance;
            }

            scores.push(((*participant).clone(), score));
        }

        // Sort by score and select top-k
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut selected = HashMap::new();
        for (participant_id, _) in scores.into_iter().take(k) {
            if let Some(gradient) = gradients.get(&participant_id) {
                selected.insert(participant_id, gradient.clone());
            }
        }

        Ok(selected)
    }

    /// Compute FoolsGold weights based on historical similarities
    fn compute_fools_gold_weights(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, T>> {
        let mut weights = HashMap::new();

        for participant_id in gradients.keys() {
            // Use reputation score as initial weight
            let base_weight = if let Some(reputation) = self.reputation_scores.get(participant_id) {
                num_traits::cast::cast(reputation.score).unwrap_or_else(|| T::zero())
            } else {
                T::one()
            };

            weights.insert(participant_id.clone(), base_weight);
        }

        Ok(weights)
    }

    /// Cluster gradients for FLAME algorithm
    fn cluster_gradients(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Vec<HashMap<String, Array1<T>>>> {
        // Simple clustering based on cosine similarity
        let mut clusters = Vec::new();
        let mut unassigned: HashMap<String, Array1<T>> = gradients.clone();

        while !unassigned.is_empty() {
            let mut current_cluster = HashMap::new();

            // Start new cluster with first unassigned gradient
            let (first_id, first_gradient) = unassigned.iter().next().unwrap();
            let first_id = first_id.clone();
            let first_gradient = first_gradient.clone();

            current_cluster.insert(first_id.clone(), first_gradient.clone());
            unassigned.remove(&first_id);

            // Add similar gradients to cluster
            let mut to_remove = Vec::new();
            for (participant_id, gradient) in &unassigned {
                let similarity = self.compute_cosine_similarity(&first_gradient, gradient)?;
                if similarity > num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()) {
                    // Similarity threshold
                    current_cluster.insert(participant_id.clone(), gradient.clone());
                    to_remove.push(participant_id.clone());
                }
            }

            for id in to_remove {
                unassigned.remove(&id);
            }

            clusters.push(current_cluster);
        }

        Ok(clusters)
    }

    /// Find the largest cluster
    fn find_largest_cluster(
        &self,
        clusters: &[HashMap<String, Array1<T>>],
    ) -> Result<HashMap<String, Array1<T>>> {
        clusters
            .iter()
            .max_by_key(|cluster| cluster.len())
            .cloned()
            .ok_or_else(|| OptimError::InvalidConfig("No clusters found".to_string()))
    }

    /// Compute cosine similarity between two gradients
    fn compute_cosine_similarity(&self, a: &Array1<T>, b: &Array1<T>) -> Result<T> {
        if a.len() != b.len() {
            return Err(OptimError::InvalidConfig(
                "Gradient dimensions don't match".to_string(),
            ));
        }

        let mut dot_product = T::zero();
        let mut norm_a = T::zero();
        let mut norm_b = T::zero();

        for (x, y) in a.iter().zip(b.iter()) {
            dot_product = dot_product + *x * *y;
            norm_a = norm_a + *x * *x;
            norm_b = norm_b + *y * *y;
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        if norm_a > T::zero() && norm_b > T::zero() {
            Ok(dot_product / (norm_a * norm_b))
        } else {
            Ok(T::zero())
        }
    }

    /// Compute outlier score for a gradient
    fn compute_outlier_score(
        &self,
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        match self.config.outlier_detection {
            OutlierDetectionMethod::ZScore => {
                let mut max_z_score = T::zero();

                for i in 0..gradient.len() {
                    if stats.std_dev[i] > T::zero() {
                        let z_score = ((gradient[i] - stats.mean[i]) / stats.std_dev[i]).abs();
                        if z_score > max_z_score {
                            max_z_score = z_score;
                        }
                    }
                }

                Ok(OutlierScore {
                    score: max_z_score.to_f64().unwrap_or(0.0),
                    method: OutlierDetectionMethod::ZScore,
                    details: format!("Max Z-score: {:.4}", max_z_score.to_f64().unwrap_or(0.0)),
                })
            }

            OutlierDetectionMethod::IQR => {
                let mut max_iqr_score = 0.0;

                for i in 0..gradient.len() {
                    let q1 = stats.median[i] - stats.iqr[i] / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
                    let q3 = stats.median[i] + stats.iqr[i] / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());

                    if gradient[i] < q1 || gradient[i] > q3 {
                        let iqr_score = if gradient[i] < q1 {
                            (q1 - gradient[i]) / stats.iqr[i]
                        } else {
                            (gradient[i] - q3) / stats.iqr[i]
                        };

                        let score = iqr_score.to_f64().unwrap_or(0.0);
                        if score > max_iqr_score {
                            max_iqr_score = score;
                        }
                    }
                }

                Ok(OutlierScore {
                    score: max_iqr_score,
                    method: OutlierDetectionMethod::IQR,
                    details: format!("Max IQR score: {:.4}", max_iqr_score),
                })
            }

            _ => {
                // Fallback to Z-score for other methods
                self.compute_outlier_score(gradient, stats)
            }
        }
    }

    /// Get reputation updates
    fn get_reputation_updates(&self) -> HashMap<String, ReputationScore> {
        self.reputation_scores.clone()
    }
}

/// Anomaly score for participant
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous)
    pub score: f64,

    /// Detection method used
    pub method: String,

    /// Additional details
    pub details: String,
}

/// Outlier score for participant
#[derive(Debug, Clone)]
pub struct OutlierScore {
    /// Outlier score
    pub score: f64,

    /// Detection method used
    pub method: OutlierDetectionMethod,

    /// Additional details
    pub details: String,
}

/// Verification score for gradient
#[derive(Debug, Clone)]
pub struct VerificationScore {
    /// Verification score (0.0 = failed, 1.0 = passed)
    pub score: f64,

    /// Individual rule scores
    pub rule_scores: HashMap<String, f64>,

    /// Overall verification status
    pub passed: bool,
}

/// Byzantine aggregation result
#[derive(Debug, Clone)]
pub struct ByzantineAggregationResult<T: Float> {
    /// Aggregated gradient
    pub aggregate: Array1<T>,

    /// List of honest participants
    pub honest_participants: Vec<String>,

    /// List of detected Byzantine participants
    pub byzantine_participants: Vec<String>,

    /// Updated reputation scores
    pub reputation_updates: HashMap<String, ReputationScore>,

    /// Aggregation method used
    pub aggregation_method: ByzantineAggregationMethod,

    /// Confidence score of the aggregation
    pub confidence_score: f64,
}

impl ReputationScore {
    /// Create new reputation score with default values
    pub fn new() -> Self {
        Self {
            score: 0.7, // Start with medium trust
            successful_aggregations: 0,
            detected_anomalies: 0,
            gradient_quality: 0.5,
            consistency_score: 0.5,
            trust_level: TrustLevel::Medium,
        }
    }
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand> AnomalyDetector<T> {
    /// Create new anomaly detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            gradient_stats: GradientStatistics::new(),
            pattern_model: PatternModel::new(),
        }
    }

    /// Detect anomaly in gradient
    pub fn detect_anomaly(&mut self, gradient: &Array1<T>) -> Result<AnomalyScore> {
        // Update gradient statistics
        self.gradient_stats.update(gradient)?;

        // Compute anomaly score based on deviation from normal patterns
        let norm_deviation = self.compute_norm_deviation(gradient)?;
        let pattern_deviation = self.pattern_model.compute_pattern_deviation(gradient)?;

        let combined_score = (norm_deviation + pattern_deviation) / 2.0;

        Ok(AnomalyScore {
            score: combined_score,
            method: "Combined norm and pattern analysis".to_string(),
            details: format!(
                "Norm dev: {:.4}, Pattern dev: {:.4}",
                norm_deviation, pattern_deviation
            ),
        })
    }

    /// Compute norm deviation score
    fn compute_norm_deviation(&self, gradient: &Array1<T>) -> Result<f64> {
        let gradient_norm = self.compute_l2_norm(gradient);

        if self.gradient_stats.norm_history.is_empty() {
            return Ok(0.0);
        }

        // Compute mean and std of historical norms
        let mean_norm = self
            .gradient_stats
            .norm_history
            .iter()
            .fold(T::zero(), |acc, &x| acc + x)
            / T::from(self.gradient_stats.norm_history.len()).unwrap();

        let variance = self
            .gradient_stats
            .norm_history
            .iter()
            .map(|&x| {
                let diff = x - mean_norm;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(self.gradient_stats.norm_history.len()).unwrap();

        let std_norm = variance.sqrt();

        if std_norm > T::zero() {
            let z_score = ((gradient_norm - mean_norm) / std_norm).abs();
            Ok(z_score.to_f64().unwrap_or(0.0) / 3.0) // Normalize to [0,1] approximately
        } else {
            Ok(0.0)
        }
    }

    /// Compute L2 norm of gradient
    fn compute_l2_norm(&self, gradient: &Array1<T>) -> T {
        gradient
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand> GradientStatistics<T> {
    /// Create new gradient statistics
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            covariance: Array2::zeros((0, 0)),
            norm_history: Vec::new(),
            direction_patterns: Array2::zeros((0, 0)),
        }
    }

    /// Update statistics with new gradient
    pub fn update(&mut self, gradient: &Array1<T>) -> Result<()> {
        // Update norm history
        let norm = gradient
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt();

        self.norm_history.push(norm);

        // Keep only recent history (last 1000 gradients)
        if self.norm_history.len() > 1000 {
            self.norm_history.remove(0);
        }

        // Initialize mean if this is the first gradient
        if self.mean.len() == 0 {
            self.mean = gradient.clone();
        } else if self.mean.len() == gradient.len() {
            // Update running mean
            let alpha = num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()); // Learning rate for running average
            self.mean = &self.mean * (T::one() - alpha) + gradient * alpha;
        }

        Ok(())
    }
}

impl<T: Float + Send + Sync> PatternModel<T> {
    /// Create new pattern model
    pub fn new() -> Self {
        Self {
            normal_patterns: Vec::new(),
            attack_patterns: Vec::new(),
            matching_threshold: 0.8,
        }
    }

    /// Compute pattern deviation score
    pub fn compute_pattern_deviation(&self, gradient: &Array1<T>) -> Result<f64> {
        // If no patterns learned yet, return neutral score
        if self.normal_patterns.is_empty() {
            return Ok(0.5);
        }

        // Find closest normal pattern
        let mut min_distance = T::infinity();
        for pattern in &self.normal_patterns {
            if pattern.len() == gradient.len() {
                let distance = self.compute_pattern_distance(gradient, pattern)?;
                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }

        // Normalize distance to [0,1] score
        let max_expected_distance = num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()); // Tunable parameter
        let deviation_score = (min_distance / max_expected_distance).min(T::one());

        Ok(deviation_score.to_f64().unwrap_or(0.5))
    }

    /// Compute distance between gradient and pattern
    fn compute_pattern_distance(&self, gradient: &Array1<T>, pattern: &Array1<T>) -> Result<T> {
        if gradient.len() != pattern.len() {
            return Err(OptimError::InvalidConfig("Dimension mismatch".to_string()));
        }

        let mut sum = T::zero();
        for (g, p) in gradient.iter().zip(pattern.iter()) {
            let diff = *g - *p;
            sum = sum + diff * diff;
        }

        Ok(sum.sqrt())
    }
}

impl<T: Float + Send + Sync> StatisticalAnalysis<T> {
    /// Create new statistical analysis engine
    pub fn new(_windowsize: usize) -> Self {
        Self {
            window_size: _windowsize,
            measures: StatisticalMeasures::new(),
        }
    }

    /// Compute statistical measures for gradients
    pub fn compute_statistics(
        &mut self,
        gradients: &[&Array1<T>],
    ) -> Result<StatisticalMeasures<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients provided".to_string(),
            ));
        }

        let first_gradient = gradients[0];
        let dim = first_gradient.len();

        let mut mean = Array1::zeros(dim);
        let mut median = Array1::zeros(dim);
        let mut std_dev = Array1::zeros(dim);
        let mut iqr = Array1::zeros(dim);
        let mut skewness = Array1::zeros(dim);
        let mut kurtosis = Array1::zeros(dim);

        // Compute statistics for each dimension
        for i in 0..dim {
            let mut values: Vec<T> = gradients.iter().map(|g| g[i]).collect();

            // Mean
            let sum: T = values.iter().copied().fold(T::zero(), |acc, x| acc + x);
            mean[i] = sum / T::from(values.len()).unwrap();

            // Sort for median and IQR
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Median
            median[i] = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
            } else {
                values[values.len() / 2]
            };

            // Standard deviation
            let variance: T = gradients
                .iter()
                .map(|g| {
                    let diff = g[i] - mean[i];
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(gradients.len()).unwrap();
            std_dev[i] = variance.sqrt();

            // Interquartile range
            let q1_idx = values.len() / 4;
            let q3_idx = 3 * values.len() / 4;
            iqr[i] = values[q3_idx] - values[q1_idx];

            // Skewness and kurtosis (simplified calculations)
            skewness[i] = T::zero(); // Placeholder
            kurtosis[i] = T::zero(); // Placeholder
        }

        self.measures = StatisticalMeasures {
            mean,
            std_dev,
            median,
            iqr,
            skewness,
            kurtosis,
        };

        Ok(self.measures.clone())
    }
}

impl<T: Float + Send + Sync> StatisticalMeasures<T> {
    /// Create new statistical measures
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            std_dev: Array1::zeros(0),
            median: Array1::zeros(0),
            iqr: Array1::zeros(0),
            skewness: Array1::zeros(0),
            kurtosis: Array1::zeros(0),
        }
    }
}

impl<T: Float + Send + Sync> GradientVerifier<T> {
    /// Create new gradient verifier
    pub fn new() -> Self {
        let mut verification_rules = Vec::new();

        // Add some basic verification rules
        verification_rules.push(VerificationRule {
            name: "Finite values".to_string(),
            rule_fn: Box::new(|gradient: &Array1<T>| gradient.iter().all(|&x| x.is_finite())),
            weight: 1.0,
        });

        Self {
            expected_properties: GradientProperties::new(),
            verification_rules,
        }
    }

    /// Verify gradient using all rules
    pub fn verify_gradient(&self, gradient: &Array1<T>) -> Result<VerificationScore> {
        let mut rule_scores = HashMap::new();
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;

        for rule in &self.verification_rules {
            let passed = (rule.rule_fn)(gradient);
            let score = if passed { 1.0 } else { 0.0 };

            rule_scores.insert(rule.name.clone(), score);
            weighted_score += score * rule.weight;
            total_weight += rule.weight;
        }

        let overall_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            1.0
        };

        Ok(VerificationScore {
            score: overall_score,
            rule_scores,
            passed: overall_score >= 0.8,
        })
    }
}

impl<T: Float + Send + Sync> GradientProperties<T> {
    /// Create new gradient properties
    pub fn new() -> Self {
        Self {
            norm_range: (T::zero(), num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())),
            sparsity_threshold: 0.1,
            direction_consistency: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::collections::HashMap;

    #[test]
    fn test_byzantine_config() {
        let config = ByzantineConfig {
            max_byzantine: 2,
            min_participants: 5,
            aggregation_method: ByzantineAggregationMethod::Krum,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: true,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.7,
        };

        assert_eq!(config.max_byzantine, 2);
        assert_eq!(config.min_participants, 5);
    }

    #[test]
    fn test_reputation_score() {
        let mut reputation = ReputationScore::new();
        assert_eq!(reputation.score, 0.7);
        assert_eq!(reputation.successful_aggregations, 0);
        assert!(matches!(reputation.trust_level, TrustLevel::Medium));

        reputation.successful_aggregations += 1;
        reputation.score = 0.9;
        assert_eq!(reputation.successful_aggregations, 1);
    }

    #[test]
    fn test_trimmed_mean_aggregation() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 3,
            aggregation_method: ByzantineAggregationMethod::TrimmedMean,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: false,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.7,
        };

        let aggregator = ByzantineTolerantAggregator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("client1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        gradients.insert("client2".to_string(), Array1::from(vec![1.1, 2.1, 3.1]));
        gradients.insert("client3".to_string(), Array1::from(vec![0.9, 1.9, 2.9]));
        gradients.insert("client4".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        gradients.insert("client5".to_string(), Array1::from(vec![10.0, 20.0, 30.0])); // Outlier

        let result = aggregator.trimmed_mean_aggregation(&gradients).unwrap();

        // Should be close to [1.0, 2.0, 3.0] after trimming outliers
        assert!((result[0] - 1.0).abs() < 0.2);
        assert!((result[1] - 2.0).abs() < 0.2);
        assert!((result[2] - 3.0).abs() < 0.2);
    }

    #[test]
    fn test_coordinate_median_aggregation() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 3,
            aggregation_method: ByzantineAggregationMethod::CoordinateMedian,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: false,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.7,
        };

        let aggregator = ByzantineTolerantAggregator::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("client1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        gradients.insert("client2".to_string(), Array1::from(vec![2.0, 3.0, 4.0]));
        gradients.insert("client3".to_string(), Array1::from(vec![3.0, 4.0, 5.0]));

        let result = aggregator
            .coordinate_median_aggregation(&gradients)
            .unwrap();

        // Median should be [2.0, 3.0, 4.0]
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 3,
            aggregation_method: ByzantineAggregationMethod::Krum,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: false,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.7,
        };

        let aggregator = ByzantineTolerantAggregator::new(config);

        let a = Array1::from(vec![1.0, 2.0, 3.0]);
        let b = Array1::from(vec![4.0, 5.0, 6.0]);

        let distance = aggregator.compute_euclidean_distance(&a, &b).unwrap();
        let expected = ((3.0_f64.powi(2) + 3.0_f64.powi(2) + 3.0_f64.powi(2)) as f64).sqrt();

        assert!((distance - expected).abs() < 1e-10);
    }
}
