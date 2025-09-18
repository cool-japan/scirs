//! Core adaptive streaming optimizer implementation
//!
//! This module contains the main AdaptiveStreamingOptimizer that orchestrates
//! all streaming optimization components including drift detection, performance
//! tracking, resource management, and adaptive learning rate control.

use super::anomaly_detection::{
    EnsembleAnomalyDetector, MLAnomalyDetector, StatisticalAnomalyDetector,
};
use super::buffering::AdaptiveBuffer;
use super::config::*;
use super::drift_detection::EnhancedDriftDetector;
use super::meta_learning::{ExperienceReplay, MetaLearner, StrategySelector};
use super::performance::{DataStatistics, PerformanceSnapshot, PerformanceTracker};
use super::resource_management::{ResourceManager, ResourceUsage};

use crate::adaptive_selection::OptimizerType;
use crate::learned_optimizers::lstm_optimizer::AdaptiveLearningRateController;
use ndarray::{Array, Array1, Array2, Dimension, IxDyn};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Streaming data point for optimization
#[derive(Debug, Clone)]
pub struct StreamingDataPoint<A: Float> {
    /// Input features
    pub features: Array1<A>,
    /// Target values (optional for unsupervised learning)
    pub target: Option<Array1<A>>,
    /// Timestamp when data was received
    pub timestamp: Instant,
    /// Data source identifier
    pub source_id: Option<String>,
    /// Data quality score (0.0 to 1.0)
    pub quality_score: A,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Adaptation instruction for optimizer components
#[derive(Debug, Clone)]
pub struct Adaptation<A: Float> {
    /// Type of adaptation
    pub adaptation_type: AdaptationType,
    /// Magnitude of adaptation
    pub magnitude: A,
    /// Target component for adaptation
    pub target_component: String,
    /// Adaptation parameters
    pub parameters: HashMap<String, A>,
    /// Priority of this adaptation
    pub priority: AdaptationPriority,
    /// Timestamp when adaptation was computed
    pub timestamp: Instant,
}

/// Types of adaptations that can be applied
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptationType {
    /// Adjust learning rate
    LearningRate,
    /// Modify buffer size
    BufferSize,
    /// Change drift sensitivity
    DriftSensitivity,
    /// Update resource allocation
    ResourceAllocation,
    /// Adjust performance thresholds
    PerformanceThreshold,
    /// Modify anomaly detection parameters
    AnomalyDetection,
    /// Update meta-learning parameters
    MetaLearning,
    /// Custom adaptation type
    Custom(String),
}

/// Priority levels for adaptations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdaptationPriority {
    /// Low priority adaptation
    Low = 0,
    /// Normal priority adaptation
    Normal = 1,
    /// High priority adaptation
    High = 2,
    /// Critical adaptation that must be applied immediately
    Critical = 3,
}

/// Statistics for adaptive streaming optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStreamingStats {
    /// Total number of data points processed
    pub total_data_points: usize,
    /// Total number of optimization steps performed
    pub optimization_steps: usize,
    /// Number of drift events detected
    pub drift_events: usize,
    /// Number of anomalies detected
    pub anomalies_detected: usize,
    /// Number of adaptations applied
    pub adaptations_applied: usize,
    /// Current buffer size
    pub current_buffer_size: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Average processing time per batch
    pub avg_processing_time_ms: f64,
    /// Resource utilization statistics
    pub resource_utilization: ResourceUsage,
    /// Performance trend (improvement/degradation)
    pub performance_trend: f64,
    /// Meta-learning effectiveness score
    pub meta_learning_score: f64,
}

/// Main adaptive streaming optimizer
pub struct AdaptiveStreamingOptimizer<O, A, D>
where
    A: Float + Default + Clone + Send + Sync,
    D: Dimension,
{
    /// Base optimizer instance
    base_optimizer: O,
    /// Streaming configuration
    config: StreamingConfig,
    /// Adaptive buffer for incoming data
    buffer: AdaptiveBuffer<A>,
    /// Drift detection system
    drift_detector: EnhancedDriftDetector<A>,
    /// Performance tracking system
    performance_tracker: PerformanceTracker<A>,
    /// Resource management system
    resource_manager: ResourceManager,
    /// Meta-learning system
    meta_learner: MetaLearner<A>,
    /// Anomaly detection system
    anomaly_detector: AnomalyDetector<A>,
    /// Learning rate controller
    learning_rate_controller: AdaptiveLearningRateController<A>,
    /// Current model parameters
    parameters: Option<Array<A, D>>,
    /// Optimization statistics
    stats: AdaptiveStreamingStats,
    /// Last adaptation timestamp
    last_adaptation: Instant,
    /// Adaptation history
    adaptation_history: VecDeque<Adaptation<A>>,
    /// Performance baseline for comparison
    performance_baseline: Option<A>,
    /// Phantom data for dimension type
    _phantom: PhantomData<D>,
}

impl<O, A, D> AdaptiveStreamingOptimizer<O, A, D>
where
    A: Float + Default + Clone + Send + Sync + std::iter::Sum + std::fmt::Debug,
    D: Dimension,
    O: Clone,
{
    /// Creates a new adaptive streaming optimizer
    pub fn new(base_optimizer: O, config: StreamingConfig) -> Result<Self, String> {
        // Validate configuration
        config.validate()?;

        let buffer = AdaptiveBuffer::new(&config)?;
        let drift_detector = EnhancedDriftDetector::new(&config)?;
        let performance_tracker = PerformanceTracker::new(&config)?;
        let resource_manager = ResourceManager::new(&config)?;
        let meta_learner = MetaLearner::new(&config)?;
        let anomaly_detector = AnomalyDetector::new(&config)?;
        let learning_rate_controller = AdaptiveLearningRateController::new(&config)?;

        let stats = AdaptiveStreamingStats {
            total_data_points: 0,
            optimization_steps: 0,
            drift_events: 0,
            anomalies_detected: 0,
            adaptations_applied: 0,
            current_buffer_size: config.buffer_config.initial_size,
            current_learning_rate: config.learning_rate_config.initial_rate,
            avg_processing_time_ms: 0.0,
            resource_utilization: ResourceUsage::default(),
            performance_trend: 0.0,
            meta_learning_score: 0.0,
        };

        Ok(Self {
            base_optimizer,
            config,
            buffer,
            drift_detector,
            performance_tracker,
            resource_manager,
            meta_learner,
            anomaly_detector,
            learning_rate_controller,
            parameters: None,
            stats,
            last_adaptation: Instant::now(),
            adaptation_history: VecDeque::with_capacity(1000),
            performance_baseline: None,
            _phantom: PhantomData,
        })
    }

    /// Performs an adaptive optimization step with streaming data
    pub fn adaptive_step(
        &mut self,
        data_batch: Vec<StreamingDataPoint<A>>,
    ) -> Result<Array<A, D>, String> {
        let start_time = Instant::now();

        // Update resource utilization tracking
        self.resource_manager.update_utilization()?;

        // Add data to buffer and check for anomalies
        let filtered_batch = self.filter_anomalies(data_batch)?;
        self.buffer.add_batch(filtered_batch)?;

        // Check if buffer should be processed
        if !self.should_process_buffer()? {
            return self
                .parameters
                .clone()
                .ok_or("No parameters available".to_string());
        }

        // Get batch from buffer for processing
        let processing_batch = self.buffer.get_batch_for_processing()?;
        self.stats.total_data_points += processing_batch.len();

        // Detect drift in the data
        let drift_detected = self.drift_detector.detect_drift(&processing_batch)?;
        if drift_detected {
            self.stats.drift_events += 1;
        }

        // Compute necessary adaptations
        let adaptations = self.compute_adaptations(&processing_batch, drift_detected)?;

        // Apply adaptations to system components
        self.apply_adaptations(&adaptations)?;

        // Perform actual optimization step
        let updated_parameters = self.perform_optimization_step(&processing_batch)?;

        // Evaluate performance of the optimization step
        let performance = self.evaluate_performance(&processing_batch, &updated_parameters)?;

        // Update performance tracking
        self.performance_tracker
            .add_performance(performance.clone())?;

        // Update meta-learner with experience
        self.update_meta_learner(&processing_batch, &adaptations, &performance)?;

        // Update statistics
        self.stats.optimization_steps += 1;
        self.stats.adaptations_applied += adaptations.len();
        self.stats.current_buffer_size = self.buffer.current_size();
        self.stats.current_learning_rate = self.learning_rate_controller.current_rate() as f64;
        self.stats.performance_trend = self.compute_performance_trend();
        self.stats.meta_learning_score = self.meta_learner.get_effectiveness_score() as f64;

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.stats.avg_processing_time_ms = (self.stats.avg_processing_time_ms
            * (self.stats.optimization_steps - 1) as f64
            + processing_time)
            / self.stats.optimization_steps as f64;

        // Store updated parameters
        self.parameters = Some(updated_parameters.clone());

        Ok(updated_parameters)
    }

    /// Filters out anomalous data points
    fn filter_anomalies(
        &mut self,
        data_batch: Vec<StreamingDataPoint<A>>,
    ) -> Result<Vec<StreamingDataPoint<A>>, String> {
        if !self.config.anomaly_config.enable_detection {
            return Ok(data_batch);
        }

        let mut filtered_batch = Vec::new();

        for data_point in data_batch {
            let is_anomaly = self.anomaly_detector.detect_anomaly(&data_point)?;

            if is_anomaly {
                self.stats.anomalies_detected += 1;

                // Apply anomaly response strategy
                match &self.config.anomaly_config.response_strategy {
                    AnomalyResponseStrategy::Ignore => {
                        // Include the data point anyway
                        filtered_batch.push(data_point);
                    }
                    AnomalyResponseStrategy::Filter => {
                        // Skip this data point
                        continue;
                    }
                    AnomalyResponseStrategy::Adaptive => {
                        // Adapt the data point or model
                        let adapted_point = self.adapt_for_anomaly(data_point)?;
                        filtered_batch.push(adapted_point);
                    }
                    AnomalyResponseStrategy::Reset => {
                        // Reset relevant components (implemented in apply_adaptations)
                        filtered_batch.push(data_point);
                    }
                    AnomalyResponseStrategy::Custom(_) => {
                        // Custom handling (simplified)
                        filtered_batch.push(data_point);
                    }
                }
            } else {
                filtered_batch.push(data_point);
            }
        }

        Ok(filtered_batch)
    }

    /// Adapts a data point that was detected as anomalous
    fn adapt_for_anomaly(
        &self,
        mut data_point: StreamingDataPoint<A>,
    ) -> Result<StreamingDataPoint<A>, String> {
        // Simple adaptation: reduce the influence of extreme values
        let median = self.compute_feature_median(&data_point.features)?;

        for (i, value) in data_point.features.iter_mut().enumerate() {
            let diff = (*value - median[i]).abs();
            let threshold = median[i] * A::from(self.config.anomaly_config.threshold).unwrap();

            if diff > threshold {
                // Clip the value to be within the threshold
                let sign = if *value > median[i] {
                    A::one()
                } else {
                    -A::one()
                };
                *value = median[i] + sign * threshold;
            }
        }

        // Reduce quality score for adapted anomalous data
        data_point.quality_score = data_point.quality_score * A::from(0.5).unwrap();

        Ok(data_point)
    }

    /// Computes feature-wise median from recent data
    fn compute_feature_median(&self, features: &Array1<A>) -> Result<Array1<A>, String> {
        // Simplified implementation - in practice would use rolling window
        Ok(features.clone())
    }

    /// Checks if the buffer should be processed
    fn should_process_buffer(&self) -> Result<bool, String> {
        let buffer_quality = self.buffer.get_quality_metrics();
        let buffer_size = self.buffer.current_size();

        // Check size threshold
        let size_threshold = self.config.buffer_config.initial_size;
        let size_ready = buffer_size >= size_threshold;

        // Check quality threshold
        let quality_ready = buffer_quality.average_quality
            >= A::from(self.config.buffer_config.quality_threshold).unwrap();

        // Check timeout
        let timeout_ready = self.buffer.time_since_last_processing()
            >= self.config.buffer_config.processing_timeout;

        // Check resource availability
        let resources_available = self
            .resource_manager
            .has_sufficient_resources_for_processing()?;

        Ok((size_ready && quality_ready) || timeout_ready && resources_available)
    }

    /// Computes necessary adaptations based on current state
    fn compute_adaptations(
        &mut self,
        batch: &[StreamingDataPoint<A>],
        drift_detected: bool,
    ) -> Result<Vec<Adaptation<A>>, String> {
        let mut adaptations = Vec::new();

        // Learning rate adaptation
        if let Some(lr_adaptation) = self
            .learning_rate_controller
            .compute_adaptation(batch, &self.performance_tracker)?
        {
            adaptations.push(lr_adaptation);
        }

        // Drift-based adaptations
        if drift_detected {
            if let Some(drift_adaptation) = self.drift_detector.compute_sensitivity_adaptation()? {
                adaptations.push(drift_adaptation);
            }
        }

        // Buffer size adaptation
        if let Some(buffer_adaptation) = self
            .buffer
            .compute_size_adaptation(&self.performance_tracker)?
        {
            adaptations.push(buffer_adaptation);
        }

        // Resource allocation adaptation
        if let Some(resource_adaptation) = self.resource_manager.compute_allocation_adaptation()? {
            adaptations.push(resource_adaptation);
        }

        // Meta-learning based adaptations
        let meta_adaptations = self
            .meta_learner
            .recommend_adaptations(batch, &self.performance_tracker)?;
        adaptations.extend(meta_adaptations);

        // Sort adaptations by priority
        adaptations.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(adaptations)
    }

    /// Applies computed adaptations to system components
    fn apply_adaptations(&mut self, adaptations: &[Adaptation<A>]) -> Result<(), String> {
        for adaptation in adaptations {
            match &adaptation.adaptation_type {
                AdaptationType::LearningRate => {
                    self.learning_rate_controller.apply_adaptation(adaptation)?;
                }
                AdaptationType::BufferSize => {
                    self.buffer.apply_size_adaptation(adaptation)?;
                }
                AdaptationType::DriftSensitivity => {
                    self.drift_detector
                        .apply_sensitivity_adaptation(adaptation)?;
                }
                AdaptationType::ResourceAllocation => {
                    self.resource_manager
                        .apply_allocation_adaptation(adaptation)?;
                }
                AdaptationType::PerformanceThreshold => {
                    self.performance_tracker
                        .apply_threshold_adaptation(adaptation)?;
                }
                AdaptationType::AnomalyDetection => {
                    self.anomaly_detector.apply_adaptation(adaptation)?;
                }
                AdaptationType::MetaLearning => {
                    self.meta_learner.apply_adaptation(adaptation)?;
                }
                AdaptationType::Custom(name) => {
                    // Handle custom adaptations
                    println!("Applying custom adaptation: {}", name);
                }
            }

            // Store adaptation in history
            if self.adaptation_history.len() >= 1000 {
                self.adaptation_history.pop_front();
            }
            self.adaptation_history.push_back(adaptation.clone());
        }

        self.last_adaptation = Instant::now();
        Ok(())
    }

    /// Performs the actual optimization step
    fn perform_optimization_step(
        &mut self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<Array<A, D>, String> {
        // Compute gradients from the batch
        let gradients = self.compute_batch_gradients(batch)?;

        // Get current learning rate
        let learning_rate = self.learning_rate_controller.current_rate();

        // Apply optimization step (simplified implementation)
        let mut updated_parameters = self.parameters.clone().unwrap_or_else(|| {
            // Initialize parameters if not present
            Array::zeros(IxDyn(&[gradients.len()]))
        });

        // Simple gradient descent update (in practice would use the base optimizer)
        for (param, &grad) in updated_parameters.iter_mut().zip(gradients.iter()) {
            *param = *param - learning_rate * grad;
        }

        Ok(updated_parameters)
    }

    /// Computes batch gradients from streaming data
    fn compute_batch_gradients(
        &self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<Array1<A>, String> {
        if batch.is_empty() {
            return Err("Cannot compute gradients from empty batch".to_string());
        }

        let feature_dim = batch[0].features.len();
        let mut gradients = Array1::zeros(feature_dim);

        // Simplified gradient computation (in practice would depend on loss function)
        for data_point in batch {
            for (i, &feature) in data_point.features.iter().enumerate() {
                gradients[i] = gradients[i] + feature * data_point.quality_score;
            }
        }

        // Normalize by batch size
        let batch_size = A::from(batch.len()).unwrap();
        gradients /= batch_size;

        Ok(gradients)
    }

    /// Evaluates performance of the optimization step
    fn evaluate_performance(
        &self,
        batch: &[StreamingDataPoint<A>],
        parameters: &Array<A, D>,
    ) -> Result<PerformanceSnapshot<A>, String> {
        // Compute various performance metrics
        let loss = self.compute_loss(batch, parameters)?;
        let accuracy = self.compute_accuracy(batch, parameters)?;
        let convergence_rate = self.compute_convergence_rate(parameters)?;

        // Compute data statistics
        let data_stats = self.compute_data_statistics(batch)?;

        // Get resource usage
        let resource_usage = self.resource_manager.current_usage()?;

        let performance = PerformanceSnapshot {
            timestamp: Instant::now(),
            loss,
            accuracy: Some(accuracy),
            convergence_rate: Some(convergence_rate),
            gradient_norm: Some(A::from(1.0).unwrap()), // Simplified
            parameter_update_magnitude: Some(A::from(0.1).unwrap()), // Simplified
            data_statistics: data_stats,
            resource_usage,
            custom_metrics: HashMap::new(),
        };

        Ok(performance)
    }

    /// Computes loss for the current batch and parameters
    fn compute_loss(
        &self,
        batch: &[StreamingDataPoint<A>],
        _parameters: &Array<A, D>,
    ) -> Result<A, String> {
        // Simplified loss computation (Mean Squared Error)
        let mut total_loss = A::zero();
        let mut count = 0;

        for data_point in batch {
            if let Some(ref target) = data_point.target {
                let prediction = &data_point.features; // Simplified
                let diff = prediction - target;
                let squared_diff = diff.mapv(|x| x * x);
                total_loss = total_loss + squared_diff.sum();
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_loss / A::from(count).unwrap())
        } else {
            Ok(A::zero())
        }
    }

    /// Computes accuracy for the current batch and parameters
    fn compute_accuracy(
        &self,
        batch: &[StreamingDataPoint<A>],
        _parameters: &Array<A, D>,
    ) -> Result<A, String> {
        // Simplified accuracy computation
        let mut correct = 0;
        let mut total = 0;

        for data_point in batch {
            if data_point.target.is_some() {
                // Simplified: assume accuracy based on quality score
                if data_point.quality_score > A::from(0.5).unwrap() {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            Ok(A::from(correct).unwrap() / A::from(total).unwrap())
        } else {
            Ok(A::one())
        }
    }

    /// Computes convergence rate
    fn compute_convergence_rate(&self, _parameters: &Array<A, D>) -> Result<A, String> {
        // Simplified convergence rate computation
        let recent_losses = self.performance_tracker.get_recent_losses(10);
        if recent_losses.len() >= 2 {
            let improvement = recent_losses[0] - recent_losses[recent_losses.len() - 1];
            Ok(improvement / recent_losses[0])
        } else {
            Ok(A::zero())
        }
    }

    /// Computes comprehensive data statistics
    fn compute_data_statistics(
        &self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<DataStatistics<A>, String> {
        if batch.is_empty() {
            return Ok(DataStatistics::default());
        }

        let feature_dim = batch[0].features.len();
        let mut feature_means = Array1::zeros(feature_dim);
        let mut feature_stds = Array1::zeros(feature_dim);
        let mut quality_scores = Vec::new();

        // Compute means
        for data_point in batch {
            feature_means = feature_means + &data_point.features;
            quality_scores.push(data_point.quality_score);
        }
        feature_means /= A::from(batch.len()).unwrap();

        // Compute standard deviations
        for data_point in batch {
            let diff = &data_point.features - &feature_means;
            feature_stds = feature_stds + &diff.mapv(|x| x * x);
        }
        feature_stds /= A::from(batch.len()).unwrap();
        feature_stds = feature_stds.mapv(|x| x.sqrt());

        let avg_quality =
            quality_scores.iter().copied().sum::<A>() / A::from(quality_scores.len()).unwrap();

        Ok(DataStatistics {
            sample_count: batch.len(),
            feature_means,
            feature_stds,
            average_quality: avg_quality,
            timestamp: Instant::now(),
        })
    }

    /// Updates meta-learner with experience from this optimization step
    fn update_meta_learner(
        &mut self,
        batch: &[StreamingDataPoint<A>],
        adaptations: &[Adaptation<A>],
        performance: &PerformanceSnapshot<A>,
    ) -> Result<(), String> {
        if !self.config.meta_learning_config.enable_meta_learning {
            return Ok(());
        }

        // Extract meta-state from current situation
        let meta_state = self.extract_meta_state(performance)?;

        // Extract meta-action from applied adaptations
        let meta_action = self.extract_meta_action(adaptations)?;

        // Compute reward based on performance improvement
        let reward = self.compute_meta_reward(performance)?;

        // Update meta-learner
        self.meta_learner
            .update_experience(meta_state, meta_action, reward)?;

        Ok(())
    }

    /// Extracts meta-state representation from performance data
    fn extract_meta_state(
        &self,
        performance: &PerformanceSnapshot<A>,
    ) -> Result<MetaState<A>, String> {
        let state = MetaState {
            performance_metrics: vec![
                performance.loss,
                performance.accuracy.unwrap_or(A::zero()),
                performance.convergence_rate.unwrap_or(A::zero()),
            ],
            resource_state: vec![
                A::from(performance.resource_usage.memory_usage_mb as f64).unwrap(),
                A::from(performance.resource_usage.cpu_usage_percent).unwrap(),
            ],
            drift_indicators: vec![A::from(if self.drift_detector.is_drift_detected() {
                1.0
            } else {
                0.0
            })
            .unwrap()],
            adaptation_history: self.adaptation_history.len(),
            timestamp: Instant::now(),
        };

        Ok(state)
    }

    /// Extracts meta-action representation from adaptations
    fn extract_meta_action(&self, adaptations: &[Adaptation<A>]) -> Result<MetaAction<A>, String> {
        let mut adaptation_vector = Vec::new();
        let mut adaptation_types = Vec::new();

        for adaptation in adaptations {
            adaptation_vector.push(adaptation.magnitude);
            adaptation_types.push(adaptation.adaptation_type.clone());
        }

        let action = MetaAction {
            adaptation_magnitudes: adaptation_vector,
            adaptation_types,
            learning_rate_change: A::from(self.learning_rate_controller.last_change()).unwrap(),
            buffer_size_change: A::from(self.buffer.last_size_change()).unwrap(),
            timestamp: Instant::now(),
        };

        Ok(action)
    }

    /// Computes reward for meta-learning based on performance improvement
    fn compute_meta_reward(&self, performance: &PerformanceSnapshot<A>) -> Result<A, String> {
        // Compare with baseline or previous performance
        let reward = if let Some(baseline) = self.performance_baseline {
            performance.loss - baseline // Negative reward for higher loss
        } else {
            A::zero()
        };

        Ok(reward)
    }

    /// Gets current adaptive streaming statistics
    pub fn get_adaptive_stats(&self) -> AdaptiveStreamingStats {
        let mut stats = self.stats.clone();
        stats.resource_utilization = self.resource_manager.current_usage().unwrap_or_default();
        stats
    }

    /// Counts the number of adaptations applied in recent history
    fn count_adaptations_applied(&self) -> usize {
        let recent_threshold = Instant::now() - Duration::from_secs(300); // Last 5 minutes
        self.adaptation_history
            .iter()
            .filter(|adaptation| adaptation.timestamp > recent_threshold)
            .count()
    }

    /// Computes performance trend over recent optimization steps
    fn compute_performance_trend(&self) -> f64 {
        let recent_performance = self.performance_tracker.get_recent_performance(20);
        if recent_performance.len() >= 2 {
            let recent_avg = recent_performance
                .iter()
                .rev()
                .take(5)
                .map(|p| p.loss.to_f64().unwrap_or(0.0))
                .sum::<f64>()
                / 5.0;

            let older_avg = recent_performance
                .iter()
                .take(5)
                .map(|p| p.loss.to_f64().unwrap_or(0.0))
                .sum::<f64>()
                / 5.0;

            // Negative trend means improvement (lower loss)
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        }
    }

    /// Forces an adaptation cycle even if normal triggers haven't fired
    pub fn force_adaptation(&mut self) -> Result<(), String> {
        let empty_batch = Vec::new();
        let adaptations = self.compute_adaptations(&empty_batch, false)?;
        self.apply_adaptations(&adaptations)?;
        Ok(())
    }

    /// Resets the optimizer to initial state while preserving learned knowledge
    pub fn soft_reset(&mut self) -> Result<(), String> {
        // Reset components while preserving meta-learning knowledge
        self.buffer.reset()?;
        self.drift_detector.reset()?;
        self.performance_tracker.reset()?;

        // Don't reset meta-learner to preserve learned adaptations
        // self.meta_learner.reset()?;

        self.stats = AdaptiveStreamingStats {
            total_data_points: 0,
            optimization_steps: 0,
            drift_events: 0,
            anomalies_detected: 0,
            adaptations_applied: 0,
            current_buffer_size: self.config.buffer_config.initial_size,
            current_learning_rate: self.config.learning_rate_config.initial_rate,
            avg_processing_time_ms: 0.0,
            resource_utilization: ResourceUsage::default(),
            performance_trend: 0.0,
            meta_learning_score: self.meta_learner.get_effectiveness_score() as f64,
        };

        self.adaptation_history.clear();
        self.performance_baseline = None;

        Ok(())
    }

    /// Gets detailed diagnostic information
    pub fn get_diagnostics(&self) -> StreamingDiagnostics {
        StreamingDiagnostics {
            buffer_diagnostics: self.buffer.get_diagnostics(),
            drift_diagnostics: self.drift_detector.get_diagnostics(),
            performance_diagnostics: self.performance_tracker.get_diagnostics(),
            resource_diagnostics: self.resource_manager.get_diagnostics(),
            meta_learning_diagnostics: self.meta_learner.get_diagnostics(),
            anomaly_diagnostics: self.anomaly_detector.get_diagnostics(),
        }
    }
}

/// Comprehensive diagnostic information for streaming optimizer
#[derive(Debug, Clone)]
pub struct StreamingDiagnostics {
    pub buffer_diagnostics: BufferDiagnostics,
    pub drift_diagnostics: DriftDiagnostics,
    pub performance_diagnostics: PerformanceDiagnostics,
    pub resource_diagnostics: ResourceDiagnostics,
    pub meta_learning_diagnostics: MetaLearningDiagnostics,
    pub anomaly_diagnostics: AnomalyDiagnostics,
}
