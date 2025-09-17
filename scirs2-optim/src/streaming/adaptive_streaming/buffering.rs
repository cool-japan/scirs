//! Adaptive buffering strategies for streaming optimization
//!
//! This module provides sophisticated buffer management including adaptive sizing,
//! quality-based filtering, priority queuing, and intelligent data retention
//! strategies for streaming optimization scenarios.

use super::config::*;
use super::optimizer::{StreamingDataPoint, Adaptation, AdaptationType, AdaptationPriority};
use super::performance::{PerformanceSnapshot, PerformanceTracker};

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

/// Adaptive buffer for managing streaming data with quality-based retention
pub struct AdaptiveBuffer<A: Float> {
    /// Buffer configuration
    config: BufferConfig,
    /// Main data buffer with priority queue
    buffer: BinaryHeap<PrioritizedDataPoint<A>>,
    /// Secondary buffer for low-quality data
    secondary_buffer: VecDeque<StreamingDataPoint<A>>,
    /// Buffer quality metrics
    quality_metrics: BufferQualityMetrics<A>,
    /// Buffer sizing strategy
    sizing_strategy: BufferSizingStrategy<A>,
    /// Data retention policy
    retention_policy: DataRetentionPolicy<A>,
    /// Buffer statistics
    statistics: BufferStatistics<A>,
    /// Last processing timestamp
    last_processing: Instant,
    /// Size change tracking
    size_change_log: VecDeque<SizeChangeEvent>,
}

/// Data point with priority information for buffering
#[derive(Debug, Clone)]
pub struct PrioritizedDataPoint<A: Float> {
    /// The actual data point
    pub data_point: StreamingDataPoint<A>,
    /// Priority score (higher = more important)
    pub priority_score: A,
    /// Buffer insertion timestamp
    pub buffer_timestamp: Instant,
    /// Expected processing time
    pub expected_processing_time: Duration,
    /// Data freshness score
    pub freshness_score: A,
    /// Relevance score for current model
    pub relevance_score: A,
}

/// Buffer quality metrics for adaptive management
#[derive(Debug, Clone)]
pub struct BufferQualityMetrics<A: Float> {
    /// Average quality score of buffered data
    pub average_quality: A,
    /// Quality variance
    pub quality_variance: A,
    /// Minimum quality in buffer
    pub min_quality: A,
    /// Maximum quality in buffer
    pub max_quality: A,
    /// Data freshness distribution
    pub freshness_distribution: Vec<A>,
    /// Priority distribution
    pub priority_distribution: Vec<A>,
    /// Quality trend over time
    pub quality_trend: QualityTrend<A>,
}

/// Quality trend analysis
#[derive(Debug, Clone)]
pub struct QualityTrend<A: Float> {
    /// Recent quality changes
    pub recent_changes: VecDeque<A>,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend magnitude
    pub trend_magnitude: A,
    /// Trend confidence
    pub confidence: A,
}

/// Trend direction for quality analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Quality improving
    Improving,
    /// Quality degrading
    Degrading,
    /// Quality stable
    Stable,
    /// Quality oscillating
    Oscillating,
}

/// Buffer sizing strategy implementation
pub struct BufferSizingStrategy<A: Float> {
    /// Current strategy type
    strategy_type: BufferSizeStrategy,
    /// Target size
    target_size: usize,
    /// Size adjustment parameters
    adjustment_params: SizeAdjustmentParams<A>,
    /// Performance feedback
    performance_feedback: VecDeque<SizingPerformanceFeedback<A>>,
    /// Sizing history
    sizing_history: VecDeque<SizingEvent>,
}

/// Parameters for size adjustment
#[derive(Debug, Clone)]
pub struct SizeAdjustmentParams<A: Float> {
    /// Growth rate for increasing buffer size
    pub growth_rate: A,
    /// Shrinkage rate for decreasing buffer size
    pub shrinkage_rate: A,
    /// Stability threshold (minimum change for adjustment)
    pub stability_threshold: A,
    /// Performance sensitivity
    pub performance_sensitivity: A,
    /// Quality sensitivity
    pub quality_sensitivity: A,
    /// Memory pressure sensitivity
    pub memory_sensitivity: A,
}

/// Performance feedback for buffer sizing
#[derive(Debug, Clone)]
pub struct SizingPerformanceFeedback<A: Float> {
    /// Buffer size when feedback was recorded
    pub buffer_size: usize,
    /// Processing latency
    pub processing_latency: Duration,
    /// Throughput (items per second)
    pub throughput: A,
    /// Quality score achieved
    pub quality_score: A,
    /// Memory usage
    pub memory_usage: usize,
    /// Timestamp of feedback
    pub timestamp: Instant,
}

/// Buffer sizing event
#[derive(Debug, Clone)]
pub struct SizingEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Old buffer size
    pub old_size: usize,
    /// New buffer size
    pub new_size: usize,
    /// Reason for size change
    pub reason: SizingReason,
    /// Performance impact
    pub performance_impact: Option<f64>,
}

/// Reasons for buffer size changes
#[derive(Debug, Clone)]
pub enum SizingReason {
    /// Performance optimization
    PerformanceOptimization,
    /// Quality improvement
    QualityImprovement,
    /// Memory pressure
    MemoryPressure,
    /// Latency requirements
    LatencyRequirement,
    /// Throughput optimization
    ThroughputOptimization,
    /// Manual adjustment
    Manual,
    /// Configuration change
    Configuration,
}

/// Data retention policy for buffer management
pub struct DataRetentionPolicy<A: Float> {
    /// Retention strategy
    strategy: RetentionStrategy,
    /// Age-based retention parameters
    age_policy: AgeBasedRetention,
    /// Quality-based retention parameters
    quality_policy: QualityBasedRetention<A>,
    /// Relevance-based retention parameters
    relevance_policy: RelevanceBasedRetention<A>,
    /// Combined retention scoring
    retention_scorer: RetentionScorer<A>,
}

/// Data retention strategies
#[derive(Debug, Clone)]
pub enum RetentionStrategy {
    /// First In, First Out
    FIFO,
    /// Last In, First Out
    LIFO,
    /// Least Recently Used
    LRU,
    /// Priority-based retention
    Priority,
    /// Quality-based retention
    Quality,
    /// Age-based retention
    Age,
    /// Hybrid retention combining multiple factors
    Hybrid,
    /// Adaptive retention based on performance
    Adaptive,
}

/// Age-based retention configuration
#[derive(Debug, Clone)]
pub struct AgeBasedRetention {
    /// Maximum age for data retention
    pub max_age: Duration,
    /// Soft age limit (start considering for removal)
    pub soft_age_limit: Duration,
    /// Age weight in retention scoring
    pub age_weight: f64,
    /// Enable adaptive age limits
    pub adaptive_limits: bool,
}

/// Quality-based retention configuration
#[derive(Debug, Clone)]
pub struct QualityBasedRetention<A: Float> {
    /// Minimum quality threshold
    pub min_quality_threshold: A,
    /// Quality weight in retention scoring
    pub quality_weight: A,
    /// Enable adaptive quality thresholds
    pub adaptive_thresholds: bool,
    /// Quality distribution targets
    pub quality_targets: QualityDistributionTargets<A>,
}

/// Target quality distribution for buffer content
#[derive(Debug, Clone)]
pub struct QualityDistributionTargets<A: Float> {
    /// Target percentage of high-quality data
    pub high_quality_target: A,
    /// Target percentage of medium-quality data
    pub medium_quality_target: A,
    /// Target percentage of low-quality data
    pub low_quality_target: A,
    /// Quality boundaries
    pub high_quality_threshold: A,
    pub medium_quality_threshold: A,
}

/// Relevance-based retention configuration
#[derive(Debug, Clone)]
pub struct RelevanceBasedRetention<A: Float> {
    /// Relevance calculation method
    pub relevance_method: RelevanceMethod,
    /// Relevance weight in retention scoring
    pub relevance_weight: A,
    /// Enable temporal relevance decay
    pub temporal_decay: bool,
    /// Relevance decay rate
    pub decay_rate: A,
}

/// Methods for calculating data relevance
#[derive(Debug, Clone)]
pub enum RelevanceMethod {
    /// Distance-based relevance
    Distance,
    /// Similarity-based relevance
    Similarity,
    /// Feature importance-based relevance
    FeatureImportance,
    /// Model uncertainty-based relevance
    Uncertainty,
    /// Diversity-based relevance
    Diversity,
    /// Custom relevance function
    Custom(String),
}

/// Retention scoring system
pub struct RetentionScorer<A: Float> {
    /// Scoring weights
    weights: RetentionWeights<A>,
    /// Scoring history for adaptation
    scoring_history: VecDeque<RetentionScore<A>>,
    /// Performance feedback
    performance_feedback: VecDeque<RetentionPerformanceFeedback<A>>,
}

/// Weights for different retention factors
#[derive(Debug, Clone)]
pub struct RetentionWeights<A: Float> {
    /// Age weight
    pub age_weight: A,
    /// Quality weight
    pub quality_weight: A,
    /// Relevance weight
    pub relevance_weight: A,
    /// Priority weight
    pub priority_weight: A,
    /// Freshness weight
    pub freshness_weight: A,
    /// Diversity weight
    pub diversity_weight: A,
}

/// Retention score for a data point
#[derive(Debug, Clone)]
pub struct RetentionScore<A: Float> {
    /// Overall retention score
    pub overall_score: A,
    /// Individual component scores
    pub component_scores: HashMap<String, A>,
    /// Retention decision
    pub should_retain: bool,
    /// Confidence in decision
    pub confidence: A,
    /// Scoring timestamp
    pub timestamp: Instant,
}

/// Performance feedback for retention decisions
#[derive(Debug, Clone)]
pub struct RetentionPerformanceFeedback<A: Float> {
    /// Number of items retained
    pub items_retained: usize,
    /// Number of items discarded
    pub items_discarded: usize,
    /// Quality of retained items
    pub retained_quality: A,
    /// Quality of discarded items
    pub discarded_quality: A,
    /// Performance impact
    pub performance_impact: A,
    /// Feedback timestamp
    pub timestamp: Instant,
}

/// Buffer statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct BufferStatistics<A: Float> {
    /// Total items processed
    pub total_items_processed: u64,
    /// Total items discarded
    pub total_items_discarded: u64,
    /// Average buffer utilization
    pub avg_buffer_utilization: A,
    /// Peak buffer utilization
    pub peak_buffer_utilization: A,
    /// Average processing latency
    pub avg_processing_latency: Duration,
    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics<A>,
    /// Quality statistics
    pub quality_stats: QualityStatistics<A>,
    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,
}

/// Throughput statistics
#[derive(Debug, Clone)]
pub struct ThroughputStatistics<A: Float> {
    /// Current throughput (items per second)
    pub current_throughput: A,
    /// Average throughput
    pub avg_throughput: A,
    /// Peak throughput
    pub peak_throughput: A,
    /// Throughput trend
    pub throughput_trend: TrendDirection,
    /// Throughput stability
    pub stability: A,
}

/// Quality statistics for buffer content
#[derive(Debug, Clone)]
pub struct QualityStatistics<A: Float> {
    /// Current average quality
    pub current_avg_quality: A,
    /// Historical average quality
    pub historical_avg_quality: A,
    /// Quality improvement rate
    pub quality_improvement_rate: A,
    /// Quality distribution
    pub quality_distribution: HashMap<String, A>,
    /// Quality prediction
    pub predicted_quality: Option<A>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Current memory usage in bytes
    pub current_usage_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    /// Average memory usage in bytes
    pub avg_usage_bytes: usize,
    /// Memory efficiency (useful data / total memory)
    pub memory_efficiency: f64,
    /// Memory fragmentation
    pub fragmentation: f64,
}

/// Size change tracking event
#[derive(Debug, Clone)]
pub struct SizeChangeEvent {
    /// Change timestamp
    pub timestamp: Instant,
    /// Size before change
    pub old_size: usize,
    /// Size after change
    pub new_size: usize,
    /// Change magnitude
    pub change_magnitude: i32,
    /// Reason for change
    pub reason: String,
}

impl<A: Float + Default + Clone> AdaptiveBuffer<A> {
    /// Creates a new adaptive buffer
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let buffer_config = config.buffer_config.clone();

        let quality_metrics = BufferQualityMetrics {
            average_quality: A::zero(),
            quality_variance: A::zero(),
            min_quality: A::one(),
            max_quality: A::zero(),
            freshness_distribution: Vec::new(),
            priority_distribution: Vec::new(),
            quality_trend: QualityTrend {
                recent_changes: VecDeque::with_capacity(50),
                trend_direction: TrendDirection::Stable,
                trend_magnitude: A::zero(),
                confidence: A::zero(),
            },
        };

        let sizing_strategy = BufferSizingStrategy::new(
            buffer_config.size_strategy.clone(),
            buffer_config.initial_size,
        );

        let retention_policy = DataRetentionPolicy::new(RetentionStrategy::Hybrid);

        let statistics = BufferStatistics {
            total_items_processed: 0,
            total_items_discarded: 0,
            avg_buffer_utilization: A::zero(),
            peak_buffer_utilization: A::zero(),
            avg_processing_latency: Duration::ZERO,
            throughput_stats: ThroughputStatistics {
                current_throughput: A::zero(),
                avg_throughput: A::zero(),
                peak_throughput: A::zero(),
                throughput_trend: TrendDirection::Stable,
                stability: A::zero(),
            },
            quality_stats: QualityStatistics {
                current_avg_quality: A::zero(),
                historical_avg_quality: A::zero(),
                quality_improvement_rate: A::zero(),
                quality_distribution: HashMap::new(),
                predicted_quality: None,
            },
            memory_stats: MemoryStatistics {
                current_usage_bytes: 0,
                peak_usage_bytes: 0,
                avg_usage_bytes: 0,
                memory_efficiency: 0.0,
                fragmentation: 0.0,
            },
        };

        Ok(Self {
            config: buffer_config,
            buffer: BinaryHeap::new(),
            secondary_buffer: VecDeque::new(),
            quality_metrics,
            sizing_strategy,
            retention_policy,
            statistics,
            last_processing: Instant::now(),
            size_change_log: VecDeque::with_capacity(100),
        })
    }

    /// Adds a batch of data points to the buffer
    pub fn add_batch(&mut self, batch: Vec<StreamingDataPoint<A>>) -> Result<(), String> {
        for data_point in batch {
            self.add_single_point(data_point)?;
        }

        // Update quality metrics after batch addition
        self.update_quality_metrics()?;

        // Check if buffer needs resizing
        self.check_buffer_resizing()?;

        // Apply retention policy if buffer is too large
        if self.current_size() > self.sizing_strategy.target_size {
            self.apply_retention_policy()?;
        }

        Ok(())
    }

    /// Adds a single data point to the buffer
    fn add_single_point(&mut self, data_point: StreamingDataPoint<A>) -> Result<(), String> {
        // Calculate priority score for the data point
        let priority_score = self.calculate_priority_score(&data_point)?;

        // Calculate freshness and relevance scores
        let freshness_score = self.calculate_freshness_score(&data_point);
        let relevance_score = self.calculate_relevance_score(&data_point)?;

        let prioritized_point = PrioritizedDataPoint {
            data_point,
            priority_score,
            buffer_timestamp: Instant::now(),
            expected_processing_time: Duration::from_millis(100), // Estimated
            freshness_score,
            relevance_score,
        };

        // Add to appropriate buffer based on quality
        if priority_score >= A::from(self.config.quality_threshold).unwrap() {
            self.buffer.push(prioritized_point);
        } else {
            // Add to secondary buffer for potential later processing
            self.secondary_buffer.push_back(prioritized_point.data_point);
        }

        // Update statistics
        self.statistics.total_items_processed += 1;

        Ok(())
    }

    /// Calculates priority score for a data point
    fn calculate_priority_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        let mut score = data_point.quality_score;

        // Adjust score based on recency
        let age = data_point.timestamp.elapsed().as_secs_f64();
        let recency_bonus = A::from(1.0 / (1.0 + age / 3600.0)).unwrap(); // Hour-based decay
        score = score + recency_bonus * A::from(0.1).unwrap();

        // Adjust score based on feature variance (novelty)
        let novelty_score = self.calculate_novelty_score(data_point)?;
        score = score + novelty_score * A::from(0.2).unwrap();

        Ok(score)
    }

    /// Calculates novelty score based on feature variance
    fn calculate_novelty_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        // Simple novelty calculation based on distance from recent data
        if self.buffer.is_empty() {
            return Ok(A::from(0.5).unwrap()); // Medium novelty for first data
        }

        // Calculate average distance from recent buffer content
        let recent_points: Vec<_> = self.buffer.iter().take(10).collect();
        if recent_points.is_empty() {
            return Ok(A::from(0.5).unwrap());
        }

        let mut total_distance = A::zero();
        for recent_point in &recent_points {
            let distance = self.calculate_feature_distance(
                &data_point.features,
                &recent_point.data_point.features
            )?;
            total_distance = total_distance + distance;
        }

        let avg_distance = total_distance / A::from(recent_points.len()).unwrap();

        // Normalize to 0-1 range
        let normalized_novelty = avg_distance / (avg_distance + A::one());
        Ok(normalized_novelty)
    }

    /// Calculates distance between feature vectors
    fn calculate_feature_distance(&self, features1: &ndarray::Array1<A>, features2: &ndarray::Array1<A>) -> Result<A, String> {
        if features1.len() != features2.len() {
            return Err("Feature vectors have different lengths".to_string());
        }

        let mut distance = A::zero();
        for (f1, f2) in features1.iter().zip(features2.iter()) {
            let diff = *f1 - *f2;
            distance = distance + diff * diff;
        }

        Ok(distance.sqrt())
    }

    /// Calculates freshness score based on data age
    fn calculate_freshness_score(&self, data_point: &StreamingDataPoint<A>) -> A {
        let age_seconds = data_point.timestamp.elapsed().as_secs_f64();
        let max_age = 3600.0; // 1 hour maximum age

        let freshness = (max_age - age_seconds.min(max_age)) / max_age;
        A::from(freshness.max(0.0)).unwrap()
    }

    /// Calculates relevance score for current model context
    fn calculate_relevance_score(&self, _data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        // Simplified relevance calculation
        // In practice, this would consider current model parameters, recent performance, etc.
        Ok(A::from(0.7).unwrap()) // Default moderate relevance
    }

    /// Gets a batch of data for processing
    pub fn get_batch_for_processing(&mut self) -> Result<Vec<StreamingDataPoint<A>>, String> {
        let batch_size = self.calculate_optimal_batch_size()?;
        let mut processing_batch = Vec::with_capacity(batch_size);

        // Extract high-priority items from main buffer
        while processing_batch.len() < batch_size && !self.buffer.is_empty() {
            if let Some(prioritized_point) = self.buffer.pop() {
                processing_batch.push(prioritized_point.data_point);
            }
        }

        // Fill remaining space with secondary buffer items if needed
        while processing_batch.len() < batch_size && !self.secondary_buffer.is_empty() {
            if let Some(data_point) = self.secondary_buffer.pop_front() {
                processing_batch.push(data_point);
            }
        }

        // Update last processing time
        self.last_processing = Instant::now();

        // Update throughput statistics
        self.update_throughput_stats(processing_batch.len())?;

        Ok(processing_batch)
    }

    /// Calculates optimal batch size based on current conditions
    fn calculate_optimal_batch_size(&self) -> Result<usize, String> {
        let mut batch_size = self.config.initial_size.min(32); // Default reasonable batch size

        // Adjust based on buffer fullness
        let buffer_utilization = self.current_size() as f64 / self.sizing_strategy.target_size as f64;
        if buffer_utilization > 0.8 {
            batch_size = (batch_size as f64 * 1.5) as usize; // Larger batches when buffer is full
        } else if buffer_utilization < 0.3 {
            batch_size = (batch_size as f64 * 0.7) as usize; // Smaller batches when buffer is sparse
        }

        // Adjust based on processing latency
        if self.statistics.avg_processing_latency > Duration::from_millis(500) {
            batch_size = (batch_size as f64 * 0.8) as usize; // Smaller batches for slow processing
        }

        // Ensure minimum and maximum bounds
        Ok(batch_size.max(1).min(self.current_size().min(100)))
    }

    /// Updates quality metrics for the buffer
    fn update_quality_metrics(&mut self) -> Result<(), String> {
        if self.buffer.is_empty() && self.secondary_buffer.is_empty() {
            return Ok(());
        }

        let mut quality_sum = A::zero();
        let mut quality_values = Vec::new();

        // Collect quality scores from main buffer
        for prioritized_point in &self.buffer {
            let quality = prioritized_point.data_point.quality_score;
            quality_sum = quality_sum + quality;
            quality_values.push(quality);
        }

        // Collect quality scores from secondary buffer
        for data_point in &self.secondary_buffer {
            let quality = data_point.quality_score;
            quality_sum = quality_sum + quality;
            quality_values.push(quality);
        }

        if !quality_values.is_empty() {
            let count = A::from(quality_values.len()).unwrap();
            self.quality_metrics.average_quality = quality_sum / count;

            // Update min/max quality
            self.quality_metrics.min_quality = quality_values.iter().cloned().fold(A::one(), A::min);
            self.quality_metrics.max_quality = quality_values.iter().cloned().fold(A::zero(), A::max);

            // Calculate quality variance
            let mean = self.quality_metrics.average_quality;
            let variance_sum = quality_values.iter()
                .map(|&q| (q - mean) * (q - mean))
                .sum::<A>();
            self.quality_metrics.quality_variance = variance_sum / count;

            // Update quality trend
            self.update_quality_trend(self.quality_metrics.average_quality)?;
        }

        Ok(())
    }

    /// Updates quality trend analysis
    fn update_quality_trend(&mut self, current_quality: A) -> Result<(), String> {
        let trend = &mut self.quality_metrics.quality_trend;

        // Add current quality to recent changes
        if trend.recent_changes.len() >= 50 {
            trend.recent_changes.pop_front();
        }
        trend.recent_changes.push_back(current_quality);

        // Analyze trend if we have enough data
        if trend.recent_changes.len() >= 10 {
            let recent: Vec<A> = trend.recent_changes.iter().cloned().collect();
            let first_half_avg = recent.iter().take(recent.len() / 2).cloned().sum::<A>() / A::from(recent.len() / 2).unwrap();
            let second_half_avg = recent.iter().skip(recent.len() / 2).cloned().sum::<A>() / A::from(recent.len() - recent.len() / 2).unwrap();

            let change = second_half_avg - first_half_avg;
            let change_threshold = A::from(0.05).unwrap(); // 5% change threshold

            trend.trend_direction = if change > change_threshold {
                TrendDirection::Improving
            } else if change < -change_threshold {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            };

            trend.trend_magnitude = change.abs();
            trend.confidence = A::from(0.8).unwrap(); // Simplified confidence
        }

        Ok(())
    }

    /// Checks if buffer needs resizing
    fn check_buffer_resizing(&mut self) -> Result<(), String> {
        if !self.config.enable_adaptive_sizing {
            return Ok(());
        }

        let current_size = self.current_size();
        let target_size = self.sizing_strategy.target_size;
        let utilization = current_size as f64 / target_size as f64;

        // Check if resize is needed
        let should_resize = if utilization > 0.9 {
            // Buffer is nearly full - consider growing
            Some(SizingReason::ThroughputOptimization)
        } else if utilization < 0.3 && target_size > self.config.min_size {
            // Buffer is underutilized - consider shrinking
            Some(SizingReason::MemoryPressure)
        } else {
            None
        };

        if let Some(reason) = should_resize {
            self.resize_buffer(reason)?;
        }

        Ok(())
    }

    /// Resizes the buffer based on current conditions
    fn resize_buffer(&mut self, reason: SizingReason) -> Result<(), String> {
        let old_size = self.sizing_strategy.target_size;
        let new_size = match reason {
            SizingReason::ThroughputOptimization => {
                // Grow buffer
                let growth_factor = 1.0 + self.sizing_strategy.adjustment_params.growth_rate.to_f64().unwrap_or(0.2);
                ((old_size as f64) * growth_factor) as usize
            },
            SizingReason::MemoryPressure => {
                // Shrink buffer
                let shrink_factor = 1.0 - self.sizing_strategy.adjustment_params.shrinkage_rate.to_f64().unwrap_or(0.2);
                ((old_size as f64) * shrink_factor) as usize
            },
            _ => old_size, // No change for other reasons
        };

        // Apply size bounds
        let bounded_size = new_size.max(self.config.min_size).min(self.config.max_size);

        if bounded_size != old_size {
            self.sizing_strategy.target_size = bounded_size;

            // Log the size change
            let change_event = SizeChangeEvent {
                timestamp: Instant::now(),
                old_size,
                new_size: bounded_size,
                change_magnitude: bounded_size as i32 - old_size as i32,
                reason: format!("{:?}", reason),
            };

            if self.size_change_log.len() >= 100 {
                self.size_change_log.pop_front();
            }
            self.size_change_log.push_back(change_event);
        }

        Ok(())
    }

    /// Applies retention policy to manage buffer size
    fn apply_retention_policy(&mut self) -> Result<(), String> {
        let target_size = self.sizing_strategy.target_size;
        let current_size = self.current_size();

        if current_size <= target_size {
            return Ok(());
        }

        let items_to_remove = current_size - target_size;
        let mut removed_count = 0;

        // Apply retention policy to secondary buffer first
        while removed_count < items_to_remove && !self.secondary_buffer.is_empty() {
            if self.should_remove_from_secondary()? {
                self.secondary_buffer.pop_front();
                removed_count += 1;
                self.statistics.total_items_discarded += 1;
            } else {
                break;
            }
        }

        // If still need to remove items, apply to main buffer
        let mut temp_buffer = Vec::new();
        while let Some(item) = self.buffer.pop() {
            temp_buffer.push(item);
        }

        // Sort by retention score and keep the best items
        temp_buffer.sort_by(|a, b| {
            let score_a = self.calculate_retention_score(&a.data_point).unwrap_or(A::zero());
            let score_b = self.calculate_retention_score(&b.data_point).unwrap_or(A::zero());
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        // Keep only the target number of items
        let items_to_keep = (temp_buffer.len()).saturating_sub(items_to_remove - removed_count);
        for item in temp_buffer.into_iter().take(items_to_keep) {
            self.buffer.push(item);
        }

        Ok(())
    }

    /// Determines if an item should be removed from secondary buffer
    fn should_remove_from_secondary(&self) -> Result<bool, String> {
        // Simple policy: remove oldest items first
        if let Some(oldest) = self.secondary_buffer.front() {
            let age = oldest.timestamp.elapsed();
            Ok(age > Duration::from_secs(3600)) // Remove items older than 1 hour
        } else {
            Ok(false)
        }
    }

    /// Calculates retention score for a data point
    fn calculate_retention_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        let age_score = self.calculate_age_score(data_point);
        let quality_score = data_point.quality_score;
        let freshness_score = self.calculate_freshness_score(data_point);

        // Weighted combination
        let retention_score = quality_score * A::from(0.5).unwrap() +
                             freshness_score * A::from(0.3).unwrap() +
                             age_score * A::from(0.2).unwrap();

        Ok(retention_score)
    }

    /// Calculates age score for retention
    fn calculate_age_score(&self, data_point: &StreamingDataPoint<A>) -> A {
        let age_seconds = data_point.timestamp.elapsed().as_secs_f64();
        let max_age = 7200.0; // 2 hours

        let age_score = (max_age - age_seconds.min(max_age)) / max_age;
        A::from(age_score.max(0.0)).unwrap()
    }

    /// Updates throughput statistics
    fn update_throughput_stats(&mut self, items_processed: usize) -> Result<(), String> {
        let time_since_last = self.last_processing.elapsed().as_secs_f64();
        if time_since_last > 0.0 {
            let current_throughput = items_processed as f64 / time_since_last;
            let throughput_value = A::from(current_throughput).unwrap();

            self.statistics.throughput_stats.current_throughput = throughput_value;

            // Update average throughput (simple moving average)
            let alpha = A::from(0.1).unwrap(); // Smoothing factor
            self.statistics.throughput_stats.avg_throughput =
                alpha * throughput_value + (A::one() - alpha) * self.statistics.throughput_stats.avg_throughput;

            // Update peak throughput
            self.statistics.throughput_stats.peak_throughput =
                self.statistics.throughput_stats.peak_throughput.max(throughput_value);
        }

        Ok(())
    }

    /// Gets current buffer size (total items across all buffers)
    pub fn current_size(&self) -> usize {
        self.buffer.len() + self.secondary_buffer.len()
    }

    /// Gets time since last processing
    pub fn time_since_last_processing(&self) -> Duration {
        self.last_processing.elapsed()
    }

    /// Gets current buffer quality metrics
    pub fn get_quality_metrics(&self) -> BufferQualityMetrics<A> {
        self.quality_metrics.clone()
    }

    /// Computes size adaptation based on performance feedback
    pub fn compute_size_adaptation(&self, performance_tracker: &PerformanceTracker<A>) -> Result<Option<Adaptation<A>>, String> {
        // Get recent performance data
        let recent_performance = performance_tracker.get_recent_performance(10);
        if recent_performance.is_empty() {
            return Ok(None);
        }

        // Calculate average processing time
        let avg_processing_time = recent_performance.iter()
            .map(|p| p.timestamp.elapsed().as_millis() as f64)
            .sum::<f64>() / recent_performance.len() as f64;

        // If processing is too slow, suggest reducing buffer size
        if avg_processing_time > 1000.0 { // More than 1 second
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::BufferSize,
                magnitude: A::from(-0.2).unwrap(), // Reduce by 20%
                target_component: "adaptive_buffer".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::Normal,
                timestamp: Instant::now(),
            };
            return Ok(Some(adaptation));
        }

        // If processing is very fast and buffer is often empty, suggest increasing size
        let avg_utilization = self.current_size() as f64 / self.sizing_strategy.target_size as f64;
        if avg_processing_time < 100.0 && avg_utilization < 0.3 {
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::BufferSize,
                magnitude: A::from(0.3).unwrap(), // Increase by 30%
                target_component: "adaptive_buffer".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::Low,
                timestamp: Instant::now(),
            };
            return Ok(Some(adaptation));
        }

        Ok(None)
    }

    /// Applies size adaptation to the buffer
    pub fn apply_size_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::BufferSize {
            let current_target = self.sizing_strategy.target_size;
            let change_factor = A::one() + adaptation.magnitude;
            let new_target = (current_target as f64 * change_factor.to_f64().unwrap_or(1.0)) as usize;

            // Apply bounds
            let bounded_target = new_target.max(self.config.min_size).min(self.config.max_size);

            if bounded_target != current_target {
                self.sizing_strategy.target_size = bounded_target;

                // Log the change
                let change_event = SizeChangeEvent {
                    timestamp: Instant::now(),
                    old_size: current_target,
                    new_size: bounded_target,
                    change_magnitude: bounded_target as i32 - current_target as i32,
                    reason: "adaptation".to_string(),
                };

                if self.size_change_log.len() >= 100 {
                    self.size_change_log.pop_front();
                }
                self.size_change_log.push_back(change_event);
            }
        }

        Ok(())
    }

    /// Gets the last size change amount
    pub fn last_size_change(&self) -> f32 {
        if let Some(last_change) = self.size_change_log.back() {
            last_change.change_magnitude as f32
        } else {
            0.0
        }
    }

    /// Resets the buffer to initial state
    pub fn reset(&mut self) -> Result<(), String> {
        self.buffer.clear();
        self.secondary_buffer.clear();

        self.quality_metrics = BufferQualityMetrics {
            average_quality: A::zero(),
            quality_variance: A::zero(),
            min_quality: A::one(),
            max_quality: A::zero(),
            freshness_distribution: Vec::new(),
            priority_distribution: Vec::new(),
            quality_trend: QualityTrend {
                recent_changes: VecDeque::with_capacity(50),
                trend_direction: TrendDirection::Stable,
                trend_magnitude: A::zero(),
                confidence: A::zero(),
            },
        };

        self.statistics.total_items_processed = 0;
        self.statistics.total_items_discarded = 0;
        self.last_processing = Instant::now();
        self.size_change_log.clear();

        Ok(())
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> BufferDiagnostics {
        BufferDiagnostics {
            current_size: self.current_size(),
            target_size: self.sizing_strategy.target_size,
            utilization: self.current_size() as f64 / self.sizing_strategy.target_size as f64,
            average_quality: self.quality_metrics.average_quality.to_f64().unwrap_or(0.0),
            total_processed: self.statistics.total_items_processed,
            total_discarded: self.statistics.total_items_discarded,
            size_changes: self.size_change_log.len(),
        }
    }
}

// Implement Ord for PrioritizedDataPoint to work with BinaryHeap
impl<A: Float> Ord for PrioritizedDataPoint<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_score.partial_cmp(&other.priority_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl<A: Float> PartialOrd for PrioritizedDataPoint<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Float> PartialEq for PrioritizedDataPoint<A> {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl<A: Float> Eq for PrioritizedDataPoint<A> {}

impl<A: Float> BufferSizingStrategy<A> {
    fn new(strategy_type: BufferSizeStrategy, initial_size: usize) -> Self {
        Self {
            strategy_type,
            target_size: initial_size,
            adjustment_params: SizeAdjustmentParams {
                growth_rate: A::from(0.2).unwrap(),
                shrinkage_rate: A::from(0.15).unwrap(),
                stability_threshold: A::from(0.05).unwrap(),
                performance_sensitivity: A::from(0.1).unwrap(),
                quality_sensitivity: A::from(0.1).unwrap(),
                memory_sensitivity: A::from(0.2).unwrap(),
            },
            performance_feedback: VecDeque::with_capacity(100),
            sizing_history: VecDeque::with_capacity(100),
        }
    }
}

impl<A: Float> DataRetentionPolicy<A> {
    fn new(strategy: RetentionStrategy) -> Self {
        Self {
            strategy,
            age_policy: AgeBasedRetention {
                max_age: Duration::from_secs(7200), // 2 hours
                soft_age_limit: Duration::from_secs(3600), // 1 hour
                age_weight: 0.3,
                adaptive_limits: true,
            },
            quality_policy: QualityBasedRetention {
                min_quality_threshold: A::from(0.3).unwrap(),
                quality_weight: A::from(0.5).unwrap(),
                adaptive_thresholds: true,
                quality_targets: QualityDistributionTargets {
                    high_quality_target: A::from(0.3).unwrap(),
                    medium_quality_target: A::from(0.5).unwrap(),
                    low_quality_target: A::from(0.2).unwrap(),
                    high_quality_threshold: A::from(0.8).unwrap(),
                    medium_quality_threshold: A::from(0.5).unwrap(),
                },
            },
            relevance_policy: RelevanceBasedRetention {
                relevance_method: RelevanceMethod::Similarity,
                relevance_weight: A::from(0.2).unwrap(),
                temporal_decay: true,
                decay_rate: A::from(0.1).unwrap(),
            },
            retention_scorer: RetentionScorer::new(),
        }
    }
}

impl<A: Float> RetentionScorer<A> {
    fn new() -> Self {
        Self {
            weights: RetentionWeights {
                age_weight: A::from(0.2).unwrap(),
                quality_weight: A::from(0.3).unwrap(),
                relevance_weight: A::from(0.2).unwrap(),
                priority_weight: A::from(0.15).unwrap(),
                freshness_weight: A::from(0.1).unwrap(),
                diversity_weight: A::from(0.05).unwrap(),
            },
            scoring_history: VecDeque::with_capacity(1000),
            performance_feedback: VecDeque::with_capacity(100),
        }
    }
}

/// Diagnostic information for buffer management
#[derive(Debug, Clone)]
pub struct BufferDiagnostics {
    pub current_size: usize,
    pub target_size: usize,
    pub utilization: f64,
    pub average_quality: f64,
    pub total_processed: u64,
    pub total_discarded: u64,
    pub size_changes: usize,
}