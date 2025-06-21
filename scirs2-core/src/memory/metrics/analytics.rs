//! Advanced memory analytics for pattern detection and optimization recommendations
//!
//! This module provides sophisticated analysis of memory usage patterns to detect
//! potential issues, memory leaks, and optimization opportunities.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::memory::metrics::{MemoryEvent, MemoryEventType};

#[cfg(feature = "memory_metrics")]
use serde::{Deserialize, Serialize};

/// Memory leak detection configuration
#[derive(Debug, Clone)]
pub struct LeakDetectionConfig {
    /// Window size for leak detection analysis
    pub analysis_window_size: usize,
    /// Minimum threshold for considering a component as potentially leaking (bytes)
    pub leak_threshold_bytes: usize,
    /// Minimum time threshold for leak detection (seconds)
    pub leak_threshold_duration: Duration,
    /// Growth rate threshold for leak detection (bytes/second)
    pub growth_rate_threshold: f64,
    /// Minimum number of data points for reliable analysis
    pub min_data_points: usize,
}

impl Default for LeakDetectionConfig {
    fn default() -> Self {
        Self {
            analysis_window_size: 100,
            leak_threshold_bytes: 1024 * 1024, // 1 MB
            leak_threshold_duration: Duration::from_secs(30),
            growth_rate_threshold: 1024.0, // 1 KB/sec
            min_data_points: 10,
        }
    }
}

/// Memory pattern analysis results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct MemoryPatternAnalysis {
    /// Component being analyzed
    pub component: String,
    /// Detected allocation patterns
    pub patterns: Vec<AllocationPattern>,
    /// Memory efficiency metrics
    pub efficiency: MemoryEfficiencyMetrics,
    /// Potential issues found
    pub potential_issues: Vec<MemoryIssue>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Detected allocation patterns
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub enum AllocationPattern {
    /// Steady growth in memory usage
    SteadyGrowth {
        /// Growth rate in bytes per second
        rate: f64,
        /// Confidence level (0.0 to 1.0)
        confidence: f64,
    },
    /// Periodic allocation/deallocation cycles
    PeriodicCycle {
        /// Cycle duration
        cycle_duration: Duration,
        /// Peak size during cycle
        peak_size: usize,
        /// Confidence level
        confidence: f64,
    },
    /// Burst allocations followed by steady usage
    BurstAllocation {
        /// Burst size in bytes
        burst_size: usize,
        /// Burst duration
        burst_duration: Duration,
        /// Confidence level
        confidence: f64,
    },
    /// Memory usage plateau
    Plateau {
        /// Plateau size in bytes
        size: usize,
        /// Duration of plateau
        duration: Duration,
        /// Confidence level
        confidence: f64,
    },
}

/// Memory efficiency metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct MemoryEfficiencyMetrics {
    /// Memory reuse ratio (total allocated / peak usage)
    pub reuse_ratio: f64,
    /// Allocation frequency (allocations per second)
    pub allocation_frequency: f64,
    /// Average allocation lifetime
    pub avg_allocation_lifetime: Duration,
    /// Memory fragmentation estimate (0.0 to 1.0)
    pub fragmentation_estimate: f64,
    /// Buffer pool efficiency (if applicable)
    pub buffer_pool_efficiency: Option<f64>,
}

/// Potential memory issues
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub enum MemoryIssue {
    /// Potential memory leak detected
    MemoryLeak {
        /// Growth rate in bytes per second
        growth_rate: f64,
        /// Duration of observed growth
        duration: Duration,
        /// Severity level (0.0 to 1.0)
        severity: f64,
    },
    /// High allocation frequency
    HighAllocationFrequency {
        /// Allocations per second
        frequency: f64,
        /// Potential performance impact
        impact: String,
    },
    /// Large peak memory usage
    HighPeakUsage {
        /// Peak size in bytes
        peak_size: usize,
        /// Percentage of total system memory (if available)
        system_percentage: Option<f64>,
    },
    /// Memory fragmentation
    MemoryFragmentation {
        /// Estimated fragmentation ratio
        fragmentation_ratio: f64,
        /// Potential waste in bytes
        potential_waste: usize,
    },
    /// Inefficient buffer pool usage
    IneffientBufferPool {
        /// Pool efficiency ratio
        efficiency: f64,
        /// Number of pool misses
        pool_misses: usize,
    },
}

/// Optimization recommendations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub enum OptimizationRecommendation {
    /// Use buffer pooling
    UseBufferPooling {
        /// Expected memory savings
        expected_savings: usize,
        /// Suggested pool sizes
        suggested_pool_sizes: Vec<usize>,
    },
    /// Batch allocations
    BatchAllocations {
        /// Current allocation frequency
        current_frequency: f64,
        /// Suggested batch size
        suggested_batch_size: usize,
    },
    /// Pre-allocate memory
    PreAllocateMemory {
        /// Suggested pre-allocation size
        suggested_size: usize,
        /// Expected performance improvement
        performance_gain: String,
    },
    /// Use memory-efficient data structures
    UseMemoryEfficientStructures {
        /// Current structure type
        current_type: String,
        /// Suggested alternative
        suggested_alternative: String,
        /// Expected memory reduction
        memory_reduction: usize,
    },
    /// Implement memory compaction
    ImplementCompaction {
        /// Estimated fragmentation reduction
        fragmentation_reduction: f64,
        /// Suggested compaction frequency
        suggested_frequency: Duration,
    },
}

/// Memory leak detection result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct LeakDetectionResult {
    /// Component being analyzed
    pub component: String,
    /// Whether a leak was detected
    pub leak_detected: bool,
    /// Growth rate (bytes per second)
    pub growth_rate: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Duration of analysis
    pub analysis_duration: Duration,
    /// Current memory usage
    pub current_usage: usize,
    /// Projected usage after 1 hour
    pub projected_usage_1h: usize,
    /// Projected usage after 24 hours
    pub projected_usage_24h: usize,
}

/// Advanced memory analytics engine
pub struct MemoryAnalytics {
    /// Configuration for leak detection
    leak_config: LeakDetectionConfig,
    /// Historical memory usage data per component
    usage_history: HashMap<String, VecDeque<(Instant, usize)>>,
    /// Allocation event history per component
    allocation_history: HashMap<String, VecDeque<(Instant, usize, MemoryEventType)>>,
}

impl MemoryAnalytics {
    /// Create a new memory analytics engine
    pub fn new(leak_config: LeakDetectionConfig) -> Self {
        Self {
            leak_config,
            usage_history: HashMap::new(),
            allocation_history: HashMap::new(),
        }
    }

    /// Update analytics with a new memory event
    pub fn record_event(&mut self, event: MemoryEvent) {
        let component = event.component.clone();
        let timestamp = event.timestamp;

        // Update allocation history
        let alloc_history = self
            .allocation_history
            .entry(component.clone())
            .or_default();
        alloc_history.push_back((timestamp, event.size, event.event_type));

        // Limit history size
        while alloc_history.len() > self.leak_config.analysis_window_size {
            alloc_history.pop_front();
        }

        // Calculate current usage for this component
        let current_usage = self.calculate_current_usage(&component);

        // Update usage history
        let usage_history = self.usage_history.entry(component).or_default();
        usage_history.push_back((timestamp, current_usage));

        // Limit history size
        while usage_history.len() > self.leak_config.analysis_window_size {
            usage_history.pop_front();
        }
    }

    /// Calculate current memory usage for a component
    fn calculate_current_usage(&self, component: &str) -> usize {
        if let Some(history) = self.allocation_history.get(component) {
            let mut usage = 0usize;
            for (_timestamp, size, event_type) in history {
                match event_type {
                    MemoryEventType::Allocation => usage += size,
                    MemoryEventType::Deallocation => usage = usage.saturating_sub(*size),
                    MemoryEventType::Resize => {
                        // For resize events, we need additional metadata
                        // For now, treat as allocation
                        usage += size;
                    }
                    MemoryEventType::Access | MemoryEventType::Transfer => {
                        // These events don't affect memory usage calculations
                    }
                }
            }
            usage
        } else {
            0
        }
    }

    /// Perform leak detection for a specific component
    pub fn detect_memory_leak(&self, component: &str) -> Option<LeakDetectionResult> {
        let usage_history = self.usage_history.get(component)?;

        if usage_history.len() < self.leak_config.min_data_points {
            return None;
        }

        // Calculate linear regression to detect growth trend
        let (growth_rate, confidence) = self.calculate_growth_rate(usage_history);

        let analysis_duration = usage_history
            .back()?
            .0
            .duration_since(usage_history.front()?.0);

        if analysis_duration < self.leak_config.leak_threshold_duration {
            return None;
        }

        let current_usage = usage_history.back()?.1;
        let leak_detected = growth_rate > self.leak_config.growth_rate_threshold
            && confidence > 0.7
            && current_usage > self.leak_config.leak_threshold_bytes;

        Some(LeakDetectionResult {
            component: component.to_string(),
            leak_detected,
            growth_rate,
            confidence,
            analysis_duration,
            current_usage,
            projected_usage_1h: current_usage + (growth_rate * 3600.0) as usize,
            projected_usage_24h: current_usage + (growth_rate * 86400.0) as usize,
        })
    }

    /// Calculate growth rate using linear regression
    fn calculate_growth_rate(&self, history: &VecDeque<(Instant, usize)>) -> (f64, f64) {
        if history.len() < 2 {
            return (0.0, 0.0);
        }

        let start_time = history.front().unwrap().0;
        let points: Vec<(f64, f64)> = history
            .iter()
            .map(|(timestamp, usage)| {
                let x = timestamp.duration_since(start_time).as_secs_f64();
                let y = *usage as f64;
                (x, y)
            })
            .collect();

        self.linear_regression(&points)
    }

    /// Perform linear regression to find slope (growth rate) and R-squared (confidence)
    fn linear_regression(&self, points: &[(f64, f64)]) -> (f64, f64) {
        let n = points.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = points.iter().map(|(_, y)| y * y).sum();

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let numerator = sum_xy - n * mean_x * mean_y;
        let denominator = sum_x2 - n * mean_x * mean_x;

        if denominator.abs() < f64::EPSILON {
            return (0.0, 0.0);
        }

        let slope = numerator / denominator;

        // Calculate R-squared
        let ss_tot = sum_y2 - n * mean_y * mean_y;
        let ss_res: f64 = points
            .iter()
            .map(|(x, y)| {
                let predicted = slope * (x - mean_x) + mean_y;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() < f64::EPSILON {
            0.0
        } else {
            1.0 - (ss_res / ss_tot).max(0.0)
        };

        (slope, r_squared)
    }

    /// Perform comprehensive pattern analysis for a component
    pub fn analyze_patterns(&self, component: &str) -> Option<MemoryPatternAnalysis> {
        let usage_history = self.usage_history.get(component)?;
        let allocation_history = self.allocation_history.get(component)?;

        if usage_history.len() < self.leak_config.min_data_points {
            return None;
        }

        let patterns = self.detect_allocation_patterns(usage_history, allocation_history);
        let efficiency = self.calculate_efficiency_metrics(component, allocation_history);
        let potential_issues = self.identify_potential_issues(component);
        let recommendations = self.generate_recommendations(&efficiency, &potential_issues);

        Some(MemoryPatternAnalysis {
            component: component.to_string(),
            patterns,
            efficiency,
            potential_issues,
            recommendations,
        })
    }

    /// Detect allocation patterns in the usage history
    fn detect_allocation_patterns(
        &self,
        usage_history: &VecDeque<(Instant, usize)>,
        allocation_history: &VecDeque<(Instant, usize, MemoryEventType)>,
    ) -> Vec<AllocationPattern> {
        let mut patterns = Vec::new();

        // Detect steady growth
        let (growth_rate, confidence) = self.calculate_growth_rate(usage_history);
        if growth_rate > 100.0 && confidence > 0.8 {
            patterns.push(AllocationPattern::SteadyGrowth {
                rate: growth_rate,
                confidence,
            });
        }

        // Detect periodic cycles
        if let Some(cycle) = self.detect_periodic_cycles(usage_history) {
            patterns.push(cycle);
        }

        // Detect burst allocations
        if let Some(burst) = self.detect_burst_allocations(allocation_history) {
            patterns.push(burst);
        }

        // Detect plateaus
        if let Some(plateau) = self.detect_plateaus(usage_history) {
            patterns.push(plateau);
        }

        patterns
    }

    /// Detect periodic allocation cycles
    fn detect_periodic_cycles(
        &self,
        usage_history: &VecDeque<(Instant, usize)>,
    ) -> Option<AllocationPattern> {
        // This is a simplified cycle detection algorithm
        // In a real implementation, you might use FFT or autocorrelation

        if usage_history.len() < 10 {
            return None;
        }

        // Look for repeating patterns in memory usage
        let values: Vec<usize> = usage_history.iter().map(|(_, usage)| *usage).collect();

        // Simple pattern detection: look for similar values at regular intervals
        for cycle_len in 3..values.len() / 3 {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in cycle_len..values.len() {
                let diff = (values[i] as f64 - values[i - cycle_len] as f64).abs();
                let avg = (values[i] + values[i - cycle_len]) as f64 / 2.0;
                if avg > 0.0 {
                    correlation += 1.0 - (diff / avg).min(1.0);
                    count += 1;
                }
            }

            if count > 0 {
                correlation /= count as f64;
                if correlation > 0.8 {
                    let cycle_duration = Duration::from_secs((cycle_len * 5) as u64); // Approximate
                    let peak_size = values.iter().max().copied().unwrap_or(0);

                    return Some(AllocationPattern::PeriodicCycle {
                        cycle_duration,
                        peak_size,
                        confidence: correlation,
                    });
                }
            }
        }

        None
    }

    /// Detect burst allocation patterns
    fn detect_burst_allocations(
        &self,
        allocation_history: &VecDeque<(Instant, usize, MemoryEventType)>,
    ) -> Option<AllocationPattern> {
        if allocation_history.len() < 5 {
            return None;
        }

        // Look for periods of high allocation activity
        let mut burst_size = 0usize;
        let mut burst_start: Option<Instant> = None;
        let mut current_burst_size = 0usize;

        for (timestamp, size, event_type) in allocation_history {
            match event_type {
                MemoryEventType::Allocation => {
                    if burst_start.is_none() {
                        burst_start = Some(*timestamp);
                        current_burst_size = *size;
                    } else {
                        current_burst_size += size;
                    }
                }
                MemoryEventType::Deallocation => {
                    if let Some(start) = burst_start {
                        let duration = timestamp.duration_since(start);
                        if duration > Duration::from_millis(100) && current_burst_size > burst_size
                        {
                            burst_size = current_burst_size;
                        }
                        burst_start = None;
                        current_burst_size = 0;
                    }
                }
                _ => {}
            }
        }

        if burst_size > 1024 * 1024 {
            // 1 MB threshold
            Some(AllocationPattern::BurstAllocation {
                burst_size,
                burst_duration: Duration::from_millis(500), // Approximate
                confidence: 0.9,
            })
        } else {
            None
        }
    }

    /// Detect memory usage plateaus
    fn detect_plateaus(
        &self,
        usage_history: &VecDeque<(Instant, usize)>,
    ) -> Option<AllocationPattern> {
        if usage_history.len() < 10 {
            return None;
        }

        let values: Vec<usize> = usage_history.iter().map(|(_, usage)| *usage).collect();

        // Look for periods of stable memory usage
        let mut plateau_start = 0;
        let mut max_plateau_len = 0;
        let mut plateau_value = 0;

        for i in 1..values.len() {
            let diff_ratio = if values[i - 1] > 0 {
                (values[i] as f64 - values[i - 1] as f64).abs() / values[i - 1] as f64
            } else {
                0.0
            };

            if diff_ratio < 0.05 {
                // Less than 5% change
                if plateau_start == 0 {
                    plateau_start = i - 1;
                }
            } else {
                if plateau_start > 0 {
                    let plateau_len = i - plateau_start;
                    if plateau_len > max_plateau_len {
                        max_plateau_len = plateau_len;
                        plateau_value = values[plateau_start];
                    }
                }
                plateau_start = 0;
            }
        }

        // Check final plateau
        if plateau_start > 0 {
            let plateau_len = values.len() - plateau_start;
            if plateau_len > max_plateau_len {
                max_plateau_len = plateau_len;
                plateau_value = values[plateau_start];
            }
        }

        if max_plateau_len >= 5 && plateau_value > 0 {
            Some(AllocationPattern::Plateau {
                size: plateau_value,
                duration: Duration::from_secs((max_plateau_len * 5) as u64), // Approximate
                confidence: 0.8,
            })
        } else {
            None
        }
    }

    /// Calculate efficiency metrics for a component
    fn calculate_efficiency_metrics(
        &self,
        component: &str,
        allocation_history: &VecDeque<(Instant, usize, MemoryEventType)>,
    ) -> MemoryEfficiencyMetrics {
        if allocation_history.is_empty() {
            return MemoryEfficiencyMetrics {
                reuse_ratio: 0.0,
                allocation_frequency: 0.0,
                avg_allocation_lifetime: Duration::from_secs(0),
                fragmentation_estimate: 0.0,
                buffer_pool_efficiency: None,
            };
        }

        // Calculate basic metrics
        let total_allocated = allocation_history
            .iter()
            .filter_map(|(_, size, event_type)| {
                if matches!(event_type, MemoryEventType::Allocation) {
                    Some(*size)
                } else {
                    None
                }
            })
            .sum::<usize>();

        let current_usage = self.calculate_current_usage(component);
        let reuse_ratio = if current_usage > 0 {
            total_allocated as f64 / current_usage as f64
        } else {
            0.0
        };

        let allocation_count = allocation_history
            .iter()
            .filter(|(_, _, event_type)| matches!(event_type, MemoryEventType::Allocation))
            .count();

        let duration = if let (Some(first), Some(last)) =
            (allocation_history.front(), allocation_history.back())
        {
            last.0.duration_since(first.0)
        } else {
            Duration::from_secs(1)
        };

        let allocation_frequency = allocation_count as f64 / duration.as_secs_f64();

        // Estimate average allocation lifetime
        let avg_allocation_lifetime = if allocation_count > 0 {
            duration / allocation_count as u32
        } else {
            Duration::from_secs(0)
        };

        // Estimate fragmentation (simplified heuristic)
        let fragmentation_estimate = self.estimate_fragmentation(allocation_history);

        MemoryEfficiencyMetrics {
            reuse_ratio,
            allocation_frequency,
            avg_allocation_lifetime,
            fragmentation_estimate,
            buffer_pool_efficiency: None, // Would require buffer pool instrumentation
        }
    }

    /// Estimate memory fragmentation
    fn estimate_fragmentation(
        &self,
        allocation_history: &VecDeque<(Instant, usize, MemoryEventType)>,
    ) -> f64 {
        // This is a simplified fragmentation estimate
        // In reality, you'd need more detailed memory layout information

        let allocation_sizes: Vec<usize> = allocation_history
            .iter()
            .filter_map(|(_, size, event_type)| {
                if matches!(event_type, MemoryEventType::Allocation) {
                    Some(*size)
                } else {
                    None
                }
            })
            .collect();

        if allocation_sizes.is_empty() {
            return 0.0;
        }

        // Calculate coefficient of variation as a proxy for fragmentation
        let mean = allocation_sizes.iter().sum::<usize>() as f64 / allocation_sizes.len() as f64;
        let variance = allocation_sizes
            .iter()
            .map(|&size| {
                let diff = size as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / allocation_sizes.len() as f64;

        let std_dev = variance.sqrt();

        if mean > 0.0 {
            (std_dev / mean).min(1.0)
        } else {
            0.0
        }
    }

    /// Identify potential memory issues
    fn identify_potential_issues(&self, component: &str) -> Vec<MemoryIssue> {
        let mut issues = Vec::new();

        // Check for memory leaks
        if let Some(leak_result) = self.detect_memory_leak(component) {
            if leak_result.leak_detected {
                issues.push(MemoryIssue::MemoryLeak {
                    growth_rate: leak_result.growth_rate,
                    duration: leak_result.analysis_duration,
                    severity: leak_result.confidence,
                });
            }
        }

        // Check for high allocation frequency
        if let Some(allocation_history) = self.allocation_history.get(component) {
            let efficiency = self.calculate_efficiency_metrics(component, allocation_history);

            if efficiency.allocation_frequency > 100.0 {
                // More than 100 allocations per second
                issues.push(MemoryIssue::HighAllocationFrequency {
                    frequency: efficiency.allocation_frequency,
                    impact: "High allocation frequency can cause performance degradation"
                        .to_string(),
                });
            }

            if efficiency.fragmentation_estimate > 0.7 {
                let current_usage = self.calculate_current_usage(component);
                let potential_waste =
                    (current_usage as f64 * efficiency.fragmentation_estimate) as usize;

                issues.push(MemoryIssue::MemoryFragmentation {
                    fragmentation_ratio: efficiency.fragmentation_estimate,
                    potential_waste,
                });
            }
        }

        // Check for high peak usage
        if let Some(usage_history) = self.usage_history.get(component) {
            if let Some(peak_usage) = usage_history.iter().map(|(_, usage)| *usage).max() {
                if peak_usage > 100 * 1024 * 1024 {
                    // 100 MB threshold
                    issues.push(MemoryIssue::HighPeakUsage {
                        peak_size: peak_usage,
                        system_percentage: None, // Would require system memory detection
                    });
                }
            }
        }

        issues
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        efficiency: &MemoryEfficiencyMetrics,
        issues: &[MemoryIssue],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Recommend buffer pooling for high allocation frequency
        if efficiency.allocation_frequency > 50.0 {
            recommendations.push(OptimizationRecommendation::UseBufferPooling {
                expected_savings: (efficiency.allocation_frequency * 1024.0) as usize, // Rough estimate
                suggested_pool_sizes: vec![1024, 4096, 16384, 65536],
            });
        }

        // Recommend batching for frequent small allocations
        if efficiency.allocation_frequency > 20.0
            && efficiency.avg_allocation_lifetime.as_secs() < 10
        {
            recommendations.push(OptimizationRecommendation::BatchAllocations {
                current_frequency: efficiency.allocation_frequency,
                suggested_batch_size: (efficiency.allocation_frequency * 2.0) as usize,
            });
        }

        // Recommend pre-allocation for predictable patterns
        if efficiency.reuse_ratio > 2.0 {
            recommendations.push(OptimizationRecommendation::PreAllocateMemory {
                suggested_size: (efficiency.allocation_frequency * 1024.0) as usize,
                performance_gain: "Reduced allocation overhead".to_string(),
            });
        }

        // Recommend compaction for high fragmentation
        if efficiency.fragmentation_estimate > 0.5 {
            recommendations.push(OptimizationRecommendation::ImplementCompaction {
                fragmentation_reduction: efficiency.fragmentation_estimate * 0.7, // Estimated reduction
                suggested_frequency: Duration::from_secs(300),                    // 5 minutes
            });
        }

        // Specific recommendations based on detected issues
        for issue in issues {
            if let MemoryIssue::HighPeakUsage { peak_size, .. } = issue {
                recommendations.push(OptimizationRecommendation::UseMemoryEfficientStructures {
                    current_type: "Unknown".to_string(),
                    suggested_alternative: "Streaming or memory-mapped structures".to_string(),
                    memory_reduction: peak_size / 2, // Rough estimate
                });
            }
        }

        recommendations
    }

    /// Get leak detection results for all components
    pub fn get_leak_detection_results(&self) -> Vec<LeakDetectionResult> {
        self.usage_history
            .keys()
            .filter_map(|component| self.detect_memory_leak(component))
            .collect()
    }

    /// Get pattern analysis for all components
    pub fn get_pattern_analysis_results(&self) -> Vec<MemoryPatternAnalysis> {
        self.usage_history
            .keys()
            .filter_map(|component| self.analyze_patterns(component))
            .collect()
    }

    /// Clear all analytics data
    pub fn clear(&mut self) {
        self.usage_history.clear();
        self.allocation_history.clear();
    }
}

impl Default for MemoryAnalytics {
    fn default() -> Self {
        Self::new(LeakDetectionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_analytics_creation() {
        let analytics = MemoryAnalytics::new(LeakDetectionConfig::default());
        assert!(analytics.usage_history.is_empty());
        assert!(analytics.allocation_history.is_empty());
    }

    #[test]
    fn test_leak_detection_insufficient_data() {
        let analytics = MemoryAnalytics::new(LeakDetectionConfig::default());
        let result = analytics.detect_memory_leak("test_component");
        assert!(result.is_none());
    }

    #[test]
    fn test_linear_regression() {
        let analytics = MemoryAnalytics::new(LeakDetectionConfig::default());

        // Test with perfect linear growth
        let points = vec![(0.0, 0.0), (1.0, 100.0), (2.0, 200.0), (3.0, 300.0)];

        let (slope, r_squared) = analytics.linear_regression(&points);
        assert!((slope - 100.0).abs() < 1.0);
        assert!(r_squared > 0.99);
    }

    #[test]
    fn test_pattern_analysis_with_insufficient_data() {
        let analytics = MemoryAnalytics::new(LeakDetectionConfig::default());
        let result = analytics.analyze_patterns("test_component");
        assert!(result.is_none());
    }
}
