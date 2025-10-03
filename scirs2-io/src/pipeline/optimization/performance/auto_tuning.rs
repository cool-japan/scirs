//! Auto-tuning system for dynamic parameter optimization using machine learning
//!
//! This module provides AI-driven parameter optimization for pipeline performance,
//! using reinforcement learning and adaptive algorithms to continuously improve
//! system performance based on real-world feedback.

use crate::error::Result;
use super::super::resource::allocation::SystemMetrics;
use crate::pipeline::{ExecutionRecord, PipelinePerformanceMetrics};
use std::time::Duration;
use serde::{Deserialize, Serialize};
use super::predictive_modeling::{ParameterOptimizationModel, TrainingExample};

/// Auto-tuner for dynamic parameter optimization using machine learning
#[derive(Debug)]
pub struct AutoTuner {
    parameter_model: ParameterOptimizationModel,
    learning_rate: f64,
    exploration_rate: f64,
    performance_baseline: f64,
}

impl AutoTuner {
    pub fn new() -> Self {
        Self {
            parameter_model: ParameterOptimizationModel::new(),
            learning_rate: 0.01,
            exploration_rate: 0.1,
            performance_baseline: 0.0,
        }
    }

    pub fn optimize_parameters(
        &mut self,
        system_metrics: &SystemMetrics,
        historical_data: &[&ExecutionRecord],
        data_size: usize,
    ) -> Result<OptimalParameters> {
        // Extract features from system state and historical data
        let features = self.extract_features(system_metrics, historical_data, data_size)?;

        // Use model to predict optimal parameters
        let predicted_params = self.parameter_model.predict(&features)?;

        // Apply exploration for continuous learning
        let final_params = self.apply_exploration(predicted_params)?;

        Ok(final_params)
    }

    pub fn update_model(
        &mut self,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> Result<()> {
        // Update model with observed performance
        let performance_score = self.calculate_performance_score(metrics);
        self.parameter_model.update(config, performance_score)?;

        // Update baseline performance
        if self.performance_baseline == 0.0 {
            self.performance_baseline = performance_score;
        } else {
            self.performance_baseline = self.performance_baseline * 0.9 + performance_score * 0.1;
        }

        Ok(())
    }

    fn extract_features(
        &self,
        system_metrics: &SystemMetrics,
        historical_data: &[&ExecutionRecord],
        data_size: usize,
    ) -> Result<Vec<f64>> {
        let mut features = vec![
            system_metrics.cpu_usage,
            system_metrics.memory_usage.utilization,
            system_metrics.io_utilization,
            system_metrics.cache_performance.l1_hit_rate,
            system_metrics.cache_performance.l2_hit_rate,
            (data_size as f64).log10(),
        ];

        // Historical performance features
        if !historical_data.is_empty() {
            let avg_throughput: f64 = historical_data
                .iter()
                .map(|r| r.metrics.throughput)
                .sum::<f64>()
                / historical_data.len() as f64;
            features.push(avg_throughput);

            let avg_memory: f64 = historical_data
                .iter()
                .map(|r| r.metrics.peak_memory_usage as f64)
                .sum::<f64>()
                / historical_data.len() as f64;
            features.push(avg_memory.log10());
        } else {
            features.push(0.0);
            features.push(0.0);
        }

        Ok(features)
    }

    fn apply_exploration(&self, mut params: OptimalParameters) -> Result<OptimalParameters> {
        let mut rng = scirs2_core::random::rng();
        if rng.random::<f64>() < self.exploration_rate {
            // Apply random perturbation for exploration
            params.thread_count =
                ((params.thread_count as f64) * (1.0 + (rng.random::<f64>() - 0.5) * 0.2)) as usize;
            params.chunk_size =
                ((params.chunk_size as f64) * (1.0 + (rng.random::<f64>() - 0.5) * 0.2)) as usize;
            params.compression_level = (params.compression_level as f64
                * (1.0 + (rng.random::<f64>() - 0.5) * 0.2))
                .clamp(1.0, 9.0) as u8;
        }

        Ok(params)
    }

    fn calculate_performance_score(&self, metrics: &PipelinePerformanceMetrics) -> f64 {
        // Composite performance score considering multiple factors
        let throughput_score = metrics.throughput / 1000.0; // Normalize throughput
        let memory_efficiency = 1.0 / (metrics.peak_memory_usage as f64 / 1024.0 / 1024.0); // Inverse of MB used
        let cpu_efficiency = metrics.cpu_utilization;
        let cache_efficiency = metrics.cache_hit_rate;

        // Weighted combination
        throughput_score * 0.4
            + memory_efficiency * 0.3
            + cpu_efficiency * 0.2
            + cache_efficiency * 0.1
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimal parameters determined by the auto-tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalParameters {
    pub thread_count: usize,
    pub chunk_size: usize,
    pub simd_enabled: bool,
    pub gpu_enabled: bool,
    pub prefetch_strategy: PrefetchStrategy,
    pub compression_level: u8,
    pub io_buffer_size: usize,
    pub batch_processing: BatchProcessingMode,
}

/// Prefetch strategy for I/O optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    Sequential { distance: usize },
    Random { probability: f64 },
    Adaptive { learning_window: usize },
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchProcessingMode {
    Fixed { batch_size: usize },
    Dynamic {
        min_batch_size: usize,
        max_batch_size: usize,
        latency_target: Duration,
    },
}

/// Optimized pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPipelineConfig {
    pub thread_count: usize,
    pub chunk_size: usize,
    pub simd_optimization: bool,
    pub gpu_acceleration: bool,
    pub compression_level: u8,
    pub io_buffer_size: usize,
    pub memory_strategy: super::super::memory::pool_management::MemoryStrategy,
    pub auto_scaling: bool,
    pub cache_strategy: CacheStrategy,
    pub prefetch_strategy: PrefetchStrategy,
    pub batch_processing: BatchProcessingMode,
}

/// Cache optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    NoCache,
    LRU { capacity: usize },
    LFU { capacity: usize },
    Adaptive { initial_capacity: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::optimization::resource::allocation::{SystemMetrics, MemoryUsage, CachePerformance, NumaTopology};

    #[test]
    fn test_autotuner_creation() {
        let tuner = AutoTuner::new();
        assert_eq!(tuner.learning_rate, 0.01);
        assert_eq!(tuner.exploration_rate, 0.1);
    }

    #[test]
    fn test_feature_extraction() {
        let tuner = AutoTuner::new();
        let system_metrics = SystemMetrics {
            cpu_usage: 0.5,
            memory_usage: MemoryUsage {
                total: 8 * 1024 * 1024 * 1024,
                available: 4 * 1024 * 1024 * 1024,
                used: 4 * 1024 * 1024 * 1024,
                utilization: 0.5,
            },
            io_utilization: 0.3,
            network_bandwidth_usage: 0.2,
            cache_performance: CachePerformance {
                l1_hit_rate: 0.95,
                l2_hit_rate: 0.85,
                l3_hit_rate: 0.75,
                tlb_hit_rate: 0.99,
            },
            numa_topology: NumaTopology::default(),
        };

        let features = tuner.extract_features(&system_metrics, &[], 1024).unwrap();
        assert_eq!(features.len(), 8);
    }

    #[test]
    fn test_exploration_application() {
        let tuner = AutoTuner::new();
        let original_params = OptimalParameters {
            thread_count: 4,
            chunk_size: 1024,
            simd_enabled: true,
            gpu_enabled: false,
            prefetch_strategy: PrefetchStrategy::Sequential { distance: 4 },
            compression_level: 6,
            io_buffer_size: 64 * 1024,
            batch_processing: BatchProcessingMode::Fixed { batch_size: 100 },
        };

        let result = tuner.apply_exploration(original_params);
        assert!(result.is_ok());
    }
}