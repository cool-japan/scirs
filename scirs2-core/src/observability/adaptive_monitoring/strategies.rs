//! Optimization strategies and adaptive performance tuning

use super::types::{
    ComprehensivePerformanceMetrics, ImpactLevel, OptimizationAction, OptimizationActionType,
    OptimizationPriority, OptimizationRecommendation, OptimizationStrategy,
    PerformanceLearningModel, PerformancePredictions, RecommendationCategory,
    RecommendationPriority,
};
use crate::error::CoreResult;
use std::collections::HashMap;
use std::time::Instant;

/// Intelligent optimization engine with adaptive learning capabilities
#[allow(dead_code)]
#[derive(Debug)]
pub struct OptimizationEngine {
    optimization_history: Vec<OptimizationAction>,
    learning_model: PerformanceLearningModel,
    current_strategy: OptimizationStrategy,
    strategy_effectiveness: HashMap<OptimizationStrategy, f64>,
}

impl OptimizationEngine {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            optimization_history: Vec::new(),
            learning_model: PerformanceLearningModel::new()?,
            current_strategy: OptimizationStrategy::Conservative,
            strategy_effectiveness: HashMap::new(),
        })
    }

    pub fn apply_strategy(
        &mut self,
        current_metrics: &ComprehensivePerformanceMetrics,
        predictions: &PerformancePredictions,
    ) -> CoreResult<()> {
        // Analyze current performance
        let performance_score = self.calculate_performance_score(current_metrics);

        // Check if optimization is needed
        if self.needs_optimization(current_metrics, predictions)? {
            let optimization_action =
                self.determine_optimization_action(current_metrics, predictions)?;
            self.execute_optimization(optimization_action)?;
        }

        // Update learning model
        self.learning_model.update_with_metrics(current_metrics)?;

        // Adapt strategy based on effectiveness
        self.adapt_strategy(performance_score)?;

        Ok(())
    }

    fn calculate_performance_score(&self, metrics: &ComprehensivePerformanceMetrics) -> f64 {
        let cpu_score = 1.0 - metrics.cpu_utilization;
        let memory_score = 1.0 - metrics.memory_utilization;
        let latency_score = 1.0 / (1.0 + metrics.average_latency_ms / 100.0);
        let throughput_score = metrics.operations_per_second / 10000.0;

        (cpu_score + memory_score + latency_score + throughput_score) / 4.0
    }

    fn needs_optimization(
        &self,
        current_metrics: &ComprehensivePerformanceMetrics,
        predictions: &PerformancePredictions,
    ) -> CoreResult<bool> {
        // Check current performance thresholds
        if current_metrics.cpu_utilization > 0.8 || current_metrics.memory_utilization > 0.8 {
            return Ok(true);
        }

        // Check predicted performance issues
        if predictions.predicted_cpu_spike || predictions.predicted_memory_pressure {
            return Ok(true);
        }

        // Check for performance degradation trends
        if current_metrics.operations_per_second < 100.0
            || current_metrics.average_latency_ms > 1000.0
        {
            return Ok(true);
        }

        Ok(false)
    }

    fn select_optimization_action(
        &self,
        current_metrics: &ComprehensivePerformanceMetrics,
        predictions: &PerformancePredictions,
    ) -> CoreResult<OptimizationAction> {
        let mut actions = Vec::new();

        // CPU optimization
        if current_metrics.cpu_utilization > 0.8 {
            actions.push(OptimizationActionType::ReduceThreads);
        } else if current_metrics.cpu_utilization < 0.3 {
            actions.push(OptimizationActionType::IncreaseParallelism);
        }

        // Memory optimization
        if current_metrics.memory_utilization > 0.8 {
            actions.push(OptimizationActionType::ReduceMemoryUsage);
        }

        // Cache optimization
        if current_metrics.cache_miss_rate > 0.1 {
            actions.push(OptimizationActionType::OptimizeCacheUsage);
        }

        // Predictive optimization
        if predictions.predicted_cpu_spike {
            actions.push(OptimizationActionType::PreemptiveCpuOptimization);
        }

        if predictions.predicted_memory_pressure {
            actions.push(OptimizationActionType::PreemptiveMemoryOptimization);
        }

        Ok(OptimizationAction {
            actions,
            timestamp: Instant::now(),
            reason: "Adaptive optimization based on current metrics and predictions".to_string(),
            priority: OptimizationPriority::Medium,
            expected_impact: ImpactLevel::Medium,
            success: false,
        })
    }

    fn execute_optimization(&mut self, mut action: OptimizationAction) -> CoreResult<()> {
        let mut execution_success = true;

        for action_type in &action.actions {
            let result = match action_type {
                OptimizationActionType::ReduceThreads => self.reduce_thread_count(),
                OptimizationActionType::IncreaseParallelism => self.increase_parallelism(),
                OptimizationActionType::ReduceMemoryUsage => self.reduce_memory_usage(),
                OptimizationActionType::OptimizeCacheUsage => self.optimize_cache_usage(),
                OptimizationActionType::PreemptiveCpuOptimization => {
                    self.preemptive_cpu_optimization()
                }
                OptimizationActionType::PreemptiveMemoryOptimization => {
                    self.preemptive_memory_optimization()
                }
                OptimizationActionType::ReduceCpuUsage => self.reduce_cpu_usage(),
                OptimizationActionType::OptimizePerformance => self.optimize_performance(),
            };

            if result.is_err() {
                execution_success = false;
            }
        }

        action.success = execution_success;
        self.optimization_history.push(action);
        Ok(())
    }

    fn reduce_thread_count(&self) -> CoreResult<()> {
        // Reduce thread count by 20%
        #[cfg(feature = "parallel")]
        {
            let current_threads = crate::parallel_ops::get_num_threads();
            let new_threads = ((current_threads as f64) * 0.8) as usize;
            crate::parallel_ops::set_num_threads(new_threads.max(1));
        }
        Ok(())
    }

    fn increase_parallelism(&self) -> CoreResult<()> {
        // Increase thread count by 20%
        #[cfg(feature = "parallel")]
        {
            let current_threads = crate::parallel_ops::get_num_threads();
            let max_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let new_threads = ((current_threads as f64) * 1.2) as usize;
            crate::parallel_ops::set_num_threads(new_threads.min(max_threads));
        }
        Ok(())
    }

    fn reduce_memory_usage(&self) -> CoreResult<()> {
        // Trigger garbage collection or memory cleanup
        // This would integrate with memory management systems
        Ok(())
    }

    fn optimize_cache_usage(&self) -> CoreResult<()> {
        // Optimize cache usage patterns
        // This would adjust cache-aware algorithms
        Ok(())
    }

    fn preemptive_cpu_optimization(&self) -> CoreResult<()> {
        // Preemptively optimize for predicted CPU spike
        self.reduce_thread_count()?;
        Ok(())
    }

    fn preemptive_memory_optimization(&self) -> CoreResult<()> {
        // Preemptively optimize for predicted memory pressure
        self.reduce_memory_usage()?;
        Ok(())
    }

    fn reduce_cpu_usage(&self) -> CoreResult<()> {
        // Implement CPU usage reduction
        Ok(())
    }

    fn optimize_performance(&self) -> CoreResult<()> {
        // Implement general performance optimization
        Ok(())
    }

    fn update_effectiveness(&mut self, score: f64) -> CoreResult<()> {
        // Update strategy effectiveness
        let current_effectiveness = self
            .strategy_effectiveness
            .entry(self.current_strategy)
            .or_insert(0.5);
        *current_effectiveness = (*current_effectiveness * 0.9) + (score * 0.1);

        // Consider switching strategy if current one is not effective
        if *current_effectiveness < 0.3 {
            self.current_strategy = match self.current_strategy {
                OptimizationStrategy::Conservative => OptimizationStrategy::Aggressive,
                OptimizationStrategy::Aggressive => OptimizationStrategy::Balanced,
                OptimizationStrategy::Balanced => OptimizationStrategy::Conservative,
            };
        }

        Ok(())
    }

    pub fn get_recommendations(&self) -> CoreResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Analyze optimization history
        if self.optimization_history.len() >= 10 {
            let recent_actions: Vec<_> = self.optimization_history.iter().rev().take(10).collect();

            // Check for repeated actions (might indicate ineffective optimization)
            let action_counts = self.count_action_types(&recent_actions);
            for (action_type, count) in action_counts {
                if count >= 5 {
                    recommendations.push(OptimizationRecommendation {
                        category: RecommendationCategory::Optimization,
                        title: format!("Frequent {action_type:?} actions detected"),
                        description: "Consider investigating root cause of performance issues"
                            .to_string(),
                        priority: RecommendationPriority::High,
                        estimated_impact: ImpactLevel::Medium,
                    });
                }
            }
        }

        // Strategy recommendations
        if let Some(&effectiveness) = self.strategy_effectiveness.get(&self.current_strategy) {
            if effectiveness < 0.5 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Strategy,
                    title: "Current optimization strategy showing low effectiveness".to_string(),
                    description: format!(
                        "Consider switching from {:?} strategy",
                        self.current_strategy
                    ),
                    priority: RecommendationPriority::Medium,
                    estimated_impact: ImpactLevel::High,
                });
            }
        }

        Ok(recommendations)
    }

    fn count_action_types(
        &self,
        actions: &[&OptimizationAction],
    ) -> HashMap<OptimizationActionType, usize> {
        let mut counts = HashMap::new();
        for action in actions {
            for action_type in &action.actions {
                *counts.entry(*action_type).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Adaptive optimization method
    pub fn adaptive_optimize(
        &mut self,
        current_metrics: &ComprehensivePerformanceMetrics,
        predictions: &PerformancePredictions,
    ) -> CoreResult<()> {
        // Apply optimization strategy
        self.apply_strategy(current_metrics, predictions)?;

        // Update effectiveness tracking
        let performance_score = self.calculate_performance_score(current_metrics);
        self.strategy_effectiveness
            .insert(self.current_strategy, performance_score);

        Ok(())
    }

    /// Determine the optimization action based on current metrics and predictions
    pub fn determine_optimization_action(
        &mut self,
        current_metrics: &ComprehensivePerformanceMetrics,
        predictions: &PerformancePredictions,
    ) -> CoreResult<OptimizationAction> {
        // Analyze metrics to determine action
        let mut actions = Vec::new();

        // Check CPU usage
        if current_metrics.cpu_utilization > 0.8 {
            actions.push(OptimizationActionType::ReduceCpuUsage);
        }

        // Check memory usage
        if current_metrics.memory_utilization > 0.8 {
            actions.push(OptimizationActionType::ReduceMemoryUsage);
        }

        // Check for performance issues based on predictions
        if predictions.predicted_performance_change < -0.1 {
            actions.push(OptimizationActionType::OptimizePerformance);
        }

        Ok(OptimizationAction {
            timestamp: std::time::Instant::now(),
            actions,
            priority: OptimizationPriority::Medium,
            reason: "Performance optimization based on metrics analysis".to_string(),
            expected_impact: ImpactLevel::Medium,
            success: false, // Will be updated after execution
        })
    }

    /// Adapt the optimization strategy based on performance score
    pub fn adapt_strategy(&mut self, performance_score: f64) -> CoreResult<()> {
        // Simple strategy adaptation logic
        if performance_score < 0.3 {
            self.current_strategy = OptimizationStrategy::Aggressive;
        } else if performance_score < 0.7 {
            self.current_strategy = OptimizationStrategy::Balanced;
        } else {
            self.current_strategy = OptimizationStrategy::Conservative;
        }

        // Update effectiveness for the current strategy
        self.update_effectiveness(performance_score)?;

        Ok(())
    }

    /// Get current optimization strategy
    pub fn get_current_strategy(&self) -> OptimizationStrategy {
        self.current_strategy
    }

    /// Get strategy effectiveness scores
    pub fn get_strategy_effectiveness(&self) -> HashMap<OptimizationStrategy, f64> {
        self.strategy_effectiveness.clone()
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationAction] {
        &self.optimization_history
    }

    /// Set optimization strategy manually
    pub fn set_strategy(&mut self, strategy: OptimizationStrategy) -> CoreResult<()> {
        self.current_strategy = strategy;
        Ok(())
    }

    /// Clear optimization history
    pub fn clear_history(&mut self) -> CoreResult<()> {
        self.optimization_history.clear();
        Ok(())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let total_actions = self.optimization_history.len();
        let successful_actions = self
            .optimization_history
            .iter()
            .filter(|action| action.success)
            .count();

        let success_rate = if total_actions > 0 {
            successful_actions as f64 / total_actions as f64
        } else {
            0.0
        };

        // Count actions by type
        let action_type_counts = self.count_action_types(
            &self
                .optimization_history
                .iter()
                .collect::<Vec<_>>()
        );

        OptimizationStats {
            total_actions,
            successful_actions,
            success_rate,
            current_strategy: self.current_strategy,
            strategy_effectiveness: self.strategy_effectiveness.clone(),
            action_type_counts,
        }
    }

    /// Check if optimization engine is healthy (good performance)
    pub fn is_healthy(&self) -> bool {
        if let Some(&effectiveness) = self.strategy_effectiveness.get(&self.current_strategy) {
            effectiveness > 0.5
        } else {
            true // Assume healthy if no data
        }
    }

    /// Get optimization summary for the last N actions
    pub fn get_recent_optimization_summary(&self, limit: usize) -> OptimizationSummary {
        let recent_actions: Vec<_> = self
            .optimization_history
            .iter()
            .rev()
            .take(limit)
            .collect();

        let total = recent_actions.len();
        let successful = recent_actions.iter().filter(|action| action.success).count();
        let action_counts = self.count_action_types(&recent_actions);

        OptimizationSummary {
            total_recent_actions: total,
            successful_recent_actions: successful,
            recent_success_rate: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
            most_common_action: action_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(action_type, _)| *action_type),
            recent_action_counts: action_counts,
        }
    }
}

/// Statistics about optimization engine performance
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub total_actions: usize,
    pub successful_actions: usize,
    pub success_rate: f64,
    pub current_strategy: OptimizationStrategy,
    pub strategy_effectiveness: HashMap<OptimizationStrategy, f64>,
    pub action_type_counts: HashMap<OptimizationActionType, usize>,
}

/// Summary of recent optimization activities
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimizationSummary {
    pub total_recent_actions: usize,
    pub successful_recent_actions: usize,
    pub recent_success_rate: f64,
    pub most_common_action: Option<OptimizationActionType>,
    pub recent_action_counts: HashMap<OptimizationActionType, usize>,
}