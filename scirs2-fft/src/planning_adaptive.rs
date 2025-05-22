//! Adaptive FFT Planning
//!
//! This module extends the basic planning system with runtime adaptivity,
//! allowing the planner to adjust its strategy based on observed performance
//! and system conditions.

use crate::error::FFTResult;
use crate::planning::{FftPlan, PlannerBackend, PlanningStrategy};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration options for adaptive planning
#[derive(Debug, Clone)]
pub struct AdaptivePlanningConfig {
    /// Whether adaptivity is enabled
    pub enabled: bool,

    /// Minimum number of FFTs before adapting strategy
    pub min_samples: usize,

    /// Time between strategy evaluations
    pub evaluation_interval: Duration,

    /// Maximum number of strategies to try
    pub max_strategy_switches: usize,

    /// Whether to enable backend switching
    pub enable_backend_switching: bool,

    /// Threshold for considering a strategy better (ratio)
    pub improvement_threshold: f64,
}

impl Default for AdaptivePlanningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_samples: 5,
            evaluation_interval: Duration::from_secs(10),
            max_strategy_switches: 3,
            enable_backend_switching: true,
            improvement_threshold: 1.1, // 10% improvement needed
        }
    }
}

/// Performance metrics for a specific strategy
#[derive(Debug, Clone)]
struct StrategyMetrics {
    /// Total execution time
    total_time: Duration,

    /// Number of executions
    count: usize,

    /// Average execution time
    avg_time: Duration,

    /// Last time this strategy was evaluated
    /// Used for future time-based strategy optimization
    #[allow(dead_code)]
    last_evaluated: Instant,
}

impl StrategyMetrics {
    /// Create new empty metrics
    fn new() -> Self {
        Self {
            total_time: Duration::from_nanos(0),
            count: 0,
            avg_time: Duration::from_nanos(0),
            last_evaluated: Instant::now(),
        }
    }

    /// Record a new execution time
    fn record(&mut self, time: Duration) {
        self.total_time += time;
        self.count += 1;
        self.avg_time =
            Duration::from_nanos((self.total_time.as_nanos() / self.count as u128) as u64);
    }
}

/// Adaptive planner that switches strategies based on runtime performance
pub struct AdaptivePlanner {
    /// The FFT size this planner is optimized for
    size: Vec<usize>,

    /// Direction of the transform (forward or inverse)
    forward: bool,

    /// Current strategy being used
    current_strategy: PlanningStrategy,

    /// Current backend being used
    current_backend: PlannerBackend,

    /// Performance metrics for each strategy
    metrics: HashMap<PlanningStrategy, StrategyMetrics>,

    /// Last time the strategy was switched
    last_strategy_switch: Instant,

    /// Number of strategy switches performed
    strategy_switches: usize,

    /// Configuration options
    config: AdaptivePlanningConfig,

    /// Currently active plan
    current_plan: Option<Arc<FftPlan>>,
}

use std::collections::HashMap;

impl AdaptivePlanner {
    /// Create a new adaptive planner
    pub fn new(size: &[usize], forward: bool, config: Option<AdaptivePlanningConfig>) -> Self {
        let config = config.unwrap_or_default();
        let mut metrics = HashMap::new();

        // Initialize metrics for all strategies
        metrics.insert(PlanningStrategy::AlwaysNew, StrategyMetrics::new());
        metrics.insert(PlanningStrategy::CacheFirst, StrategyMetrics::new());
        metrics.insert(PlanningStrategy::SerializedFirst, StrategyMetrics::new());
        metrics.insert(PlanningStrategy::AutoTuned, StrategyMetrics::new());

        Self {
            size: size.to_vec(),
            forward,
            current_strategy: PlanningStrategy::CacheFirst, // Start with a reasonable default
            current_backend: PlannerBackend::default(),
            metrics,
            last_strategy_switch: Instant::now(),
            strategy_switches: 0,
            config,
            current_plan: None,
        }
    }

    /// Get the current strategy
    pub fn current_strategy(&self) -> PlanningStrategy {
        self.current_strategy
    }

    /// Get the current backend
    pub fn current_backend(&self) -> PlannerBackend {
        self.current_backend.clone()
    }

    /// Get the current plan
    pub fn get_plan(&mut self) -> FFTResult<Arc<FftPlan>> {
        // If we have a current plan, return it
        if let Some(plan) = &self.current_plan {
            return Ok(plan.clone());
        }

        // Otherwise create a new plan using the regular planner
        use crate::planning::{AdvancedFftPlanner, PlanningConfig};

        let config = PlanningConfig {
            strategy: self.current_strategy,
            ..Default::default()
        };

        let mut planner = AdvancedFftPlanner::with_config(config);
        let plan = planner.plan_fft(&self.size, self.forward, self.current_backend.clone())?;

        self.current_plan = Some(plan.clone());
        Ok(plan)
    }

    /// Record execution time and potentially adapt strategy
    pub fn record_execution(&mut self, execution_time: Duration) -> FFTResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Record metrics for current strategy
        if let Some(metrics) = self.metrics.get_mut(&self.current_strategy) {
            metrics.record(execution_time);
        }

        // Check if we should evaluate strategies
        let should_evaluate =
            // We have enough samples
            self.metrics[&self.current_strategy].count >= self.config.min_samples &&
            // It's been long enough since last evaluation
            self.last_strategy_switch.elapsed() >= self.config.evaluation_interval &&
            // We haven't switched too many times
            self.strategy_switches < self.config.max_strategy_switches;

        if should_evaluate {
            self.evaluate_strategies()?;
        }

        Ok(())
    }

    /// Evaluate all strategies and potentially switch
    fn evaluate_strategies(&mut self) -> FFTResult<()> {
        // Find the strategy with the best performance
        let mut best_strategy = self.current_strategy;
        let mut best_time = self.metrics[&self.current_strategy].avg_time;

        for (strategy, metrics) in &self.metrics {
            // Skip strategies with no data
            if metrics.count == 0 {
                continue;
            }

            // Improvement must exceed threshold
            let improvement_ratio =
                best_time.as_nanos() as f64 / metrics.avg_time.as_nanos() as f64;
            if improvement_ratio > self.config.improvement_threshold {
                best_strategy = *strategy;
                best_time = metrics.avg_time;
            }
        }

        // If we found a better strategy, switch to it
        if best_strategy != self.current_strategy {
            self.current_strategy = best_strategy;
            self.last_strategy_switch = Instant::now();
            self.strategy_switches += 1;

            // Clear the current plan to force creation with new strategy
            self.current_plan = None;
        }

        // Backend switching would go here if enabled
        if self.config.enable_backend_switching {
            // This would require additional metrics tracking
            // Not implemented in this simplified version
        }

        Ok(())
    }

    /// Get performance statistics for all strategies
    pub fn get_statistics(&self) -> HashMap<PlanningStrategy, (Duration, usize)> {
        let mut stats = HashMap::new();

        for (strategy, metrics) in &self.metrics {
            stats.insert(*strategy, (metrics.avg_time, metrics.count));
        }

        stats
    }
}

/// Executor that uses adaptive planning
pub struct AdaptiveExecutor {
    /// Adaptive planner instance
    planner: Arc<Mutex<AdaptivePlanner>>,
}

impl AdaptiveExecutor {
    /// Create a new adaptive executor
    pub fn new(size: &[usize], forward: bool, config: Option<AdaptivePlanningConfig>) -> Self {
        let planner = AdaptivePlanner::new(size, forward, config);

        Self {
            planner: Arc::new(Mutex::new(planner)),
        }
    }

    /// Execute an FFT with adaptive planning
    pub fn execute(
        &self,
        input: &[num_complex::Complex64],
        output: &mut [num_complex::Complex64],
    ) -> FFTResult<()> {
        let start = Instant::now();

        // Get the current plan
        let plan = {
            let mut planner = self.planner.lock().unwrap();
            planner.get_plan()?
        };

        // Create an executor for the plan
        let executor = crate::planning::FftPlanExecutor::new(plan);

        // Execute the plan
        executor.execute(input, output)?;

        // Record execution time
        let execution_time = start.elapsed();

        {
            let mut planner = self.planner.lock().unwrap();
            planner.record_execution(execution_time)?;
        }

        Ok(())
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> PlanningStrategy {
        let planner = self.planner.lock().unwrap();
        planner.current_strategy()
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> HashMap<PlanningStrategy, (Duration, usize)> {
        let planner = self.planner.lock().unwrap();
        planner.get_statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_adaptive_planner_basics() {
        let mut planner = AdaptivePlanner::new(&[16], true, None);

        // Should start with CacheFirst strategy
        assert_eq!(planner.current_strategy(), PlanningStrategy::CacheFirst);

        // Record some executions
        for _ in 0..10 {
            planner
                .record_execution(Duration::from_micros(100))
                .unwrap();
        }

        // Check that metrics were recorded
        let stats = planner.get_statistics();
        assert_eq!(stats[&PlanningStrategy::CacheFirst].1, 10);
    }

    #[test]
    fn test_adaptive_executor() {
        let executor = AdaptiveExecutor::new(&[16], true, None);

        // Create test data
        let input = vec![Complex64::new(1.0, 0.0); 16];
        let mut output = vec![Complex64::default(); 16];

        // Execute several times
        for _ in 0..5 {
            executor.execute(&input, &mut output).unwrap();
        }

        // Check statistics
        let stats = executor.get_statistics();
        assert!(stats[&executor.current_strategy()].1 >= 5);
    }
}
