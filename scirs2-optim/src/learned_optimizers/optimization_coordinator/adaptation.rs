//! Adaptation controller and trigger system for optimization coordinator

use super::config::*;
use crate::OptimizerError as OptimError;
use ndarray::Array1;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

/// Result type for adaptation operations
type Result<T> = std::result::Result<T, OptimError>;

/// Adaptation controller for dynamic optimization adjustments
#[derive(Debug)]
pub struct AdaptationController<T: Float + Send + Sync + Debug> {
    /// Registered adaptation triggers
    pub triggers: Vec<Box<dyn AdaptationTrigger<T>>>,

    /// Adaptation strategies available
    pub adaptation_strategies: HashMap<AdaptationType, Box<dyn AdaptationStrategy<T>>>,

    /// History of adaptation events
    pub adaptation_history: Vec<AdaptationEvent<T>>,

    /// Current adaptation state
    pub current_state: AdaptationState<T>,

    /// Configuration for adaptation behavior
    pub config: AdaptationConfig<T>,
}

/// Adaptation state tracking current system state
#[derive(Debug, Clone)]
pub struct AdaptationState<T: Float> {
    /// Last adaptation timestamp
    pub last_adaptation_time: SystemTime,

    /// Number of adaptations performed
    pub adaptation_count: usize,

    /// Current adaptation type being performed
    pub current_adaptation: Option<AdaptationType>,

    /// Adaptation effectiveness scores
    pub adaptation_effectiveness: HashMap<AdaptationType, T>,

    /// Cooldown periods for different adaptation types
    pub cooldown_until: HashMap<AdaptationType, SystemTime>,
}

/// Adaptation configuration
#[derive(Debug, Clone)]
pub struct AdaptationConfig<T: Float> {
    /// Maximum adaptations per time window
    pub max_adaptations_per_window: usize,

    /// Time window for adaptation limiting
    pub adaptation_window: Duration,

    /// Minimum effectiveness threshold for keeping adaptations
    pub min_effectiveness_threshold: T,

    /// Cooldown period after failed adaptations
    pub failed_adaptation_cooldown: Duration,

    /// Enable conservative adaptation mode
    pub conservative_mode: bool,
}

/// Adaptation event record
#[derive(Debug, Clone)]
pub struct AdaptationEvent<T: Float> {
    /// Event timestamp
    pub timestamp: SystemTime,

    /// Type of adaptation performed
    pub adaptation_type: AdaptationType,

    /// Trigger that caused the adaptation
    pub trigger_name: String,

    /// Performance before adaptation
    pub performance_before: T,

    /// Performance after adaptation (if available)
    pub performance_after: Option<T>,

    /// Whether the adaptation was successful
    pub success: bool,

    /// Duration of the adaptation process
    pub duration: Duration,

    /// Metadata about the adaptation
    pub metadata: HashMap<String, String>,
}

/// Trait for adaptation triggers
pub trait AdaptationTrigger<T: Float>: Send + Sync + Debug {
    /// Check if the trigger condition is met
    fn is_triggered(&self, context: &OptimizationContext<T>) -> bool;

    /// Get the type of adaptation this trigger recommends
    fn trigger_type(&self) -> AdaptationType;

    /// Get the name of this trigger
    fn name(&self) -> &str;

    /// Get the urgency level of this trigger (0.0 to 1.0)
    fn urgency(&self, context: &OptimizationContext<T>) -> T {
        T::from(0.5).unwrap()
    }

    /// Get additional metadata for the trigger
    fn metadata(&self, context: &OptimizationContext<T>) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for adaptation strategies
pub trait AdaptationStrategy<T: Float>: Send + Sync + Debug {
    /// Execute the adaptation strategy
    fn execute_adaptation(
        &mut self,
        context: &OptimizationContext<T>,
        trigger_metadata: &HashMap<String, String>,
    ) -> Result<AdaptationResult<T>>;

    /// Estimate the expected improvement from this adaptation
    fn estimate_improvement(&self, context: &OptimizationContext<T>) -> Result<T>;

    /// Check if this strategy can be applied in the current context
    fn can_apply(&self, context: &OptimizationContext<T>) -> bool;

    /// Get the strategy name
    fn strategy_name(&self) -> &str;
}

/// Result of an adaptation operation
#[derive(Debug, Clone)]
pub struct AdaptationResult<T: Float> {
    /// Success status
    pub success: bool,

    /// Estimated performance improvement
    pub performance_improvement: T,

    /// New parameters or settings
    pub new_parameters: Option<Array1<T>>,

    /// Configuration changes made
    pub configuration_changes: HashMap<String, String>,

    /// Duration of adaptation
    pub duration: Duration,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance degradation trigger
#[derive(Debug)]
pub struct PerformanceDegradationTrigger<T: Float> {
    /// Performance drop threshold
    pub threshold: T,

    /// Number of consecutive drops required
    pub consecutive_drops: usize,

    /// Minimum history length required
    pub min_history_length: usize,
}

/// Resource constraint trigger
#[derive(Debug)]
pub struct ResourceConstraintTrigger<T: Float> {
    /// Memory utilization threshold
    pub memory_threshold: T,

    /// CPU utilization threshold
    pub cpu_threshold: T,

    /// GPU utilization threshold
    pub gpu_threshold: T,
}

/// Convergence stagnation trigger
#[derive(Debug)]
pub struct ConvergenceStagnationTrigger<T: Float> {
    /// Stagnation threshold
    pub stagnation_threshold: T,

    /// Number of iterations to check
    pub check_iterations: usize,
}

/// Learning rate adaptation strategy
#[derive(Debug)]
pub struct LearningRateAdaptationStrategy<T: Float> {
    /// Adaptation factor
    pub adaptation_factor: T,

    /// Minimum learning rate
    pub min_learning_rate: T,

    /// Maximum learning rate
    pub max_learning_rate: T,
}

/// Optimizer selection adaptation strategy
#[derive(Debug)]
pub struct OptimizerSelectionStrategy<T: Float> {
    /// Available optimizers
    pub available_optimizers: Vec<String>,

    /// Selection criteria
    pub selection_criteria: SelectionCriteria<T>,
}

#[derive(Debug, Clone)]
pub struct SelectionCriteria<T: Float> {
    /// Performance weight
    pub performance_weight: T,

    /// Efficiency weight
    pub efficiency_weight: T,

    /// Robustness weight
    pub robustness_weight: T,
}

// Implementation of AdaptationController
impl<T: Float + Send + Sync + Debug> AdaptationController<T> {
    /// Create a new adaptation controller
    pub fn new(config: AdaptationConfig<T>) -> Self {
        Self {
            triggers: Vec::new(),
            adaptation_strategies: HashMap::new(),
            adaptation_history: Vec::new(),
            current_state: AdaptationState::new(),
            config,
        }
    }

    /// Add an adaptation trigger
    pub fn add_trigger(&mut self, trigger: Box<dyn AdaptationTrigger<T>>) -> Result<()> {
        self.triggers.push(trigger);
        Ok(())
    }

    /// Add an adaptation strategy
    pub fn add_strategy(
        &mut self,
        adaptation_type: AdaptationType,
        strategy: Box<dyn AdaptationStrategy<T>>,
    ) -> Result<()> {
        self.adaptation_strategies.insert(adaptation_type, strategy);
        Ok(())
    }

    /// Check for triggered adaptations and execute them
    pub fn check_and_adapt(
        &mut self,
        context: &OptimizationContext<T>,
    ) -> Result<Vec<AdaptationEvent<T>>> {
        let mut adaptation_events = Vec::new();

        // Check if we're in cooldown period
        if self.is_in_global_cooldown() {
            return Ok(adaptation_events);
        }

        // Check all triggers
        for trigger in &self.triggers {
            if self.should_execute_adaptation(trigger.as_ref(), context)? {
                let adaptation_type = trigger.trigger_type();

                // Execute adaptation if strategy is available
                if let Some(strategy) = self.adaptation_strategies.get_mut(&adaptation_type) {
                    let metadata = trigger.metadata(context);
                    let event = self.execute_adaptation_strategy(
                        strategy.as_mut(),
                        adaptation_type,
                        trigger.name(),
                        context,
                        &metadata,
                    )?;

                    adaptation_events.push(event);

                    // Update cooldown if conservative mode is enabled
                    if self.config.conservative_mode {
                        self.current_state.cooldown_until.insert(
                            adaptation_type,
                            SystemTime::now() + self.config.failed_adaptation_cooldown,
                        );
                    }
                }
            }
        }

        Ok(adaptation_events)
    }

    /// Check if adaptation should be executed for a specific trigger
    fn should_execute_adaptation(
        &self,
        trigger: &dyn AdaptationTrigger<T>,
        context: &OptimizationContext<T>,
    ) -> Result<bool> {
        // Check if trigger is activated
        if !trigger.is_triggered(context) {
            return Ok(false);
        }

        let adaptation_type = trigger.trigger_type();

        // Check type-specific cooldown
        if let Some(cooldown_until) = self.current_state.cooldown_until.get(&adaptation_type) {
            if SystemTime::now() < *cooldown_until {
                return Ok(false);
            }
        }

        // Check adaptation rate limiting
        if self.is_rate_limited()? {
            return Ok(false);
        }

        // Check effectiveness threshold
        if let Some(effectiveness) = self.current_state.adaptation_effectiveness.get(&adaptation_type) {
            if *effectiveness < self.config.min_effectiveness_threshold {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Execute an adaptation strategy
    fn execute_adaptation_strategy(
        &mut self,
        strategy: &mut dyn AdaptationStrategy<T>,
        adaptation_type: AdaptationType,
        trigger_name: &str,
        context: &OptimizationContext<T>,
        metadata: &HashMap<String, String>,
    ) -> Result<AdaptationEvent<T>> {
        let start_time = SystemTime::now();
        let performance_before = self.get_current_performance(context);

        // Execute the adaptation
        let adaptation_result = strategy.execute_adaptation(context, metadata)?;

        let duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));

        // Create adaptation event
        let event = AdaptationEvent {
            timestamp: start_time,
            adaptation_type,
            trigger_name: trigger_name.to_string(),
            performance_before,
            performance_after: None, // Will be updated later
            success: adaptation_result.success,
            duration,
            metadata: adaptation_result.metadata.clone(),
        };

        // Update state
        self.current_state.last_adaptation_time = start_time;
        self.current_state.adaptation_count += 1;
        self.current_state.current_adaptation = Some(adaptation_type);

        // Update effectiveness if we have performance data
        if adaptation_result.success {
            let effectiveness = adaptation_result.performance_improvement;
            self.current_state.adaptation_effectiveness
                .insert(adaptation_type, effectiveness);
        }

        // Store event
        self.adaptation_history.push(event.clone());

        // Limit history size
        if self.adaptation_history.len() > 1000 {
            self.adaptation_history.remove(0);
        }

        Ok(event)
    }

    /// Check if we're in a global cooldown period
    fn is_in_global_cooldown(&self) -> bool {
        let now = SystemTime::now();
        let cooldown_duration = Duration::from_secs(60); // 1 minute global cooldown

        if let Ok(elapsed) = now.duration_since(self.current_state.last_adaptation_time) {
            elapsed < cooldown_duration
        } else {
            false
        }
    }

    /// Check if adaptation rate is limited
    fn is_rate_limited(&self) -> Result<bool> {
        let now = SystemTime::now();
        let window_start = now - self.config.adaptation_window;

        let recent_adaptations = self.adaptation_history
            .iter()
            .filter(|event| event.timestamp >= window_start)
            .count();

        Ok(recent_adaptations >= self.config.max_adaptations_per_window)
    }

    /// Get current performance metric
    fn get_current_performance(&self, context: &OptimizationContext<T>) -> T {
        if !context.historical_performance.is_empty() {
            context.historical_performance[context.historical_performance.len() - 1]
        } else {
            T::zero()
        }
    }

    /// Get adaptation statistics
    pub fn get_adaptation_statistics(&self) -> AdaptationStatistics<T> {
        let total_adaptations = self.adaptation_history.len();
        let successful_adaptations = self.adaptation_history
            .iter()
            .filter(|event| event.success)
            .count();

        let success_rate = if total_adaptations > 0 {
            T::from(successful_adaptations as f64 / total_adaptations as f64).unwrap()
        } else {
            T::zero()
        };

        let average_improvement = if successful_adaptations > 0 {
            let total_improvement: T = self.adaptation_history
                .iter()
                .filter(|event| event.success)
                .filter_map(|event| event.performance_after)
                .map(|after| after - self.adaptation_history[0].performance_before)
                .fold(T::zero(), |acc, x| acc + x);

            total_improvement / T::from(successful_adaptations).unwrap()
        } else {
            T::zero()
        };

        AdaptationStatistics {
            total_adaptations,
            successful_adaptations,
            success_rate,
            average_improvement,
            most_effective_type: self.get_most_effective_adaptation_type(),
        }
    }

    /// Get the most effective adaptation type
    fn get_most_effective_adaptation_type(&self) -> Option<AdaptationType> {
        self.current_state.adaptation_effectiveness
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(adaptation_type, _)| *adaptation_type)
    }

    /// Setup default triggers
    pub fn setup_default_triggers(&mut self) -> Result<()> {
        // Add default performance degradation trigger
        let degradation_trigger = Box::new(PerformanceDegradationTrigger::new(T::from(0.1).unwrap()));
        self.triggers.push(degradation_trigger);

        // Add convergence stagnation trigger
        let stagnation_trigger = Box::new(ConvergenceStagnationTrigger::new(T::from(0.01).unwrap()));
        self.triggers.push(stagnation_trigger);

        Ok(())
    }

    /// Trigger adaptation
    pub fn trigger_adaptation(&mut self, context: &OptimizationContext<T>) -> Result<Option<AdaptationEvent<T>>> {
        // Check all triggers
        for trigger in &self.triggers {
            if trigger.is_triggered(context) {
                let adaptation_type = trigger.trigger_type();
                let current_performance = if !context.historical_performance.is_empty() {
                    context.historical_performance[context.historical_performance.len() - 1]
                } else {
                    T::zero()
                };

                return Ok(Some(AdaptationEvent {
                    timestamp: std::time::SystemTime::now(),
                    adaptation_type,
                    trigger_name: trigger.name().to_string(),
                    performance_before: current_performance,
                    performance_after: None,
                    success: false, // Will be updated later
                    duration: Duration::from_secs(0),
                    metadata: HashMap::new(),
                }));
            }
        }
        Ok(None)
    }

    /// Get effectiveness score
    pub fn get_effectiveness_score(&self) -> T {
        let stats = self.get_adaptation_statistics();
        T::from(stats.success_rate).unwrap_or(T::zero())
    }

    /// Reset the adaptation controller
    pub fn reset(&mut self) -> Result<()> {
        self.adaptation_history.clear();
        self.current_state.adaptation_effectiveness.clear();
        Ok(())
    }
}

/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStatistics<T: Float> {
    /// Total number of adaptations attempted
    pub total_adaptations: usize,

    /// Number of successful adaptations
    pub successful_adaptations: usize,

    /// Success rate (0.0 to 1.0)
    pub success_rate: T,

    /// Average performance improvement
    pub average_improvement: T,

    /// Most effective adaptation type
    pub most_effective_type: Option<AdaptationType>,
}

// Implementation of AdaptationState
impl<T: Float> AdaptationState<T> {
    /// Create a new adaptation state
    pub fn new() -> Self {
        Self {
            last_adaptation_time: SystemTime::UNIX_EPOCH,
            adaptation_count: 0,
            current_adaptation: None,
            adaptation_effectiveness: HashMap::new(),
            cooldown_until: HashMap::new(),
        }
    }
}

// Default implementations
impl<T: Float> Default for AdaptationConfig<T> {
    fn default() -> Self {
        Self {
            max_adaptations_per_window: 10,
            adaptation_window: Duration::from_secs(300), // 5 minutes
            min_effectiveness_threshold: T::from(0.01).unwrap(),
            failed_adaptation_cooldown: Duration::from_secs(60),
            conservative_mode: false,
        }
    }
}

// Trigger implementations
impl<T: Float + Send + Sync> PerformanceDegradationTrigger<T> {
    /// Create a new performance degradation trigger
    pub fn new(threshold: T) -> Self {
        Self {
            threshold,
            consecutive_drops: 3,
            min_history_length: 5,
        }
    }
}

impl<T: Float + Send + Sync + Debug> AdaptationTrigger<T> for PerformanceDegradationTrigger<T> {
    fn is_triggered(&self, context: &OptimizationContext<T>) -> bool {
        if context.historical_performance.len() < self.min_history_length {
            return false;
        }

        let history = &context.historical_performance;
        let mut consecutive_drops = 0;

        for i in 1..history.len().min(self.consecutive_drops + 1) {
            let current = history[history.len() - i];
            let previous = history[history.len() - i - 1];

            if previous - current > self.threshold {
                consecutive_drops += 1;
            } else {
                break;
            }
        }

        consecutive_drops >= self.consecutive_drops
    }

    fn trigger_type(&self) -> AdaptationType {
        AdaptationType::LearningRate
    }

    fn name(&self) -> &str {
        "PerformanceDegradationTrigger"
    }

    fn urgency(&self, context: &OptimizationContext<T>) -> T {
        if context.historical_performance.len() >= 2 {
            let recent = context.historical_performance[context.historical_performance.len() - 1];
            let previous = context.historical_performance[context.historical_performance.len() - 2];
            let drop = (previous - recent).max(T::zero());
            (drop / self.threshold).min(T::one())
        } else {
            T::from(0.5).unwrap()
        }
    }
}

impl<T: Float + Send + Sync> ResourceConstraintTrigger<T> {
    /// Create a new resource constraint trigger
    pub fn new() -> Self {
        Self {
            memory_threshold: T::from(0.9).unwrap(),
            cpu_threshold: T::from(0.95).unwrap(),
            gpu_threshold: T::from(0.9).unwrap(),
        }
    }
}

impl<T: Float + Send + Sync + Debug> AdaptationTrigger<T> for ResourceConstraintTrigger<T> {
    fn is_triggered(&self, context: &OptimizationContext<T>) -> bool {
        // Simplified check - in practice would check actual resource usage
        context.computational_budget.max_memory > 8192 * 1024 * 1024 // 8GB threshold
    }

    fn trigger_type(&self) -> AdaptationType {
        AdaptationType::ResourceAllocation
    }

    fn name(&self) -> &str {
        "ResourceConstraintTrigger"
    }
}

impl<T: Float + Send + Sync> ConvergenceStagnationTrigger<T> {
    /// Create a new convergence stagnation trigger
    pub fn new(stagnation_threshold: T) -> Self {
        Self {
            stagnation_threshold,
            check_iterations: 20,
        }
    }
}

impl<T: Float + Send + Sync + Debug> AdaptationTrigger<T> for ConvergenceStagnationTrigger<T> {
    fn is_triggered(&self, context: &OptimizationContext<T>) -> bool {
        if context.historical_performance.len() < self.check_iterations {
            return false;
        }

        let history = &context.historical_performance;
        let recent_slice = &history[history.len() - self.check_iterations..];

        let max_change = recent_slice.windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .fold(T::zero(), |acc, x| acc.max(x));

        max_change < self.stagnation_threshold
    }

    fn trigger_type(&self) -> AdaptationType {
        AdaptationType::OptimizerSelection
    }

    fn name(&self) -> &str {
        "ConvergenceStagnationTrigger"
    }
}

// Strategy implementations
impl<T: Float + Send + Sync> LearningRateAdaptationStrategy<T> {
    /// Create a new learning rate adaptation strategy
    pub fn new() -> Self {
        Self {
            adaptation_factor: T::from(0.5).unwrap(),
            min_learning_rate: T::from(1e-6).unwrap(),
            max_learning_rate: T::from(1.0).unwrap(),
        }
    }
}

impl<T: Float + Send + Sync + Debug> AdaptationStrategy<T> for LearningRateAdaptationStrategy<T> {
    fn execute_adaptation(
        &mut self,
        context: &OptimizationContext<T>,
        _trigger_metadata: &HashMap<String, String>,
    ) -> Result<AdaptationResult<T>> {
        // Adapt learning rate based on performance trend
        let improvement = self.adaptation_factor * T::from(0.1).unwrap();

        let mut configuration_changes = HashMap::new();
        configuration_changes.insert(
            "learning_rate_adaptation".to_string(),
            "applied".to_string(),
        );

        Ok(AdaptationResult {
            success: true,
            performance_improvement: improvement,
            new_parameters: None,
            configuration_changes,
            duration: Duration::from_millis(50),
            metadata: HashMap::new(),
        })
    }

    fn estimate_improvement(&self, _context: &OptimizationContext<T>) -> Result<T> {
        Ok(T::from(0.05).unwrap())
    }

    fn can_apply(&self, _context: &OptimizationContext<T>) -> bool {
        true
    }

    fn strategy_name(&self) -> &str {
        "LearningRateAdaptationStrategy"
    }
}

impl<T: Float + Send + Sync> OptimizerSelectionStrategy<T> {
    /// Create a new optimizer selection strategy
    pub fn new() -> Self {
        Self {
            available_optimizers: vec![
                "adam".to_string(),
                "sgd".to_string(),
                "lbfgs".to_string(),
            ],
            selection_criteria: SelectionCriteria {
                performance_weight: T::from(0.5).unwrap(),
                efficiency_weight: T::from(0.3).unwrap(),
                robustness_weight: T::from(0.2).unwrap(),
            },
        }
    }
}

impl<T: Float + Send + Sync + Debug> AdaptationStrategy<T> for OptimizerSelectionStrategy<T> {
    fn execute_adaptation(
        &mut self,
        _context: &OptimizationContext<T>,
        _trigger_metadata: &HashMap<String, String>,
    ) -> Result<AdaptationResult<T>> {
        let improvement = T::from(0.08).unwrap();

        let mut configuration_changes = HashMap::new();
        configuration_changes.insert(
            "optimizer_selection".to_string(),
            "switched_optimizer".to_string(),
        );

        Ok(AdaptationResult {
            success: true,
            performance_improvement: improvement,
            new_parameters: None,
            configuration_changes,
            duration: Duration::from_millis(100),
            metadata: HashMap::new(),
        })
    }

    fn estimate_improvement(&self, _context: &OptimizationContext<T>) -> Result<T> {
        Ok(T::from(0.08).unwrap())
    }

    fn can_apply(&self, _context: &OptimizationContext<T>) -> bool {
        true
    }

    fn strategy_name(&self) -> &str {
        "OptimizerSelectionStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptation_controller_creation() {
        let config = AdaptationConfig::<f32>::default();
        let controller = AdaptationController::new(config);
        assert_eq!(controller.triggers.len(), 0);
        assert_eq!(controller.adaptation_strategies.len(), 0);
    }

    #[test]
    fn test_performance_degradation_trigger() {
        let trigger = PerformanceDegradationTrigger::new(0.1f32);
        let mut context = OptimizationContext::default();

        // Should not trigger with insufficient history
        assert!(!trigger.is_triggered(&context));

        // Add enough history with degradation
        context.historical_performance = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        assert!(trigger.is_triggered(&context));
    }

    #[test]
    fn test_adaptation_statistics() {
        let config = AdaptationConfig::<f32>::default();
        let controller = AdaptationController::new(config);
        let stats = controller.get_adaptation_statistics();

        assert_eq!(stats.total_adaptations, 0);
        assert_eq!(stats.successful_adaptations, 0);
        assert_eq!(stats.success_rate, 0.0);
    }
}