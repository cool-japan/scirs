//! Self-tuning parameter selection for optimization algorithms
//!
//! This module provides intelligent parameter selection and adaptation mechanisms
//! that automatically tune optimization algorithm parameters based on problem
//! characteristics and optimization progress.

use crate::error::{ScirsError, ScirsResult};
// Unused imports
// use ndarray::{Array1, Array2, ArrayView1};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Configuration for self-tuning optimization
#[derive(Debug, Clone)]
pub struct SelfTuningConfig {
    /// Strategy for parameter adaptation
    pub adaptation_strategy: AdaptationStrategy,
    /// Frequency of parameter updates (in iterations)
    pub update_frequency: usize,
    /// Learning rate for parameter adaptation
    pub learning_rate: f64,
    /// Memory window for performance tracking
    pub memory_window: usize,
    /// Whether to use Bayesian optimization for parameter tuning
    pub use_bayesian_tuning: bool,
    /// Exploration vs exploitation trade-off
    pub exploration_factor: f64,
}

impl Default for SelfTuningConfig {
    fn default() -> Self {
        Self {
            adaptation_strategy: AdaptationStrategy::PerformanceBased,
            update_frequency: 50,
            learning_rate: 0.1,
            memory_window: 100,
            use_bayesian_tuning: true,
            exploration_factor: 0.1,
        }
    }
}

/// Parameter adaptation strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptationStrategy {
    /// Adapt based on optimization performance
    PerformanceBased,
    /// Adapt based on convergence characteristics
    ConvergenceBased,
    /// Use reinforcement learning for adaptation
    ReinforcementLearning,
    /// Use Bayesian optimization for parameter search
    BayesianOptimization,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Self-tuning optimizer that adapts parameters automatically
pub struct SelfTuningOptimizer {
    config: SelfTuningConfig,
    parameter_manager: ParameterManager,
    performance_tracker: PerformanceTracker,
    adaptation_engine: AdaptationEngine,
    tuning_history: TuningHistory,
}

impl SelfTuningOptimizer {
    /// Create a new self-tuning optimizer
    pub fn new(config: SelfTuningConfig) -> Self {
        Self {
            parameter_manager: ParameterManager::new(),
            performance_tracker: PerformanceTracker::new(config.memory_window),
            adaptation_engine: AdaptationEngine::new(config.adaptation_strategy),
            tuning_history: TuningHistory::new(),
            config,
        }
    }

    /// Register tunable parameters for an optimization algorithm
    pub fn register_parameter<T>(
        &mut self,
        name: &str,
        param: TunableParameter<T>,
    ) -> ScirsResult<()>
    where
        T: Clone + PartialOrd + std::fmt::Debug + 'static + Send + Sync,
    {
        self.parameter_manager.register(name, param)
    }

    /// Get current parameter values
    pub fn get_parameters(&self) -> &HashMap<String, ParameterValue> {
        self.parameter_manager.current_values()
    }

    /// Update parameters based on optimization progress
    pub fn update_parameters(
        &mut self,
        iteration: usize,
        function_value: f64,
        gradient_norm: Option<f64>,
        improvement: f64,
    ) -> ScirsResult<bool> {
        // Record performance
        self.performance_tracker.record_performance(
            iteration,
            function_value,
            gradient_norm,
            improvement,
        );

        // Check if it's time to update parameters
        if iteration % self.config.update_frequency != 0 {
            return Ok(false);
        }

        // Analyze performance and adapt parameters
        let performance_metrics = self.performance_tracker.compute_metrics();
        let adaptation_result = self.adaptation_engine.adapt_parameters(
            &mut self.parameter_manager,
            &performance_metrics,
            &self.config,
        )?;

        // Record tuning action
        if adaptation_result.parameters_changed {
            self.tuning_history.record_adaptation(
                iteration,
                adaptation_result.clone(),
                performance_metrics.clone(),
            );
        }

        Ok(adaptation_result.parameters_changed)
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceTracker {
        &self.performance_tracker
    }

    /// Get tuning history
    pub fn tuning_history(&self) -> &TuningHistory {
        &self.tuning_history
    }

    /// Generate tuning report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("Self-Tuning Optimization Report\n");
        report.push_str("===============================\n\n");

        // Parameter summary
        report.push_str("Current Parameters:\n");
        for (name, value) in self.parameter_manager.current_values() {
            report.push_str(&format!("  {}: {:?}\n", name, value));
        }
        report.push('\n');

        // Performance summary
        let metrics = self.performance_tracker.compute_metrics();
        report.push_str("Performance Metrics:\n");
        report.push_str(&format!(
            "  Convergence Rate: {:.6}\n",
            metrics.convergence_rate
        ));
        report.push_str(&format!(
            "  Average Improvement: {:.6e}\n",
            metrics.average_improvement
        ));
        report.push_str(&format!(
            "  Stability Score: {:.3}\n",
            metrics.stability_score
        ));
        report.push('\n');

        // Adaptation history
        report.push_str(&format!(
            "Total Adaptations: {}\n",
            self.tuning_history.adaptations.len()
        ));
        if let Some(last_adaptation) = self.tuning_history.adaptations.last() {
            report.push_str(&format!(
                "Last Adaptation at Iteration: {}\n",
                last_adaptation.iteration
            ));
        }

        report
    }
}

/// Manages tunable parameters for optimization algorithms
struct ParameterManager {
    parameters: HashMap<String, Box<dyn TunableParam>>,
    current_values: HashMap<String, ParameterValue>,
    parameter_bounds: HashMap<String, (ParameterValue, ParameterValue)>,
}

impl ParameterManager {
    fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            current_values: HashMap::new(),
            parameter_bounds: HashMap::new(),
        }
    }

    fn register<T>(&mut self, name: &str, param: TunableParameter<T>) -> ScirsResult<()>
    where
        T: Clone + PartialOrd + std::fmt::Debug + 'static + Send + Sync,
    {
        let value = ParameterValue::from_typed(&param.current_value);
        let min_bound = ParameterValue::from_typed(&param.min_value);
        let max_bound = ParameterValue::from_typed(&param.max_value);

        self.current_values.insert(name.to_string(), value);
        self.parameter_bounds
            .insert(name.to_string(), (min_bound, max_bound));
        self.parameters.insert(name.to_string(), Box::new(param));

        Ok(())
    }

    fn update_parameter(&mut self, name: &str, newvalue: ParameterValue) -> ScirsResult<()> {
        if let Some((min_bound, max_bound)) = self.parameter_bounds.get(name) {
            if newvalue < *min_bound || newvalue > *max_bound {
                return Err(ScirsError::InvalidInput(
                    scirs2_core::error::ErrorContext::new(format!(
                        "Parameter {} _value {:?} is out of bounds [{:?}, {:?}]",
                        name, newvalue, min_bound, max_bound
                    )),
                ));
            }
        }

        self.current_values
            .insert(name.to_string(), newvalue.clone());

        if let Some(param) = self.parameters.get_mut(name) {
            param.set_value(newvalue)?;
        }

        Ok(())
    }

    fn current_values(&self) -> &HashMap<String, ParameterValue> {
        &self.current_values
    }

    fn get_bounds(&self, name: &str) -> Option<&(ParameterValue, ParameterValue)> {
        self.parameter_bounds.get(name)
    }
}

/// Trait for tunable parameters
trait TunableParam {
    fn set_value(&mut self, value: ParameterValue) -> ScirsResult<()>;
    fn get_value(&self) -> ParameterValue;
    fn get_bounds(&self) -> (ParameterValue, ParameterValue);
}

/// Generic tunable parameter
#[derive(Debug, Clone)]
pub struct TunableParameter<T> {
    pub current_value: T,
    pub min_value: T,
    pub max_value: T,
    pub adaptation_rate: f64,
}

impl<T> TunableParameter<T>
where
    T: Clone + PartialOrd + std::fmt::Debug + 'static + Send + Sync,
{
    /// Create a new tunable parameter
    pub fn new(current: T, min: T, max: T) -> Self {
        Self {
            current_value: current,
            min_value: min,
            max_value: max,
            adaptation_rate: 0.1,
        }
    }

    /// Set adaptation rate for this parameter
    pub fn with_adaptation_rate(mut self, rate: f64) -> Self {
        self.adaptation_rate = rate;
        self
    }
}

impl<T> TunableParam for TunableParameter<T>
where
    T: Clone + PartialOrd + std::fmt::Debug + 'static + Send + Sync,
{
    fn set_value(&mut self, value: ParameterValue) -> ScirsResult<()> {
        use std::any::{Any, TypeId};

        let type_id = TypeId::of::<T>();

        // Convert ParameterValue to the correct type T
        if type_id == TypeId::of::<f64>() {
            if let Some(f_val) = value.as_f64() {
                if let Some(self_any) =
                    (&mut self.current_value as &mut dyn Any).downcast_mut::<f64>()
                {
                    *self_any = f_val;
                    return Ok(());
                }
            }
        } else if type_id == TypeId::of::<f32>() {
            if let Some(f_val) = value.as_f64() {
                if let Some(self_any) =
                    (&mut self.current_value as &mut dyn Any).downcast_mut::<f32>()
                {
                    *self_any = f_val as f32;
                    return Ok(());
                }
            }
        } else if type_id == TypeId::of::<i64>() {
            if let Some(i_val) = value.as_i64() {
                if let Some(self_any) =
                    (&mut self.current_value as &mut dyn Any).downcast_mut::<i64>()
                {
                    *self_any = i_val;
                    return Ok(());
                }
            }
        } else if type_id == TypeId::of::<i32>() {
            if let Some(i_val) = value.as_i64() {
                if let Some(self_any) =
                    (&mut self.current_value as &mut dyn Any).downcast_mut::<i32>()
                {
                    *self_any = i_val as i32;
                    return Ok(());
                }
            }
        } else if type_id == TypeId::of::<usize>() {
            if let Some(i_val) = value.as_i64() {
                if i_val >= 0 {
                    if let Some(self_any) =
                        (&mut self.current_value as &mut dyn Any).downcast_mut::<usize>()
                    {
                        *self_any = i_val as usize;
                        return Ok(());
                    }
                }
            }
        } else if type_id == TypeId::of::<bool>() {
            if let Some(b_val) = value.as_bool() {
                if let Some(self_any) =
                    (&mut self.current_value as &mut dyn Any).downcast_mut::<bool>()
                {
                    *self_any = b_val;
                    return Ok(());
                }
            }
        } else if type_id == TypeId::of::<String>() {
            if let ParameterValue::String(ref s_val) = value {
                if let Some(self_any) =
                    (&mut self.current_value as &mut dyn Any).downcast_mut::<String>()
                {
                    *self_any = s_val.clone();
                    return Ok(());
                }
            }
        }

        Err(ScirsError::InvalidInput(
            scirs2_core::error::ErrorContext::new(format!(
                "Cannot convert parameter value {:?} to type {}",
                value,
                std::any::type_name::<T>()
            )),
        ))
    }

    fn get_value(&self) -> ParameterValue {
        ParameterValue::from_typed(&self.current_value)
    }

    fn get_bounds(&self) -> (ParameterValue, ParameterValue) {
        (
            ParameterValue::from_typed(&self.min_value),
            ParameterValue::from_typed(&self.max_value),
        )
    }
}

/// Type-erased parameter value
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
}

impl ParameterValue {
    fn from_typed<T>(value: &T) -> Self
    where
        T: std::fmt::Debug + 'static,
    {
        // Use Any trait for type introspection
        use std::any::{Any, TypeId};

        let type_id = TypeId::of::<T>();

        // Handle common types
        if type_id == TypeId::of::<f64>() {
            if let Some(f_val) = (value as &dyn Any).downcast_ref::<f64>() {
                return ParameterValue::Float(*f_val);
            }
        } else if type_id == TypeId::of::<f32>() {
            if let Some(f_val) = (value as &dyn Any).downcast_ref::<f32>() {
                return ParameterValue::Float(*f_val as f64);
            }
        } else if type_id == TypeId::of::<i64>() {
            if let Some(i_val) = (value as &dyn Any).downcast_ref::<i64>() {
                return ParameterValue::Integer(*i_val);
            }
        } else if type_id == TypeId::of::<i32>() {
            if let Some(i_val) = (value as &dyn Any).downcast_ref::<i32>() {
                return ParameterValue::Integer(*i_val as i64);
            }
        } else if type_id == TypeId::of::<usize>() {
            if let Some(u_val) = (value as &dyn Any).downcast_ref::<usize>() {
                return ParameterValue::Integer(*u_val as i64);
            }
        } else if type_id == TypeId::of::<bool>() {
            if let Some(b_val) = (value as &dyn Any).downcast_ref::<bool>() {
                return ParameterValue::Boolean(*b_val);
            }
        } else if type_id == TypeId::of::<String>() {
            if let Some(s_val) = (value as &dyn Any).downcast_ref::<String>() {
                return ParameterValue::String(s_val.clone());
            }
        } else if type_id == TypeId::of::<&str>() {
            if let Some(s_val) = (value as &dyn Any).downcast_ref::<&str>() {
                return ParameterValue::String(s_val.to_string());
            }
        }

        // Fallback - try to parse as float from debug representation
        let debug_str = format!("{:?}", value);
        if let Ok(f_val) = debug_str.parse::<f64>() {
            ParameterValue::Float(f_val)
        } else {
            // Last resort - return a default _value
            ParameterValue::Float(0.0)
        }
    }

    /// Extract as f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(f) => Some(*f),
            ParameterValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Extract as i64 if possible
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ParameterValue::Integer(i) => Some(*i),
            ParameterValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Extract as bool if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Tracks optimization performance for parameter adaptation
struct PerformanceTracker {
    memory_window: usize,
    function_values: VecDeque<f64>,
    gradient_norms: VecDeque<f64>,
    improvements: VecDeque<f64>,
    nit: VecDeque<usize>,
    timestamps: VecDeque<Instant>,
}

impl PerformanceTracker {
    fn new(_memorywindow: usize) -> Self {
        Self {
            memory_window: _memorywindow,
            function_values: VecDeque::new(),
            gradient_norms: VecDeque::new(),
            improvements: VecDeque::new(),
            nit: VecDeque::new(),
            timestamps: VecDeque::new(),
        }
    }

    fn record_performance(
        &mut self,
        iteration: usize,
        function_value: f64,
        gradient_norm: Option<f64>,
        improvement: f64,
    ) {
        // Maintain window size
        if self.function_values.len() >= self.memory_window {
            self.function_values.pop_front();
            self.improvements.pop_front();
            self.nit.pop_front();
            self.timestamps.pop_front();
            if !self.gradient_norms.is_empty() {
                self.gradient_norms.pop_front();
            }
        }

        self.function_values.push_back(function_value);
        self.improvements.push_back(improvement);
        self.nit.push_back(iteration);
        self.timestamps.push_back(Instant::now());

        if let Some(grad_norm) = gradient_norm {
            self.gradient_norms.push_back(grad_norm);
        }
    }

    fn compute_metrics(&self) -> PerformanceMetrics {
        let convergence_rate = self.compute_convergence_rate();
        let average_improvement =
            self.improvements.iter().sum::<f64>() / self.improvements.len() as f64;
        let stability_score = self.compute_stability_score();
        let progress_rate = self.compute_progress_rate();

        PerformanceMetrics {
            convergence_rate,
            average_improvement,
            stability_score,
            progress_rate,
            current_function_value: self.function_values.back().copied().unwrap_or(0.0),
            current_gradient_norm: self.gradient_norms.back().copied(),
        }
    }

    fn compute_convergence_rate(&self) -> f64 {
        if self.function_values.len() < 2 {
            return 0.0;
        }

        let mut rates = Vec::new();
        let values: Vec<f64> = self.function_values.iter().copied().collect();

        for i in 1..values.len() {
            if values[i - 1] != 0.0 && values[i - 1] != values[i] {
                let rate = (values[i - 1] - values[i]).abs() / values[i - 1].abs();
                if rate.is_finite() {
                    rates.push(rate);
                }
            }
        }

        if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        }
    }

    fn compute_stability_score(&self) -> f64 {
        if self.improvements.len() < 2 {
            return 1.0;
        }

        let improvements: Vec<f64> = self.improvements.iter().copied().collect();
        let mean = improvements.iter().sum::<f64>() / improvements.len() as f64;
        let variance = improvements
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / improvements.len() as f64;

        let std_dev = variance.sqrt();

        // Higher stability score for lower variance
        if std_dev == 0.0 {
            1.0
        } else {
            1.0 / (1.0 + std_dev)
        }
    }

    fn compute_progress_rate(&self) -> f64 {
        if self.nit.len() < 2 || self.timestamps.len() < 2 {
            return 0.0;
        }

        let time_elapsed = self
            .timestamps
            .back()
            .unwrap()
            .duration_since(*self.timestamps.front().unwrap())
            .as_secs_f64();

        if time_elapsed == 0.0 {
            return 0.0;
        }

        let iteration_count = self.nit.len() as f64;
        iteration_count / time_elapsed
    }
}

/// Performance metrics for parameter adaptation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Rate of convergence
    pub convergence_rate: f64,
    /// Average improvement per iteration
    pub average_improvement: f64,
    /// Stability of the optimization process
    pub stability_score: f64,
    /// Rate of progress (iterations per second)
    pub progress_rate: f64,
    /// Current function value
    pub current_function_value: f64,
    /// Current gradient norm (if available)
    pub current_gradient_norm: Option<f64>,
}

/// Engine for adapting parameters based on performance
struct AdaptationEngine {
    strategy: AdaptationStrategy,
    rl_agent: Option<ReinforcementLearningAgent>,
    bayesian_optimizer: Option<BayesianParameterOptimizer>,
}

impl AdaptationEngine {
    fn new(strategy: AdaptationStrategy) -> Self {
        let rl_agent = if matches!(
            strategy,
            AdaptationStrategy::ReinforcementLearning | AdaptationStrategy::Hybrid
        ) {
            Some(ReinforcementLearningAgent::new())
        } else {
            None
        };

        let bayesian_optimizer = if matches!(
            strategy,
            AdaptationStrategy::BayesianOptimization | AdaptationStrategy::Hybrid
        ) {
            Some(BayesianParameterOptimizer::new())
        } else {
            None
        };

        Self {
            strategy,
            rl_agent,
            bayesian_optimizer,
        }
    }

    fn adapt_parameters(
        &mut self,
        parameter_manager: &mut ParameterManager,
        metrics: &PerformanceMetrics,
        config: &SelfTuningConfig,
    ) -> ScirsResult<AdaptationResult> {
        match self.strategy {
            AdaptationStrategy::PerformanceBased => {
                self.performance_based_adaptation(parameter_manager, metrics, config)
            }
            AdaptationStrategy::ConvergenceBased => {
                self.convergence_based_adaptation(parameter_manager, metrics, config)
            }
            AdaptationStrategy::ReinforcementLearning => {
                if self.rl_agent.is_some() {
                    // Temporarily take the agent to avoid borrow conflicts
                    let mut agent = self.rl_agent.take().unwrap();
                    let result =
                        self.rl_based_adaptation(&mut agent, parameter_manager, metrics, config);
                    self.rl_agent = Some(agent);
                    result
                } else {
                    self.performance_based_adaptation(parameter_manager, metrics, config)
                }
            }
            AdaptationStrategy::BayesianOptimization => {
                if let Some(ref mut optimizer) = self.bayesian_optimizer {
                    self.bayesian_adaptation(parameter_manager, metrics, config)
                } else {
                    self.performance_based_adaptation(parameter_manager, metrics, config)
                }
            }
            AdaptationStrategy::Hybrid => {
                self.hybrid_adaptation(parameter_manager, metrics, config)
            }
        }
    }

    fn performance_based_adaptation(
        &self,
        parameter_manager: &mut ParameterManager,
        metrics: &PerformanceMetrics,
        config: &SelfTuningConfig,
    ) -> ScirsResult<AdaptationResult> {
        let mut changes = Vec::new();
        let mut parameters_changed = false;

        // Adapt based on convergence rate
        if metrics.convergence_rate < 0.001 {
            // Slow convergence - increase exploration
            for (name, value) in parameter_manager.current_values().clone() {
                if name.contains("learning_rate") || name.contains("step_size") {
                    let old_value = value.clone();
                    if let Some(new_value) =
                        self.increase_parameter(value, 1.1, parameter_manager.get_bounds(&name))
                    {
                        parameter_manager.update_parameter(&name, new_value.clone())?;
                        changes.push(ParameterChange {
                            name: name.clone(),
                            old_value,
                            new_value,
                            reason: "Increase step size for slow convergence".to_string(),
                        });
                        parameters_changed = true;
                    }
                }
            }
        } else if metrics.convergence_rate > 0.1 {
            // Fast convergence - might overshoot
            for (name, value) in parameter_manager.current_values().clone() {
                if name.contains("learning_rate") || name.contains("step_size") {
                    let old_value = value.clone();
                    if let Some(new_value) =
                        self.decrease_parameter(value, 0.9, parameter_manager.get_bounds(&name))
                    {
                        parameter_manager.update_parameter(&name, new_value.clone())?;
                        changes.push(ParameterChange {
                            name: name.clone(),
                            old_value,
                            new_value,
                            reason: "Decrease step size for fast convergence".to_string(),
                        });
                        parameters_changed = true;
                    }
                }
            }
        }

        Ok(AdaptationResult {
            parameters_changed,
            changes,
            strategy_used: AdaptationStrategy::PerformanceBased,
        })
    }

    fn convergence_based_adaptation(
        &self,
        parameter_manager: &mut ParameterManager,
        metrics: &PerformanceMetrics,
        config: &SelfTuningConfig,
    ) -> ScirsResult<AdaptationResult> {
        // Similar to performance-based but focuses on convergence characteristics
        self.performance_based_adaptation(parameter_manager, metrics, config)
    }

    fn rl_based_adaptation(
        &mut self,
        agent: &mut ReinforcementLearningAgent,
        parameter_manager: &mut ParameterManager,
        metrics: &PerformanceMetrics,
        config: &SelfTuningConfig,
    ) -> ScirsResult<AdaptationResult> {
        let action = agent.select_action(metrics);
        let changes = agent.apply_action(action, parameter_manager)?;

        Ok(AdaptationResult {
            parameters_changed: !changes.is_empty(),
            changes,
            strategy_used: AdaptationStrategy::ReinforcementLearning,
        })
    }

    fn bayesian_adaptation(
        &mut self,
        parameter_manager: &mut ParameterManager,
        metrics: &PerformanceMetrics,
        config: &SelfTuningConfig,
    ) -> ScirsResult<AdaptationResult> {
        let suggestions = if let Some(ref mut optimizer) = self.bayesian_optimizer {
            optimizer.suggest_parameters(parameter_manager.current_values(), metrics)?
        } else {
            return Err(ScirsError::ComputationError(
                scirs2_core::error::ErrorContext::new("Bayesian optimizer not available"),
            ));
        };
        let mut changes = Vec::new();

        for (name, new_value) in suggestions {
            if let Some(old_value) = parameter_manager.current_values().get(&name) {
                let old_value_clone = old_value.clone();
                parameter_manager.update_parameter(&name, new_value.clone())?;
                changes.push(ParameterChange {
                    name: name.clone(),
                    old_value: old_value_clone,
                    new_value,
                    reason: "Bayesian optimization suggestion".to_string(),
                });
            }
        }

        Ok(AdaptationResult {
            parameters_changed: !changes.is_empty(),
            changes,
            strategy_used: AdaptationStrategy::BayesianOptimization,
        })
    }

    fn hybrid_adaptation(
        &mut self,
        parameter_manager: &mut ParameterManager,
        metrics: &PerformanceMetrics,
        config: &SelfTuningConfig,
    ) -> ScirsResult<AdaptationResult> {
        // Combine multiple strategies based on context
        if metrics.stability_score < 0.5 {
            // Use performance-based for unstable optimization
            self.performance_based_adaptation(parameter_manager, metrics, config)
        } else if config.use_bayesian_tuning && self.bayesian_optimizer.is_some() {
            // Use Bayesian optimization for stable cases
            self.bayesian_adaptation(parameter_manager, metrics, config)
        } else {
            self.performance_based_adaptation(parameter_manager, metrics, config)
        }
    }

    fn increase_parameter(
        &self,
        value: ParameterValue,
        factor: f64,
        bounds: Option<&(ParameterValue, ParameterValue)>,
    ) -> Option<ParameterValue> {
        match value {
            ParameterValue::Float(f) => {
                let new_value = f * factor;
                if let Some((_, max_bound)) = bounds {
                    if let Some(max_f) = max_bound.as_f64() {
                        if new_value <= max_f {
                            Some(ParameterValue::Float(new_value))
                        } else {
                            None
                        }
                    } else {
                        Some(ParameterValue::Float(new_value))
                    }
                } else {
                    Some(ParameterValue::Float(new_value))
                }
            }
            ParameterValue::Integer(i) => {
                let new_value = ((i as f64) * factor) as i64;
                if let Some((_, max_bound)) = bounds {
                    if let Some(max_i) = max_bound.as_i64() {
                        if new_value <= max_i {
                            Some(ParameterValue::Integer(new_value))
                        } else {
                            None
                        }
                    } else {
                        Some(ParameterValue::Integer(new_value))
                    }
                } else {
                    Some(ParameterValue::Integer(new_value))
                }
            }
            _ => None,
        }
    }

    fn decrease_parameter(
        &self,
        value: ParameterValue,
        factor: f64,
        bounds: Option<&(ParameterValue, ParameterValue)>,
    ) -> Option<ParameterValue> {
        match value {
            ParameterValue::Float(f) => {
                let new_value = f * factor;
                if let Some((min_bound, _)) = bounds {
                    if let Some(min_f) = min_bound.as_f64() {
                        if new_value >= min_f {
                            Some(ParameterValue::Float(new_value))
                        } else {
                            None
                        }
                    } else {
                        Some(ParameterValue::Float(new_value))
                    }
                } else {
                    Some(ParameterValue::Float(new_value))
                }
            }
            ParameterValue::Integer(i) => {
                let new_value = ((i as f64) * factor) as i64;
                if let Some((min_bound, _)) = bounds {
                    if let Some(min_i) = min_bound.as_i64() {
                        if new_value >= min_i {
                            Some(ParameterValue::Integer(new_value))
                        } else {
                            None
                        }
                    } else {
                        Some(ParameterValue::Integer(new_value))
                    }
                } else {
                    Some(ParameterValue::Integer(new_value))
                }
            }
            _ => None,
        }
    }
}

/// Result of parameter adaptation
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Whether any parameters were changed
    pub parameters_changed: bool,
    /// Details of parameter changes
    pub changes: Vec<ParameterChange>,
    /// Strategy used for adaptation
    pub strategy_used: AdaptationStrategy,
}

/// Details of a parameter change
#[derive(Debug, Clone)]
pub struct ParameterChange {
    /// Name of the parameter
    pub name: String,
    /// Old parameter value
    pub old_value: ParameterValue,
    /// New parameter value
    pub new_value: ParameterValue,
    /// Reason for the change
    pub reason: String,
}

/// History of parameter tuning
struct TuningHistory {
    adaptations: Vec<AdaptationRecord>,
}

impl TuningHistory {
    fn new() -> Self {
        Self {
            adaptations: Vec::new(),
        }
    }

    fn record_adaptation(
        &mut self,
        iteration: usize,
        result: AdaptationResult,
        metrics: PerformanceMetrics,
    ) {
        self.adaptations.push(AdaptationRecord {
            iteration,
            result,
            metrics,
            timestamp: Instant::now(),
        });
    }
}

/// Record of a parameter adaptation
#[derive(Debug, Clone)]
struct AdaptationRecord {
    iteration: usize,
    result: AdaptationResult,
    metrics: PerformanceMetrics,
    timestamp: Instant,
}

/// Reinforcement learning agent for parameter adaptation
struct ReinforcementLearningAgent {
    q_table: HashMap<String, f64>,
    epsilon: f64,
    learning_rate: f64,
    discount_factor: f64,
}

impl ReinforcementLearningAgent {
    fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            epsilon: 0.1,
            learning_rate: 0.1,
            discount_factor: 0.9,
        }
    }

    fn select_action(&self, metrics: &PerformanceMetrics) -> RLAction {
        // Simplified action selection
        if metrics.convergence_rate < 0.01 {
            RLAction::IncreaseExploration
        } else if metrics.convergence_rate > 0.1 {
            RLAction::DecreaseExploration
        } else {
            RLAction::MaintainParameters
        }
    }

    fn apply_action(
        &self,
        action: RLAction,
        parameter_manager: &mut ParameterManager,
    ) -> ScirsResult<Vec<ParameterChange>> {
        let mut changes = Vec::new();

        match action {
            RLAction::IncreaseExploration => {
                // Increase step size/learning rate parameters
                for (name, value) in parameter_manager.current_values().clone() {
                    if name.contains("step_size")
                        || name.contains("learning_rate")
                        || name.contains("f_scale")
                    {
                        if let Some(new_value) = self.multiply_parameter(
                            value.clone(),
                            1.2,
                            parameter_manager.get_bounds(&name),
                        ) {
                            parameter_manager.update_parameter(&name, new_value.clone())?;
                            changes.push(ParameterChange {
                                name: name.clone(),
                                old_value: value,
                                new_value,
                                reason: "RL: Increase exploration".to_string(),
                            });
                        }
                    }
                }
            }
            RLAction::DecreaseExploration => {
                // Decrease step size/learning rate parameters
                for (name, value) in parameter_manager.current_values().clone() {
                    if name.contains("step_size")
                        || name.contains("learning_rate")
                        || name.contains("f_scale")
                    {
                        if let Some(new_value) = self.multiply_parameter(
                            value.clone(),
                            0.8,
                            parameter_manager.get_bounds(&name),
                        ) {
                            parameter_manager.update_parameter(&name, new_value.clone())?;
                            changes.push(ParameterChange {
                                name: name.clone(),
                                old_value: value,
                                new_value,
                                reason: "RL: Decrease exploration".to_string(),
                            });
                        }
                    }
                }
            }
            RLAction::MaintainParameters => {
                // No changes
            }
        }

        Ok(changes)
    }

    fn multiply_parameter(
        &self,
        value: ParameterValue,
        factor: f64,
        bounds: Option<&(ParameterValue, ParameterValue)>,
    ) -> Option<ParameterValue> {
        match value {
            ParameterValue::Float(f) => {
                let new_value = f * factor;
                if let Some((min_bound, max_bound)) = bounds {
                    if let (Some(min_f), Some(max_f)) = (min_bound.as_f64(), max_bound.as_f64()) {
                        if new_value >= min_f && new_value <= max_f {
                            Some(ParameterValue::Float(new_value))
                        } else {
                            None
                        }
                    } else {
                        Some(ParameterValue::Float(new_value))
                    }
                } else {
                    Some(ParameterValue::Float(new_value))
                }
            }
            ParameterValue::Integer(i) => {
                let new_value = ((i as f64) * factor) as i64;
                if let Some((min_bound, max_bound)) = bounds {
                    if let (Some(min_i), Some(max_i)) = (min_bound.as_i64(), max_bound.as_i64()) {
                        if new_value >= min_i && new_value <= max_i {
                            Some(ParameterValue::Integer(new_value))
                        } else {
                            None
                        }
                    } else {
                        Some(ParameterValue::Integer(new_value))
                    }
                } else {
                    Some(ParameterValue::Integer(new_value))
                }
            }
            _ => None,
        }
    }
}

/// Reinforcement learning actions
#[derive(Debug, Clone, Copy)]
enum RLAction {
    IncreaseExploration,
    DecreaseExploration,
    MaintainParameters,
}

/// Bayesian optimizer for parameter tuning
struct BayesianParameterOptimizer {
    observations: Vec<(HashMap<String, ParameterValue>, f64)>,
}

impl BayesianParameterOptimizer {
    fn new() -> Self {
        Self {
            observations: Vec::new(),
        }
    }

    fn suggest_parameters(
        &mut self,
        current_params: &HashMap<String, ParameterValue>,
        metrics: &PerformanceMetrics,
    ) -> ScirsResult<HashMap<String, ParameterValue>> {
        // Record current observation
        self.observations
            .push((current_params.clone(), metrics.current_function_value));

        let mut suggestions = HashMap::new();

        // Simplified Bayesian optimization approach
        // In practice, this would use proper Gaussian processes and acquisition functions
        if self.observations.len() >= 2 {
            // Find best observation so far
            let best_observation = self
                .observations
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(best_params) = best_observation {
                // Suggest parameter modifications based on best observation
                for (name, value) in current_params {
                    if let Some(best_value) = best_params.0.get(name) {
                        // Move toward best observed value with some exploration
                        let suggested_value =
                            self.interpolate_toward_best(value.clone(), best_value.clone(), 0.3);
                        if let Some(new_value) = suggested_value {
                            if new_value != *value {
                                suggestions.insert(name.clone(), new_value);
                            }
                        }
                    }
                }
            }
        } else {
            // Not enough observations, suggest small random perturbations
            for (name, value) in current_params {
                if name.contains("step_size") || name.contains("learning_rate") {
                    // Add small random perturbation for exploration
                    use rand::{rng, Rng};
                    let mut rng = rand::rng();
                    let perturbation_factor = 1.0 + (rng.gen_range(-0.1..=0.1));

                    let perturbed_value = match value {
                        ParameterValue::Float(f) => {
                            Some(ParameterValue::Float(f * perturbation_factor))
                        }
                        ParameterValue::Integer(i) => {
                            let new_val = ((*i as f64) * perturbation_factor) as i64;
                            Some(ParameterValue::Integer(new_val.max(1)))
                        }
                        _ => None,
                    };

                    if let Some(new_value) = perturbed_value {
                        if new_value != *value {
                            suggestions.insert(name.clone(), new_value);
                        }
                    }
                }
            }
        }

        Ok(suggestions)
    }

    fn interpolate_toward_best(
        &self,
        current: ParameterValue,
        best: ParameterValue,
        alpha: f64, // interpolation factor (0.0 = current, 1.0 = best)
    ) -> Option<ParameterValue> {
        match (current, best) {
            (ParameterValue::Float(curr), ParameterValue::Float(best_val)) => {
                let interpolated = curr * (1.0 - alpha) + best_val * alpha;
                Some(ParameterValue::Float(interpolated))
            }
            (ParameterValue::Integer(curr), ParameterValue::Integer(best_val)) => {
                let interpolated =
                    ((curr as f64) * (1.0 - alpha) + (best_val as f64) * alpha) as i64;
                Some(ParameterValue::Integer(interpolated))
            }
            _ => None,
        }
    }
}

/// Predefined parameter configurations for common algorithms
pub mod presets {
    use super::*;

    /// Get self-tuning configuration for differential evolution
    pub fn differential_evolution_config() -> SelfTuningConfig {
        SelfTuningConfig {
            adaptation_strategy: AdaptationStrategy::PerformanceBased,
            update_frequency: 25,
            learning_rate: 0.05,
            memory_window: 50,
            use_bayesian_tuning: false,
            exploration_factor: 0.15,
        }
    }

    /// Get self-tuning configuration for gradient-based methods
    pub fn gradient_based_config() -> SelfTuningConfig {
        SelfTuningConfig {
            adaptation_strategy: AdaptationStrategy::ConvergenceBased,
            update_frequency: 10,
            learning_rate: 0.2,
            memory_window: 20,
            use_bayesian_tuning: true,
            exploration_factor: 0.05,
        }
    }

    /// Get self-tuning configuration for particle swarm optimization
    pub fn particle_swarm_config() -> SelfTuningConfig {
        SelfTuningConfig {
            adaptation_strategy: AdaptationStrategy::Hybrid,
            update_frequency: 30,
            learning_rate: 0.1,
            memory_window: 75,
            use_bayesian_tuning: true,
            exploration_factor: 0.2,
        }
    }

    /// Create tunable parameters for BFGS algorithm
    pub fn bfgs_parameters() -> HashMap<String, TunableParameter<f64>> {
        let mut params = HashMap::new();

        params.insert(
            "line_search_tolerance".to_string(),
            TunableParameter::new(1e-4, 1e-8, 1e-1),
        );

        params.insert(
            "gradient_tolerance".to_string(),
            TunableParameter::new(1e-5, 1e-12, 1e-2),
        );

        params
    }

    /// Create tunable parameters for differential evolution
    pub fn differential_evolution_parameters() -> HashMap<String, TunableParameter<f64>> {
        let mut params = HashMap::new();

        params.insert("f_scale".to_string(), TunableParameter::new(0.8, 0.1, 2.0));

        params.insert(
            "crossover_rate".to_string(),
            TunableParameter::new(0.7, 0.1, 1.0),
        );

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_tuning_config() {
        let config = SelfTuningConfig::default();
        assert_eq!(
            config.adaptation_strategy,
            AdaptationStrategy::PerformanceBased
        );
        assert_eq!(config.update_frequency, 50);
        assert!(config.use_bayesian_tuning);
    }

    #[test]
    fn test_parameter_value() {
        let float_val = ParameterValue::Float(3.5);
        assert_eq!(float_val.as_f64(), Some(3.5));

        let int_val = ParameterValue::Integer(42);
        assert_eq!(int_val.as_i64(), Some(42));
        assert_eq!(int_val.as_f64(), Some(42.0));

        let bool_val = ParameterValue::Boolean(true);
        assert_eq!(bool_val.as_bool(), Some(true));
    }

    #[test]
    fn test_tunable_parameter() {
        let param = TunableParameter::new(1.0, 0.0, 10.0).with_adaptation_rate(0.2);

        assert_eq!(param.adaptation_rate, 0.2);
        assert_eq!(param.current_value, 1.0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(10);

        tracker.record_performance(1, 100.0, Some(10.0), 5.0);
        tracker.record_performance(2, 95.0, Some(8.0), 5.0);
        tracker.record_performance(3, 90.0, Some(6.0), 5.0);

        let metrics = tracker.compute_metrics();
        assert!(metrics.convergence_rate > 0.0);
        assert_eq!(metrics.average_improvement, 5.0);
        assert!(metrics.stability_score > 0.0);
    }

    #[test]
    fn test_parameter_manager() {
        let mut manager = ParameterManager::new();
        let param = TunableParameter::new(1.0, 0.0, 10.0);

        manager.register("test_param", param).unwrap();
        assert!(manager.current_values().contains_key("test_param"));

        let new_value = ParameterValue::Float(2.0);
        manager
            .update_parameter("test_param", new_value.clone())
            .unwrap();
        assert_eq!(manager.current_values()["test_param"], new_value);
    }

    #[test]
    fn test_presets() {
        let de_config = presets::differential_evolution_config();
        assert_eq!(de_config.update_frequency, 25);

        let grad_config = presets::gradient_based_config();
        assert_eq!(
            grad_config.adaptation_strategy,
            AdaptationStrategy::ConvergenceBased
        );

        let de_params = presets::differential_evolution_parameters();
        assert!(de_params.contains_key("f_scale"));
        assert!(de_params.contains_key("crossover_rate"));
    }
}
