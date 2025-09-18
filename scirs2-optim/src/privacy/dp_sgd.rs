//! Differentially Private Stochastic Gradient Descent (DP-SGD)
//!
//! This module implements DP-SGD with adaptive clipping, noise calibration,
//! and privacy budget tracking for training machine learning models with
//! formal privacy guarantees.

#![allow(dead_code)]

use ndarray::{Array, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use rand_distr::Normal;
use std::collections::{HashMap, VecDeque};

use super::moment_accountant::MomentsAccountant;
use super::{DifferentialPrivacyConfig, NoiseMechanism, PrivacyBudget};
use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// DP-SGD optimizer with privacy guarantees
pub struct DPSGDOptimizer<O, A, D>
where
    A: Float + Send + Sync + ndarray::ScalarOperand + std::fmt::Debug + Default + Clone + std::iter::Sum,
    D: ndarray::Dimension,
    O: Optimizer<A, D>,
{
    /// Base optimizer (SGD, Adam, etc.)
    baseoptimizer: O,

    /// Privacy configuration
    config: DifferentialPrivacyConfig,

    /// Moment accountant for privacy tracking
    accountant: MomentsAccountant,

    /// Random number generator for noise
    rng: scirs2_core::random::Random,

    /// Adaptive clipping state
    adaptive_clipping: Option<AdaptiveClippingState>,

    /// Privacy budget tracker
    privacy_budget: PrivacyBudgetTracker,

    /// Gradient statistics
    gradient_stats: GradientStatistics<A>,

    /// Noise calibration
    noise_calibrator: NoiseCalibrator<A>,

    /// Current step count
    step_count: usize,

    /// Batch size for current iteration
    current_batch_size: usize,

    /// Phantom data for unused type parameter
    _phantom: std::marker::PhantomData<D>,
}

/// Adaptive clipping state for DP-SGD
#[derive(Debug, Clone)]
struct AdaptiveClippingState {
    /// Current clipping threshold
    current_threshold: f64,

    /// Target quantile (e.g., 0.5 for median)
    target_quantile: f64,

    /// Learning rate for threshold adaptation
    adaptationlr: f64,

    /// History of gradient norms
    norm_history: VecDeque<f64>,

    /// Update frequency (in steps)
    update_frequency: usize,

    /// Last update step
    last_update_step: usize,

    /// Quantile estimation parameters
    quantile_estimator: QuantileEstimator,
}

/// Quantile estimator for adaptive clipping
#[derive(Debug, Clone)]
struct QuantileEstimator {
    /// P² algorithm state
    p2_state: P2AlgorithmState,

    /// Simple moving average
    moving_avg: f64,

    /// Exponential moving average
    ema: f64,

    /// EMA decay factor
    ema_decay: f64,
}

/// P² algorithm state for quantile estimation
#[derive(Debug, Clone)]
struct P2AlgorithmState {
    /// Marker positions
    markers: [f64; 5],

    /// Marker values
    values: [f64; 5],

    /// Desired marker positions
    desired_positions: [f64; 5],

    /// Increments
    increments: [f64; 5],

    /// Number of observations
    count: usize,
}

/// Privacy budget tracker
#[derive(Debug, Clone)]
struct PrivacyBudgetTracker {
    /// Total epsilon consumed
    epsilon_consumed: f64,

    /// Total delta consumed
    delta_consumed: f64,

    /// Target epsilon
    target_epsilon: f64,

    /// Target delta
    target_delta: f64,

    /// Privacy budget per step
    epsilon_per_step: f64,

    /// Delta per step
    delta_per_step: f64,

    /// Privacy consumption history
    consumption_history: Vec<PrivacyConsumption>,
}

/// Privacy consumption record
#[derive(Debug, Clone)]
pub struct PrivacyConsumption {
    step: usize,
    epsilon_spent: f64,
    delta_spent: f64,
    batchsize: usize,
    noise_multiplier: f64,
}

/// Gradient statistics for DP-SGD
#[derive(Debug, Clone)]
struct GradientStatistics<A: Float + Default + Clone + std::iter::Sum> {
    /// Recent gradient norms
    norm_history: VecDeque<A>,

    /// Clipping frequency
    clipping_frequency: f64,

    /// Average gradient norm
    avg_norm: A,

    /// Std deviation of gradient norms
    std_norm: A,

    /// Percentile statistics
    percentiles: HashMap<String, A>,

    /// Maximum history size
    max_history_size: usize,
}

/// Noise calibration for different mechanisms
#[derive(Debug, Clone)]
struct NoiseCalibrator<A: Float> {
    /// Current noise multiplier
    noise_multiplier: A,

    /// Base noise scale
    base_noise_scale: A,

    /// Adaptive noise scaling
    adaptive_scaling: bool,

    /// Noise mechanism
    mechanism: NoiseMechanism,

    /// Calibration history
    calibration_history: Vec<NoiseCalibration<A>>,
}

/// Noise calibration record
#[derive(Debug, Clone)]
pub struct NoiseCalibration<A: Float> {
    step: usize,
    noise_scale: A,
    gradientnorm: A,
    clipping_threshold: A,
    privacy_cost: A,
}

impl<O, A, D> DPSGDOptimizer<O, A, D>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + rand_distr::uniform::SampleUniform
        + ndarray::ScalarOperand
        + std::fmt::Debug
        + std::iter::Sum,
    D: ndarray::Dimension,
    O: Optimizer<A, D> + Send + Sync,
{
    /// Create a new DP-SGD optimizer
    pub fn new(baseoptimizer: O, config: DifferentialPrivacyConfig) -> Result<Self> {
        let accountant = MomentsAccountant::new(
            config.noise_multiplier,
            config.target_delta,
            config.batch_size,
            config.dataset_size,
        );

        let rng = scirs2_core::random::rng();

        let adaptive_clipping = if config.adaptive_clipping {
            Some(AdaptiveClippingState::new(
                config.adaptive_clip_init,
                config.adaptive_clip_lr,
            )?)
        } else {
            None
        };

        let privacy_budget = PrivacyBudgetTracker::new(&config);
        let gradient_stats = GradientStatistics::new();
        let noise_calibrator = NoiseCalibrator::new(&config);

        let batchsize = config.batch_size;
        Ok(Self {
            baseoptimizer,
            config,
            accountant,
            rng,
            adaptive_clipping,
            privacy_budget,
            gradient_stats,
            noise_calibrator,
            step_count: 0,
            current_batch_size: batchsize,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Perform a DP-SGD step
    pub fn dp_step(
        &mut self,
        params: &Array<A, D>,
        gradients: &mut Array<A, D>,
        batchsize: usize,
    ) -> Result<Array<A, D>> {
        self.step_count += 1;
        self.current_batch_size = batchsize;

        // Check privacy budget
        if !self.has_privacy_budget()? {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.privacy_budget.epsilon_consumed,
                target_epsilon: self.privacy_budget.target_epsilon,
            });
        }

        // Compute gradient norm before clipping
        let pre_clip_norm = self.compute_gradient_norm(gradients);

        // Update gradient statistics
        self.gradient_stats.update_norm(pre_clip_norm);

        // Get current clipping threshold
        let clipping_threshold = self.get_clipping_threshold();

        // Apply gradient clipping
        let was_clipped = self.clip_gradients(gradients, clipping_threshold)?;

        // Update clipping statistics
        if was_clipped {
            self.gradient_stats.update_clipping();
        }

        // Compute post-clipping norm
        let _post_clip_norm = self.compute_gradient_norm(gradients);

        // Add calibrated noise
        self.add_noise(gradients, clipping_threshold)?;

        // Update moment accountant
        let (epsilon_spent, delta_spent) = self.accountant.get_privacy_spent(self.step_count)?;
        self.privacy_budget.update_consumption(
            self.step_count,
            epsilon_spent,
            delta_spent,
            batchsize,
            self.config.noise_multiplier,
        );

        // Update adaptive clipping if enabled
        let should_update = self.should_update_clipping_threshold();
        if let Some(ref mut adaptive_state) = self.adaptive_clipping {
            if should_update {
                adaptive_state.update_threshold(pre_clip_norm.to_f64().unwrap_or(0.0));
            }
        }

        // Update noise calibration
        self.noise_calibrator.update_calibration(
            self.step_count,
            pre_clip_norm,
            A::from(clipping_threshold).unwrap(),
            A::from(epsilon_spent).unwrap(),
        );

        // Apply base optimizer step
        let updated_params = self.baseoptimizer.step(params, gradients)?;

        Ok(updated_params)
    }

    /// Check if privacy budget is available
    pub fn has_privacy_budget(&self) -> Result<bool> {
        Ok(
            self.privacy_budget.epsilon_consumed < self.privacy_budget.target_epsilon
                && self.privacy_budget.delta_consumed < self.privacy_budget.target_delta,
        )
    }

    /// Get current privacy budget status
    pub fn get_privacy_budget(&self) -> PrivacyBudget {
        PrivacyBudget {
            epsilon_consumed: self.privacy_budget.epsilon_consumed,
            delta_consumed: self.privacy_budget.delta_consumed,
            epsilon_remaining: (self.privacy_budget.target_epsilon
                - self.privacy_budget.epsilon_consumed)
                .max(0.0),
            delta_remaining: (self.privacy_budget.target_delta
                - self.privacy_budget.delta_consumed)
                .max(0.0),
            steps_taken: self.step_count,
            accounting_method: super::AccountingMethod::MomentsAccountant,
            estimated_steps_remaining: self.estimate_remaining_steps(),
        }
    }

    /// Get adaptive clipping statistics
    pub fn get_clipping_stats(&self) -> AdaptiveClippingStats {
        AdaptiveClippingStats {
            current_threshold: self.get_clipping_threshold(),
            target_quantile: self
                .adaptive_clipping
                .as_ref()
                .map(|ac| ac.target_quantile)
                .unwrap_or(0.5),
            clipping_frequency: self.gradient_stats.clipping_frequency,
            avg_gradient_norm: self.gradient_stats.avg_norm.to_f64().unwrap_or(0.0),
            std_gradient_norm: self.gradient_stats.std_norm.to_f64().unwrap_or(0.0),
            adaptation_rate: self
                .adaptive_clipping
                .as_ref()
                .map(|ac| ac.adaptationlr)
                .unwrap_or(0.0),
        }
    }

    /// Set batch size for next iterations
    pub fn set_batch_size(&mut self, batchsize: usize) {
        self.current_batch_size = batchsize;
        // Update moment accountant with new batch _size
        self.accountant = MomentsAccountant::new(
            self.config.noise_multiplier,
            self.config.target_delta,
            batchsize,
            self.config.dataset_size,
        );
    }

    /// Update privacy configuration
    pub fn update_privacy_config(&mut self, newconfig: DifferentialPrivacyConfig) -> Result<()> {
        // Validate that privacy budget doesn't decrease
        if newconfig.target_epsilon < self.config.target_epsilon
            || newconfig.target_delta < self.config.target_delta
        {
            return Err(OptimError::InvalidConfig(
                "Cannot decrease privacy budget mid-training".to_string(),
            ));
        }

        self.config = newconfig;
        self.privacy_budget.target_epsilon = self.config.target_epsilon;
        self.privacy_budget.target_delta = self.config.target_delta;

        // Update moment accountant
        self.accountant = MomentsAccountant::new(
            self.config.noise_multiplier,
            self.config.target_delta,
            self.current_batch_size,
            self.config.dataset_size,
        );

        Ok(())
    }

    /// Compute gradient norm
    fn compute_gradient_norm<S, DIM>(&self, gradients: &ArrayBase<S, DIM>) -> A
    where
        S: Data<Elem = A>,
        DIM: Dimension,
    {
        gradients.iter().map(|&g| g * g).sum::<A>().sqrt()
    }

    /// Get current clipping threshold
    fn get_clipping_threshold(&self) -> f64 {
        if let Some(ref adaptive_state) = self.adaptive_clipping {
            adaptive_state.current_threshold
        } else {
            self.config.l2_norm_clip
        }
    }

    /// Clip gradients to threshold
    fn clip_gradients<S, DIM>(
        &self,
        gradients: &mut ArrayBase<S, DIM>,
        threshold: f64,
    ) -> Result<bool>
    where
        S: DataMut<Elem = A>,
        DIM: Dimension,
    {
        let norm = self.compute_gradient_norm(gradients);
        let threshold_a = A::from(threshold).unwrap();

        if norm > threshold_a {
            let scale = threshold_a / norm;
            gradients.mapv_inplace(|g| g * scale);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Add calibrated noise to gradients
    fn add_noise<S, DIM>(
        &mut self,
        gradients: &mut ArrayBase<S, DIM>,
        clipping_threshold: f64,
    ) -> Result<()>
    where
        S: DataMut<Elem = A>,
        DIM: Dimension,
    {
        let noise_scale = self.config.noise_multiplier * clipping_threshold;

        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let normal = Normal::new(0.0, noise_scale)
                    .map_err(|_| OptimError::InvalidConfig("Invalid noise scale".to_string()))?;
                gradients.mapv_inplace(|g| {
                    // Use scirs2_core random for Gaussian noise
                    let noise_f64 = self.rng.sample(normal);
                    let noise = A::from(noise_f64).unwrap();
                    g + noise
                });
            }
            NoiseMechanism::Laplace => {
                // Simplified Laplace noise using Normal distribution approximation
                let normal = Normal::new(0.0, noise_scale * 1.414)
                    .map_err(|_| OptimError::InvalidConfig("Invalid noise scale".to_string()))?;
                gradients.mapv_inplace(|g| {
                    // Use scirs2_core random for Laplace-approximated noise
                    let noise_f64 = self.rng.sample(normal);
                    let noise = A::from(noise_f64).unwrap();
                    g + noise
                });
            }
            _ => {
                // Default to Gaussian
                let normal = Normal::new(0.0, noise_scale)
                    .map_err(|_| OptimError::InvalidConfig("Invalid noise scale".to_string()))?;
                gradients.mapv_inplace(|g| {
                    // Use scirs2_core random for Gaussian noise
                    let noise_f64 = self.rng.sample(normal);
                    let noise = A::from(noise_f64).unwrap();
                    g + noise
                });
            }
        }

        Ok(())
    }

    /// Check if clipping threshold should be updated
    fn should_update_clipping_threshold(&self) -> bool {
        if let Some(ref adaptive_state) = self.adaptive_clipping {
            self.step_count - adaptive_state.last_update_step >= adaptive_state.update_frequency
        } else {
            false
        }
    }

    /// Estimate remaining steps before budget exhaustion
    fn estimate_remaining_steps(&self) -> usize {
        if self.step_count == 0 {
            return usize::MAX;
        }

        let epsilon_per_step = self.privacy_budget.epsilon_consumed / self.step_count as f64;
        let remaining_epsilon =
            self.privacy_budget.target_epsilon - self.privacy_budget.epsilon_consumed;

        if epsilon_per_step > 0.0 {
            (remaining_epsilon / epsilon_per_step) as usize
        } else {
            usize::MAX
        }
    }

    /// Get privacy accounting details
    pub fn get_privacy_accounting_details(&self) -> PrivacyAccountingDetails {
        PrivacyAccountingDetails {
            moment_accountant_orders: self.accountant.get_computed_orders(),
            privacy_consumption_history: self.privacy_budget.consumption_history.clone(),
            gradient_statistics: GradientStatsSnapshot {
                avg_norm: self.gradient_stats.avg_norm.to_f64().unwrap_or(0.0),
                std_norm: self.gradient_stats.std_norm.to_f64().unwrap_or(0.0),
                clipping_frequency: self.gradient_stats.clipping_frequency,
                percentiles: self
                    .gradient_stats
                    .percentiles
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or(0.0)))
                    .collect(),
            },
            noise_calibration_history: self
                .noise_calibrator
                .calibration_history
                .iter()
                .map(|entry| NoiseCalibration {
                    step: entry.step,
                    noise_scale: entry.noise_scale.to_f64().unwrap_or(0.0),
                    gradientnorm: entry.gradientnorm.to_f64().unwrap_or(0.0),
                    clipping_threshold: entry.clipping_threshold.to_f64().unwrap_or(0.0),
                    privacy_cost: entry.privacy_cost.to_f64().unwrap_or(0.0),
                })
                .collect(),
        }
    }

    /// Validate DP-SGD configuration
    pub fn validate_configuration(&self) -> Result<ConfigurationValidation> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Check noise multiplier
        if self.config.noise_multiplier < 0.1 {
            warnings
                .push("Very low noise multiplier may not provide sufficient privacy".to_string());
        }
        if self.config.noise_multiplier > 10.0 {
            warnings.push("Very high noise multiplier may severely impact utility".to_string());
        }

        // Check clipping threshold
        if self.config.l2_norm_clip < 0.01 {
            warnings
                .push("Very low clipping threshold may impact gradient information".to_string());
        }
        if self.config.l2_norm_clip > 100.0 {
            warnings.push(
                "Very high clipping threshold may not provide effective clipping".to_string(),
            );
        }

        // Check batch size
        if self.config.batch_size < 16 {
            warnings.push("Small batch size may reduce privacy amplification benefits".to_string());
        }

        // Check dataset size
        if self.config.dataset_size < 1000 {
            warnings
                .push("Small dataset may limit achievable privacy-utility tradeoff".to_string());
        }

        // Check privacy budget
        if self.config.target_epsilon > 10.0 {
            warnings.push("Large epsilon value provides limited privacy guarantee".to_string());
        }
        if self.config.target_delta > 1.0 / self.config.dataset_size as f64 {
            errors.push("Delta should typically be much smaller than 1/n".to_string());
        }

        Ok(ConfigurationValidation {
            is_valid: errors.is_empty(),
            warnings,
            errors,
            recommended_adjustments: self.generate_recommendations(),
        })
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.gradient_stats.clipping_frequency > 0.8 {
            recommendations.push(
                "Consider increasing clipping threshold - high clipping frequency detected"
                    .to_string(),
            );
        }

        if self.gradient_stats.clipping_frequency < 0.1 {
            recommendations.push(
                "Consider decreasing clipping threshold - low clipping frequency detected"
                    .to_string(),
            );
        }

        if self.privacy_budget.epsilon_consumed / self.privacy_budget.target_epsilon > 0.9 {
            recommendations.push(
                "Privacy budget nearly exhausted - consider reducing noise multiplier".to_string(),
            );
        }

        recommendations
    }
}

// Implementation of helper structures

impl AdaptiveClippingState {
    fn new(initial_threshold: f64, adaptationlr: f64) -> Result<Self> {
        Ok(Self {
            current_threshold: initial_threshold,
            target_quantile: 0.5,
            adaptationlr,
            norm_history: VecDeque::with_capacity(1000),
            update_frequency: 50,
            last_update_step: 0,
            quantile_estimator: QuantileEstimator::new(),
        })
    }

    fn update_threshold(&mut self, gradientnorm: f64) {
        self.norm_history.push_back(gradientnorm);
        if self.norm_history.len() > 1000 {
            self.norm_history.pop_front();
        }

        // Update quantile estimate
        self.quantile_estimator.update(gradientnorm);

        // Adapt threshold towards target quantile
        let quantile_estimate = self.quantile_estimator.get_quantile(self.target_quantile);
        let error = quantile_estimate - self.current_threshold;
        self.current_threshold += self.adaptationlr * error;

        // Ensure threshold is positive
        self.current_threshold = self.current_threshold.max(1e-6);
    }
}

impl QuantileEstimator {
    fn new() -> Self {
        Self {
            p2_state: P2AlgorithmState::new(0.5),
            moving_avg: 0.0,
            ema: 0.0,
            ema_decay: 0.99,
        }
    }

    fn update(&mut self, value: f64) {
        self.p2_state.update(value);

        // Update EMA
        if self.p2_state.count == 1 {
            self.ema = value;
        } else {
            self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * value;
        }
    }

    fn get_quantile(&self, quantile: f64) -> f64 {
        if self.p2_state.count >= 5 {
            self.p2_state.get_quantile()
        } else {
            self.ema
        }
    }
}

impl P2AlgorithmState {
    fn new(quantile: f64) -> Self {
        Self {
            markers: [0.0; 5],
            values: [0.0; 5],
            desired_positions: [0.0, quantile / 2.0, quantile, (1.0 + quantile) / 2.0, 1.0],
            increments: [0.0, quantile / 2.0, quantile, (1.0 + quantile) / 2.0, 1.0],
            count: 0,
        }
    }

    fn update(&mut self, value: f64) {
        if self.count < 5 {
            self.values[self.count] = value;
            if self.count == 4 {
                self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for i in 0..5 {
                    self.markers[i] = i as f64;
                }
            }
            self.count += 1;
        } else {
            // P² algorithm update
            // Simplified implementation
            self.count += 1;
        }
    }

    fn get_quantile(&self) -> f64 {
        if self.count >= 5 {
            self.values[2] // Median for simplicity
        } else {
            0.0
        }
    }
}

impl PrivacyBudgetTracker {
    fn new(config: &DifferentialPrivacyConfig) -> Self {
        Self {
            epsilon_consumed: 0.0,
            delta_consumed: 0.0,
            target_epsilon: config.target_epsilon,
            target_delta: config.target_delta,
            epsilon_per_step: 0.0,
            delta_per_step: 0.0,
            consumption_history: Vec::new(),
        }
    }

    fn update_consumption(
        &mut self,
        step: usize,
        epsilon_spent: f64,
        delta_spent: f64,
        batchsize: usize,
        noise_multiplier: f64,
    ) {
        self.epsilon_consumed = epsilon_spent;
        self.delta_consumed = delta_spent;

        self.consumption_history.push(PrivacyConsumption {
            step,
            epsilon_spent,
            delta_spent,
            batchsize,
            noise_multiplier,
        });
    }
}

impl<A: Float + Default + Clone + std::iter::Sum> GradientStatistics<A> {
    fn new() -> Self {
        Self {
            norm_history: VecDeque::with_capacity(1000),
            clipping_frequency: 0.0,
            avg_norm: A::zero(),
            std_norm: A::zero(),
            percentiles: HashMap::new(),
            max_history_size: 1000,
        }
    }

    fn update_norm(&mut self, norm: A) {
        self.norm_history.push_back(norm);
        if self.norm_history.len() > self.max_history_size {
            self.norm_history.pop_front();
        }

        // Update statistics
        let n = A::from(self.norm_history.len()).unwrap();
        self.avg_norm = self.norm_history.iter().cloned().sum::<A>() / n;

        let variance = self
            .norm_history
            .iter()
            .map(|&x| (x - self.avg_norm) * (x - self.avg_norm))
            .sum::<A>()
            / n;
        self.std_norm = variance.sqrt();
    }

    fn update_clipping(&mut self) {
        // Simple moving average for clipping frequency
        let alpha = 0.01; // Learning rate for frequency update
        self.clipping_frequency = (1.0 - alpha) * self.clipping_frequency + alpha;
    }
}

impl<A: Float + Default + Clone> NoiseCalibrator<A> {
    fn new(config: &DifferentialPrivacyConfig) -> Self {
        Self {
            noise_multiplier: A::from(config.noise_multiplier).unwrap(),
            base_noise_scale: A::from(config.noise_multiplier * config.l2_norm_clip).unwrap(),
            adaptive_scaling: false,
            mechanism: config.noise_mechanism,
            calibration_history: Vec::new(),
        }
    }

    fn update_calibration(
        &mut self,
        step: usize,
        gradientnorm: A,
        clipping_threshold: A,
        privacy_cost: A,
    ) {
        let noise_scale = self.noise_multiplier * clipping_threshold;

        self.calibration_history.push(NoiseCalibration {
            step,
            noise_scale,
            gradientnorm,
            clipping_threshold,
            privacy_cost,
        });

        // Limit history size
        if self.calibration_history.len() > 1000 {
            self.calibration_history.remove(0);
        }
    }
}

/// Adaptive clipping statistics
#[derive(Debug, Clone)]
pub struct AdaptiveClippingStats {
    pub current_threshold: f64,
    pub target_quantile: f64,
    pub clipping_frequency: f64,
    pub avg_gradient_norm: f64,
    pub std_gradient_norm: f64,
    pub adaptation_rate: f64,
}

/// Privacy accounting details
#[derive(Debug, Clone)]
pub struct PrivacyAccountingDetails {
    pub moment_accountant_orders: Vec<f64>,
    pub privacy_consumption_history: Vec<PrivacyConsumption>,
    pub gradient_statistics: GradientStatsSnapshot,
    pub noise_calibration_history: Vec<NoiseCalibration<f64>>,
}

/// Gradient statistics snapshot
#[derive(Debug, Clone)]
pub struct GradientStatsSnapshot {
    pub avg_norm: f64,
    pub std_norm: f64,
    pub clipping_frequency: f64,
    pub percentiles: HashMap<String, f64>,
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ConfigurationValidation {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub recommended_adjustments: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_dp_sgd_creation() {
        let sgd = SGD::new(0.01);
        let config = DifferentialPrivacyConfig::default();
        let dp_sgd = DPSGDOptimizer::<_, f64, ndarray::Ix1>::new(sgd, config);
        assert!(dp_sgd.is_ok());
    }

    #[test]
    fn test_adaptive_clipping_state() {
        let state = AdaptiveClippingState::new(1.0, 0.1);
        assert!(state.is_ok());

        let state = state.unwrap();
        assert_eq!(state.current_threshold, 1.0);
        assert_eq!(state.adaptationlr, 0.1);
    }

    #[test]
    fn test_privacy_budget_tracker() {
        let config = DifferentialPrivacyConfig::default();
        let tracker = PrivacyBudgetTracker::new(&config);

        assert_eq!(tracker.target_epsilon, config.target_epsilon);
        assert_eq!(tracker.epsilon_consumed, 0.0);
    }

    #[test]
    fn test_quantile_estimator() {
        let mut estimator = QuantileEstimator::new();

        for i in 1..=10 {
            estimator.update(i as f64);
        }

        let quantile = estimator.get_quantile(0.5);
        assert!(quantile > 0.0);
    }

    #[test]
    fn test_gradient_statistics() {
        let mut stats = GradientStatistics::<f64>::new();

        stats.update_norm(1.0);
        stats.update_norm(2.0);
        stats.update_norm(3.0);

        assert_eq!(stats.avg_norm, 2.0);
        assert!(stats.std_norm > 0.0);
    }
}
