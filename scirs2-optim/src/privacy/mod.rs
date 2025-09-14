//! Differential Privacy support for optimizers
//!
//! This module provides differential privacy mechanisms for machine learning
//! optimization, including DP-SGD with moment accountant for privacy budget tracking.

#![allow(dead_code)]

use crate::error::{OptimError, Result};
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use scirs2_core::ScientificNumber;
use std::collections::VecDeque;
use std::fmt::Debug;

pub mod byzantine_tolerance;
pub mod differential_privacy; // New modular differential privacy
pub mod dp_sgd;
pub mod enhanced_audit;
pub mod federated; // New modular federated privacy
pub mod federated_privacy;
pub mod moment_accountant;
pub mod noise_mechanisms;
pub mod private_hyperparameter_optimization;
pub mod secure_multiparty;
pub mod utility_analysis;

use crate::optimizers::Optimizer;

// Re-export key utility analysis types
pub use utility_analysis::{
    AnalysisConfig, AnalysisMetadata, BudgetRecommendations, OptimalConfiguration, ParetoPoint,
    PrivacyConfiguration, PrivacyParameterSpace, PrivacyRiskAssessment, PrivacyUtilityAnalyzer,
    PrivacyUtilityResults, RobustnessResults, SensitivityResults, StatisticalTestResults,
    UtilityMetric,
};

// Re-export modular federated privacy types
pub use federated::{
    ByzantineRobustAggregator, ByzantineRobustConfig, ByzantineRobustMethod, ClientComposition,
    CompositionStats, CrossDeviceConfig, CrossDevicePrivacyManager, DeviceProfile, DeviceType,
    FederatedCompositionAnalyzer, FederatedCompositionMethod, OutlierDetectionResult,
    ReputationSystemConfig, RoundComposition, SecureAggregationConfig, SecureAggregationPlan,
    SecureAggregator, SeedSharingMethod, StatisticalTestConfig, StatisticalTestType, TemporalEvent,
    TemporalEventType,
};

// Re-export modular differential privacy types
pub use differential_privacy::{
    AmplificationConfig, AmplificationStats, PrivacyAmplificationAnalyzer, SubsamplingEvent,
};

/// Differential privacy configuration
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyConfig {
    /// Target privacy parameter epsilon
    pub target_epsilon: f64,

    /// Privacy parameter delta (typically 1/n where n is dataset size)
    pub target_delta: f64,

    /// Noise multiplier for gradient perturbation
    pub noise_multiplier: f64,

    /// L2 norm clipping threshold for gradients
    pub l2_norm_clip: f64,

    /// Batch size for sampling
    pub batch_size: usize,

    /// Dataset size for privacy accounting
    pub dataset_size: usize,

    /// Maximum number of training steps
    pub max_steps: usize,

    /// Noise mechanism to use
    pub noise_mechanism: NoiseMechanism,

    /// Enable secure aggregation (for federated learning)
    pub secure_aggregation: bool,

    /// Enable adaptive clipping
    pub adaptive_clipping: bool,

    /// Initial clipping threshold for adaptive clipping
    pub adaptive_clip_init: f64,

    /// Learning rate for adaptive clipping
    pub adaptive_clip_lr: f64,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            target_epsilon: 1.0,
            target_delta: 1e-5,
            noise_multiplier: 1.1,
            l2_norm_clip: 1.0,
            batch_size: 256,
            dataset_size: 50000,
            max_steps: 1000,
            noise_mechanism: NoiseMechanism::Gaussian,
            secure_aggregation: false,
            adaptive_clipping: false,
            adaptive_clip_init: 1.0,
            adaptive_clip_lr: 0.2,
        }
    }
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Copy)]
pub enum NoiseMechanism {
    /// Gaussian noise mechanism
    Gaussian,
    /// Laplace noise mechanism  
    Laplace,
    /// Tree aggregation with Gaussian noise
    TreeAggregation,
    /// Improved composition with amplification
    ImprovedComposition,
}

/// Privacy budget tracking information
#[derive(Debug, Clone)]
pub struct PrivacyBudget {
    /// Current epsilon consumed
    pub epsilon_consumed: f64,

    /// Current delta consumed
    pub delta_consumed: f64,

    /// Remaining epsilon budget
    pub epsilon_remaining: f64,

    /// Remaining delta budget
    pub delta_remaining: f64,

    /// Number of steps taken
    pub steps_taken: usize,

    /// Privacy accounting method used
    pub accounting_method: AccountingMethod,

    /// Estimated steps until budget exhaustion
    pub estimated_steps_remaining: usize,
}

/// Privacy accounting methods
#[derive(Debug, Clone, Copy)]
pub enum AccountingMethod {
    /// Moments accountant (most accurate)
    MomentsAccountant,
    /// Renyi differential privacy
    RenyiDP,
    /// Advanced composition
    AdvancedComposition,
    /// Zero-concentrated differential privacy
    ZCDP,
}

/// Differentially private optimizer wrapper
pub struct DifferentiallyPrivateOptimizer<O, A, D>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
    O: Optimizer<A, D>,
{
    /// Base optimizer
    base_optimizer: O,

    /// Privacy configuration
    config: DifferentialPrivacyConfig,

    /// Moment accountant for privacy tracking
    accountant: MomentsAccountant,

    /// Random number generator for noise
    rng: scirs2_core::random::Random,

    /// Adaptive clipping state
    adaptive_clip_state: Option<AdaptiveClippingState>,

    /// Gradient history for analysis
    gradient_history: VecDeque<GradientNorms>,

    /// Privacy audit trail
    audit_trail: Vec<PrivacyEvent>,

    /// Current step count
    step_count: usize,

    /// Phantom data for unused type parameters
    _phantom: std::marker::PhantomData<(A, D)>,
}

/// Adaptive clipping state
#[derive(Debug, Clone)]
struct AdaptiveClippingState {
    current_threshold: f64,
    quantile_estimate: f64,
    update_frequency: usize,
    last_update_step: usize,
}

/// Gradient norm statistics
#[derive(Debug, Clone)]
struct GradientNorms {
    step: usize,
    pre_clip_norm: f64,
    post_clip_norm: f64,
    clipping_ratio: f64,
}

/// Privacy event for audit trail
#[derive(Debug, Clone)]
pub struct PrivacyEvent {
    step: usize,
    event_type: PrivacyEventType,
    epsilon_spent: f64,
    delta_spent: f64,
    noise_scale: f64,
}

#[derive(Debug, Clone)]
enum PrivacyEventType {
    GradientRelease,
    ModelUpdate,
    ParameterQuery,
    AdaptiveClipUpdate,
}

impl<O, A, D> DifferentiallyPrivateOptimizer<O, A, D>
where
    A: Float
        + rand_distr::uniform::SampleUniform
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Send
        + Sync
        + ndarray::ScalarOperand
        + std::fmt::Debug,
    D: Dimension,
    O: Optimizer<A, D>,
{
    /// Create a new differentially private optimizer
    pub fn new(baseoptimizer: O, config: DifferentialPrivacyConfig) -> Result<Self> {
        let accountant = MomentsAccountant::new(
            config.noise_multiplier,
            config.target_delta,
            config.batch_size,
            config.dataset_size,
        );

        let rng = scirs2_core::random::rng();

        let adaptive_clip_state = if config.adaptive_clipping {
            Some(AdaptiveClippingState {
                current_threshold: config.adaptive_clip_init,
                quantile_estimate: config.l2_norm_clip,
                update_frequency: 50, // Update every 50 steps
                last_update_step: 0,
            })
        } else {
            None
        };

        Ok(Self {
            base_optimizer: baseoptimizer,
            config,
            accountant,
            rng,
            adaptive_clip_state,
            gradient_history: VecDeque::with_capacity(1000),
            audit_trail: Vec::new(),
            step_count: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Perform a differentially private step
    pub fn dp_step(
        &mut self,
        params: &Array<A, D>,
        gradients: &mut Array<A, D>,
    ) -> Result<Array<A, D>> {
        self.step_count += 1;

        // Check privacy budget
        if !self.has_privacy_budget()? {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.get_privacy_budget().epsilon_consumed,
                target_epsilon: self.config.target_epsilon,
            });
        }

        // Compute gradient norm before clipping
        let pre_clip_norm = self.compute_l2_norm(gradients);

        // Apply gradient clipping
        let clip_threshold = self.get_clipping_threshold();
        let clipping_ratio = if pre_clip_norm > clip_threshold {
            let scale = clip_threshold / pre_clip_norm;
            gradients.mapv_inplace(|g| g * A::from(scale).unwrap());
            scale
        } else {
            1.0
        };

        let post_clip_norm = self.compute_l2_norm(gradients);

        // Add calibrated noise
        self.add_calibrated_noise(gradients, clip_threshold)?;

        // Update moment accountant
        let (epsilon_spent, delta_spent) = self.accountant.get_privacy_spent(self.step_count)?;

        // Record gradient statistics
        self.gradient_history.push_back(GradientNorms {
            step: self.step_count,
            pre_clip_norm,
            post_clip_norm,
            clipping_ratio,
        });

        if self.gradient_history.len() > 1000 {
            self.gradient_history.pop_front();
        }

        // Record privacy event
        self.audit_trail.push(PrivacyEvent {
            step: self.step_count,
            event_type: PrivacyEventType::GradientRelease,
            epsilon_spent,
            delta_spent,
            noise_scale: self.config.noise_multiplier * clip_threshold,
        });

        // Update adaptive clipping if enabled
        if let Some(ref mut state) = self.adaptive_clip_state {
            if self.step_count - state.last_update_step >= state.update_frequency {
                state.last_update_step = self.step_count;
                // Update threshold based on recent gradient norms
                let target_ratio = 0.8; // Target 80% of gradients to be clipped
                let new_threshold = pre_clip_norm * target_ratio;
                state.current_threshold = new_threshold;
            }
        }

        // Apply base optimizer step
        let updated_params = self.base_optimizer.step(params, gradients)?;

        Ok(updated_params)
    }

    /// Check if privacy budget is available
    pub fn has_privacy_budget(&self) -> Result<bool> {
        let budget = self.get_privacy_budget();
        Ok(budget.epsilon_remaining > 0.0 && budget.delta_remaining > 0.0)
    }

    /// Get current privacy budget status
    pub fn get_privacy_budget(&self) -> PrivacyBudget {
        let (epsilon_consumed, delta_consumed) = self
            .accountant
            .get_privacy_spent(self.step_count)
            .unwrap_or((0.0, 0.0));

        let epsilon_remaining = (self.config.target_epsilon - epsilon_consumed).max(0.0);
        let delta_remaining = (self.config.target_delta - delta_consumed).max(0.0);

        // Estimate remaining steps
        let epsilon_per_step = if self.step_count > 0 {
            epsilon_consumed / self.step_count as f64
        } else {
            0.0
        };

        let estimated_steps_remaining = if epsilon_per_step > 0.0 {
            (epsilon_remaining / epsilon_per_step) as usize
        } else {
            usize::MAX
        };

        PrivacyBudget {
            epsilon_consumed,
            delta_consumed,
            epsilon_remaining,
            delta_remaining,
            steps_taken: self.step_count,
            accounting_method: AccountingMethod::MomentsAccountant,
            estimated_steps_remaining,
        }
    }

    fn compute_l2_norm<S, DIM>(&self, array: &ArrayBase<S, DIM>) -> f64
    where
        S: Data<Elem = A>,
        DIM: Dimension,
    {
        array
            .iter()
            .map(|&x| {
                let val = x.to_f64().unwrap_or(0.0);
                val * val
            })
            .sum::<f64>()
            .sqrt()
    }

    fn get_clipping_threshold(&self) -> f64 {
        if let Some(ref state) = self.adaptive_clip_state {
            state.current_threshold
        } else {
            self.config.l2_norm_clip
        }
    }

    fn add_calibrated_noise<S, DIM>(
        &mut self,
        gradients: &mut ArrayBase<S, DIM>,
        clip_threshold: f64,
    ) -> Result<()>
    where
        S: DataMut<Elem = A>,
        DIM: Dimension,
    {
        let noise_scale = self.config.noise_multiplier * clip_threshold;

        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let sigma_f64 = noise_scale.to_f64().unwrap_or(1.0);
                gradients.mapv_inplace(|g| {
                    // Use Box-Muller transformation for Gaussian noise
                    let u1: f64 = self.rng.gen_range(0.0..1.0);
                    let u2: f64 = self.rng.gen_range(0.0..1.0);
                    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let noise = A::from(z0 * sigma_f64).unwrap();
                    g + noise
                });
            }
            NoiseMechanism::Laplace => {
                // Implement Laplace distribution using transformation method
                let scale_f64 = noise_scale.to_f64().unwrap_or(1.0);
                gradients.mapv_inplace(|g| {
                    let u: f64 = self.rng.gen_range(0.0..1.0);
                    let laplace_sample = if u < 0.5 {
                        scale_f64 * (2.0 * u).ln()
                    } else {
                        -scale_f64 * (2.0 * (1.0 - u)).ln()
                    };
                    let noise = A::from(laplace_sample).unwrap();
                    g + noise
                });
            }
            _ => {
                // Use Gaussian as fallback
                let sigma_f64 = noise_scale.to_f64().unwrap_or(1.0);
                gradients.mapv_inplace(|g| {
                    let u1: f64 = self.rng.gen_range(0.0..1.0);
                    let u2: f64 = self.rng.gen_range(0.0..1.0);
                    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let noise = A::from(z0 * sigma_f64).unwrap();
                    g + noise
                });
            }
        }

        Ok(())
    }

    fn update_adaptive_clipping(&mut self, state: &mut AdaptiveClippingState, current_norm: f64) {
        // Use exponential moving average to track gradient _norm quantiles
        let alpha = self.config.adaptive_clip_lr;

        // Target the 50th percentile of gradient norms
        let target_quantile = 0.5;

        // Update quantile estimate
        if current_norm > state.quantile_estimate {
            state.quantile_estimate += alpha * target_quantile;
        } else {
            state.quantile_estimate -= alpha * (1.0 - target_quantile);
        }

        // Update clipping threshold
        state.current_threshold = state.quantile_estimate;
        state.last_update_step = self.step_count;
    }

    /// Get gradient clipping statistics
    pub fn get_clipping_stats(&self) -> ClippingStats {
        if self.gradient_history.is_empty() {
            return ClippingStats::default();
        }

        let total_steps = self.gradient_history.len();
        let clipped_steps = self
            .gradient_history
            .iter()
            .filter(|stats| stats.clipping_ratio < 1.0)
            .count();

        let avg_clipping_ratio: f64 = self
            .gradient_history
            .iter()
            .map(|stats| stats.clipping_ratio)
            .sum::<f64>()
            / total_steps as f64;

        let avg_pre_clip_norm: f64 = self
            .gradient_history
            .iter()
            .map(|stats| stats.pre_clip_norm)
            .sum::<f64>()
            / total_steps as f64;

        ClippingStats {
            total_steps,
            clipped_steps,
            clipping_frequency: clipped_steps as f64 / total_steps as f64,
            avg_clipping_ratio,
            avg_pre_clip_norm,
            current_threshold: self.get_clipping_threshold(),
        }
    }

    /// Get privacy audit trail
    pub fn get_audit_trail(&self) -> &[PrivacyEvent] {
        &self.audit_trail
    }

    /// Validate privacy guarantees
    pub fn validate_privacy(&self) -> PrivacyValidation {
        let budget = self.get_privacy_budget();
        let clipping_stats = self.get_clipping_stats();

        let mut warnings = Vec::new();
        let mut is_valid = true;

        // Check if privacy budget is exceeded
        if budget.epsilon_consumed > self.config.target_epsilon {
            warnings.push("Epsilon budget exceeded".to_string());
            is_valid = false;
        }

        if budget.delta_consumed > self.config.target_delta {
            warnings.push("Delta budget exceeded".to_string());
            is_valid = false;
        }

        // Check clipping frequency
        if clipping_stats.clipping_frequency < 0.1 {
            warnings.push(
                "Low clipping frequency may indicate sub-optimal privacy-utility tradeoff"
                    .to_string(),
            );
        }

        if clipping_stats.clipping_frequency > 0.9 {
            warnings.push("High clipping frequency may severely impact utility".to_string());
        }

        PrivacyValidation {
            is_valid,
            budget: budget.clone(),
            clipping_stats: clipping_stats.clone(),
            warnings,
            recommendations: self.generate_recommendations(&budget, &clipping_stats),
        }
    }

    fn generate_recommendations(
        &self,
        budget: &PrivacyBudget,
        clipping: &ClippingStats,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if clipping.clipping_frequency > 0.8 {
            recommendations.push("Consider increasing the clipping threshold".to_string());
        }

        if clipping.clipping_frequency < 0.2 {
            recommendations.push("Consider decreasing the clipping threshold".to_string());
        }

        if budget.epsilon_remaining < budget.epsilon_consumed * 0.1 {
            recommendations.push("Privacy budget nearly exhausted - consider reducing noise multiplier for remaining steps".to_string());
        }

        recommendations
    }
}

/// Gradient clipping statistics
#[derive(Debug, Clone)]
pub struct ClippingStats {
    pub total_steps: usize,
    pub clipped_steps: usize,
    pub clipping_frequency: f64,
    pub avg_clipping_ratio: f64,
    pub avg_pre_clip_norm: f64,
    pub current_threshold: f64,
}

impl Default for ClippingStats {
    fn default() -> Self {
        Self {
            total_steps: 0,
            clipped_steps: 0,
            clipping_frequency: 0.0,
            avg_clipping_ratio: 1.0,
            avg_pre_clip_norm: 0.0,
            current_threshold: 1.0,
        }
    }
}

/// Privacy validation results
#[derive(Debug, Clone)]
pub struct PrivacyValidation {
    pub is_valid: bool,
    pub budget: PrivacyBudget,
    pub clipping_stats: ClippingStats,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Moments accountant for privacy tracking
pub struct MomentsAccountant {
    noise_multiplier: f64,
    target_delta: f64,
    batch_size: usize,
    dataset_size: usize,
    sampling_probability: f64,
}

impl MomentsAccountant {
    pub fn new(
        noise_multiplier: f64,
        target_delta: f64,
        batch_size: usize,
        dataset_size: usize,
    ) -> Self {
        let sampling_probability = batch_size as f64 / dataset_size as f64;

        Self {
            noise_multiplier,
            target_delta,
            batch_size,
            dataset_size,
            sampling_probability,
        }
    }

    /// Compute privacy cost for given number of steps
    pub fn get_privacy_spent(&self, steps: usize) -> Result<(f64, f64)> {
        if steps == 0 {
            return Ok((0.0, 0.0));
        }

        // Simplified moments accountant calculation
        // In practice, this would use the full moment generating function

        let sigma = self.noise_multiplier;
        let q = self.sampling_probability;
        let t = steps as f64;

        // Gaussian mechanism with subsampling
        let alpha_max = 32.0; // Maximum order for moment computation
        let log_moments = self.compute_log_moments(sigma, q, t, alpha_max);

        // Convert to (epsilon, delta)-DP
        let epsilon = self.compute_epsilon_from_moments(&log_moments, self.target_delta);
        let delta = self.target_delta;

        Ok((epsilon, delta))
    }

    fn compute_log_moments(&self, sigma: f64, q: f64, t: f64, alphamax: f64) -> Vec<f64> {
        let mut log_moments = Vec::new();

        for alpha_int in 2..=(alphamax as usize) {
            let alpha = alpha_int as f64;

            // Log moment for Gaussian mechanism with subsampling
            let log_moment = t
                * (q * q * alpha * (alpha - 1.0) / (2.0 * sigma * sigma))
                    .exp()
                    .ln();

            log_moments.push(log_moment);
        }

        log_moments
    }

    fn compute_epsilon_from_moments(&self, logmoments: &[f64], delta: f64) -> f64 {
        let mut min_epsilon = f64::INFINITY;

        for (i, &log_moment) in logmoments.iter().enumerate() {
            let alpha = (i + 2) as f64;
            let epsilon = (log_moment - delta.ln()) / (alpha - 1.0);

            if epsilon < min_epsilon {
                min_epsilon = epsilon;
            }
        }

        min_epsilon.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_dp_config_default() {
        let config = DifferentialPrivacyConfig::default();
        assert_eq!(config.target_epsilon, 1.0);
        assert_eq!(config.noise_multiplier, 1.1);
        assert!(matches!(config.noise_mechanism, NoiseMechanism::Gaussian));
    }

    #[test]
    fn test_moments_accountant() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);

        let (epsilon, delta) = accountant.get_privacy_spent(100).unwrap();
        assert!(epsilon > 0.0);
        assert_eq!(delta, 1e-5);

        let (epsilon2, _) = accountant.get_privacy_spent(200).unwrap();
        assert!(epsilon2 > epsilon); // More steps should consume more budget
    }

    #[test]
    fn test_dp_optimizer_creation() {
        let sgd = SGD::new(0.01);
        let dp_config = DifferentialPrivacyConfig::default();

        let dp_optimizer =
            DifferentiallyPrivateOptimizer::<_, f64, ndarray::Ix1>::new(sgd, dp_config);
        assert!(dp_optimizer.is_ok());
    }

    #[test]
    fn test_privacy_budget_tracking() {
        let sgd = SGD::new(0.01);
        let dp_config = DifferentialPrivacyConfig {
            target_epsilon: 1.0,
            max_steps: 100,
            ..Default::default()
        };

        let dp_optimizer: DifferentiallyPrivateOptimizer<SGD<f64>, f64, ndarray::Ix1> =
            DifferentiallyPrivateOptimizer::new(sgd, dp_config).unwrap();
        let budget = dp_optimizer.get_privacy_budget();

        assert_eq!(budget.epsilon_consumed, 0.0);
        assert_eq!(budget.epsilon_remaining, 1.0);
        assert_eq!(budget.steps_taken, 0);
    }
}
