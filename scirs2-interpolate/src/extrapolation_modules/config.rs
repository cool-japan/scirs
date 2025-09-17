//! Configuration structures for extrapolation methods
//!
//! This module contains all configuration-related types used to customize
//! extrapolation behavior across different methods and scenarios.

use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;

use super::types::{ARFittingMethod, AdaptiveSelectionCriterion, EnsembleCombinationStrategy};

/// Parameters for specialized extrapolation methods
#[derive(Debug, Clone)]
pub struct ExtrapolationParameters<T: Float> {
    /// Decay/growth rate for exponential extrapolation
    pub exponential_rate: T,

    /// Offset for exponential extrapolation
    pub exponential_offset: T,

    /// Exponent for power law extrapolation
    pub power_exponent: T,

    /// Scale factor for power law extrapolation
    pub power_scale: T,

    /// Period for periodic extrapolation
    pub period: T,
}

impl<T: Float> Default for ExtrapolationParameters<T> {
    fn default() -> Self {
        Self {
            exponential_rate: T::one(),
            exponential_offset: T::zero(),
            power_exponent: -T::one(), // Default to 1/x decay
            power_scale: T::one(),
            period: T::from(2.0 * std::f64::consts::PI).unwrap(),
        }
    }
}

impl<T: Float> ExtrapolationParameters<T> {
    /// Creates default parameters for extrapolation methods
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the decay/growth rate for exponential extrapolation
    pub fn with_exponential_rate(mut self, rate: T) -> Self {
        self.exponential_rate = rate;
        self
    }

    /// Set the offset for exponential extrapolation
    pub fn with_exponential_offset(mut self, offset: T) -> Self {
        self.exponential_offset = offset;
        self
    }

    /// Set the exponent for power law extrapolation
    pub fn with_power_exponent(mut self, exponent: T) -> Self {
        self.power_exponent = exponent;
        self
    }

    /// Set the scale factor for power law extrapolation
    pub fn with_power_scale(mut self, scale: T) -> Self {
        self.power_scale = scale;
        self
    }

    /// Set the period for periodic extrapolation
    pub fn with_period(mut self, period: T) -> Self {
        self.period = period;
        self
    }
}

/// Configuration for confidence-based extrapolation
#[derive(Debug, Clone)]
pub struct ConfidenceExtrapolationConfig<T: Float> {
    /// Number of bootstrap samples for uncertainty estimation
    pub bootstrap_samples: usize,

    /// Confidence level (e.g., 0.95 for 95% confidence intervals)
    pub confidence_level: T,

    /// Maximum extrapolation distance as multiple of domain size
    pub max_extrapolation_ratio: T,

    /// Whether to use bias correction in bootstrap
    pub bias_correction: bool,
}

impl<T: Float> Default for ConfidenceExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            bootstrap_samples: 1000,
            confidence_level: T::from(0.95).unwrap(),
            max_extrapolation_ratio: T::from(0.5).unwrap(),
            bias_correction: true,
        }
    }
}

/// Result from confidence-based extrapolation
#[derive(Debug, Clone)]
pub struct ConfidenceExtrapolationResult<T: Float> {
    /// Predicted value
    pub value: T,
    /// Lower confidence bound
    pub lower_bound: T,
    /// Upper confidence bound
    pub upper_bound: T,
    /// Confidence level used
    pub confidence_level: T,
}

/// Configuration for ensemble extrapolation
#[derive(Debug, Clone)]
pub struct EnsembleExtrapolationConfig<T: Float> {
    /// Methods to include in the ensemble
    pub methods: Vec<super::types::ExtrapolationMethod>,
    /// Combination strategy
    pub combination_strategy: EnsembleCombinationStrategy,
    /// Weights for weighted combination (if applicable)
    pub weights: Option<Vec<T>>,
    /// Whether to include confidence estimation
    pub include_confidence: bool,
}

impl<T: Float> Default for EnsembleExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            methods: vec![
                super::types::ExtrapolationMethod::Linear,
                super::types::ExtrapolationMethod::Quadratic,
                super::types::ExtrapolationMethod::Cubic,
            ],
            combination_strategy: EnsembleCombinationStrategy::Mean,
            weights: None,
            include_confidence: true,
        }
    }
}

/// Configuration for adaptive extrapolation
#[derive(Debug, Clone)]
pub struct AdaptiveExtrapolationConfig {
    /// Selection criterion for choosing methods
    pub selection_criterion: AdaptiveSelectionCriterion,
    /// Number of nearby points to consider for local analysis
    pub local_window_size: usize,
    /// Minimum confidence required to use a method
    pub minimum_confidence: f64,
    /// Whether to cache method selections for performance
    pub cache_selections: bool,
}

impl Default for AdaptiveExtrapolationConfig {
    fn default() -> Self {
        Self {
            selection_criterion: AdaptiveSelectionCriterion::CrossValidationError,
            local_window_size: 10,
            minimum_confidence: 0.5,
            cache_selections: true,
        }
    }
}

/// Configuration for autoregressive extrapolation
#[derive(Debug, Clone)]
pub struct AutoregressiveExtrapolationConfig<T: Float> {
    /// Order of the autoregressive model
    pub ar_order: usize,
    /// Fitting method for AR parameters
    pub fitting_method: ARFittingMethod,
    /// Maximum number of time steps to extrapolate
    pub max_steps: usize,
    /// Whether to apply trend adjustment
    pub trend_adjustment: bool,
    /// Regularization parameter for AR fitting
    pub regularization: T,
}

impl<T: Float> Default for AutoregressiveExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            ar_order: 3,
            fitting_method: ARFittingMethod::YuleWalker,
            max_steps: 100,
            trend_adjustment: true,
            regularization: T::from(1e-6).unwrap_or(T::zero()),
        }
    }
}

/// Configuration builder for complex extrapolation setups
#[derive(Debug, Clone)]
pub struct ExtrapolationConfigBuilder<T: Float> {
    /// Base parameters
    pub parameters: ExtrapolationParameters<T>,
    /// Confidence configuration
    pub confidence_config: Option<ConfidenceExtrapolationConfig<T>>,
    /// Ensemble configuration
    pub ensemble_config: Option<EnsembleExtrapolationConfig<T>>,
    /// Adaptive configuration
    pub adaptive_config: Option<AdaptiveExtrapolationConfig>,
    /// Autoregressive configuration
    pub ar_config: Option<AutoregressiveExtrapolationConfig<T>>,
    /// Custom method-specific parameters
    pub custom_parameters: HashMap<String, T>,
}

impl<T: Float> Default for ExtrapolationConfigBuilder<T> {
    fn default() -> Self {
        Self {
            parameters: ExtrapolationParameters::default(),
            confidence_config: None,
            ensemble_config: None,
            adaptive_config: None,
            ar_config: None,
            custom_parameters: HashMap::new(),
        }
    }
}

impl<T: Float> ExtrapolationConfigBuilder<T> {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set base extrapolation parameters
    pub fn with_parameters(mut self, parameters: ExtrapolationParameters<T>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Enable confidence-based extrapolation
    pub fn with_confidence(mut self, config: ConfidenceExtrapolationConfig<T>) -> Self {
        self.confidence_config = Some(config);
        self
    }

    /// Enable ensemble extrapolation
    pub fn with_ensemble(mut self, config: EnsembleExtrapolationConfig<T>) -> Self {
        self.ensemble_config = Some(config);
        self
    }

    /// Enable adaptive extrapolation
    pub fn with_adaptive(mut self, config: AdaptiveExtrapolationConfig) -> Self {
        self.adaptive_config = Some(config);
        self
    }

    /// Enable autoregressive extrapolation
    pub fn with_autoregressive(mut self, config: AutoregressiveExtrapolationConfig<T>) -> Self {
        self.ar_config = Some(config);
        self
    }

    /// Add custom parameter
    pub fn with_custom_parameter(mut self, name: String, value: T) -> Self {
        self.custom_parameters.insert(name, value);
        self
    }

    /// Build the final configuration
    pub fn build(self) -> ExtrapolationConfig<T> {
        ExtrapolationConfig {
            parameters: self.parameters,
            confidence_config: self.confidence_config,
            ensemble_config: self.ensemble_config,
            adaptive_config: self.adaptive_config,
            ar_config: self.ar_config,
            custom_parameters: self.custom_parameters,
        }
    }
}

/// Complete extrapolation configuration
#[derive(Debug, Clone)]
pub struct ExtrapolationConfig<T: Float> {
    /// Base parameters
    pub parameters: ExtrapolationParameters<T>,
    /// Confidence configuration
    pub confidence_config: Option<ConfidenceExtrapolationConfig<T>>,
    /// Ensemble configuration
    pub ensemble_config: Option<EnsembleExtrapolationConfig<T>>,
    /// Adaptive configuration
    pub adaptive_config: Option<AdaptiveExtrapolationConfig>,
    /// Autoregressive configuration
    pub ar_config: Option<AutoregressiveExtrapolationConfig<T>>,
    /// Custom method-specific parameters
    pub custom_parameters: HashMap<String, T>,
}

impl<T: Float> Default for ExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            parameters: ExtrapolationParameters::default(),
            confidence_config: None,
            ensemble_config: None,
            adaptive_config: None,
            ar_config: None,
            custom_parameters: HashMap::new(),
        }
    }
}

impl<T: Float> ExtrapolationConfig<T> {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration builder
    pub fn builder() -> ExtrapolationConfigBuilder<T> {
        ExtrapolationConfigBuilder::new()
    }

    /// Check if confidence estimation is enabled
    pub fn has_confidence(&self) -> bool {
        self.confidence_config.is_some()
    }

    /// Check if ensemble methods are enabled
    pub fn has_ensemble(&self) -> bool {
        self.ensemble_config.is_some()
    }

    /// Check if adaptive selection is enabled
    pub fn has_adaptive(&self) -> bool {
        self.adaptive_config.is_some()
    }

    /// Check if autoregressive modeling is enabled
    pub fn has_autoregressive(&self) -> bool {
        self.ar_config.is_some()
    }

    /// Get custom parameter value
    pub fn get_custom_parameter(&self, name: &str) -> Option<&T> {
        self.custom_parameters.get(name)
    }
}
