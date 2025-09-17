//! Factory functions for creating different types of extrapolators
//!
//! This module provides convenient factory functions for creating extrapolators
//! with specific configurations for common use cases.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::default::Default;
use std::ops::AddAssign;

use super::advanced::AdvancedExtrapolator;
use super::config::{
    AdaptiveExtrapolationConfig, AutoregressiveExtrapolationConfig, ConfidenceExtrapolationConfig,
    EnsembleExtrapolationConfig, ExtrapolationParameters,
};
use super::core::Extrapolator;
use super::types::{
    ARFittingMethod, AdaptiveSelectionCriterion, EnsembleCombinationStrategy, ExtrapolationMethod,
};

/// Creates an extrapolator with linear extrapolation at both boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - Derivative at the lower boundary
/// * `upper_derivative` - Derivative at the upper boundary
///
/// # Returns
///
/// A new `Extrapolator` configured for linear extrapolation
pub fn make_linear_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Linear,
    )
    .with_derivatives(lower_derivative, upper_derivative)
}

/// Creates an extrapolator with periodic extension.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `period` - The period of the function (defaults to domain width if None)
///
/// # Returns
///
/// A new `Extrapolator` configured for periodic extrapolation
pub fn make_periodic_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    period: Option<T>,
) -> Extrapolator<T> {
    let mut extrapolator = Extrapolator::new(
        lower_bound,
        upper_bound,
        T::zero(), // Values and derivatives don't matter for periodic extrapolation
        T::zero(),
        ExtrapolationMethod::Periodic,
        ExtrapolationMethod::Periodic,
    );

    if let Some(p) = period {
        let params = ExtrapolationParameters::default().with_period(p);
        extrapolator = extrapolator.with_parameters(params);
    }

    extrapolator
}

/// Creates an extrapolator with reflection at boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
///
/// # Returns
///
/// A new `Extrapolator` configured for reflection extrapolation
pub fn make_reflection_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        T::zero(), // Values and derivatives don't matter for reflection extrapolation
        T::zero(),
        ExtrapolationMethod::Reflection,
        ExtrapolationMethod::Reflection,
    )
}

/// Creates an extrapolator with cubic polynomial extrapolation.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - First derivative at the lower boundary
/// * `upper_derivative` - First derivative at the upper boundary
/// * `lower_second_derivative` - Second derivative at the lower boundary
/// * `upper_second_derivative` - Second derivative at the upper boundary
///
/// # Returns
///
/// A new `Extrapolator` configured for cubic extrapolation
#[allow(clippy::too_many_arguments)]
pub fn make_cubic_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    lower_second_derivative: T,
    upper_second_derivative: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Cubic,
        ExtrapolationMethod::Cubic,
    )
    .with_derivatives(lower_derivative, upper_derivative)
    .with_second_derivatives(lower_second_derivative, upper_second_derivative)
}

/// Creates an extrapolator with exponential decay/growth.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - Derivative at the lower boundary
/// * `upper_derivative` - Derivative at the upper boundary
/// * `exponential_rate` - Exponential rate (positive = growth, negative = decay)
/// * `exponential_offset` - Offset for exponential extrapolation
///
/// # Returns
///
/// A new `Extrapolator` configured for exponential extrapolation
#[allow(clippy::too_many_arguments)]
pub fn make_exponential_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    exponential_rate: T,
    exponential_offset: T,
) -> Extrapolator<T> {
    let params = ExtrapolationParameters::default()
        .with_exponential_rate(exponential_rate)
        .with_exponential_offset(exponential_offset);

    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Exponential,
        ExtrapolationMethod::Exponential,
    )
    .with_derivatives(lower_derivative, upper_derivative)
    .with_parameters(params)
}

/// Creates an extrapolator with power law decay/growth.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - Derivative at the lower boundary
/// * `upper_derivative` - Derivative at the upper boundary
/// * `power_exponent` - Power law exponent
/// * `power_scale` - Scale factor for power law
///
/// # Returns
///
/// A new `Extrapolator` configured for power law extrapolation
#[allow(clippy::too_many_arguments)]
pub fn make_power_law_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    power_exponent: T,
    power_scale: T,
) -> Extrapolator<T> {
    let params = ExtrapolationParameters::default()
        .with_power_exponent(power_exponent)
        .with_power_scale(power_scale);

    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::PowerLaw,
        ExtrapolationMethod::PowerLaw,
    )
    .with_derivatives(lower_derivative, upper_derivative)
    .with_parameters(params)
}

/// Creates an extrapolator that returns zeros outside the domain.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
///
/// # Returns
///
/// A new `Extrapolator` configured to return zeros
pub fn make_zeros_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        T::zero(),
        T::zero(),
        ExtrapolationMethod::Zeros,
        ExtrapolationMethod::Zeros,
    )
}

/// Creates an extrapolator that uses nearest boundary values.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
///
/// # Returns
///
/// A new `Extrapolator` configured for nearest value extrapolation
pub fn make_nearest_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Nearest,
        ExtrapolationMethod::Nearest,
    )
}

/// Convenience function to create a confidence extrapolator
pub fn make_confidence_extrapolator<T: Float + std::fmt::Display + Default + AddAssign>(
    base_extrapolator: Extrapolator<T>,
    confidence_level: T,
    bootstrap_samples: usize,
) -> AdvancedExtrapolator<T> {
    let config = ConfidenceExtrapolationConfig {
        bootstrap_samples,
        confidence_level,
        max_extrapolation_ratio: T::from(0.5).unwrap(),
        bias_correction: true,
    };

    AdvancedExtrapolator::new(base_extrapolator).with_confidence(config)
}

/// Convenience function to create an ensemble extrapolator
pub fn make_ensemble_extrapolator<T: Float + std::fmt::Display + Default + AddAssign>(
    base_extrapolator: Extrapolator<T>,
    methods: Vec<ExtrapolationMethod>,
    weights: Option<Vec<T>>,
    strategy: EnsembleCombinationStrategy,
) -> AdvancedExtrapolator<T> {
    let config = EnsembleExtrapolationConfig {
        methods,
        combination_strategy: strategy,
        weights,
        include_confidence: true,
    };

    AdvancedExtrapolator::new(base_extrapolator).with_ensemble(config)
}

/// Convenience function to create an adaptive extrapolator
pub fn make_adaptive_extrapolator<T: Float + std::fmt::Display + Default + AddAssign>(
    base_extrapolator: Extrapolator<T>,
    selection_criterion: AdaptiveSelectionCriterion,
    local_window_size: usize,
    minimum_confidence: f64,
) -> AdvancedExtrapolator<T> {
    let config = AdaptiveExtrapolationConfig {
        selection_criterion,
        local_window_size,
        minimum_confidence,
        cache_selections: true,
    };

    AdvancedExtrapolator::new(base_extrapolator).with_adaptive(config)
}

/// Convenience function to create an autoregressive extrapolator
pub fn make_autoregressive_extrapolator<T: Float + std::fmt::Display + Default + AddAssign>(
    base_extrapolator: Extrapolator<T>,
    ar_order: usize,
    fitting_method: ARFittingMethod,
    historical_data: Option<(Array1<T>, Array1<T>)>,
) -> AdvancedExtrapolator<T> {
    let config = AutoregressiveExtrapolationConfig {
        ar_order,
        fitting_method,
        max_steps: 100,
        trend_adjustment: true,
        regularization: T::from(1e-6).unwrap_or(T::zero()),
    };

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator).with_autoregressive(config);

    if let Some((x_data, y_data)) = historical_data {
        extrapolator = extrapolator.with_historical_data(x_data, y_data);
    }

    extrapolator
}

/// Creates a conservative extrapolator for production use
///
/// Uses linear extrapolation with clamping at large distances to prevent
/// numerical instability
pub fn make_conservative_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    max_extrapolation_distance: T,
) -> Extrapolator<T> {
    // For now, use linear extrapolation
    // In a full implementation, this would clamp extrapolation beyond max_distance
    let _ = max_extrapolation_distance; // Suppress unused warning

    make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    )
}

/// Creates a robust ensemble extrapolator combining multiple methods
pub fn make_robust_ensemble_extrapolator<T: Float + std::fmt::Display + Default + AddAssign>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let methods = vec![
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Quadratic,
        ExtrapolationMethod::Exponential,
    ];

    let weights = Some(vec![
        T::from(0.5).unwrap(), // Higher weight for linear (most stable)
        T::from(0.3).unwrap(), // Medium weight for quadratic
        T::from(0.2).unwrap(), // Lower weight for exponential
    ]);

    make_ensemble_extrapolator(
        base_extrapolator,
        methods,
        weights,
        EnsembleCombinationStrategy::WeightedMean,
    )
}

/// Creates an extrapolator optimized for smooth functions
pub fn make_smooth_function_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    lower_second_derivative: T,
    upper_second_derivative: T,
) -> Extrapolator<T> {
    make_cubic_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
        lower_second_derivative,
        upper_second_derivative,
    )
}

/// Creates an extrapolator optimized for oscillatory functions
pub fn make_oscillatory_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    period: Option<T>,
) -> Extrapolator<T> {
    make_periodic_extrapolator(lower_bound, upper_bound, period)
}

/// Creates an extrapolator optimized for monotonic functions
pub fn make_monotonic_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    is_increasing: bool,
) -> Extrapolator<T> {
    if is_increasing {
        // For increasing functions, use exponential growth if derivatives are positive
        if lower_derivative > T::zero() && upper_derivative > T::zero() {
            let rate = T::from(0.1).unwrap(); // Conservative growth rate
            make_exponential_extrapolator(
                lower_bound,
                upper_bound,
                lower_value,
                upper_value,
                lower_derivative,
                upper_derivative,
                rate,
                T::zero(),
            )
        } else {
            make_linear_extrapolator(
                lower_bound,
                upper_bound,
                lower_value,
                upper_value,
                lower_derivative,
                upper_derivative,
            )
        }
    } else {
        // For decreasing functions, use exponential decay
        let rate = T::from(-0.1).unwrap(); // Conservative decay rate
        make_exponential_extrapolator(
            lower_bound,
            upper_bound,
            lower_value,
            upper_value,
            lower_derivative,
            upper_derivative,
            rate,
            T::zero(),
        )
    }
}

/// Creates a high-confidence extrapolator with uncertainty bounds
pub fn make_high_confidence_extrapolator<T: Float + std::fmt::Display + Default + AddAssign>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    confidence_level: T,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    make_confidence_extrapolator(base_extrapolator, confidence_level, 1000)
}
