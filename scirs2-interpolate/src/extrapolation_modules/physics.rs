//! Physics-informed extrapolation methods
//!
//! This module provides extrapolation methods that incorporate physical laws
//! and boundary conditions to ensure physically realistic extrapolation.

use scirs2_core::numeric::{Float, FromPrimitive};
use std::default::Default;
use std::ops::AddAssign;

use super::advanced::AdvancedExtrapolator;
use super::core::Extrapolator;
use super::factory::make_linear_extrapolator;
use super::types::{BoundaryType, DataCharacteristics, ExtrapolationMethod, PhysicsLaw};

/// Physics-informed extrapolation that respects physical laws
#[allow(clippy::too_many_arguments)]
pub fn make_physics_informed_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    physics_law: PhysicsLaw,
) -> AdvancedExtrapolator<T> {
    // Create base linear extrapolator
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Configure physics-based constraints
    match physics_law {
        PhysicsLaw::MassConservation => {
            // Ensure non-negative extrapolation for mass quantities
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.parameters.exponential_rate = T::from(-0.1).unwrap();
            // Decay rate
        }
        PhysicsLaw::EnergyConservation => {
            // Energy-conserving polynomial extrapolation
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
        }
        PhysicsLaw::MomentumConservation => {
            // Linear momentum conservation
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
    }

    extrapolator
}

/// Boundary condition preserving extrapolation
#[allow(clippy::too_many_arguments)]
pub fn make_boundary_preserving_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    boundary_type: BoundaryType,
) -> AdvancedExtrapolator<T> {
    // Create base extrapolator with appropriate methods
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    match boundary_type {
        BoundaryType::Dirichlet => {
            // Fixed value boundaries - use cubic for smooth transition
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Cubic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Cubic;
        }
        BoundaryType::Neumann => {
            // Fixed derivative boundaries - use quadratic
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
        }
        BoundaryType::Robin => {
            // Mixed boundaries - use linear combination
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
        BoundaryType::Absorbing => {
            // Absorbing boundaries - exponential decay
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.parameters.exponential_rate = T::from(-1.0).unwrap();
        }
    }

    extrapolator
}

/// Adaptive extrapolation that selects method based on local data characteristics
#[allow(clippy::too_many_arguments)]
pub fn make_smart_adaptive_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    data_characteristics: &DataCharacteristics<T>,
) -> AdvancedExtrapolator<T> {
    // Create base extrapolator with appropriate methods
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Select extrapolation method based on data analysis
    if data_characteristics.is_periodic {
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Periodic;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Periodic;
        extrapolator.base_extrapolator.parameters.period = data_characteristics
            .estimated_period
            .unwrap_or_else(|| T::from(2.0 * std::f64::consts::PI).unwrap());
    } else if data_characteristics.is_monotonic {
        if data_characteristics.is_exponential_like {
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;
        } else {
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
    } else if data_characteristics.is_oscillatory {
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Sinusoidal;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Sinusoidal;
    } else {
        // Default to quadratic for smooth data
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
    }

    extrapolator
}

/// Creates extrapolator for conservation laws (mass, energy, momentum)
pub fn make_conservation_law_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    conservation_type: PhysicsLaw,
    is_conserved_quantity: bool,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    match (conservation_type, is_conserved_quantity) {
        (PhysicsLaw::MassConservation, true) => {
            // Mass must remain non-negative and tend to zero at infinity
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.parameters.exponential_rate = T::from(-0.5).unwrap();
        }
        (PhysicsLaw::EnergyConservation, true) => {
            // Total energy should be conserved (polynomial behavior)
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
        }
        (PhysicsLaw::MomentumConservation, true) => {
            // Linear momentum conservation implies linear behavior
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
        _ => {
            // For non-conserved quantities, use more flexible methods
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Cubic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Cubic;
        }
    }

    extrapolator
}

/// Creates extrapolator for wave equations with appropriate dispersion
pub fn make_wave_equation_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    wave_speed: T,
    frequency: Option<T>,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    if let Some(freq) = frequency {
        // For known frequency, use sinusoidal extrapolation
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Sinusoidal;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Sinusoidal;

        // Set period based on frequency and wave speed
        let wavelength = wave_speed / freq;
        extrapolator.base_extrapolator.parameters.period = wavelength;
    } else {
        // For unknown frequency, use periodic with estimated period
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Periodic;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Periodic;

        // Estimate period from domain size
        let domain_width = upper_bound - lower_bound;
        extrapolator.base_extrapolator.parameters.period = domain_width;
    }

    extrapolator
}

/// Creates extrapolator for diffusion equations with exponential decay
pub fn make_diffusion_equation_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    diffusion_coefficient: T,
    time_scale: T,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Diffusion typically leads to exponential decay
    extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
    extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;

    // Set decay rate based on diffusion coefficient and time scale
    let decay_rate = -diffusion_coefficient / time_scale;
    extrapolator.base_extrapolator.parameters.exponential_rate = decay_rate;

    extrapolator
}

/// Creates extrapolator for boundary layer problems with rapid transitions
pub fn make_boundary_layer_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    layer_thickness: T,
    outer_value: T,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Boundary layers typically have exponential transitions to outer values
    extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
    extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;

    // Set decay rate based on boundary layer thickness
    let decay_rate = T::one() / layer_thickness;
    extrapolator.base_extrapolator.parameters.exponential_rate = -decay_rate;
    extrapolator.base_extrapolator.parameters.exponential_offset = outer_value;

    extrapolator
}

/// Creates extrapolator that enforces physical constraints (positivity, bounds)
pub fn make_constrained_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + Default + AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    min_value: Option<T>,
    max_value: Option<T>,
    enforce_monotonicity: bool,
) -> AdvancedExtrapolator<T> {
    let base_extrapolator = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Choose methods based on constraints
    if let Some(min_val) = min_value {
        if min_val >= T::zero() {
            // Enforce non-negativity with exponential decay
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;
        }
    }

    if enforce_monotonicity {
        // Use linear extrapolation to preserve monotonicity
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
    }

    if max_value.is_some() || min_value.is_some() {
        // Use clamped extrapolation for bounded quantities
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Clamped;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Clamped;
    }

    extrapolator
}

/// Analyze data characteristics for physics-informed extrapolation
pub fn analyze_physics_characteristics<T: Float + FromPrimitive>(
    values: &[T],
    gradients: Option<&[T]>,
    second_derivatives: Option<&[T]>,
) -> DataCharacteristics<T> {
    let mut characteristics = DataCharacteristics::new();

    if values.is_empty() {
        return characteristics;
    }

    // Analyze monotonicity
    if values.len() > 1 {
        let mut increasing_count = 0;
        let mut decreasing_count = 0;

        for i in 1..values.len() {
            if values[i] > values[i - 1] {
                increasing_count += 1;
            } else if values[i] < values[i - 1] {
                decreasing_count += 1;
            }
        }

        let total_transitions = increasing_count + decreasing_count;
        characteristics.is_monotonic = total_transitions > 0
            && (increasing_count as f64 / total_transitions as f64 > 0.9
                || decreasing_count as f64 / total_transitions as f64 > 0.9);
    }

    // Analyze for exponential-like behavior
    if values.len() > 2 {
        let mut ratios = Vec::new();
        for i in 2..values.len() {
            if values[i - 1] != T::zero() && values[i - 2] != T::zero() {
                let ratio1 = values[i] / values[i - 1];
                let ratio2 = values[i - 1] / values[i - 2];
                if ratio1 != T::zero() && ratio2 != T::zero() {
                    ratios.push((ratio1 / ratio2 - T::one()).abs());
                }
            }
        }

        if !ratios.is_empty() {
            let avg_ratio_variation: T =
                ratios.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(ratios.len()).unwrap();
            characteristics.is_exponential_like = avg_ratio_variation < T::from(0.1).unwrap();
        }
    }

    // Simple oscillation detection
    if values.len() > 4 {
        let mut sign_changes = 0;
        if let Some(grads) = gradients {
            for i in 1..grads.len() {
                if (grads[i] > T::zero()) != (grads[i - 1] > T::zero()) {
                    sign_changes += 1;
                }
            }
            characteristics.is_oscillatory = sign_changes > 2;
        }
    }

    // Estimate characteristic scale
    let max_val = values
        .iter()
        .fold(values[0], |acc, &x| if x > acc { x } else { acc });
    let min_val = values
        .iter()
        .fold(values[0], |acc, &x| if x < acc { x } else { acc });
    characteristics.characteristic_scale = max_val - min_val;

    characteristics
}
