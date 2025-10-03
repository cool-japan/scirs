//! Data analysis utilities for interpolation problem assessment
//!
//! This module provides specialized analysis functions for understanding
//! the characteristics of interpolation data and suggesting optimal approaches.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

use super::types::{BoundaryAnalysis, DataPointsAnalysis, FunctionValuesAnalysis};
use crate::error::{InterpolateError, InterpolateResult};

/// Suggest data-based regularization parameter
pub fn suggest_data_based_regularization<F>(min_distance: F, distance_ratio: F) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let base_reg = super::types::machine_epsilon::<F>()
        * F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap());

    // Adjust based on point spacing
    let spacing_factor = if min_distance > F::zero() {
        F::one() / min_distance.sqrt()
    } else {
        F::from(1e6).unwrap_or_else(|| F::from(1e6).unwrap())
    };

    // Adjust based on distance ratio (how uniform the spacing is)
    let ratio_factor = if distance_ratio > F::one() {
        distance_ratio.ln().max(F::one())
    } else {
        F::one()
    };

    base_reg * spacing_factor * ratio_factor
}

/// Comprehensive analysis of interpolation data characteristics
pub fn analyze_interpolation_data<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<InterpolationDataReport<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    if points.nrows() != values.len() {
        return Err(InterpolateError::ShapeMismatch {
            expected: format!("{} points", values.len()),
            actual: format!("{} points", points.nrows()),
            object: "data analysis".to_string(),
        });
    }

    let data_points = super::edge_cases::analyze_data_points(points)?;
    let function_values = super::edge_cases::analyze_function_values(values)?;
    let boundary = super::edge_cases::analyze_boundary_conditions(points, values)?;

    // Additional specialized analyses
    let scaling_analysis = analyze_data_scaling(points, values)?;
    let noise_analysis = analyze_noise_characteristics(values)?;
    let interpolation_method_recommendation =
        recommend_interpolation_method(&data_points, &function_values)?;

    Ok(InterpolationDataReport {
        data_points,
        function_values,
        boundary,
        scaling_analysis,
        noise_analysis,
        interpolation_method_recommendation,
    })
}

/// Complete interpolation data analysis report
#[derive(Debug, Clone)]
pub struct InterpolationDataReport<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    pub data_points: DataPointsAnalysis<F>,
    pub function_values: FunctionValuesAnalysis<F>,
    pub boundary: BoundaryAnalysis<F>,
    pub scaling_analysis: DataScalingAnalysis<F>,
    pub noise_analysis: NoiseAnalysis<F>,
    pub interpolation_method_recommendation: InterpolationMethodRecommendation,
}

/// Data scaling analysis
#[derive(Debug, Clone)]
pub struct DataScalingAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Scale factor for input coordinates
    pub coordinate_scale: F,
    /// Scale factor for function values
    pub value_scale: F,
    /// Whether scaling is recommended
    pub scaling_recommended: bool,
    /// Condition number improvement estimate with scaling
    pub condition_improvement_factor: F,
}

/// Noise characteristics analysis
#[derive(Debug, Clone)]
pub struct NoiseAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Estimated noise level
    pub estimated_noise_level: F,
    /// Signal-to-noise ratio estimate
    pub signal_noise_ratio: F,
    /// Whether the function appears noisy
    pub is_noisy: bool,
    /// Recommended denoising strategy
    pub denoising_strategy: DenoisingStrategy,
}

/// Denoising strategy recommendations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenoisingStrategy {
    /// No denoising needed
    None,
    /// Apply smoothing splines
    SmoothingSplines,
    /// Use robust interpolation
    RobustInterpolation,
    /// Apply wavelet denoising
    WaveletDenoising,
    /// Use total variation regularization
    TotalVariationRegularization,
}

/// Interpolation method recommendations
#[derive(Debug, Clone)]
pub struct InterpolationMethodRecommendation {
    /// Primary recommended method
    pub primary_method: InterpolationMethod,
    /// Alternative methods to consider
    pub alternative_methods: Vec<InterpolationMethod>,
    /// Explanation for the recommendation
    pub recommendation_reason: String,
}

/// Available interpolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// B-spline interpolation
    BSpline,
    /// NURBS interpolation
    NURBS,
    /// Radial basis function interpolation
    RadialBasisFunction,
    /// Kriging interpolation
    Kriging,
    /// Thin plate spline
    ThinPlateSpline,
    /// Piecewise polynomial
    PiecewisePolynomial,
}

/// Analyze data scaling characteristics
fn analyze_data_scaling<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<DataScalingAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let num_points = points.nrows();
    let dimension = points.ncols();

    if num_points == 0 {
        return Ok(DataScalingAnalysis {
            coordinate_scale: F::one(),
            value_scale: F::one(),
            scaling_recommended: false,
            condition_improvement_factor: F::one(),
        });
    }

    // Analyze coordinate scaling
    let mut coord_ranges = Vec::new();
    for j in 0..dimension {
        let mut min_coord = F::infinity();
        let mut max_coord = F::neg_infinity();
        for i in 0..num_points {
            let coord = points[(i, j)];
            min_coord = min_coord.min(coord);
            max_coord = max_coord.max(coord);
        }
        coord_ranges.push(max_coord - min_coord);
    }

    let max_coord_range = coord_ranges.iter().fold(F::zero(), |acc, &x| acc.max(x));
    let coordinate_scale = if max_coord_range > F::zero() {
        F::one() / max_coord_range
    } else {
        F::one()
    };

    // Analyze value scaling
    let min_value = values.iter().fold(F::infinity(), |acc, &x| acc.min(x));
    let max_value = values.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
    let value_range = max_value - min_value;
    let value_scale = if value_range > F::zero() {
        F::one() / value_range
    } else {
        F::one()
    };

    // Determine if scaling is recommended
    let coord_scale_factor = max_coord_range;
    let value_scale_factor = value_range;
    let scaling_recommended = coord_scale_factor
        >= F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap())
        || coord_scale_factor <= F::from(0.001).unwrap_or_else(|| F::from(0.001).unwrap())
        || value_scale_factor >= F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap())
        || value_scale_factor <= F::from(0.001).unwrap_or_else(|| F::from(0.001).unwrap());

    // Estimate condition number improvement
    let condition_improvement_factor = if scaling_recommended {
        let coord_improvement = coord_scale_factor.min(F::one() / coord_scale_factor);
        let value_improvement = value_scale_factor.min(F::one() / value_scale_factor);
        coord_improvement * value_improvement
    } else {
        F::one()
    };

    Ok(DataScalingAnalysis {
        coordinate_scale,
        value_scale,
        scaling_recommended,
        condition_improvement_factor,
    })
}

/// Analyze noise characteristics in function values
pub fn analyze_noise_characteristics<F>(
    values: &ArrayView1<F>,
) -> InterpolateResult<NoiseAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = values.len();
    if n < 3 {
        return Ok(NoiseAnalysis {
            estimated_noise_level: F::zero(),
            signal_noise_ratio: F::infinity(),
            is_noisy: false,
            denoising_strategy: DenoisingStrategy::None,
        });
    }

    // Estimate noise using finite differences
    let mut differences = Vec::new();
    for i in 1..n {
        differences.push(values[i] - values[i - 1]);
    }

    // Second differences to separate noise from signal
    let mut second_differences = Vec::new();
    for i in 1..(n - 1) {
        let second_diff = values[i + 1] - F::from(2.0).unwrap() * values[i] + values[i - 1];
        second_differences.push(second_diff.abs());
    }

    // Estimate noise level using median absolute deviation of second differences
    let mut sorted_second_diffs = second_differences.clone();
    sorted_second_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let estimated_noise_level = if !sorted_second_diffs.is_empty() {
        let median_idx = sorted_second_diffs.len() / 2;
        sorted_second_diffs[median_idx]
            / F::from(1.4826).unwrap_or_else(|| F::from(1.4826).unwrap()) // MAD to std conversion
    } else {
        F::zero()
    };

    // Estimate signal level
    let signal_range = values.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x))
        - values.iter().fold(F::infinity(), |acc, &x| acc.min(x));

    let signal_noise_ratio = if estimated_noise_level > F::zero() {
        signal_range / estimated_noise_level
    } else {
        F::infinity()
    };

    // Determine if data is noisy
    let noise_threshold = F::from(10.0).unwrap_or_else(|| F::from(10.0).unwrap());
    let is_noisy = signal_noise_ratio < noise_threshold;

    // Recommend denoising strategy
    let denoising_strategy = if !is_noisy {
        DenoisingStrategy::None
    } else if signal_noise_ratio > F::from(5.0).unwrap_or_else(|| F::from(5.0).unwrap()) {
        DenoisingStrategy::SmoothingSplines
    } else if signal_noise_ratio > F::from(2.0).unwrap_or_else(|| F::from(2.0).unwrap()) {
        DenoisingStrategy::RobustInterpolation
    } else {
        DenoisingStrategy::WaveletDenoising
    };

    Ok(NoiseAnalysis {
        estimated_noise_level,
        signal_noise_ratio,
        is_noisy,
        denoising_strategy,
    })
}

/// Recommend interpolation method based on data characteristics
fn recommend_interpolation_method<F>(
    data_points: &DataPointsAnalysis<F>,
    function_values: &FunctionValuesAnalysis<F>,
) -> InterpolateResult<InterpolationMethodRecommendation>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut primary_method = InterpolationMethod::CubicSpline;
    let mut alternative_methods = Vec::new();
    let mut reasons = Vec::new();

    // Consider data point characteristics
    if data_points.num_points < 10 {
        primary_method = InterpolationMethod::Linear;
        alternative_methods.push(InterpolationMethod::CubicSpline);
        reasons.push("Few data points favor simpler methods".to_string());
    } else if data_points.num_points > 1000 {
        primary_method = InterpolationMethod::BSpline;
        alternative_methods.push(InterpolationMethod::ThinPlateSpline);
        reasons.push("Large datasets benefit from B-spline efficiency".to_string());
    }

    // Consider function characteristics (but don't override small dataset preference)
    if !function_values.is_smooth {
        if function_values.has_outliers {
            if data_points.num_points >= 10 {
                primary_method = InterpolationMethod::RadialBasisFunction;
                alternative_methods.push(InterpolationMethod::Kriging);
                reasons.push("Non-smooth functions with outliers need robust methods".to_string());
            }
        } else if data_points.num_points >= 10 {
            primary_method = InterpolationMethod::PiecewisePolynomial;
            alternative_methods.push(InterpolationMethod::BSpline);
            reasons.push("Non-smooth functions benefit from piecewise approaches".to_string());
        }
    } else if function_values.is_monotonic && data_points.num_points >= 10 {
        primary_method = InterpolationMethod::CubicSpline;
        alternative_methods.push(InterpolationMethod::BSpline);
        reasons.push("Smooth monotonic functions are ideal for spline interpolation".to_string());
    }

    // Consider geometric characteristics
    if data_points.is_collinear {
        primary_method = InterpolationMethod::Linear;
        alternative_methods.push(InterpolationMethod::CubicSpline);
        reasons.push("Collinear points suggest 1D interpolation".to_string());
    } else if data_points.clustering_score > F::from(0.7).unwrap_or_else(|| F::from(0.7).unwrap()) {
        primary_method = InterpolationMethod::Kriging;
        alternative_methods.push(InterpolationMethod::RadialBasisFunction);
        reasons.push("Clustered data benefits from spatial interpolation methods".to_string());
    }

    // Ensure we have alternatives
    if alternative_methods.is_empty() {
        match primary_method {
            InterpolationMethod::Linear => {
                alternative_methods.push(InterpolationMethod::CubicSpline);
                alternative_methods.push(InterpolationMethod::BSpline);
            }
            InterpolationMethod::CubicSpline => {
                alternative_methods.push(InterpolationMethod::BSpline);
                alternative_methods.push(InterpolationMethod::ThinPlateSpline);
            }
            InterpolationMethod::BSpline => {
                alternative_methods.push(InterpolationMethod::CubicSpline);
                alternative_methods.push(InterpolationMethod::NURBS);
            }
            _ => {
                alternative_methods.push(InterpolationMethod::CubicSpline);
                alternative_methods.push(InterpolationMethod::BSpline);
            }
        }
    }

    let recommendation_reason = if reasons.is_empty() {
        "Based on standard interpolation guidelines".to_string()
    } else {
        reasons.join("; ")
    };

    Ok(InterpolationMethodRecommendation {
        primary_method,
        alternative_methods,
        recommendation_reason,
    })
}

/// Analyze sampling density and suggest optimal point distribution
pub fn analyze_sampling_density<F>(
    points: &ArrayView2<F>,
    target_accuracy: F,
) -> InterpolateResult<SamplingDensityAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let num_points = points.nrows();
    let dimension = points.ncols();

    if num_points < 2 {
        return Ok(SamplingDensityAnalysis {
            current_density: F::zero(),
            recommended_density: F::one(),
            density_adequate: false,
            suggested_additional_points: 10,
            sampling_strategy: SamplingStrategy::Uniform,
        });
    }

    // Calculate current sampling density
    let (min_distance, max_distance, _) = super::edge_cases::analyze_point_distances(points)?;
    let current_density = F::one() / min_distance.max(super::types::machine_epsilon::<F>());

    // Estimate required density based on target accuracy
    let accuracy_factor = F::one() / target_accuracy.max(super::types::machine_epsilon::<F>());
    let recommended_density = current_density * accuracy_factor.sqrt();

    // Check if current density is adequate
    let density_adequate = current_density
        >= recommended_density * F::from(0.5).unwrap_or_else(|| F::from(0.5).unwrap());

    // Suggest additional points if needed
    let suggested_additional_points = if density_adequate {
        0
    } else {
        let density_ratio = recommended_density / current_density;
        (num_points as f64 * (density_ratio.to_f64().unwrap_or(2.0) - 1.0)).ceil() as usize
    };

    // Recommend sampling strategy
    let distance_ratio = if min_distance > F::zero() {
        max_distance / min_distance
    } else {
        F::infinity()
    };

    let sampling_strategy =
        if distance_ratio > F::from(100.0).unwrap_or_else(|| F::from(100.0).unwrap()) {
            SamplingStrategy::Adaptive
        } else if dimension > 2 {
            SamplingStrategy::QuasiRandom
        } else {
            SamplingStrategy::Uniform
        };

    Ok(SamplingDensityAnalysis {
        current_density,
        recommended_density,
        density_adequate,
        suggested_additional_points,
        sampling_strategy,
    })
}

/// Sampling density analysis results
#[derive(Debug, Clone)]
pub struct SamplingDensityAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    pub current_density: F,
    pub recommended_density: F,
    pub density_adequate: bool,
    pub suggested_additional_points: usize,
    pub sampling_strategy: SamplingStrategy,
}

/// Sampling strategy recommendations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Uniform grid sampling
    Uniform,
    /// Adaptive sampling based on function behavior
    Adaptive,
    /// Quasi-random sampling (Halton, Sobol, etc.)
    QuasiRandom,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Importance sampling
    ImportanceSampling,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_data_scaling_analysis() {
        let points = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1000.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0],
        )
        .unwrap();
        let values = Array1::from_vec(vec![0.0, 0.001, 0.001, 0.002]);

        let scaling = analyze_data_scaling(&points.view(), &values.view()).unwrap();
        assert!(scaling.scaling_recommended);
        assert!(scaling.coordinate_scale < 1.0);
        assert!(scaling.value_scale > 1.0);
    }

    #[test]
    fn test_noise_analysis() {
        // Create smooth data
        let smooth_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let smooth_analysis = analyze_noise_characteristics(&smooth_values.view()).unwrap();
        assert!(!smooth_analysis.is_noisy);
        assert_eq!(smooth_analysis.denoising_strategy, DenoisingStrategy::None);

        // Create noisy data
        let noisy_values = Array1::from_vec(vec![0.0, 1.1, 1.9, 3.05, 3.95]);
        let noisy_analysis = analyze_noise_characteristics(&noisy_values.view()).unwrap();
        assert!(noisy_analysis.estimated_noise_level > 0.0);
    }

    #[test]
    fn test_interpolation_method_recommendation() {
        // Test small dataset
        let small_data = DataPointsAnalysis {
            num_points: 5,
            min_distance: 1.0,
            max_distance: 4.0,
            distance_ratio: 4.0,
            is_collinear: false,
            clustering_score: 0.3,
            has_linear_dependencies: false,
        };

        let smooth_function = FunctionValuesAnalysis {
            value_range: 10.0,
            is_smooth: true,
            smoothness_score: 0.9,
            is_monotonic: true,
            has_outliers: false,
            recommended_smoothing: None,
        };

        let recommendation = recommend_interpolation_method(&small_data, &smooth_function).unwrap();
        assert_eq!(recommendation.primary_method, InterpolationMethod::Linear);
    }

    #[test]
    fn test_sampling_density_analysis() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let analysis = analyze_sampling_density(&points.view(), 0.01).unwrap();
        assert!(analysis.current_density > 0.0);
        assert!(analysis.recommended_density > 0.0);
    }

    #[test]
    fn test_complete_data_analysis() {
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        )
        .unwrap();
        let values = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);

        let report = analyze_interpolation_data(&points.view(), &values.view()).unwrap();

        assert!(report.data_points.is_collinear);
        assert!(report.function_values.is_smooth);
        assert!(report.function_values.is_monotonic);
        assert!(!report.noise_analysis.is_noisy);
    }
}
