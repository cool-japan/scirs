//! Edge case detection and analysis for interpolation problems
//!
//! This module provides comprehensive analysis of data point distributions,
//! function values, and boundary conditions to detect potential numerical
//! issues in interpolation problems.

use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use statrs::statistics::Statistics;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

use super::types::{
    BoundaryAnalysis, BoundaryTreatment, DataPointsAnalysis, EdgeCaseAnalysis, ExtrapolationRisk,
    FunctionValuesAnalysis,
};
use crate::error::{InterpolateError, InterpolateResult};

/// Comprehensive analysis of interpolation edge cases
pub fn analyze_interpolation_edge_cases<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<EdgeCaseAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    if points.nrows() != values.len() {
        return Err(InterpolateError::ShapeMismatch {
            expected: format!("{} points", values.len()),
            actual: format!("{} points", points.nrows()),
            object: "edge case analysis".to_string(),
        });
    }

    if points.nrows() == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Cannot analyze empty dataset".to_string(),
        });
    }

    // Analyze different aspects of the problem
    let data_points = analyze_data_points(points)?;
    let function_values = analyze_function_values(values)?;
    let boundary = analyze_boundary_conditions(points, values)?;

    // Generate overall recommendation
    let overall_recommendation =
        generate_overall_recommendation(&data_points, &function_values, &boundary);

    // Determine if problem is solvable
    let is_solvable = assess_solvability(&data_points, &function_values, &boundary);

    Ok(EdgeCaseAnalysis {
        data_points,
        function_values,
        boundary,
        overall_recommendation,
        is_solvable,
    })
}

/// Analyze data point distribution and geometric properties
pub fn analyze_data_points<F>(points: &ArrayView2<F>) -> InterpolateResult<DataPointsAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let num_points = points.nrows();
    let dimension = points.ncols();

    if num_points < 2 {
        return Ok(DataPointsAnalysis {
            num_points,
            min_distance: F::zero(),
            max_distance: F::zero(),
            distance_ratio: F::one(),
            is_collinear: false,
            clustering_score: F::zero(),
            has_linear_dependencies: false,
        });
    }

    // Analyze point distances
    let (min_distance, max_distance, _closest_pair) = analyze_point_distances(points)?;

    // Calculate distance ratio
    let distance_ratio = if min_distance > F::zero() {
        max_distance / min_distance
    } else {
        F::infinity()
    };

    // Check for collinearity in 2D
    let is_collinear = if dimension == 2 && num_points >= 3 {
        check_collinearity_2d(points)
    } else {
        false
    };

    // Calculate clustering score
    let clustering_score = calculate_clustering_score(points);

    // Check for near-linear dependencies
    let has_linear_dependencies = check_near_linear_dependencies(points)?;

    Ok(DataPointsAnalysis {
        num_points,
        min_distance,
        max_distance,
        distance_ratio,
        is_collinear,
        clustering_score,
        has_linear_dependencies,
    })
}

/// Analyze function values for smoothness and outliers
pub fn analyze_function_values<F>(
    values: &ArrayView1<F>,
) -> InterpolateResult<FunctionValuesAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = values.len();
    if n == 0 {
        return Ok(FunctionValuesAnalysis::default());
    }

    // Calculate value range
    let min_val = values.iter().fold(F::infinity(), |acc, &x| acc.min(x));
    let max_val = values.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
    let value_range = max_val - min_val;

    // Estimate smoothness
    let smoothness_score = estimate_smoothness(values);
    let is_smooth = smoothness_score > F::from(0.7).unwrap_or_else(|| F::from(0.7).unwrap());

    // Check monotonicity
    let is_monotonic = check_monotonicity(values);

    // Detect outliers using simple statistical method
    let has_outliers = detect_outliers_simple(values);

    // Suggest smoothing parameter if needed
    let recommended_smoothing = if !is_smooth || has_outliers {
        Some(suggest_smoothing_parameter(values, smoothness_score))
    } else {
        None
    };

    Ok(FunctionValuesAnalysis {
        value_range,
        is_smooth,
        smoothness_score,
        is_monotonic,
        has_outliers,
        recommended_smoothing,
    })
}

/// Analyze boundary conditions and extrapolation risks
pub fn analyze_boundary_conditions<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<BoundaryAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let num_points = points.nrows();
    if num_points < 2 {
        return Ok(BoundaryAnalysis::default());
    }

    // Check for boundary effects by examining edge behavior
    let has_boundary_effects = assess_boundary_effects(points, values)?;

    // Recommend boundary treatment
    let recommended_boundary_treatment = recommend_boundary_treatment(points, values)?;

    // Assess extrapolation risk
    let extrapolation_risk = assess_extrapolation_risk(points, values)?;

    let mut analysis = BoundaryAnalysis::default();
    analysis.has_boundary_effects = has_boundary_effects;
    analysis.recommended_boundary_treatment = recommended_boundary_treatment;
    analysis.extrapolation_risk = extrapolation_risk;
    Ok(analysis)
}

/// Analyze distances between all pairs of points
pub fn analyze_point_distances<F>(points: &ArrayView2<F>) -> InterpolateResult<(F, F, usize)>
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = points.nrows();
    let dim = points.ncols();

    if n < 2 {
        return Ok((F::zero(), F::zero(), 0));
    }

    let mut min_distance = F::infinity();
    let mut max_distance = F::zero();
    let mut closest_pair = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance_sq = F::zero();
            for k in 0..dim {
                let diff = points[(i, k)] - points[(j, k)];
                distance_sq += diff * diff;
            }
            let distance = distance_sq.sqrt();

            if distance < min_distance {
                min_distance = distance;
                closest_pair = i;
            }
            if distance > max_distance {
                max_distance = distance;
            }
        }
    }

    Ok((min_distance, max_distance, closest_pair))
}

/// Check for near-linear dependencies in point set
fn check_near_linear_dependencies<F>(points: &ArrayView2<F>) -> InterpolateResult<bool>
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = points.nrows();
    let dim = points.ncols();

    if n <= dim {
        return Ok(false); // Cannot be linearly dependent
    }

    // Compute Gram matrix for dependency analysis
    let gram = compute_gram_matrix(points);

    // Check condition number of Gram matrix as proxy for linear dependence
    let min_eigenvalue = estimate_min_eigenvalue(&gram.view());
    let max_eigenvalue = estimate_max_eigenvalue(&gram.view());

    let condition_number = if min_eigenvalue > F::zero() {
        max_eigenvalue / min_eigenvalue
    } else {
        F::infinity()
    };

    // Large condition number indicates near-linear dependence
    let threshold = F::from(1e12).unwrap_or_else(|| F::from(1e12).unwrap());
    Ok(condition_number > threshold)
}

/// Compute Gram matrix G = X^T X for dependency analysis
fn compute_gram_matrix<F>(points: &ArrayView2<F>) -> scirs2_core::ndarray::Array2<F>
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = points.nrows();
    let dim = points.ncols();
    let mut gram = scirs2_core::ndarray::Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let mut dot_product = F::zero();
            for k in 0..dim {
                dot_product += points[(i, k)] * points[(j, k)];
            }
            gram[(i, j)] = dot_product;
        }
    }

    gram
}

/// Simple eigenvalue estimation using power iteration
fn estimate_max_eigenvalue<F>(matrix: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = matrix.nrows();
    if n == 0 {
        return F::zero();
    }

    // Simple estimation using matrix norm
    let mut max_row_sum = F::zero();
    for i in 0..n {
        let mut row_sum = F::zero();
        for j in 0..n {
            row_sum += matrix[(i, j)].abs();
        }
        max_row_sum = max_row_sum.max(row_sum);
    }

    max_row_sum
}

/// Simple minimum eigenvalue estimation
fn estimate_min_eigenvalue<F>(matrix: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive,
{
    let n = matrix.nrows();
    if n == 0 {
        return F::zero();
    }

    // Rough estimation using minimum diagonal element
    let mut min_diag = F::infinity();
    for i in 0..n {
        min_diag = min_diag.min(matrix[(i, i)]);
    }

    min_diag.max(F::zero())
}

/// Check if 2D points are collinear
fn check_collinearity_2d<F>(points: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive,
{
    let n = points.nrows();
    if n < 3 || points.ncols() != 2 {
        return false;
    }

    let tolerance = super::types::machine_epsilon::<F>()
        * F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap());

    // Check if all points lie on the same line
    let x1 = points[(0, 0)];
    let y1 = points[(0, 1)];
    let x2 = points[(1, 0)];
    let y2 = points[(1, 1)];

    for i in 2..n {
        let x3 = points[(i, 0)];
        let y3 = points[(i, 1)];

        // Calculate cross product to check collinearity
        let cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        if cross_product.abs() > tolerance {
            return false;
        }
    }

    true
}

/// Calculate clustering score (0 = well distributed, 1 = highly clustered)
fn calculate_clustering_score<F>(points: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = points.nrows();
    if n < 3 {
        return F::zero();
    }

    // Calculate mean distance
    let mut total_distance = F::zero();
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = F::zero();
            for k in 0..points.ncols() {
                let diff = points[(i, k)] - points[(j, k)];
                dist_sq += diff * diff;
            }
            total_distance += dist_sq.sqrt();
            count += 1;
        }
    }

    let mean_distance = total_distance / F::from(count).unwrap();

    // Calculate variance of distances
    let mut variance = F::zero();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = F::zero();
            for k in 0..points.ncols() {
                let diff = points[(i, k)] - points[(j, k)];
                dist_sq += diff * diff;
            }
            let distance = dist_sq.sqrt();
            let diff_from_mean = distance - mean_distance;
            variance += diff_from_mean * diff_from_mean;
        }
    }
    variance = variance / F::from(count).unwrap();

    // Clustering score based on coefficient of variation
    if mean_distance > F::zero() {
        let cv = variance.sqrt() / mean_distance;
        cv / (F::one() + cv) // Normalize to [0, 1]
    } else {
        F::one() // All points are identical = highly clustered
    }
}

/// Estimate smoothness of function values
fn estimate_smoothness<F>(values: &ArrayView1<F>) -> F
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = values.len();
    if n < 3 {
        return F::one(); // Assume smooth for small datasets
    }

    // Calculate second differences
    let mut second_diffs = Vec::new();
    for i in 1..(n - 1) {
        let second_diff = values[i + 1] - F::from(2.0).unwrap() * values[i] + values[i - 1];
        second_diffs.push(second_diff.abs());
    }

    if second_diffs.is_empty() {
        return F::one();
    }

    // Calculate smoothness as inverse of mean second difference
    let mean_second_diff: F = second_diffs.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from(second_diffs.len()).unwrap();

    // Normalize to [0, 1] range
    let max_possible_diff =
        values.iter().fold(F::zero(), |acc, &x| acc.max(x.abs())) * F::from(2.0).unwrap();

    if max_possible_diff > F::zero() {
        F::one() - (mean_second_diff / max_possible_diff).min(F::one())
    } else {
        F::one()
    }
}

/// Check if function values are monotonic
fn check_monotonicity<F>(values: &ArrayView1<F>) -> bool
where
    F: Float + FromPrimitive,
{
    let n = values.len();
    if n < 2 {
        return true;
    }

    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..n {
        if values[i] < values[i - 1] {
            increasing = false;
        }
        if values[i] > values[i - 1] {
            decreasing = false;
        }
    }

    increasing || decreasing
}

/// Simple outlier detection using statistical methods
fn detect_outliers_simple<F>(values: &ArrayView1<F>) -> bool
where
    F: Float + FromPrimitive + AddAssign,
{
    let n = values.len();
    if n < 4 {
        return false; // Too few points to detect outliers
    }

    // Use IQR method which is more robust for outlier detection
    let mut sorted_values: Vec<F> = values.iter().cloned().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;
    let q1 = sorted_values[q1_idx];
    let q3 = sorted_values[q3_idx];
    let iqr = q3 - q1;

    // Use 1.5 * IQR rule for outlier detection
    let lower_bound = q1 - F::from(1.5).unwrap() * iqr;
    let upper_bound = q3 + F::from(1.5).unwrap() * iqr;

    for &value in values.iter() {
        if value < lower_bound || value > upper_bound {
            return true;
        }
    }

    false
}

/// Suggest smoothing parameter based on function characteristics
fn suggest_smoothing_parameter<F>(values: &ArrayView1<F>, smoothness_score: F) -> F
where
    F: Float + FromPrimitive + AddAssign,
{
    // Base smoothing on function range and smoothness
    let range = values.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x))
        - values.iter().fold(F::infinity(), |acc, &x| acc.min(x));

    let base_smoothing = range * F::from(0.01).unwrap_or_else(|| F::from(0.01).unwrap());

    // Adjust based on smoothness score
    let smoothness_factor = F::one() - smoothness_score;
    base_smoothing
        * (F::one() + smoothness_factor * F::from(10.0).unwrap_or_else(|| F::from(10.0).unwrap()))
}

/// Assess if boundary effects are significant
fn assess_boundary_effects<F>(
    _points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<bool>
where
    F: Float + FromPrimitive,
{
    let n = values.len();
    if n < 4 {
        return Ok(false);
    }

    // Check if boundary values are significantly different from interior pattern
    let boundary_threshold = F::from(0.1).unwrap_or_else(|| F::from(0.1).unwrap());

    // Compare first/last values with interior trend
    let interior_range = values[n / 4] - values[3 * n / 4];
    let boundary_deviation = (values[0] - values[1]).abs() + (values[n - 1] - values[n - 2]).abs();

    Ok(boundary_deviation > boundary_threshold * interior_range.abs())
}

/// Recommend boundary treatment based on data characteristics
fn recommend_boundary_treatment<F>(
    _points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<BoundaryTreatment>
where
    F: Float + FromPrimitive,
{
    let n = values.len();
    if n < 2 {
        return Ok(BoundaryTreatment::Natural);
    }

    // Check if values suggest periodicity
    if n > 4 {
        let first_val = values[0];
        let last_val = values[n - 1];
        let tolerance = (values.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x))
            - values.iter().fold(F::infinity(), |acc, &x| acc.min(x)))
            * F::from(0.1).unwrap_or_else(|| F::from(0.1).unwrap());

        if (first_val - last_val).abs() < tolerance {
            return Ok(BoundaryTreatment::Periodic);
        }
    }

    // Check if function has large derivatives at boundaries
    if n > 2 {
        let left_slope = values[1] - values[0];
        let right_slope = values[n - 1] - values[n - 2];

        if left_slope.abs() > F::zero() || right_slope.abs() > F::zero() {
            return Ok(BoundaryTreatment::Clamped);
        }
    }

    Ok(BoundaryTreatment::Natural)
}

/// Assess risk level for extrapolation
fn assess_extrapolation_risk<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<ExtrapolationRisk>
where
    F: Float + FromPrimitive + AddAssign,
{
    let smoothness = estimate_smoothness(values);
    let is_monotonic = check_monotonicity(values);
    let clustering_score = calculate_clustering_score(points);

    // Risk assessment based on multiple factors
    let mut risk_score = F::zero();

    // Penalize non-smooth functions
    if smoothness < F::from(0.5).unwrap_or_else(|| F::from(0.5).unwrap()) {
        risk_score += F::from(0.3).unwrap_or_else(|| F::from(0.3).unwrap());
    }

    // Penalize non-monotonic functions
    if !is_monotonic {
        risk_score += F::from(0.2).unwrap_or_else(|| F::from(0.2).unwrap());
    }

    // Penalize clustered data
    if clustering_score > F::from(0.5).unwrap_or_else(|| F::from(0.5).unwrap()) {
        risk_score += F::from(0.3).unwrap_or_else(|| F::from(0.3).unwrap());
    }

    // Convert risk score to categorical risk
    if risk_score < F::from(0.2).unwrap_or_else(|| F::from(0.2).unwrap()) {
        Ok(ExtrapolationRisk::Low)
    } else if risk_score < F::from(0.5).unwrap_or_else(|| F::from(0.5).unwrap()) {
        Ok(ExtrapolationRisk::Medium)
    } else if risk_score < F::from(0.8).unwrap_or_else(|| F::from(0.8).unwrap()) {
        Ok(ExtrapolationRisk::High)
    } else {
        Ok(ExtrapolationRisk::Critical)
    }
}

/// Generate overall recommendation based on all analyses
fn generate_overall_recommendation<F>(
    data_points: &DataPointsAnalysis<F>,
    function_values: &FunctionValuesAnalysis<F>,
    boundary: &BoundaryAnalysis<F>,
) -> String
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut recommendations = Vec::new();

    // Data point recommendations
    if data_points.is_collinear {
        recommendations
            .push("Points are collinear - consider adding non-collinear points".to_string());
    }

    if data_points.clustering_score > F::from(0.7).unwrap_or_else(|| F::from(0.7).unwrap()) {
        recommendations
            .push("Points are highly clustered - consider more uniform distribution".to_string());
    }

    if data_points.distance_ratio > F::from(1000.0).unwrap_or_else(|| F::from(1000.0).unwrap()) {
        recommendations
            .push("Large variation in point spacing - consider regularization".to_string());
    }

    // Function value recommendations
    if !function_values.is_smooth {
        recommendations
            .push("Function appears non-smooth - consider smoothing or regularization".to_string());
    }

    if function_values.has_outliers {
        recommendations.push(
            "Outliers detected in function values - consider robust interpolation".to_string(),
        );
    }

    // Boundary recommendations
    if boundary.has_boundary_effects {
        recommendations.push(format!(
            "Boundary effects detected - consider {} boundary conditions",
            match boundary.recommended_boundary_treatment {
                BoundaryTreatment::Natural => "natural",
                BoundaryTreatment::Clamped => "clamped",
                BoundaryTreatment::Periodic => "periodic",
                BoundaryTreatment::Custom => "custom",
            }
        ));
    }

    match boundary.extrapolation_risk {
        ExtrapolationRisk::Medium => {
            recommendations.push("Moderate extrapolation risk - use with caution".to_string())
        }
        ExtrapolationRisk::High => {
            recommendations.push("High extrapolation risk - avoid extrapolation".to_string())
        }
        ExtrapolationRisk::Critical => {
            recommendations.push("Critical extrapolation risk - do not extrapolate".to_string())
        }
        ExtrapolationRisk::Low => {}
    }

    if recommendations.is_empty() {
        "No significant issues detected - standard interpolation should work well".to_string()
    } else {
        recommendations.join("; ")
    }
}

/// Assess if the interpolation problem is solvable with current setup
fn assess_solvability<F>(
    data_points: &DataPointsAnalysis<F>,
    function_values: &FunctionValuesAnalysis<F>,
    boundary: &BoundaryAnalysis<F>,
) -> bool
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    // Check for critical issues that make the problem unsolvable

    // Too few points
    if data_points.num_points < 2 {
        return false;
    }

    // All points identical (zero distance)
    if data_points.max_distance <= F::zero() {
        return false;
    }

    // Extreme clustering that would cause numerical issues
    if data_points.clustering_score > F::from(0.95).unwrap_or_else(|| F::from(0.95).unwrap()) {
        return false;
    }

    // Critical extrapolation risk
    if boundary.extrapolation_risk == ExtrapolationRisk::Critical {
        return false;
    }

    // Function values have extreme issues
    if function_values.value_range <= F::zero() {
        return false; // All function values are identical
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_analyze_well_distributed_points() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let analysis = analyze_data_points(&points.view()).unwrap();
        assert_eq!(analysis.num_points, 4);
        assert!(!analysis.is_collinear);
        assert!(analysis.clustering_score < 0.5);
    }

    #[test]
    fn test_collinearity_detection() {
        let collinear_points =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();

        let analysis = analyze_data_points(&collinear_points.view()).unwrap();
        assert!(analysis.is_collinear);
    }

    #[test]
    fn test_smoothness_estimation() {
        let smooth_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let smoothness = estimate_smoothness(&smooth_values.view());
        assert!(smoothness > 0.8);

        let non_smooth_values = Array1::from_vec(vec![0.0, 10.0, 0.0, 10.0, 0.0]);
        let smoothness2 = estimate_smoothness(&non_smooth_values.view());
        assert!(smoothness2 < 0.5);
    }

    #[test]
    fn test_monotonicity_check() {
        let increasing = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(check_monotonicity(&increasing.view()));

        let decreasing = Array1::from_vec(vec![4.0, 3.0, 2.0, 1.0]);
        assert!(check_monotonicity(&decreasing.view()));

        let non_monotonic = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0]);
        assert!(!check_monotonicity(&non_monotonic.view()));
    }

    #[test]
    fn test_outlier_detection() {
        let normal_values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(!detect_outliers_simple(&normal_values.view()));

        let with_outlier = Array1::from_vec(vec![1.0, 2.0, 100.0, 4.0, 5.0]);
        assert!(detect_outliers_simple(&with_outlier.view()));
    }

    #[test]
    fn test_complete_edge_case_analysis() {
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        )
        .unwrap();
        let values = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);

        let analysis = analyze_interpolation_edge_cases(&points.view(), &values.view()).unwrap();

        assert!(analysis.is_solvable);
        assert!(analysis.data_points.is_collinear); // Points on a line
        assert!(analysis.function_values.is_smooth); // Polynomial is smooth
        assert!(analysis.function_values.is_monotonic); // Increasing function
    }
}
