//! Piecewise trend estimation methods
//!
//! This module provides methods for estimating piecewise trends in time series data,
//! including automatic and manual breakpoint detection, and various segment models.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::{
    BreakpointCriterion, BreakpointMethod, ConfidenceIntervalOptions, PiecewiseTrendOptions,
    SegmentModelType, TrendWithConfidenceInterval,
};
use crate::error::{Result, TimeSeriesError};

/// Estimates a piecewise trend with automatic or manual breakpoint detection
///
/// This function fits a piecewise trend to time series data, automatically detecting
/// breakpoints or using provided breakpoints.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the piecewise trend estimation
///
/// # Returns
///
/// The estimated trend as a time series with the same length as the input
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::trends::{estimate_piecewise_trend, PiecewiseTrendOptions, BreakpointMethod, SegmentModelType};
///
/// // Create a sample time series with a piecewise trend and noise
/// let n = 100;
/// let mut ts = Array1::zeros(n);
///
/// // First segment: linear trend
/// for i in 0..40 {
///     ts[i] = i as f64 * 0.1 + 0.5 + 0.1 * (i as f64 / 10.0).sin();
/// }
///
/// // Second segment: constant
/// for i in 40..70 {
///     ts[i] = 4.5 + 0.1 * (i as f64 / 10.0).sin();
/// }
///
/// // Third segment: decreasing
/// for i in 70..100 {
///     ts[i] = 4.5 - (i - 70) as f64 * 0.15 + 0.1 * (i as f64 / 10.0).sin();
/// }
///
/// // Configure piecewise trend options
/// let options = PiecewiseTrendOptions {
///     breakpoint_method: BreakpointMethod::BinarySegmentation,
///     segment_model: SegmentModelType::Linear,
///     min_segment_length: 10,
///     max_breakpoints: Some(3),
///     ..Default::default()
/// };
///
/// // Estimate piecewise trend
/// let trend = estimate_piecewise_trend(&ts, &options).unwrap();
///
/// // The trend should have the same length as the input
/// assert_eq!(trend.len(), ts.len());
/// ```
pub fn estimate_piecewise_trend<F>(
    ts: &Array1<F>,
    options: &PiecewiseTrendOptions,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    if n < 2 * options.min_segment_length {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Time series too short for piecewise trend estimation with min_segment_length={}",
                options.min_segment_length
            ),
            required: 2 * options.min_segment_length,
            actual: n,
        });
    }

    // Detect breakpoints
    let breakpoints = match options.breakpoint_method {
        BreakpointMethod::Custom => {
            if let Some(custom_points) = &options.custom_breakpoints {
                // Validate custom breakpoints
                for &bp in custom_points {
                    if bp == 0 || bp >= n {
                        return Err(TimeSeriesError::InvalidInput(format!(
                            "Invalid breakpoint: {} (must be between 1 and {})",
                            bp,
                            n - 1
                        )));
                    }
                }

                let mut bps = custom_points.clone();
                bps.sort_unstable();

                // Check minimum segment length
                let mut prev = 0;
                for &bp in &bps {
                    if bp - prev < options.min_segment_length {
                        return Err(TimeSeriesError::InvalidInput(format!(
                            "Segment between breakpoints {} and {} is too short (min: {})",
                            prev, bp, options.min_segment_length
                        )));
                    }
                    prev = bp;
                }

                if n - prev < options.min_segment_length {
                    return Err(TimeSeriesError::InvalidInput(format!(
                        "Final segment after breakpoint {} is too short (min: {})",
                        prev, options.min_segment_length
                    )));
                }

                bps
            } else {
                return Err(TimeSeriesError::InvalidInput(
                    "Custom breakpoint method selected but no breakpoints provided".to_string(),
                ));
            }
        }
        BreakpointMethod::BinarySegmentation => {
            detect_breakpoints_binary_segmentation(ts, options)?
        }
        BreakpointMethod::PELT => detect_breakpoints_pelt(ts, options)?,
        BreakpointMethod::BottomUp => detect_breakpoints_bottom_up(ts, options)?,
    };

    // Fit piecewise model to the segments
    fit_piecewise_model(ts, &breakpoints, options)
}

/// Detects breakpoints using the binary segmentation algorithm
fn detect_breakpoints_binary_segmentation<F>(
    ts: &Array1<F>,
    options: &PiecewiseTrendOptions,
) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let min_segment = options.min_segment_length;
    let max_breaks = options.max_breakpoints.unwrap_or(n / min_segment);

    // Initialize with no breakpoints
    let mut breakpoints = Vec::new();
    let mut segments = vec![(0, n - 1)]; // (start, end) indices of segments

    // Binary segmentation: iteratively split segments at the point of maximum improvement
    while breakpoints.len() < max_breaks && !segments.is_empty() {
        let mut best_improvement = F::zero();
        let mut best_segment_idx = 0;
        let mut best_breakpoint = 0;

        // For each segment, find the best potential breakpoint
        for (seg_idx, &(start, end)) in segments.iter().enumerate() {
            if end - start + 1 < 2 * min_segment {
                // Segment is too small to split further
                continue;
            }

            let segment_ts = ts.slice(ndarray::s![start..=end]);

            // Try splitting at each possible breakpoint
            let mut max_improvement_in_segment = F::zero();
            let mut best_point_in_segment = 0;

            for bp in (start + min_segment)..(end + 1 - min_segment) {
                let left_ts = ts.slice(ndarray::s![start..=bp]);
                let right_ts = ts.slice(ndarray::s![(bp + 1)..=end]);

                // Calculate improvement in criterion by splitting here
                let improvement = calculate_split_improvement(
                    &segment_ts,
                    &left_ts,
                    &right_ts,
                    options.criterion,
                    options.segment_model,
                    options.penalty.map(|p| F::from_f64(p).unwrap()),
                )?;

                if improvement > max_improvement_in_segment {
                    max_improvement_in_segment = improvement;
                    best_point_in_segment = bp;
                }
            }

            // Compare with the best improvement found so far
            if max_improvement_in_segment > best_improvement {
                best_improvement = max_improvement_in_segment;
                best_segment_idx = seg_idx;
                best_breakpoint = best_point_in_segment;
            }
        }

        if best_improvement <= F::zero() {
            // No further improvement possible
            break;
        }

        // Add the best breakpoint
        breakpoints.push(best_breakpoint);

        // Update the segments list
        let (start, end) = segments[best_segment_idx];
        segments.remove(best_segment_idx);
        segments.push((start, best_breakpoint));
        segments.push((best_breakpoint + 1, end));

        // Sort breakpoints for consistent output
        breakpoints.sort_unstable();
    }

    Ok(breakpoints)
}

/// Detects breakpoints using the PELT (Pruned Exact Linear Time) algorithm
fn detect_breakpoints_pelt<F>(ts: &Array1<F>, options: &PiecewiseTrendOptions) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let min_segment = options.min_segment_length;
    let penalty_value = options
        .penalty
        .map(|p| F::from_f64(p).unwrap())
        .unwrap_or_else(|| {
            // Default penalty based on criterion
            match options.criterion {
                BreakpointCriterion::AIC => F::from_f64(2.0).unwrap(),
                BreakpointCriterion::BIC => F::from_f64((n as f64).ln()).unwrap(),
                BreakpointCriterion::ModifiedBIC => F::from_f64((n as f64).ln().powf(1.5)).unwrap(),
                BreakpointCriterion::RSS => F::from_f64(15.0).unwrap(), // Arbitrary default
            }
        });

    // Store the optimal cost up to each point and the corresponding last breakpoint
    let mut cost = vec![F::infinity(); n + 1];
    let mut last_bp = vec![0; n + 1];
    cost[0] = F::zero();

    // Candidate breakpoints that haven't been pruned
    let mut candidates = vec![0];

    // Dynamic programming approach
    for t in min_segment..n {
        let mut min_cost = F::infinity();
        let mut min_bp = 0;

        for &s in &candidates {
            if t - s < min_segment {
                continue;
            }

            let segment_ts = ts.slice(ndarray::s![s..t]);
            let segment_cost =
                calculate_segment_cost(&segment_ts, options.segment_model, options.criterion)?;

            let total_cost = cost[s] + segment_cost + penalty_value;

            if total_cost < min_cost {
                min_cost = total_cost;
                min_bp = s;
            }
        }

        cost[t] = min_cost;
        last_bp[t] = min_bp;

        // Pruning step: remove candidates that can't be optimal
        let mut new_candidates = Vec::new();
        for &s in &candidates {
            if s > t - min_segment || cost[s] + F::from_f64(0.1).unwrap() < cost[t] {
                new_candidates.push(s);
            }
        }

        new_candidates.push(t + 1 - min_segment);
        candidates = new_candidates;
    }

    // Backtrack to find the optimal breakpoints
    let mut bps = Vec::new();
    let mut t = n - 1;

    while t > 0 {
        let last_breakpoint = last_bp[t];
        if last_breakpoint > 0 {
            bps.push(last_breakpoint);
            t = last_breakpoint - 1;
        } else {
            break;
        }
    }

    // Reverse to get chronological order
    bps.reverse();

    // Limit to max_breakpoints if specified
    if let Some(max_breaks) = options.max_breakpoints {
        if bps.len() > max_breaks {
            bps.truncate(max_breaks);
        }
    }

    Ok(bps)
}

/// Detects breakpoints using the bottom-up segmentation algorithm
fn detect_breakpoints_bottom_up<F>(
    ts: &Array1<F>,
    options: &PiecewiseTrendOptions,
) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let min_segment = options.min_segment_length;
    let max_breaks = options.max_breakpoints.unwrap_or(n / min_segment);

    // Start with maximum number of breakpoints
    let mut all_breakpoints: Vec<usize> = (min_segment..(n - min_segment + 1))
        .step_by(min_segment)
        .collect();

    if all_breakpoints.is_empty() {
        return Ok(Vec::new());
    }

    // Cost of merging each pair of adjacent segments
    let mut merge_costs = Vec::with_capacity(all_breakpoints.len() - 1);

    // Iteratively merge the pair with the lowest cost
    while all_breakpoints.len() > max_breaks {
        merge_costs.clear();

        // Calculate costs of merging each pair
        let mut segments = Vec::with_capacity(all_breakpoints.len() + 1);
        segments.push(0);
        segments.extend_from_slice(&all_breakpoints);
        segments.push(n - 1);

        for i in 0..(segments.len() - 2) {
            let start = segments[i];
            let mid = segments[i + 1];
            let end = segments[i + 2];

            let left_ts = ts.slice(ndarray::s![start..=mid]);
            let right_ts = ts.slice(ndarray::s![(mid + 1)..=end]);
            let merged_ts = ts.slice(ndarray::s![start..=end]);

            let left_cost =
                calculate_segment_cost(&left_ts, options.segment_model, options.criterion)?;

            let right_cost =
                calculate_segment_cost(&right_ts, options.segment_model, options.criterion)?;

            let merged_cost =
                calculate_segment_cost(&merged_ts, options.segment_model, options.criterion)?;

            // Cost of merging = increase in cost
            let merge_cost = merged_cost - (left_cost + right_cost);
            merge_costs.push((i, merge_cost));
        }

        // Find the pair with the lowest merge cost
        if let Some((idx, _)) = merge_costs.iter().min_by(|(_, cost1), (_, cost2)| {
            cost1
                .partial_cmp(cost2)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            // Remove the breakpoint
            all_breakpoints.remove(*idx);
        } else {
            break;
        }
    }

    Ok(all_breakpoints)
}

/// Calculates the improvement in criterion from splitting a segment
fn calculate_split_improvement<F>(
    segment_ts: &ArrayView1<F>,
    left_ts: &ArrayView1<F>,
    right_ts: &ArrayView1<F>,
    criterion: BreakpointCriterion,
    model_type: SegmentModelType,
    penalty: Option<F>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n_segment = segment_ts.len();
    let _n_left = left_ts.len();
    let _n_right = right_ts.len();

    // Calculate costs for the entire segment and the two sub-segments
    let segment_cost = calculate_segment_cost(segment_ts, model_type, criterion)?;
    let left_cost = calculate_segment_cost(left_ts, model_type, criterion)?;
    let right_cost = calculate_segment_cost(right_ts, model_type, criterion)?;

    // Calculate improvement (reduction in cost)
    let mut improvement = segment_cost - (left_cost + right_cost);

    // Apply penalty if specified
    if let Some(penalty_val) = penalty {
        improvement = improvement - penalty_val;
    } else {
        // Default penalty based on criterion
        match criterion {
            BreakpointCriterion::AIC => {
                // AIC penalty: 2k, where k is the increase in number of parameters
                let params_per_segment = match model_type {
                    SegmentModelType::Constant => 1,
                    SegmentModelType::Linear => 2,
                    SegmentModelType::Quadratic => 3,
                    SegmentModelType::Cubic => 4,
                    SegmentModelType::Spline => 5, // Approximate for spline
                };
                improvement = improvement - F::from_f64(2.0 * params_per_segment as f64).unwrap();
            }
            BreakpointCriterion::BIC => {
                // BIC penalty: k * ln(n), where k is the increase in number of parameters
                let params_per_segment = match model_type {
                    SegmentModelType::Constant => 1,
                    SegmentModelType::Linear => 2,
                    SegmentModelType::Quadratic => 3,
                    SegmentModelType::Cubic => 4,
                    SegmentModelType::Spline => 5, // Approximate for spline
                };
                improvement = improvement
                    - F::from_f64(params_per_segment as f64 * (n_segment as f64).ln()).unwrap();
            }
            BreakpointCriterion::ModifiedBIC => {
                // Modified BIC with stronger penalty: k * ln(n)^1.5
                let params_per_segment = match model_type {
                    SegmentModelType::Constant => 1,
                    SegmentModelType::Linear => 2,
                    SegmentModelType::Quadratic => 3,
                    SegmentModelType::Cubic => 4,
                    SegmentModelType::Spline => 5, // Approximate for spline
                };
                improvement = improvement
                    - F::from_f64(params_per_segment as f64 * (n_segment as f64).ln().powf(1.5))
                        .unwrap();
            }
            BreakpointCriterion::RSS => {
                // For RSS, we need an explicit penalty to avoid overfitting
                improvement = improvement - F::from_f64(15.0).unwrap(); // Arbitrary default
            }
        }
    }

    Ok(improvement)
}

/// Calculates the cost of a single segment using the specified criterion
fn calculate_segment_cost<F>(
    segment_ts: &ArrayView1<F>,
    model_type: SegmentModelType,
    criterion: BreakpointCriterion,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = segment_ts.len();

    // Fit the specified model to the segment
    let fitted = match model_type {
        SegmentModelType::Constant => {
            // Constant model: y = mean
            let mean = segment_ts.sum() / F::from_usize(n).unwrap();
            Array1::from_elem(n, mean)
        }
        SegmentModelType::Linear => {
            // Linear model: y = a + b*x
            fit_linear_model(segment_ts)?
        }
        SegmentModelType::Quadratic => {
            // Quadratic model: y = a + b*x + c*x^2
            fit_polynomial_model(segment_ts, 2)?
        }
        SegmentModelType::Cubic => {
            // Cubic model: y = a + b*x + c*x^2 + d*x^3
            fit_polynomial_model(segment_ts, 3)?
        }
        SegmentModelType::Spline => {
            // Simplified spline model for cost calculation
            fit_polynomial_model(segment_ts, 3)?
        }
    };

    // Calculate residuals and residual sum of squares (RSS)
    let mut rss = F::zero();
    for i in 0..n {
        let residual = segment_ts[i] - fitted[i];
        rss = rss + residual * residual;
    }

    // Calculate cost based on criterion
    match criterion {
        BreakpointCriterion::RSS => {
            // Pure RSS
            Ok(rss)
        }
        BreakpointCriterion::AIC => {
            // AIC = n * ln(RSS/n) + 2k
            let params = match model_type {
                SegmentModelType::Constant => 1,
                SegmentModelType::Linear => 2,
                SegmentModelType::Quadratic => 3,
                SegmentModelType::Cubic => 4,
                SegmentModelType::Spline => 5,
            };
            let n_f = F::from_usize(n).unwrap();
            let aic = n_f * (rss / n_f).ln() + F::from_usize(2 * params).unwrap();
            Ok(aic)
        }
        BreakpointCriterion::BIC => {
            // BIC = n * ln(RSS/n) + k * ln(n)
            let params = match model_type {
                SegmentModelType::Constant => 1,
                SegmentModelType::Linear => 2,
                SegmentModelType::Quadratic => 3,
                SegmentModelType::Cubic => 4,
                SegmentModelType::Spline => 5,
            };
            let n_f = F::from_usize(n).unwrap();
            let bic = n_f * (rss / n_f).ln() + F::from_usize(params).unwrap() * n_f.ln();
            Ok(bic)
        }
        BreakpointCriterion::ModifiedBIC => {
            // Modified BIC = n * ln(RSS/n) + k * ln(n)^1.5
            let params = match model_type {
                SegmentModelType::Constant => 1,
                SegmentModelType::Linear => 2,
                SegmentModelType::Quadratic => 3,
                SegmentModelType::Cubic => 4,
                SegmentModelType::Spline => 5,
            };
            let n_f = F::from_usize(n).unwrap();
            let mbic = n_f * (rss / n_f).ln()
                + F::from_usize(params).unwrap() * n_f.ln().powf(F::from_f64(1.5).unwrap());
            Ok(mbic)
        }
    }
}

/// Fits a linear model to a segment
fn fit_linear_model<F>(segment_ts: &ArrayView1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = segment_ts.len();

    // Create x values: 0, 1, 2, ...
    let x_values: Vec<F> = (0..n).map(|i| F::from_usize(i).unwrap()).collect();

    // Calculate means
    let mean_x = F::from_usize(n - 1).unwrap() / F::from_f64(2.0).unwrap();
    let mean_y = segment_ts.sum() / F::from_usize(n).unwrap();

    // Calculate covariance and variance
    let mut cov_xy = F::zero();
    let mut var_x = F::zero();

    for i in 0..n {
        let x_dev = x_values[i] - mean_x;
        let y_dev = segment_ts[i] - mean_y;

        cov_xy = cov_xy + x_dev * y_dev;
        var_x = var_x + x_dev * x_dev;
    }

    // Calculate slope and intercept
    let slope = if var_x > F::zero() {
        cov_xy / var_x
    } else {
        F::zero()
    };

    let intercept = mean_y - slope * mean_x;

    // Generate fitted values
    let mut fitted = Array1::<F>::zeros(n);
    for i in 0..n {
        fitted[i] = intercept + slope * x_values[i];
    }

    Ok(fitted)
}

/// Fits a polynomial model of specified degree to a segment
fn fit_polynomial_model<F>(segment_ts: &ArrayView1<F>, degree: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = segment_ts.len();

    if n <= degree {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Segment length must be greater than polynomial degree ({})",
                degree
            ),
            required: degree + 1,
            actual: n,
        });
    }

    // Create x values: 0, 1, 2, ...
    let x_values: Vec<F> = (0..n).map(|i| F::from_usize(i).unwrap()).collect();

    // Create design matrix
    let mut x_design = Array2::<F>::zeros((n, degree + 1));

    for i in 0..n {
        let mut x_power = F::one();
        for j in 0..=degree {
            x_design[[i, j]] = x_power;
            x_power = x_power * x_values[i];
        }
    }

    // Calculate X'X and X'y
    let mut xtx = Array2::<F>::zeros((degree + 1, degree + 1));
    let mut xty = vec![F::zero(); degree + 1];

    for i in 0..=degree {
        for j in 0..=degree {
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + x_design[[k, i]] * x_design[[k, j]];
            }
            xtx[[i, j]] = sum;
        }

        let mut sum = F::zero();
        for k in 0..n {
            sum = sum + x_design[[k, i]] * segment_ts[k];
        }
        xty[i] = sum;
    }

    // Solve using LU decomposition
    let coeffs = solve_linear_system(xtx, xty)?;

    // Generate fitted values
    let mut fitted = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut y_pred = F::zero();
        let mut x_power = F::one();

        for coeff in coeffs.iter().take(degree + 1) {
            y_pred = y_pred + *coeff * x_power;
            x_power = x_power * x_values[i];
        }

        fitted[i] = y_pred;
    }

    Ok(fitted)
}

/// Solves a linear system using LU decomposition
fn solve_linear_system<F>(a: Array2<F>, b: Vec<F>) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = a.shape()[0];
    if n != b.len() {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Matrix and vector dimensions do not match: A is {}x{}, b is {}",
            n,
            a.shape()[1],
            b.len()
        )));
    }

    // Gaussian elimination with partial pivoting
    let mut a_lu = a.to_owned();
    let mut b_mod = b.clone();
    let mut perm = (0..n).collect::<Vec<_>>();

    for k in 0..(n - 1) {
        // Find pivot
        let mut max_val = a_lu[[k, k]].abs();
        let mut max_row = k;

        for i in (k + 1)..n {
            let val = a_lu[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Swap rows if needed
        if max_row != k {
            for j in k..n {
                let temp = a_lu[[k, j]];
                a_lu[[k, j]] = a_lu[[max_row, j]];
                a_lu[[max_row, j]] = temp;
            }

            b_mod.swap(k, max_row);
            perm.swap(k, max_row);
        }

        // Eliminate below
        for i in (k + 1)..n {
            let factor = a_lu[[i, k]] / a_lu[[k, k]];
            a_lu[[i, k]] = factor; // Store multiplier

            for j in (k + 1)..n {
                a_lu[[i, j]] = a_lu[[i, j]] - factor * a_lu[[k, j]];
            }

            b_mod[i] = b_mod[i] - factor * b_mod[k];
        }
    }

    // Back substitution
    let mut x = vec![F::zero(); n];
    x[n - 1] = b_mod[n - 1] / a_lu[[n - 1, n - 1]];

    for i in (0..(n - 1)).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum = sum + a_lu[[i, j]] * x[j];
        }
        x[i] = (b_mod[i] - sum) / a_lu[[i, i]];
    }

    // Reorder solution according to permutation
    let mut x_perm = vec![F::zero(); n];
    for i in 0..n {
        x_perm[perm[i]] = x[i];
    }

    Ok(x_perm)
}

/// Fits a piecewise model to segments defined by breakpoints
fn fit_piecewise_model<F>(
    ts: &Array1<F>,
    breakpoints: &[usize],
    options: &PiecewiseTrendOptions,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let model_type = options.segment_model;
    let allow_discontinuities = options.allow_discontinuities;

    let mut trend = Array1::<F>::zeros(n);

    // Combine breakpoints with start and end points
    let mut all_points = Vec::with_capacity(breakpoints.len() + 2);
    all_points.push(0);
    all_points.extend_from_slice(breakpoints);
    all_points.push(n - 1);

    // Fit each segment
    for i in 0..(all_points.len() - 1) {
        let start = all_points[i];
        let end = all_points[i + 1];

        // Extract segment data
        let segment_data = if start == 0 || allow_discontinuities {
            ts.slice(ndarray::s![start..=end])
        } else {
            // Include one point before start for continuity
            ts.slice(ndarray::s![(start - 1)..=end])
        };

        // Fit model to segment
        let fitted = match model_type {
            SegmentModelType::Constant => {
                let mean = segment_data.sum() / F::from_usize(segment_data.len()).unwrap();
                Array1::from_elem(segment_data.len(), mean)
            }
            SegmentModelType::Linear => fit_linear_model(&segment_data.view())?,
            SegmentModelType::Quadratic => fit_polynomial_model(&segment_data.view(), 2)?,
            SegmentModelType::Cubic => fit_polynomial_model(&segment_data.view(), 3)?,
            SegmentModelType::Spline => {
                // For simplicity, use cubic for spline model
                fit_polynomial_model(&segment_data.view(), 3)?
            }
        };

        // Copy fitted values to the trend
        let offset = if start == 0 || allow_discontinuities {
            0
        } else {
            1
        };

        for j in start..=end {
            let idx = j - start + offset;
            if idx < fitted.len() {
                trend[j] = fitted[idx];
            }
        }
    }

    Ok(trend)
}

/// Estimates a piecewise trend with confidence intervals
///
/// This function is a wrapper around `estimate_piecewise_trend` that also computes
/// confidence intervals for the estimated trend.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `options` - Options controlling the piecewise trend estimation
/// * `ci_options` - Options controlling the confidence interval calculation
///
/// # Returns
///
/// A `TrendWithConfidenceInterval` struct containing the estimated trend and confidence bounds
pub fn estimate_piecewise_trend_with_ci<F>(
    ts: &Array1<F>,
    options: &PiecewiseTrendOptions,
    ci_options: &ConfidenceIntervalOptions,
) -> Result<TrendWithConfidenceInterval<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    // First, compute the main trend estimate
    let trend = estimate_piecewise_trend(ts, options)?;

    // Then compute confidence intervals
    let (lower, upper) =
        super::confidence::compute_trend_confidence_interval(ts, &trend, ci_options, |data| {
            estimate_piecewise_trend(data, options)
        })?;

    Ok(TrendWithConfidenceInterval {
        trend,
        lower,
        upper,
    })
}
