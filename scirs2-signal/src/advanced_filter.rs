//! Advanced filter design algorithms
//!
//! This module implements sophisticated filter design methods including Parks-McClellan
//! optimal equiripple design, arbitrary magnitude response approximation, and advanced
//! filter optimization techniques.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Filter design specifications
#[derive(Debug, Clone)]
pub struct FilterSpec {
    /// Filter type
    pub filter_type: FilterType,
    /// Sampling frequency (Hz)
    pub sample_rate: f64,
    /// Passband frequencies (Hz)
    pub passband_freqs: Vec<f64>,
    /// Stopband frequencies (Hz)
    pub stopband_freqs: Vec<f64>,
    /// Passband ripple (dB)
    pub passband_ripple: f64,
    /// Stopband attenuation (dB)
    pub stopband_attenuation: f64,
    /// Filter order (None for automatic)
    pub order: Option<usize>,
}

/// Filter types for advanced design
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// Lowpass filter
    Lowpass,
    /// Highpass filter
    Highpass,
    /// Bandpass filter
    Bandpass,
    /// Bandstop filter
    Bandstop,
    /// Hilbert transformer
    Hilbert,
    /// Differentiator
    Differentiator,
    /// Arbitrary magnitude response
    Arbitrary,
}

/// Parks-McClellan algorithm configuration
#[derive(Debug, Clone)]
pub struct ParksMcClellanConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Grid density for frequency sampling
    pub grid_density: usize,
    /// Enable extrapolation beyond specified bands
    pub extrapolate: bool,
}

impl Default for ParksMcClellanConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            grid_density: 16,
            extrapolate: false,
        }
    }
}

/// Arbitrary filter response specification
#[derive(Debug, Clone)]
pub struct ArbitraryResponse {
    /// Frequency points (normalized, 0 to 1)
    pub frequencies: Array1<f64>,
    /// Desired magnitude response at frequency points
    pub magnitude: Array1<f64>,
    /// Weight for each frequency point
    pub weights: Array1<f64>,
    /// Phase response (optional, for complex filters)
    pub phase: Option<Array1<f64>>,
}

/// Result of filter design
#[derive(Debug, Clone)]
pub struct FilterDesignResult {
    /// Filter coefficients (numerator)
    pub numerator: Array1<f64>,
    /// Filter coefficients (denominator, for IIR filters)
    pub denominator: Option<Array1<f64>>,
    /// Actual filter order
    pub order: usize,
    /// Frequency response at design grid
    pub frequency_response: Option<(Array1<f64>, Array1<Complex64>)>,
    /// Design error (maximum deviation from desired response)
    pub design_error: f64,
    /// Number of iterations used
    pub iterations: usize,
}

/// Parks-McClellan optimal equiripple FIR filter design
///
/// # Arguments
///
/// * `spec` - Filter specifications
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// * Optimal FIR filter coefficients
pub fn parks_mcclellan(
    spec: &FilterSpec,
    config: &ParksMcClellanConfig,
) -> SignalResult<FilterDesignResult> {
    // Determine filter order if not specified
    let filter_order = match spec.order {
        Some(order) => order,
        None => estimate_filter_order(spec)?,
    };

    // Create frequency grid
    let (freq_grid, desired_response, weights) = create_design_grid(spec, config)?;

    // Initialize extremal frequencies
    let mut extremal_freqs = initialize_extremal_frequencies(&freq_grid, filter_order + 2);

    let mut best_coefficients = Array1::zeros(filter_order + 1);
    let mut best_error = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Solve the interpolation problem at extremal frequencies
        let coefficients = solve_interpolation_problem(
            &extremal_freqs,
            &desired_response,
            &weights,
            &freq_grid,
            filter_order,
        )?;

        // Compute error function on the entire grid
        let error_function =
            compute_error_function(&coefficients, &freq_grid, &desired_response, &weights)?;

        // Find new extremal frequencies
        let new_extremal =
            find_extremal_frequencies(&error_function, &freq_grid, filter_order + 2)?;

        // Check convergence
        let max_error = error_function
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));

        if (max_error - best_error).abs() < config.tolerance {
            best_coefficients = coefficients;
            best_error = max_error;
            break;
        }

        if max_error < best_error {
            best_coefficients = coefficients.clone();
            best_error = max_error;
        }

        extremal_freqs = new_extremal;
    }

    // Compute frequency response for result
    let freq_response = compute_frequency_response(&best_coefficients, 512)?;

    Ok(FilterDesignResult {
        numerator: best_coefficients,
        denominator: None,
        order: filter_order,
        frequency_response: Some(freq_response),
        design_error: best_error,
        iterations,
    })
}

/// Design filter with arbitrary magnitude response
///
/// # Arguments
///
/// * `response` - Desired arbitrary response specification
/// * `order` - Filter order
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// * Filter coefficients approximating the arbitrary response
pub fn arbitrary_magnitude_design(
    response: &ArbitraryResponse,
    order: usize,
    config: &ParksMcClellanConfig,
) -> SignalResult<FilterDesignResult> {
    if response.frequencies.len() != response.magnitude.len()
        || response.frequencies.len() != response.weights.len()
    {
        return Err(SignalError::ValueError(
            "Frequency, magnitude, and weight arrays must have same length".to_string(),
        ));
    }

    // Create dense frequency grid by interpolation
    let dense_freqs = Array1::linspace(0.0, 1.0, order * config.grid_density);
    let dense_magnitude =
        interpolate_response(&response.frequencies, &response.magnitude, &dense_freqs)?;
    let dense_weights =
        interpolate_response(&response.frequencies, &response.weights, &dense_freqs)?;

    // Use Parks-McClellan algorithm with arbitrary response
    let mut extremal_freqs = initialize_extremal_frequencies(&dense_freqs, order + 2);

    let mut best_coefficients = Array1::zeros(order + 1);
    let mut best_error = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Solve interpolation problem
        let coefficients = solve_interpolation_problem(
            &extremal_freqs,
            &dense_magnitude,
            &dense_weights,
            &dense_freqs,
            order,
        )?;

        // Compute error function
        let error_function = compute_error_function(
            &coefficients,
            &dense_freqs,
            &dense_magnitude,
            &dense_weights,
        )?;

        // Find new extremal frequencies
        let new_extremal = find_extremal_frequencies(&error_function, &dense_freqs, order + 2)?;

        // Check convergence
        let max_error = error_function
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));

        if (max_error - best_error).abs() < config.tolerance {
            best_coefficients = coefficients;
            best_error = max_error;
            break;
        }

        if max_error < best_error {
            best_coefficients = coefficients.clone();
            best_error = max_error;
        }

        extremal_freqs = new_extremal;
    }

    let freq_response = compute_frequency_response(&best_coefficients, 512)?;

    Ok(FilterDesignResult {
        numerator: best_coefficients,
        denominator: None,
        order,
        frequency_response: Some(freq_response),
        design_error: best_error,
        iterations,
    })
}

/// Least squares filter design with arbitrary constraints
///
/// # Arguments
///
/// * `response` - Desired response specification
/// * `order` - Filter order
///
/// # Returns
///
/// * Least squares optimal filter coefficients
pub fn least_squares_design(
    response: &ArbitraryResponse,
    order: usize,
) -> SignalResult<FilterDesignResult> {
    let n_freqs = response.frequencies.len();
    let n_coeffs = order + 1;

    // Create design matrix
    let mut design_matrix = Array2::zeros((n_freqs, n_coeffs));

    for (i, &freq) in response.frequencies.iter().enumerate() {
        for j in 0..n_coeffs {
            let omega = PI * freq;
            if j == 0 {
                design_matrix[[i, j]] = 1.0;
            } else {
                design_matrix[[i, j]] = 2.0 * (omega * j as f64).cos();
            }
        }
    }

    // Weight the equations
    for i in 0..n_freqs {
        let weight = response.weights[i].sqrt();
        for j in 0..n_coeffs {
            design_matrix[[i, j]] *= weight;
        }
    }

    // Weighted desired response
    let weighted_desired: Array1<f64> = response
        .magnitude
        .iter()
        .zip(response.weights.iter())
        .map(|(&mag, &weight)| mag * weight.sqrt())
        .collect();

    // Solve least squares problem: A * h = b
    // For small problems, we can use the normal equations approach: A^T A x = A^T b
    use scirs2_linalg::solve;

    let at = design_matrix.t();
    let ata = at.dot(&design_matrix);
    let atb = at.dot(&weighted_desired);

    // Add small regularization to ensure numerical stability
    let mut ata_reg = ata.clone();
    let eps = 1e-10;
    for i in 0..ata_reg.nrows() {
        ata_reg[[i, i]] += eps;
    }

    let coefficients = solve(&ata_reg.view(), &atb.view(), None).map_err(|e| {
        SignalError::Compute(format!("Failed to solve least squares problem: {}", e))
    })?;

    // Compute design error
    let estimated_response: Array1<f64> = design_matrix.dot(&coefficients);
    let error: f64 = estimated_response
        .iter()
        .zip(weighted_desired.iter())
        .map(|(&est, &des)| (est - des).powi(2))
        .sum::<f64>()
        .sqrt();

    let freq_response = compute_frequency_response(&coefficients, 512)?;

    Ok(FilterDesignResult {
        numerator: coefficients,
        denominator: None,
        order,
        frequency_response: Some(freq_response),
        design_error: error,
        iterations: 1,
    })
}

/// Constrained least squares design (for linear phase filters)
///
/// # Arguments
///
/// * `response` - Desired response specification
/// * `order` - Filter order
/// * `phase_constraints` - Linear phase constraints
///
/// # Returns
///
/// * Constrained optimal filter coefficients
pub fn constrained_least_squares_design(
    response: &ArbitraryResponse,
    order: usize,
    phase_constraints: &[(f64, f64)], // (frequency, phase) pairs
) -> SignalResult<FilterDesignResult> {
    let n_freqs = response.frequencies.len();
    let n_coeffs = order + 1;
    let n_constraints = phase_constraints.len();

    // Augmented system: [A; C] * h = [b; d]
    let mut augmented_matrix = Array2::zeros((n_freqs + n_constraints, n_coeffs));
    let mut augmented_rhs = Array1::zeros(n_freqs + n_constraints);

    // Fill magnitude equations
    for (i, &freq) in response.frequencies.iter().enumerate() {
        let weight = response.weights[i].sqrt();
        for j in 0..n_coeffs {
            let omega = PI * freq;
            if j == 0 {
                augmented_matrix[[i, j]] = weight;
            } else {
                augmented_matrix[[i, j]] = weight * 2.0 * (omega * j as f64).cos();
            }
        }
        augmented_rhs[i] = response.magnitude[i] * weight;
    }

    // Fill phase constraint equations
    for (i, &(freq, phase)) in phase_constraints.iter().enumerate() {
        let row_idx = n_freqs + i;
        for j in 0..n_coeffs {
            let omega = PI * freq;
            augmented_matrix[[row_idx, j]] = -2.0 * (omega * j as f64).sin();
        }
        augmented_rhs[row_idx] = phase;
    }

    // Solve constrained least squares using normal equations
    use scirs2_linalg::solve;

    let at = augmented_matrix.t();
    let ata = at.dot(&augmented_matrix);
    let atb = at.dot(&augmented_rhs);

    // Add small regularization to ensure numerical stability
    let mut ata_reg = ata.clone();
    let eps = 1e-10;
    for i in 0..ata_reg.nrows() {
        ata_reg[[i, i]] += eps;
    }

    let coefficients = solve(&ata_reg.view(), &atb.view(), None).map_err(|e| {
        SignalError::Compute(format!("Failed to solve constrained least squares: {}", e))
    })?;

    // Compute error on magnitude response only
    let mut error = 0.0;
    for (i, &freq) in response.frequencies.iter().enumerate() {
        let mut magnitude = coefficients[0];
        for j in 1..n_coeffs {
            magnitude += 2.0 * coefficients[j] * (PI * freq * j as f64).cos();
        }
        error += response.weights[i] * (magnitude - response.magnitude[i]).powi(2);
    }
    error = error.sqrt();

    let freq_response = compute_frequency_response(&coefficients, 512)?;

    Ok(FilterDesignResult {
        numerator: coefficients,
        denominator: None,
        order,
        frequency_response: Some(freq_response),
        design_error: error,
        iterations: 1,
    })
}

/// Minimax (Chebyshev) filter design
///
/// # Arguments
///
/// * `response` - Desired response specification
/// * `order` - Filter order
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// * Minimax optimal filter coefficients
pub fn minimax_design(
    response: &ArbitraryResponse,
    order: usize,
    config: &ParksMcClellanConfig,
) -> SignalResult<FilterDesignResult> {
    // This is essentially the Parks-McClellan algorithm for arbitrary responses
    arbitrary_magnitude_design(response, order, config)
}

// Helper functions

/// Estimate required filter order from specifications
fn estimate_filter_order(spec: &FilterSpec) -> SignalResult<usize> {
    if spec.passband_freqs.is_empty() || spec.stopband_freqs.is_empty() {
        return Err(SignalError::ValueError(
            "Missing frequency specifications".to_string(),
        ));
    }

    // Kaiser's formula for order estimation
    let delta_p = (10.0_f64.powf(spec.passband_ripple / 20.0) - 1.0)
        / (10.0_f64.powf(spec.passband_ripple / 20.0) + 1.0);
    let delta_s = 10.0_f64.powf(-spec.stopband_attenuation / 20.0);
    let delta = delta_p.min(delta_s);

    let transition_width = match spec.filter_type {
        FilterType::Lowpass | FilterType::Highpass => {
            (spec.stopband_freqs[0] - spec.passband_freqs[0]).abs()
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            let width1 = (spec.passband_freqs[0] - spec.stopband_freqs[0]).abs();
            let width2 = (spec.stopband_freqs[1] - spec.passband_freqs[1]).abs();
            width1.min(width2)
        }
        _ => spec.sample_rate * 0.1, // Default 10% of sample rate
    };

    let normalized_width = transition_width / spec.sample_rate;

    // Kaiser's formula
    let a = -20.0 * delta.log10();
    let order = if a > 50.0 {
        ((a - 13.0) / (14.6 * normalized_width)).ceil() as usize
    } else if a > 21.0 {
        ((a - 7.95) / (14.36 * normalized_width)).ceil() as usize
    } else {
        (0.9222 / normalized_width).ceil() as usize
    };

    Ok(order.max(1))
}

/// Create frequency grid for filter design
fn create_design_grid(
    spec: &FilterSpec,
    config: &ParksMcClellanConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let nyquist = spec.sample_rate / 2.0;

    // Normalize frequencies
    let passband_norm: Vec<f64> = spec.passband_freqs.iter().map(|&f| f / nyquist).collect();
    let stopband_norm: Vec<f64> = spec.stopband_freqs.iter().map(|&f| f / nyquist).collect();

    // Create frequency grid based on filter type
    let (freq_bands, desired_bands, weight_bands) = match spec.filter_type {
        FilterType::Lowpass => {
            let freqs = vec![
                Array1::linspace(0.0, passband_norm[0], config.grid_density),
                Array1::linspace(stopband_norm[0], 1.0, config.grid_density),
            ];
            let desired = vec![
                Array1::ones(config.grid_density),
                Array1::zeros(config.grid_density),
            ];
            let weights = vec![
                Array1::ones(config.grid_density),
                Array1::ones(config.grid_density),
            ];
            (freqs, desired, weights)
        }
        FilterType::Highpass => {
            let freqs = vec![
                Array1::linspace(0.0, stopband_norm[0], config.grid_density),
                Array1::linspace(passband_norm[0], 1.0, config.grid_density),
            ];
            let desired = vec![
                Array1::zeros(config.grid_density),
                Array1::ones(config.grid_density),
            ];
            let weights = vec![
                Array1::ones(config.grid_density),
                Array1::ones(config.grid_density),
            ];
            (freqs, desired, weights)
        }
        FilterType::Bandpass => {
            let freqs = vec![
                Array1::linspace(0.0, stopband_norm[0], config.grid_density),
                Array1::linspace(passband_norm[0], passband_norm[1], config.grid_density),
                Array1::linspace(stopband_norm[1], 1.0, config.grid_density),
            ];
            let desired = vec![
                Array1::zeros(config.grid_density),
                Array1::ones(config.grid_density),
                Array1::zeros(config.grid_density),
            ];
            let weights = vec![
                Array1::ones(config.grid_density),
                Array1::ones(config.grid_density),
                Array1::ones(config.grid_density),
            ];
            (freqs, desired, weights)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unsupported filter type".to_string(),
            ))
        }
    };

    // Concatenate all bands
    let total_points: usize = freq_bands.iter().map(|band| band.len()).sum();
    let mut frequencies = Array1::zeros(total_points);
    let mut desired = Array1::zeros(total_points);
    let mut weights = Array1::zeros(total_points);

    let mut offset = 0;
    for (i, freq_band) in freq_bands.iter().enumerate() {
        let band_len = freq_band.len();
        frequencies
            .slice_mut(s![offset..offset + band_len])
            .assign(freq_band);
        desired
            .slice_mut(s![offset..offset + band_len])
            .assign(&desired_bands[i]);
        weights
            .slice_mut(s![offset..offset + band_len])
            .assign(&weight_bands[i]);
        offset += band_len;
    }

    Ok((frequencies, desired, weights))
}

/// Initialize extremal frequencies for Parks-McClellan algorithm
fn initialize_extremal_frequencies(freq_grid: &Array1<f64>, num_extremal: usize) -> Array1<f64> {
    let grid_len = freq_grid.len();
    let mut extremal = Array1::zeros(num_extremal);

    // Evenly space extremal frequencies across the grid
    for i in 0..num_extremal {
        let idx = (i * (grid_len - 1)) / (num_extremal - 1);
        extremal[i] = freq_grid[idx];
    }

    extremal
}

/// Solve interpolation problem at extremal frequencies
fn solve_interpolation_problem(
    extremal_freqs: &Array1<f64>,
    desired_response: &Array1<f64>,
    weights: &Array1<f64>,
    freq_grid: &Array1<f64>,
    order: usize,
) -> SignalResult<Array1<f64>> {
    let num_extremal = extremal_freqs.len();

    // Find desired response and weights at extremal frequencies
    let mut extremal_desired = Array1::zeros(num_extremal);
    let mut extremal_weights = Array1::zeros(num_extremal);

    for (i, &freq) in extremal_freqs.iter().enumerate() {
        // Find closest frequency in grid
        let closest_idx = freq_grid
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - freq).abs())
                    .partial_cmp(&((**b - freq).abs()))
                    .unwrap()
            })
            .unwrap()
            .0;

        extremal_desired[i] = desired_response[closest_idx];
        extremal_weights[i] = weights[closest_idx];
    }

    // Set up interpolation matrix (simplified Remez exchange)
    let mut interpolation_matrix = Array2::zeros((num_extremal, order + 2));
    let mut rhs = Array1::zeros(num_extremal);

    for (i, &freq) in extremal_freqs.iter().enumerate() {
        // Cosine basis functions
        for j in 0..=order {
            interpolation_matrix[[i, j]] = (PI * freq * j as f64).cos();
        }
        // Error term
        interpolation_matrix[[i, order + 1]] = if i % 2 == 0 { 1.0 } else { -1.0 };

        rhs[i] = extremal_desired[i];
    }

    // Solve the system
    use scirs2_linalg::solve;
    let solution = solve(&interpolation_matrix.view(), &rhs.view(), None).map_err(|e| {
        SignalError::Compute(format!("Failed to solve interpolation system: {}", e))
    })?;

    // Extract filter coefficients (excluding error term)
    Ok(solution.slice(s![0..=order]).to_owned())
}

/// Compute error function on frequency grid
fn compute_error_function(
    coefficients: &Array1<f64>,
    freq_grid: &Array1<f64>,
    desired_response: &Array1<f64>,
    weights: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let mut error = Array1::zeros(freq_grid.len());

    for (i, &freq) in freq_grid.iter().enumerate() {
        // Compute filter response at this frequency
        let mut response = coefficients[0];
        for (j, &coeff) in coefficients.iter().enumerate().skip(1) {
            response += coeff * (PI * freq * j as f64).cos();
        }

        // Weighted error
        error[i] = weights[i] * (response - desired_response[i]);
    }

    Ok(error)
}

/// Find extremal frequencies (peaks) in error function
fn find_extremal_frequencies(
    error_function: &Array1<f64>,
    freq_grid: &Array1<f64>,
    num_extremal: usize,
) -> SignalResult<Array1<f64>> {
    // Find local maxima and minima
    let mut extremal_indices = Vec::new();

    // Add first point
    extremal_indices.push(0);

    // Find interior extrema
    for i in 1..error_function.len() - 1 {
        let is_max =
            error_function[i] > error_function[i - 1] && error_function[i] > error_function[i + 1];
        let is_min =
            error_function[i] < error_function[i - 1] && error_function[i] < error_function[i + 1];

        if is_max || is_min {
            extremal_indices.push(i);
        }
    }

    // Add last point
    extremal_indices.push(error_function.len() - 1);

    // Sort by absolute error magnitude and take the largest ones
    extremal_indices.sort_by(|&a, &b| {
        error_function[b]
            .abs()
            .partial_cmp(&error_function[a].abs())
            .unwrap()
    });

    // Take the required number of extremal points
    extremal_indices.truncate(num_extremal);
    extremal_indices.sort();

    // Convert indices to frequencies
    let mut extremal_freqs = Array1::zeros(extremal_indices.len());
    for (i, &idx) in extremal_indices.iter().enumerate() {
        extremal_freqs[i] = freq_grid[idx];
    }

    Ok(extremal_freqs)
}

/// Interpolate response values at new frequency points
fn interpolate_response(
    freq_points: &Array1<f64>,
    response_values: &Array1<f64>,
    new_freq_points: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let mut interpolated = Array1::zeros(new_freq_points.len());

    for (i, &new_freq) in new_freq_points.iter().enumerate() {
        // Find surrounding points for linear interpolation
        if new_freq <= freq_points[0] {
            interpolated[i] = response_values[0];
        } else if new_freq >= freq_points[freq_points.len() - 1] {
            interpolated[i] = response_values[response_values.len() - 1];
        } else {
            // Find interpolation points
            let mut lower_idx = 0;
            for j in 0..freq_points.len() - 1 {
                if freq_points[j] <= new_freq && new_freq <= freq_points[j + 1] {
                    lower_idx = j;
                    break;
                }
            }

            let upper_idx = lower_idx + 1;
            let t = (new_freq - freq_points[lower_idx])
                / (freq_points[upper_idx] - freq_points[lower_idx]);

            interpolated[i] =
                response_values[lower_idx] * (1.0 - t) + response_values[upper_idx] * t;
        }
    }

    Ok(interpolated)
}

/// Compute frequency response of FIR filter
fn compute_frequency_response(
    coefficients: &Array1<f64>,
    num_points: usize,
) -> SignalResult<(Array1<f64>, Array1<Complex64>)> {
    let frequencies = Array1::linspace(0.0, 1.0, num_points);
    let mut response = Array1::zeros(num_points);

    for (i, &freq) in frequencies.iter().enumerate() {
        let mut real_part = 0.0;
        let mut imag_part = 0.0;

        for (j, &coeff) in coefficients.iter().enumerate() {
            let angle = PI * freq * j as f64;
            real_part += coeff * angle.cos();
            imag_part -= coeff * angle.sin();
        }

        response[i] = Complex64::new(real_part, imag_part);
    }

    Ok((frequencies, response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filter_order_estimation() {
        let spec = FilterSpec {
            filter_type: FilterType::Lowpass,
            sample_rate: 1000.0,
            passband_freqs: vec![100.0],
            stopband_freqs: vec![200.0],
            passband_ripple: 1.0,
            stopband_attenuation: 60.0,
            order: None,
        };

        let estimated_order = estimate_filter_order(&spec).unwrap();
        assert!(estimated_order > 0);
        assert!(estimated_order < 1000); // Reasonable bounds
    }

    #[test]
    fn test_design_grid_creation() {
        let spec = FilterSpec {
            filter_type: FilterType::Lowpass,
            sample_rate: 1000.0,
            passband_freqs: vec![100.0],
            stopband_freqs: vec![200.0],
            passband_ripple: 1.0,
            stopband_attenuation: 60.0,
            order: Some(32),
        };

        let config = ParksMcClellanConfig::default();
        let (frequencies, desired, weights) = create_design_grid(&spec, &config).unwrap();

        assert_eq!(frequencies.len(), desired.len());
        assert_eq!(frequencies.len(), weights.len());
        assert!(!frequencies.is_empty());

        // Check frequency ordering
        for i in 1..frequencies.len() {
            assert!(frequencies[i] >= frequencies[i - 1]);
        }
    }

    #[test]
    fn test_arbitrary_response() {
        let frequencies = Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let magnitude = Array1::from_vec(vec![1.0, 1.0, 0.5, 0.5, 0.0, 0.0]);
        let weights = Array1::ones(6);

        let response = ArbitraryResponse {
            frequencies,
            magnitude,
            weights,
            phase: None,
        };

        let config = ParksMcClellanConfig::default();
        let result = arbitrary_magnitude_design(&response, 16, &config).unwrap();

        assert_eq!(result.numerator.len(), 17); // order + 1
        assert_eq!(result.order, 16);
        assert!(result.design_error >= 0.0);
    }

    #[test]
    fn test_least_squares_design() {
        let frequencies = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let magnitude = Array1::from_vec(vec![1.0, 0.7, 0.5, 0.3, 0.0]);
        let weights = Array1::ones(5);

        let response = ArbitraryResponse {
            frequencies,
            magnitude,
            weights,
            phase: None,
        };

        let result = least_squares_design(&response, 8).unwrap();

        assert_eq!(result.numerator.len(), 9);
        assert_eq!(result.order, 8);
        assert!(result.design_error >= 0.0);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_extremal_frequency_initialization() {
        let freq_grid = Array1::linspace(0.0, 1.0, 100);
        let extremal = initialize_extremal_frequencies(&freq_grid, 5);

        assert_eq!(extremal.len(), 5);
        assert_relative_eq!(extremal[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(extremal[4], 1.0, epsilon = 1e-10);

        // Check ordering
        for i in 1..extremal.len() {
            assert!(extremal[i] > extremal[i - 1]);
        }
    }

    #[test]
    fn test_frequency_response_computation() {
        let coefficients = Array1::from_vec(vec![0.5, 0.5]); // Simple averaging filter
        let (frequencies, response) = compute_frequency_response(&coefficients, 64).unwrap();

        assert_eq!(frequencies.len(), 64);
        assert_eq!(response.len(), 64);

        // At DC (freq = 0), response should be sum of coefficients
        assert_relative_eq!(response[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(response[0].im, 0.0, epsilon = 1e-10);

        // At Nyquist (freq = 1), response should be alternating sum
        assert_relative_eq!(response[63].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_response_interpolation() {
        let freq_points = Array1::from_vec(vec![0.0, 0.5, 1.0]);
        let response_values = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let new_freq_points = Array1::from_vec(vec![0.25, 0.75]);

        let interpolated =
            interpolate_response(&freq_points, &response_values, &new_freq_points).unwrap();

        assert_eq!(interpolated.len(), 2);
        assert_relative_eq!(interpolated[0], 0.5, epsilon = 1e-10); // Linear interpolation
        assert_relative_eq!(interpolated[1], 0.5, epsilon = 1e-10);
    }
}
