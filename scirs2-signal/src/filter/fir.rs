//! FIR (Finite Impulse Response) filter design functions
//!
//! This module provides comprehensive FIR filter design capabilities including
//! window-based design (firwin) and optimal equiripple design (Parks-McClellan/Remez).
//! FIR filters offer linear phase response and guaranteed stability.

use crate::error::{SignalError, SignalResult};
use num_traits::{Float, NumCast};
use std::fmt::Debug;

use super::common::validation::validate_cutoff_frequency;

/// FIR filter design using window method
///
/// Designs a linear phase FIR filter using the window method. The filter
/// is obtained by truncating and windowing the ideal impulse response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is Nyquist frequency)
/// * `window` - Window function name ("hamming", "hann", "blackman", "kaiser", etc.)
/// * `pass_zero` - If true, the filter is lowpass; if false, highpass
///
/// # Returns
///
/// * Filter coefficients as a vector
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::fir::firwin;
///
/// // Design a 65-tap lowpass filter with Hamming window
/// let h = firwin(65, 0.3, "hamming", true).unwrap();
///
/// // Design a highpass filter
/// let h = firwin(65, 0.3, "hamming", false).unwrap();
/// ```
pub fn firwin<T>(numtaps: usize, cutoff: T, window: &str, pass_zero: bool) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    let wc = validate_cutoff_frequency(cutoff)?;

    // Calculate the ideal impulse response
    let mid = (numtaps - 1) as f64 / 2.0;
    let mut h = vec![0.0; numtaps];

    for (i, item) in h.iter_mut().enumerate() {
        let n = i as f64 - mid;

        if n == 0.0 {
            // At n=0, use L'Hôpital's rule result
            *item = if pass_zero {
                wc / std::f64::consts::PI
            } else {
                1.0 - wc / std::f64::consts::PI
            };
        } else {
            // General case: sinc function
            let sinc_val = (wc * std::f64::consts::PI * n).sin() / (std::f64::consts::PI * n);
            *item = if pass_zero {
                sinc_val
            } else {
                // Highpass: subtract lowpass from delta function
                if i == numtaps / 2 {
                    1.0 - sinc_val
                } else {
                    -sinc_val
                }
            };
        }
    }

    // Apply window function
    let window_coeffs = generate_window(numtaps, window)?;
    for (i, coeff) in h.iter_mut().enumerate() {
        *coeff *= window_coeffs[i];
    }

    // Normalize to ensure unity gain at DC (for lowpass) or Nyquist (for highpass)
    let sum: f64 = h.iter().sum();
    if pass_zero && sum.abs() > 1e-10 {
        for coeff in &mut h {
            *coeff /= sum;
        }
    } else if !pass_zero {
        // For highpass, normalize for unity gain at Nyquist
        let nyquist_response: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &coeff)| coeff * (-1.0_f64).powi(i as i32))
            .sum();
        if nyquist_response.abs() > 1e-10 {
            for coeff in &mut h {
                *coeff /= nyquist_response;
            }
        }
    }

    Ok(h)
}

/// Parks-McClellan optimal FIR filter design (Remez exchange algorithm)
///
/// Design a linear phase FIR filter using the Parks-McClellan algorithm.
/// The algorithm finds the filter coefficients that minimize the maximum
/// error between the desired and actual frequency response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `bands` - Frequency bands specified as pairs of band edges (0 to 1, where 1 is Nyquist)
/// * `desired` - Desired gain for each band
/// * `weights` - Relative weights for each band (optional)
/// * `max_iter` - Maximum number of iterations (default: 25)
/// * `grid_density` - Grid density for frequency sampling (default: 16)
///
/// # Returns
///
/// * Filter coefficients as a vector
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::fir::remez;
///
/// // Design a 65-tap lowpass filter
/// // Passband: 0-0.4, Stopband: 0.45-1.0
/// let bands = vec![0.0, 0.4, 0.45, 1.0];
/// let desired = vec![1.0, 1.0, 0.0, 0.0];
/// let h = remez(65, &bands, &desired, None, None, None).unwrap();
/// ```
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weights: Option<&[f64]>,
    max_iter: Option<usize>,
    grid_density: Option<usize>,
) -> SignalResult<Vec<f64>> {
    // Validate inputs
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if bands.len() % 2 != 0 || bands.len() < 2 {
        return Err(SignalError::ValueError(
            "Bands must be specified as pairs of edges".to_string(),
        ));
    }

    if desired.len() != bands.len() {
        return Err(SignalError::ValueError(
            "Desired array must have same length as bands".to_string(),
        ));
    }

    // Check that bands are monotonically increasing
    for i in 1..bands.len() {
        if bands[i] <= bands[i - 1] {
            return Err(SignalError::ValueError(
                "Band edges must be monotonically increasing".to_string(),
            ));
        }
    }

    // Check that bands are within [0, 1]
    if bands[0] < 0.0 || bands[bands.len() - 1] > 1.0 {
        return Err(SignalError::ValueError(
            "Band edges must be between 0 and 1".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(25);
    let grid_density = grid_density.unwrap_or(16);

    // Calculate filter order
    let filter_order = numtaps - 1;

    // Number of extremal frequencies
    let r = (filter_order + 2) / 2;

    // Set up the dense frequency grid
    let grid_size = grid_density * filter_order;
    let mut omega_grid = Vec::with_capacity(grid_size);
    let mut desired_grid = Vec::with_capacity(grid_size);
    let mut weight_grid = Vec::with_capacity(grid_size);

    // Build the frequency grid for each band
    let num_bands = bands.len() / 2;
    for band_idx in 0..num_bands {
        let band_start = bands[2 * band_idx];
        let band_end = bands[2 * band_idx + 1];
        let band_points = ((band_end - band_start) * grid_size as f64).round() as usize;

        for i in 0..band_points {
            let omega =
                band_start + (band_end - band_start) * (i as f64) / (band_points as f64 - 1.0);
            omega_grid.push(omega * std::f64::consts::PI);

            // Linear interpolation for desired response
            let t = (omega - band_start) / (band_end - band_start);
            let des = desired[2 * band_idx] * (1.0 - t) + desired[2 * band_idx + 1] * t;
            desired_grid.push(des);

            // Set weights
            if let Some(w) = weights {
                let wt = w[2 * band_idx] * (1.0 - t) + w[2 * band_idx + 1] * t;
                weight_grid.push(wt);
            } else {
                weight_grid.push(1.0);
            }
        }
    }

    // Initialize extremal frequencies uniformly
    let mut extremal_freqs = Vec::with_capacity(r);
    for i in 0..r {
        extremal_freqs.push(i * (omega_grid.len() - 1) / (r - 1));
    }

    // Remez exchange algorithm
    let mut h = vec![0.0; numtaps];
    let mut best_error = f64::MAX;

    for _iter in 0..max_iter {
        // Step 1: Calculate the polynomial using the extremal frequencies
        let mut a_matrix = vec![vec![0.0; r]; r];
        let mut b_vector = vec![0.0; r];

        for (i, &ext_idx) in extremal_freqs.iter().enumerate() {
            let omega = omega_grid[ext_idx];

            // Fill the matrix for the linear system
            for j in 0..(r - 1) {
                a_matrix[i][j] = (j as f64 * omega).cos();
            }
            // Last column alternates signs
            a_matrix[i][r - 1] = if i % 2 == 0 { 1.0 } else { -1.0 } / weight_grid[ext_idx];

            b_vector[i] = desired_grid[ext_idx];
        }

        // Solve the linear system to get polynomial coefficients
        let coeffs = solve_linear_system(&a_matrix, &b_vector)?;

        // Step 2: Calculate error on the dense grid
        let mut errors = Vec::with_capacity(omega_grid.len());
        let mut max_error = 0.0;

        for i in 0..omega_grid.len() {
            let omega = omega_grid[i];

            // Evaluate the polynomial
            let mut p_omega = 0.0;
            for (j, &coeff) in coeffs.iter().enumerate().take(r - 1) {
                p_omega += coeff * (j as f64 * omega).cos();
            }

            let error = (desired_grid[i] - p_omega) * weight_grid[i];
            errors.push(error.abs());

            if error.abs() > max_error {
                max_error = error.abs();
            }
        }

        // Step 3: Find new extremal frequencies
        let mut new_extremal = Vec::new();

        // Find local maxima in the error function
        for i in 1..(errors.len() - 1) {
            if errors[i] >= errors[i - 1] && errors[i] >= errors[i + 1] {
                new_extremal.push(i);
            }
        }

        // Add boundaries if they are extremal
        if errors[0] > errors[1] {
            new_extremal.insert(0, 0);
        }
        if errors[errors.len() - 1] > errors[errors.len() - 2] {
            new_extremal.push(errors.len() - 1);
        }

        // Select r extremal points with alternating signs
        if new_extremal.len() >= r {
            // Sort by error magnitude
            new_extremal.sort_by(|&a, &b| errors[b].partial_cmp(&errors[a]).unwrap());

            // Keep the r largest errors
            new_extremal.truncate(r);
            new_extremal.sort();

            extremal_freqs = new_extremal;
        }

        // Check convergence
        if max_error < best_error {
            best_error = max_error;

            // Convert polynomial coefficients to filter coefficients
            for (i, coeff) in h.iter_mut().enumerate() {
                let n = i as f64 - (numtaps as f64 - 1.0) / 2.0;

                *coeff = 0.0;
                for (j, &c) in coeffs.iter().enumerate().take(r - 1) {
                    if j == 0 {
                        *coeff += c;
                    } else {
                        let freq = j as f64 * std::f64::consts::PI / (numtaps as f64 - 1.0);
                        *coeff += 2.0 * c * (freq * n).cos();
                    }
                }
                *coeff /= numtaps as f64;
            }
        }

        // Check if converged
        if max_error - best_error < 1e-10 {
            break;
        }
    }

    // Make filter symmetric
    let mid = numtaps / 2;
    for i in 0..mid {
        let avg = (h[i] + h[numtaps - 1 - i]) / 2.0;
        h[i] = avg;
        h[numtaps - 1 - i] = avg;
    }

    Ok(h)
}

/// Generate a window function
///
/// Creates a window function of the specified type and length.
///
/// # Arguments
///
/// * `length` - Window length
/// * `window_type` - Window type ("hamming", "hann", "blackman", "kaiser", etc.)
///
/// # Returns
///
/// * Window coefficients as a vector
fn generate_window(length: usize, window_type: &str) -> SignalResult<Vec<f64>> {
    let mut window = vec![0.0; length];

    match window_type.to_lowercase().as_str() {
        "hamming" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = length as f64;
                *w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos();
            }
        }
        "hann" | "hanning" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = length as f64;
                *w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos());
            }
        }
        "blackman" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = length as f64;
                let arg = 2.0 * std::f64::consts::PI * n / (total - 1.0);
                *w = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
            }
        }
        "rectangular" | "boxcar" => {
            window.fill(1.0);
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}. Supported types: hamming, hann, blackman, rectangular",
                window_type
            )));
        }
    }

    Ok(window)
}

/// Solve a linear system Ax = b using Gaussian elimination
///
/// Internal helper function for the Remez exchange algorithm.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err(SignalError::ValueError(
            "Invalid matrix dimensions".to_string(),
        ));
    }

    // Create augmented matrix
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug.swap(i, max_row);

        // Check for zero pivot
        if aug[i][i].abs() < 1e-10 {
            return Err(SignalError::ValueError("Singular matrix".to_string()));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}
