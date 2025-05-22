//! Empirical Mode Decomposition (EMD) and its variants
//!
//! This module provides implementations of Empirical Mode Decomposition (EMD)
//! and related techniques for analyzing non-stationary and nonlinear signals.
//! EMD decomposes a signal into a set of Intrinsic Mode Functions (IMFs),
//! which represent different oscillatory modes embedded in the signal.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};
use num_traits::{Float, NumCast};
use std::cmp::max;
use std::fmt::Debug;

/// Configuration parameters for Empirical Mode Decomposition
#[derive(Debug, Clone)]
pub struct EmdConfig {
    /// Maximum number of IMFs to extract
    pub max_imfs: usize,
    /// Sifting stop criterion threshold
    pub sift_threshold: f64,
    /// Maximum number of sifting iterations
    pub max_sift_iterations: usize,
    /// Boundary condition handling method
    pub boundary_condition: String,
    /// Envelope interpolation method
    pub interpolation: String,
    /// Minimum number of extrema to continue decomposition
    pub min_extrema: usize,
}

impl Default for EmdConfig {
    fn default() -> Self {
        Self {
            max_imfs: 10,
            sift_threshold: 0.05,
            max_sift_iterations: 100,
            boundary_condition: "mirror".to_string(),
            interpolation: "cubic".to_string(),
            min_extrema: 3,
        }
    }
}

/// Result of EMD decomposition
#[derive(Debug, Clone)]
pub struct EmdResult {
    /// Intrinsic Mode Functions (IMFs)
    pub imfs: Array2<f64>,
    /// Residual trend
    pub residue: Array1<f64>,
    /// Number of iterations for each IMF
    pub iterations: Vec<usize>,
    /// Energy of each IMF
    pub energies: Vec<f64>,
}

/// Performs Empirical Mode Decomposition (EMD) on a signal
///
/// EMD decomposes a signal into a sum of Intrinsic Mode Functions (IMFs),
/// which represent different oscillatory modes embedded in the signal.
/// Each IMF satisfies two conditions:
/// 1. The number of extrema and zero crossings must differ by at most one
/// 2. The mean of the upper and lower envelopes must be close to zero
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - EMD configuration parameters
///
/// # Returns
///
/// * EMD result containing IMFs, residue, and additional information
///
/// # Examples
///
/// ```
/// use scirs2_signal::emd::{emd, EmdConfig};
/// use std::f64::consts::PI;
///
/// // Generate a test signal (sum of two sinusoids)
/// let n = 1000;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / 100.0).collect();
/// let signal: Vec<f64> = t.iter().map(|&ti| {
///     (2.0 * PI * 2.0 * ti).sin() + 0.5 * (2.0 * PI * 10.0 * ti).sin()
/// }).collect();
///
/// // Configure and apply EMD
/// let mut config = EmdConfig::default();
/// config.max_imfs = 3;
///
/// let result = emd(&signal, &config).unwrap();
///
/// // The number of IMFs should be at most max_imfs
/// assert!(result.imfs.shape()[0] <= config.max_imfs);
/// assert_eq!(result.imfs.shape()[1], signal.len());
/// ```
pub fn emd<T>(signal: &[T], config: &EmdConfig) -> SignalResult<EmdResult>
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Convert input to f64
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Convert to Array1 for easier manipulation
    let mut residue = Array1::from(signal_f64.clone());
    let n = residue.len();

    // Store IMFs, iterations for each IMF, and energy of each IMF
    let mut imfs = Vec::new();
    let mut iterations = Vec::new();
    let mut energies = Vec::new();

    // Main EMD loop to extract IMFs
    for _ in 0..config.max_imfs {
        // Check if residue has enough extrema to continue
        let (num_maxima, num_minima) = count_extrema(&residue);
        let total_extrema = num_maxima + num_minima;

        if total_extrema < config.min_extrema {
            break;
        }

        // Extract an IMF from current residue
        let (imf, num_iter) = extract_imf(&residue, config)?;

        // Calculate IMF energy
        let energy = imf.iter().map(|&x| x * x).sum::<f64>() / n as f64;

        // Store results
        imfs.push(imf.clone());
        iterations.push(num_iter);
        energies.push(energy);

        // Update residue for next iteration
        residue -= &imf;

        // Check energy ratio of residue to determine stopping
        let residue_energy = residue.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        let original_energy = signal_f64.iter().map(|&x| x * x).sum::<f64>() / n as f64;

        if residue_energy < 1e-10 || residue_energy / original_energy < 1e-2 {
            break;
        }
    }

    // Convert imfs vector to 2D array
    let num_imfs = imfs.len();
    if num_imfs == 0 {
        return Err(SignalError::ComputationError(
            "Failed to extract any IMFs".to_string(),
        ));
    }

    let mut imfs_array = Array2::zeros((num_imfs, n));
    for (i, imf) in imfs.iter().enumerate() {
        imfs_array.slice_mut(s![i, ..]).assign(imf);
    }

    Ok(EmdResult {
        imfs: imfs_array,
        residue,
        iterations,
        energies,
    })
}

/// Extracts an Intrinsic Mode Function (IMF) from a signal
///
/// This internal function performs the sifting process to extract a single IMF
/// according to the EMD algorithm.
///
/// # Arguments
///
/// * `signal` - Input signal from which to extract IMF
/// * `config` - EMD configuration parameters
///
/// # Returns
///
/// * Tuple of (IMF, number of iterations)
fn extract_imf(signal: &Array1<f64>, config: &EmdConfig) -> SignalResult<(Array1<f64>, usize)> {
    let n = signal.len();
    let mut h = signal.clone();
    let mut iteration = 0;

    // Sifting process
    loop {
        // Make a copy to check for convergence
        let h_prev = h.clone();

        // Find local extrema
        let (maxima_idx, maxima_val) = find_local_maxima(&h);
        let (minima_idx, minima_val) = find_local_minima(&h);

        // Check if we have enough extrema to continue
        if maxima_idx.len() < 2 || minima_idx.len() < 2 {
            // If too few extrema in the first iteration, return the original signal
            if iteration == 0 {
                return Ok((h.clone(), 0));
            }
            // Otherwise, we've found an IMF
            break;
        }

        // Apply boundary conditions to extrema
        let (ext_maxima_idx, ext_maxima_val) =
            extend_extrema(&maxima_idx, &maxima_val, n, &config.boundary_condition)?;

        let (ext_minima_idx, ext_minima_val) =
            extend_extrema(&minima_idx, &minima_val, n, &config.boundary_condition)?;

        // Compute upper and lower envelopes using spline interpolation
        let upper_env =
            interpolate_envelope(&ext_maxima_idx, &ext_maxima_val, n, &config.interpolation)?;

        let lower_env =
            interpolate_envelope(&ext_minima_idx, &ext_minima_val, n, &config.interpolation)?;

        // Compute mean envelope
        let mut mean_env = Array1::zeros(n);
        for i in 0..n {
            mean_env[i] = (upper_env[i] + lower_env[i]) / 2.0;
        }

        // Update signal by subtracting the mean envelope
        h = &h - &mean_env;

        // Increment iteration counter
        iteration += 1;

        // Check stopping criteria
        if iteration >= config.max_sift_iterations {
            break;
        }

        // Compute stopping criterion (SD)
        let sd = compute_sifting_criterion(&h, &h_prev);
        if sd < config.sift_threshold {
            break;
        }
    }

    Ok((h, iteration))
}

/// Computes the stopping criterion for the sifting process
///
/// # Arguments
///
/// * `current` - Current signal
/// * `previous` - Previous signal
///
/// # Returns
///
/// * Sifting criterion value
fn compute_sifting_criterion(current: &Array1<f64>, previous: &Array1<f64>) -> f64 {
    let n = current.len();
    let mut sum_squared_diff = 0.0;
    let mut sum_squared_prev = 0.0;

    for i in 0..n {
        let diff = current[i] - previous[i];
        sum_squared_diff += diff * diff;
        sum_squared_prev += previous[i] * previous[i];
    }

    if sum_squared_prev > 0.0 {
        sum_squared_diff / sum_squared_prev
    } else {
        f64::MAX
    }
}

/// Finds local maxima in a signal
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Tuple of (indices, values) of local maxima
fn find_local_maxima(signal: &Array1<f64>) -> (Vec<usize>, Vec<f64>) {
    let n = signal.len();
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Handle short signals
    if n <= 2 {
        let max_idx = signal
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        indices.push(max_idx);
        values.push(signal[max_idx]);
        return (indices, values);
    }

    // First point
    if signal[0] > signal[1] {
        indices.push(0);
        values.push(signal[0]);
    }

    // Internal points
    for i in 1..n - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            indices.push(i);
            values.push(signal[i]);
        }
    }

    // Last point
    if signal[n - 1] > signal[n - 2] {
        indices.push(n - 1);
        values.push(signal[n - 1]);
    }

    (indices, values)
}

/// Finds local minima in a signal
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Tuple of (indices, values) of local minima
fn find_local_minima(signal: &Array1<f64>) -> (Vec<usize>, Vec<f64>) {
    let n = signal.len();
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Handle short signals
    if n <= 2 {
        let min_idx = signal
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        indices.push(min_idx);
        values.push(signal[min_idx]);
        return (indices, values);
    }

    // First point
    if signal[0] < signal[1] {
        indices.push(0);
        values.push(signal[0]);
    }

    // Internal points
    for i in 1..n - 1 {
        if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
            indices.push(i);
            values.push(signal[i]);
        }
    }

    // Last point
    if signal[n - 1] < signal[n - 2] {
        indices.push(n - 1);
        values.push(signal[n - 1]);
    }

    (indices, values)
}

/// Counts the number of extrema (maxima and minima) in a signal
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Tuple of (number of maxima, number of minima)
fn count_extrema(signal: &Array1<f64>) -> (usize, usize) {
    let (maxima_idx, _) = find_local_maxima(signal);
    let (minima_idx, _) = find_local_minima(signal);

    (maxima_idx.len(), minima_idx.len())
}

/// Extends extrema points based on boundary conditions
///
/// # Arguments
///
/// * `indices` - Indices of extrema
/// * `values` - Values of extrema
/// * `n` - Length of the signal
/// * `boundary_condition` - Boundary condition type
///
/// # Returns
///
/// * Extended extrema as tuple of (indices, values)
fn extend_extrema(
    indices: &[usize],
    values: &[f64],
    n: usize,
    boundary_condition: &str,
) -> SignalResult<(Vec<usize>, Vec<f64>)> {
    if indices.is_empty() {
        return Err(SignalError::ValueError(
            "Cannot extend empty extrema list".to_string(),
        ));
    }

    let mut ext_indices = indices.to_vec();
    let mut ext_values = values.to_vec();

    match boundary_condition.to_lowercase().as_str() {
        "mirror" => {
            // Add mirrored points at the boundaries
            if indices[0] > 0 {
                // Mirror before the first point
                ext_indices.insert(0, 0);
                ext_values.insert(0, values[0]);
            }

            if indices[indices.len() - 1] < n - 1 {
                // Mirror after the last point
                ext_indices.push(n - 1);
                ext_values.push(values[values.len() - 1]);
            }
        }
        "periodic" => {
            // Add periodic extension at the boundaries
            if indices[0] > 0 {
                // Add last extrema at the beginning with modified index
                ext_indices.insert(0, 0);
                ext_values.insert(0, values[values.len() - 1]);
            }

            if indices[indices.len() - 1] < n - 1 {
                // Add first extrema at the end with modified index
                ext_indices.push(n - 1);
                ext_values.push(values[0]);
            }
        }
        "symmetric" => {
            // Add symmetric extension
            if indices[0] > 0 {
                // Add symmetric point at the beginning
                ext_indices.insert(0, 0);

                // Extrapolate value
                let first_idx = indices[0] as f64;
                let second_idx = indices[1] as f64;
                let first_val = values[0];
                let second_val = values[1];

                let slope = (second_val - first_val) / (second_idx - first_idx);
                let extrapolated_val = first_val - slope * first_idx;

                ext_values.insert(0, extrapolated_val);
            }

            if indices[indices.len() - 1] < n - 1 {
                // Add symmetric point at the end
                ext_indices.push(n - 1);

                // Extrapolate value
                let last_idx = indices[indices.len() - 1] as f64;
                let penultimate_idx = indices[indices.len() - 2] as f64;
                let last_val = values[values.len() - 1];
                let penultimate_val = values[values.len() - 2];

                let slope = (last_val - penultimate_val) / (last_idx - penultimate_idx);
                let extrapolated_val = last_val + slope * ((n - 1) as f64 - last_idx);

                ext_values.push(extrapolated_val);
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown boundary condition: {}",
                boundary_condition
            )));
        }
    }

    Ok((ext_indices, ext_values))
}

/// Interpolates an envelope through extrema points
///
/// # Arguments
///
/// * `indices` - Indices of extrema points
/// * `values` - Values of extrema points
/// * `n` - Length of the signal
/// * `method` - Interpolation method
///
/// # Returns
///
/// * Interpolated envelope
fn interpolate_envelope(
    indices: &[usize],
    values: &[f64],
    n: usize,
    method: &str,
) -> SignalResult<Array1<f64>> {
    if indices.is_empty() {
        return Err(SignalError::ValueError(
            "Cannot interpolate empty extrema list".to_string(),
        ));
    }

    let mut envelope = Array1::zeros(n);

    match method.to_lowercase().as_str() {
        "linear" => {
            // Linear interpolation
            for i in 0..n {
                // Find the surrounding extrema
                let mut left_idx = 0;
                let mut right_idx = indices.len() - 1;

                for j in 0..indices.len() - 1 {
                    if i >= indices[j] && i <= indices[j + 1] {
                        left_idx = j;
                        right_idx = j + 1;
                        break;
                    }
                }

                // Interpolate
                if i <= indices[0] {
                    envelope[i] = values[0];
                } else if i >= indices[indices.len() - 1] {
                    envelope[i] = values[values.len() - 1];
                } else {
                    let x_left = indices[left_idx] as f64;
                    let x_right = indices[right_idx] as f64;
                    let y_left = values[left_idx];
                    let y_right = values[right_idx];

                    envelope[i] =
                        y_left + (i as f64 - x_left) * (y_right - y_left) / (x_right - x_left);
                }
            }
        }
        "cubic" => {
            // Cubic spline interpolation
            // For simplicity, we'll use a natural cubic spline

            let n_points = indices.len();
            if n_points < 3 {
                // Fall back to linear interpolation for too few points
                return interpolate_envelope(indices, values, n, "linear");
            }

            // Compute second derivatives for natural cubic spline
            let mut a = vec![0.0; n_points];
            let mut b = vec![0.0; n_points];
            let mut _c = vec![0.0; n_points]; // Unused, prefixed with underscore
            let mut d = vec![0.0; n_points];
            let mut h = vec![0.0; n_points - 1];

            // Step 1: Calculate h values
            for i in 0..n_points - 1 {
                h[i] = indices[i + 1] as f64 - indices[i] as f64;
            }

            // Step 2: Set up tridiagonal system
            let mut alpha = vec![0.0; n_points - 1];
            for i in 1..n_points - 1 {
                alpha[i] = 3.0
                    * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1]);
            }

            // Step 3: Solve the tridiagonal system
            let mut l = vec![0.0; n_points];
            let mut mu = vec![0.0; n_points];
            let mut z = vec![0.0; n_points];

            l[0] = 1.0;
            mu[0] = 0.0;
            z[0] = 0.0; // Natural spline boundary condition

            for i in 1..n_points - 1 {
                l[i] = 2.0 * (indices[i + 1] as f64 - indices[i - 1] as f64) - h[i - 1] * mu[i - 1];
                mu[i] = h[i] / l[i];
                z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
            }

            l[n_points - 1] = 1.0;
            z[n_points - 1] = 0.0; // Natural spline boundary condition

            // Step 4: Back-substitution
            let mut c_values = vec![0.0; n_points];
            c_values[n_points - 1] = 0.0; // Natural spline boundary condition

            for j in (0..n_points - 1).rev() {
                c_values[j] = z[j] - mu[j] * c_values[j + 1];
            }

            // Step 5: Compute remaining coefficients
            for i in 0..n_points - 1 {
                b[i] = (values[i + 1] - values[i]) / h[i]
                    - h[i] * (c_values[i + 1] + 2.0 * c_values[i]) / 3.0;
                d[i] = (c_values[i + 1] - c_values[i]) / (3.0 * h[i]);
                a[i] = values[i];
            }

            // Now interpolate using the cubic spline
            for i in 0..n {
                let x = i as f64;

                // Find the appropriate segment
                if x <= indices[0] as f64 {
                    envelope[i] = values[0];
                } else if x >= indices[n_points - 1] as f64 {
                    envelope[i] = values[n_points - 1];
                } else {
                    let mut segment = 0;
                    for j in 0..n_points - 1 {
                        if x >= indices[j] as f64 && x < indices[j + 1] as f64 {
                            segment = j;
                            break;
                        }
                    }

                    // Apply cubic spline formula
                    let dx = x - indices[segment] as f64;
                    envelope[i] = a[segment]
                        + b[segment] * dx
                        + c_values[segment] * dx.powi(2)
                        + d[segment] * dx.powi(3);
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown interpolation method: {}",
                method
            )));
        }
    }

    Ok(envelope)
}

/// Performs Ensemble Empirical Mode Decomposition (EEMD)
///
/// EEMD is a noise-assisted variant of EMD, which addresses the
/// mode mixing problem by adding different realizations of white
/// noise to the signal before decomposition.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - EMD configuration parameters
/// * `ensemble_size` - Number of ensemble trials
/// * `noise_std` - Standard deviation of the added white noise
///
/// # Returns
///
/// * EMD result containing IMFs, residue, and additional information
///
/// # Examples
///
/// ```
/// use scirs2_signal::emd::{eemd, EmdConfig};
/// use std::f64::consts::PI;
///
/// // Generate a test signal (sum of two sinusoids)
/// let n = 1000;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / 100.0).collect();
/// let signal: Vec<f64> = t.iter().map(|&ti| {
///     (2.0 * PI * 2.0 * ti).sin() + 0.5 * (2.0 * PI * 10.0 * ti).sin()
/// }).collect();
///
/// // Configure and apply EEMD
/// let mut config = EmdConfig::default();
/// config.max_imfs = 3;
///
/// let ensemble_size = 4; // Small value for testing (use 50-100 in practice)
/// let noise_std = 0.1;
///
/// let result = eemd(&signal, &config, ensemble_size, noise_std).unwrap();
///
/// // The number of IMFs should be at most max_imfs
/// assert!(result.imfs.shape()[0] <= config.max_imfs);
/// assert_eq!(result.imfs.shape()[1], signal.len());
/// ```
pub fn eemd<T>(
    signal: &[T],
    config: &EmdConfig,
    ensemble_size: usize,
    noise_std: f64,
) -> SignalResult<EmdResult>
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if ensemble_size < 1 {
        return Err(SignalError::ValueError(
            "Ensemble size must be at least 1".to_string(),
        ));
    }

    if noise_std < 0.0 {
        return Err(SignalError::ValueError(
            "Noise standard deviation must be non-negative".to_string(),
        ));
    }

    // Convert input to f64
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let n = signal_f64.len();
    let mut all_imfs = Vec::new();
    let mut max_imf_count = 0;

    // Run EMD on ensemble of signals with added noise
    for _ in 0..ensemble_size {
        // Add white noise
        let noisy_signal: Vec<f64> = if noise_std > 0.0 {
            // Direct use of random without creating an unused rng variable
            signal_f64
                .iter()
                .map(|&x| x + noise_std * (rand::random::<f64>() * 2.0 - 1.0))
                .collect()
        } else {
            signal_f64.clone()
        };

        // Apply EMD
        let emd_result = emd(&noisy_signal, config)?;
        let imfs = emd_result.imfs;

        // Keep track of maximum IMF count
        max_imf_count = max(max_imf_count, imfs.shape()[0]);

        // Store IMFs for ensemble averaging
        all_imfs.push(imfs);
    }

    // Prepare average IMFs
    let mut avg_imfs = Array2::zeros((max_imf_count, n));
    let mut imf_counts = vec![0; max_imf_count];

    // Accumulate IMFs from all ensemble members
    for imfs in &all_imfs {
        let num_imfs = imfs.shape()[0];
        for i in 0..num_imfs {
            // Accumulate IMF values
            for j in 0..n {
                avg_imfs[[i, j]] += imfs[[i, j]];
            }
            imf_counts[i] += 1;
        }
    }

    // Compute average by dividing by the number of accumulated IMFs
    for i in 0..max_imf_count {
        if imf_counts[i] > 0 {
            for j in 0..n {
                avg_imfs[[i, j]] /= imf_counts[i] as f64;
            }
        }
    }

    // Compute residue (signal - sum of IMFs)
    let mut residue = Array1::from(signal_f64.clone());
    for i in 0..max_imf_count {
        for j in 0..n {
            residue[j] -= avg_imfs[[i, j]];
        }
    }

    // Compute energies for each IMF
    let mut energies = Vec::with_capacity(max_imf_count);
    for i in 0..max_imf_count {
        let energy = avg_imfs.slice(s![i, ..]).map(|&x| x * x).sum() / n as f64;
        energies.push(energy);
    }

    // Since we're averaging, we don't have iteration counts per IMF
    let iterations = vec![ensemble_size; max_imf_count];

    Ok(EmdResult {
        imfs: avg_imfs,
        residue,
        iterations,
        energies,
    })
}

/// Computes the Hilbert-Huang spectrum of a signal
///
/// The Hilbert-Huang spectrum represents the energy distribution
/// in the time-frequency domain based on EMD decomposition and
/// the Hilbert transform.
///
/// # Arguments
///
/// * `emd_result` - Result of EMD or EEMD decomposition
/// * `sample_rate` - Sampling rate of the original signal
/// * `num_freqs` - Number of frequency bins in the output spectrum
///
/// # Returns
///
/// * Tuple of (time points, frequencies, Hilbert-Huang spectrum)
pub fn hilbert_huang_spectrum(
    emd_result: &EmdResult,
    sample_rate: f64,
    num_freqs: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)> {
    use crate::hilbert;

    // Validate input
    if emd_result.imfs.shape()[0] == 0 {
        return Err(SignalError::ValueError("No IMFs in EMD result".to_string()));
    }

    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }

    if num_freqs < 2 {
        return Err(SignalError::ValueError(
            "Number of frequency bins must be at least 2".to_string(),
        ));
    }

    let num_imfs = emd_result.imfs.shape()[0];
    let n = emd_result.imfs.shape()[1];

    // Create time vector
    let time_points: Vec<f64> = (0..n).map(|i| i as f64 / sample_rate).collect();

    // Create frequency vector (logarithmically spaced)
    let min_freq = 0.1; // Minimum frequency in Hz
    let max_freq = sample_rate / 2.0; // Nyquist frequency

    let log_min = min_freq.ln();
    let log_max = max_freq.ln();
    let log_step = (log_max - log_min) / (num_freqs - 1) as f64;

    let frequencies: Vec<f64> = (0..num_freqs)
        .map(|i| (log_min + i as f64 * log_step).exp())
        .collect();

    // Initialize the Hilbert-Huang spectrum
    let mut hhs = Array2::zeros((num_freqs, n));

    // Process each IMF
    for i in 0..num_imfs {
        // Extract IMF
        let imf: Vec<f64> = emd_result.imfs.slice(s![i, ..]).to_vec();

        // Compute analytic signal using Hilbert transform
        let analytic_signal = hilbert::hilbert(&imf)?;

        // Compute instantaneous amplitude (envelope)
        let amplitude: Vec<f64> = analytic_signal.iter().map(|c| c.norm()).collect();

        // Compute instantaneous phase
        let phase: Vec<f64> = analytic_signal.iter().map(|c| c.arg()).collect();

        // Compute instantaneous frequency
        let mut inst_freq = Vec::with_capacity(n);

        // First point (forward difference)
        let first_diff =
            (phase[1] - phase[0] + 2.0 * std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
        inst_freq.push(first_diff * sample_rate / (2.0 * std::f64::consts::PI));

        // Middle points (central difference for better accuracy)
        for j in 1..n - 1 {
            let mut diff = phase[j + 1] - phase[j - 1];

            // Unwrap phase difference
            while diff > std::f64::consts::PI {
                diff -= 2.0 * std::f64::consts::PI;
            }
            while diff < -std::f64::consts::PI {
                diff += 2.0 * std::f64::consts::PI;
            }

            inst_freq.push(diff * sample_rate / (4.0 * std::f64::consts::PI));
        }

        // Last point (backward difference)
        let last_diff = (phase[n - 1] - phase[n - 2] + 2.0 * std::f64::consts::PI)
            % (2.0 * std::f64::consts::PI);
        inst_freq.push(last_diff * sample_rate / (2.0 * std::f64::consts::PI));

        // Add energy contribution to HHS
        for j in 0..n {
            // Skip points with negative or zero frequency
            if inst_freq[j] <= 0.0 {
                continue;
            }

            // Find closest frequency bin
            let log_freq = inst_freq[j].ln();
            let idx = ((log_freq - log_min) / log_step).round() as usize;

            if idx < num_freqs {
                // Add squared amplitude (energy)
                hhs[[idx, j]] += amplitude[j] * amplitude[j];
            }
        }
    }

    Ok((time_points, frequencies, hhs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_find_local_extrema() {
        // Create a sine wave with explicit maxima and minima
        // Using exact periods to ensure we hit exact maxima/minima values
        let n = 100;
        let period = 20; // samples per period
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
            .collect();

        let signal_array = Array1::from(signal);

        // Find maxima and minima
        let (maxima_idx, maxima_val) = find_local_maxima(&signal_array);
        let (minima_idx, minima_val) = find_local_minima(&signal_array);

        // Sine wave with period 20 should have several maxima and minima
        assert!(!maxima_idx.is_empty(), "No maxima found");
        assert!(!minima_idx.is_empty(), "No minima found");

        // Print debug information
        println!("Maxima indices: {:?}", maxima_idx);
        println!("Maxima values: {:?}", maxima_val);
        println!("Minima indices: {:?}", minima_idx);
        println!("Minima values: {:?}", minima_val);

        // The last index (99) can be problematic because it's at the boundary
        // Let's filter out any extrema that are at indices 0 or n-1
        let filtered_maxima_idx: Vec<_> = maxima_idx
            .iter()
            .filter(|&&idx| idx > 0 && idx < n - 1)
            .cloned()
            .collect();

        let filtered_maxima_val: Vec<_> = maxima_idx
            .iter()
            .zip(maxima_val.iter())
            .filter(|&(&idx, _)| idx > 0 && idx < n - 1)
            .map(|(_, &val)| val)
            .collect();

        let filtered_minima_idx: Vec<_> = minima_idx
            .iter()
            .filter(|&&idx| idx > 0 && idx < n - 1)
            .cloned()
            .collect();

        let filtered_minima_val: Vec<_> = minima_idx
            .iter()
            .zip(minima_val.iter())
            .filter(|&(&idx, _)| idx > 0 && idx < n - 1)
            .map(|(_, &val)| val)
            .collect();

        println!("Filtered maxima indices: {:?}", filtered_maxima_idx);
        println!("Filtered minima indices: {:?}", filtered_minima_idx);

        // Check if the expected maximum indices align with our period
        // Period is 20 samples, so peaks should be at indices 5, 25, 45, 65, 85
        for &idx in &filtered_maxima_idx {
            // Verify that extrema are detected near the expected sample points
            // Allow for ±1 sample point error due to discrete sampling
            assert!(
                idx % period == period / 4
                    || idx % period == period / 4 - 1
                    || idx % period == period / 4 + 1,
                "Maximum at index {} is not near expected position (expected {} ± 1)",
                idx,
                period / 4
            );
        }

        // Period is 20, so minima should be at indices 15, 35, 55, 75, 95
        for &idx in &filtered_minima_idx {
            assert!(
                idx % period == 3 * period / 4
                    || idx % period == 3 * period / 4 - 1
                    || idx % period == 3 * period / 4 + 1,
                "Minimum at index {} is not near expected position (expected {} ± 1)",
                idx,
                3 * period / 4
            );
        }

        // Now check values with more relaxed epsilon
        for val in &filtered_maxima_val {
            assert!(*val > 0.9, "Maximum value {} is not close to 1.0", val);
        }

        for val in &filtered_minima_val {
            assert!(*val < -0.9, "Minimum value {} is not close to -1.0", val);
        }
    }

    #[test]
    fn test_interpolate_envelope() {
        // Create test data with known extrema
        let indices = vec![0, 5, 10, 15, 20];
        let values = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let n = 21;

        // Test linear interpolation
        let linear_env = interpolate_envelope(&indices, &values, n, "linear").unwrap();

        // Check interpolated values at extrema points
        for (i, idx) in indices.iter().enumerate() {
            assert_relative_eq!(linear_env[*idx], values[i], epsilon = 1e-10);
        }

        // Check a midpoint
        assert_relative_eq!(linear_env[2], 0.4, epsilon = 1e-10); // Linear interpolation between 0.0 and 1.0

        // Test cubic interpolation
        let cubic_env = interpolate_envelope(&indices, &values, n, "cubic").unwrap();

        // Check interpolated values at extrema points
        for (i, idx) in indices.iter().enumerate() {
            assert_relative_eq!(cubic_env[*idx], values[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_emd_basic() {
        // Create a simple test signal (sum of two sinusoids)
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / 20.0).sin() + 0.5 * (2.0 * PI * i as f64 / 5.0).sin())
            .collect();

        let config = EmdConfig::default();

        // Apply EMD
        let result = emd(&signal, &config).unwrap();

        // Check basic properties
        assert!(result.imfs.shape()[0] > 0); // At least one IMF
        assert_eq!(result.imfs.shape()[1], n); // Correct signal length
        assert_eq!(result.residue.len(), n); // Correct residue length
        assert_eq!(result.iterations.len(), result.imfs.shape()[0]); // One iteration count per IMF
        assert_eq!(result.energies.len(), result.imfs.shape()[0]); // One energy value per IMF
    }

    #[test]
    fn test_eemd_basic() {
        // Create a simple test signal (sum of two sinusoids)
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / 20.0).sin() + 0.5 * (2.0 * PI * i as f64 / 5.0).sin())
            .collect();

        let config = EmdConfig::default();

        // Apply EEMD with minimal ensemble for testing
        let ensemble_size = 2;
        let noise_std = 0.1;

        let result = eemd(&signal, &config, ensemble_size, noise_std).unwrap();

        // Check basic properties
        assert!(result.imfs.shape()[0] > 0); // At least one IMF
        assert_eq!(result.imfs.shape()[1], n); // Correct signal length
        assert_eq!(result.residue.len(), n); // Correct residue length
        assert_eq!(result.iterations.len(), result.imfs.shape()[0]); // One iteration count per IMF
        assert_eq!(result.energies.len(), result.imfs.shape()[0]); // One energy value per IMF
    }
}
