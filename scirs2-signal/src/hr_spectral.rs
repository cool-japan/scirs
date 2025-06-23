//! High-resolution spectral estimation algorithms
//!
//! This module implements advanced spectral estimation methods that provide
//! superior frequency resolution compared to traditional periodogram-based approaches.
//! These methods are particularly effective for analyzing sinusoidal signals in noise
//! and can resolve closely spaced frequency components.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
// use ndarray_linalg::{Eig, Inverse};
use num_complex::Complex64;
use num_traits::Zero;
use scirs2_linalg::complex::complex_inverse;
use scirs2_linalg::complex::decompositions::{complex_eig, complex_eigh};
use scirs2_linalg::solve as compute_solve;
use std::f64::consts::PI;

/// Configuration for high-resolution spectral estimation
#[derive(Debug, Clone)]
pub struct HrSpectralConfig {
    /// Number of frequency points to evaluate
    pub num_freqs: usize,
    /// Frequency range [start, end] in normalized units (0 to 1)
    pub freq_range: [f64; 2],
    /// Number of sources/signals to estimate (for subspace methods)
    pub num_sources: Option<usize>,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
    /// Threshold for determining signal subspace dimension
    pub eigenvalue_threshold: f64,
}

impl Default for HrSpectralConfig {
    fn default() -> Self {
        Self {
            num_freqs: 512,
            freq_range: [0.0, 1.0],
            num_sources: None,
            regularization: 1e-12,
            eigenvalue_threshold: 1e-6,
        }
    }
}

/// High-resolution spectral estimation methods
#[derive(Debug, Clone, Copy)]
pub enum HrSpectralMethod {
    /// Multiple Signal Classification (MUSIC)
    Music,
    /// Estimation of Signal Parameters via Rotational Invariance Techniques (ESPRIT)
    Esprit,
    /// Minimum Variance Distortionless Response (MVDR/Capon)
    MinimumVariance,
    /// Pisarenko harmonic decomposition
    Pisarenko,
    /// Prony's method for exponential modeling
    Prony,
}

/// Result of high-resolution spectral estimation
#[derive(Debug, Clone)]
pub struct HrSpectralResult {
    /// Frequency grid (normalized frequencies)
    pub frequencies: Array1<f64>,
    /// Spectral estimate (power or pseudospectrum)
    pub spectrum: Array1<f64>,
    /// Estimated source frequencies (for parametric methods)
    pub source_frequencies: Option<Array1<f64>>,
    /// Signal subspace dimension used
    pub signal_dimension: usize,
    /// Eigenvalues of the correlation matrix
    pub eigenvalues: Array1<f64>,
}

/// MUSIC (Multiple Signal Classification) algorithm
///
/// # Arguments
///
/// * `data` - Input signal data (can be a matrix for multiple snapshots)
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * High-resolution spectral estimate using MUSIC algorithm
pub fn music(data: &Array2<f64>, config: &HrSpectralConfig) -> SignalResult<HrSpectralResult> {
    let (n_samples, n_snapshots) = data.dim();

    if n_samples == 0 || n_snapshots == 0 {
        return Err(SignalError::ValueError(
            "Data matrix cannot be empty".to_string(),
        ));
    }

    // Convert to complex for processing
    let complex_data: Array2<Complex64> = data.mapv(|x| Complex64::new(x, 0.0));

    // Estimate correlation matrix
    let correlation_matrix = estimate_correlation_matrix(&complex_data)?;

    // Eigendecomposition - use eigh since correlation matrix is Hermitian
    let eig_result = complex_eigh(&correlation_matrix.view())
        .map_err(|e| SignalError::Compute(format!("Eigendecomposition failed: {}", e)))?;
    let eigenvalues = eig_result.eigenvalues;
    let eigenvectors = eig_result.eigenvectors;

    // Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<(f64, ArrayView1<Complex64>)> = eigenvalues
        .iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(&val, vec)| (val.norm(), vec))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Determine signal subspace dimension
    let signal_dim = determine_signal_dimension(&eigen_pairs, config)?;

    // Extract noise subspace
    let noise_eigenvectors: Array2<Complex64> = Array2::from_shape_vec(
        (n_samples, n_samples - signal_dim),
        eigen_pairs
            .iter()
            .skip(signal_dim)
            .flat_map(|(_, vec)| vec.iter().cloned())
            .collect(),
    )
    .map_err(|_| SignalError::Compute("Failed to construct noise subspace".to_string()))?;

    // Compute MUSIC pseudospectrum
    let frequencies = create_frequency_grid(config);
    let spectrum = compute_music_spectrum(&noise_eigenvectors, &frequencies)?;

    // Extract eigenvalues for result
    let eigenvals: Array1<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();

    Ok(HrSpectralResult {
        frequencies,
        spectrum,
        source_frequencies: None,
        signal_dimension: signal_dim,
        eigenvalues: eigenvals,
    })
}

/// ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
///
/// # Arguments
///
/// * `data` - Input signal data matrix
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * High-resolution spectral estimate using ESPRIT algorithm
pub fn esprit(data: &Array2<f64>, config: &HrSpectralConfig) -> SignalResult<HrSpectralResult> {
    let (n_samples, _n_snapshots) = data.dim();

    if n_samples < 2 {
        return Err(SignalError::ValueError(
            "ESPRIT requires at least 2 samples".to_string(),
        ));
    }

    // Convert to complex
    let complex_data: Array2<Complex64> = data.mapv(|x| Complex64::new(x, 0.0));

    // Estimate correlation matrix
    let correlation_matrix = estimate_correlation_matrix(&complex_data)?;

    // Eigendecomposition - use eigh since correlation matrix is Hermitian
    let eig_result = complex_eigh(&correlation_matrix.view())
        .map_err(|e| SignalError::Compute(format!("Eigendecomposition failed: {}", e)))?;
    let eigenvalues = eig_result.eigenvalues;
    let eigenvectors = eig_result.eigenvectors;

    // Sort eigenvalues and eigenvectors
    let mut eigen_pairs: Vec<(f64, ArrayView1<Complex64>)> = eigenvalues
        .iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(&val, vec)| (val.norm(), vec))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Determine signal subspace dimension
    let signal_dim = determine_signal_dimension(&eigen_pairs, config)?;

    if signal_dim == 0 {
        return Err(SignalError::ValueError(
            "No signal components detected".to_string(),
        ));
    }

    // Extract signal subspace
    let signal_eigenvectors: Array2<Complex64> = Array2::from_shape_vec(
        (n_samples, signal_dim),
        eigen_pairs
            .iter()
            .take(signal_dim)
            .flat_map(|(_, vec)| vec.iter().cloned())
            .collect(),
    )
    .map_err(|_| SignalError::Compute("Failed to construct signal subspace".to_string()))?;

    // Create subarray matrices for ESPRIT
    let s1 = signal_eigenvectors
        .slice(s![0..n_samples - 1, ..])
        .to_owned();
    let s2 = signal_eigenvectors.slice(s![1..n_samples, ..]).to_owned();

    // Solve the generalized eigenvalue problem: S2 = S1 * Phi
    let phi_matrix = solve_esprit_equation(&s1, &s2)?;

    // Extract frequencies from eigenvalues of Phi
    let eig_result = complex_eig(&phi_matrix.view())
        .map_err(|e| SignalError::Compute(format!("Eigendecomposition failed: {}", e)))?;
    let phi_eigenvalues = eig_result.eigenvalues;
    let mut source_freqs: Vec<f64> = phi_eigenvalues
        .iter()
        .map(|&z| z.arg() / (2.0 * PI))
        .collect();

    // Normalize frequencies to [0, 1]
    for freq in &mut source_freqs {
        if *freq < 0.0 {
            *freq += 1.0;
        }
    }
    source_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Create spectrum by placing peaks at estimated frequencies
    let frequencies = create_frequency_grid(config);
    let spectrum = create_line_spectrum(&frequencies, &source_freqs)?;

    let eigenvals: Array1<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();

    Ok(HrSpectralResult {
        frequencies,
        spectrum,
        source_frequencies: Some(Array1::from_vec(source_freqs)),
        signal_dimension: signal_dim,
        eigenvalues: eigenvals,
    })
}

/// Minimum Variance Distortionless Response (MVDR/Capon) beamformer
///
/// # Arguments
///
/// * `data` - Input signal data matrix
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * High-resolution spectral estimate using MVDR method
pub fn minimum_variance(
    data: &Array2<f64>,
    config: &HrSpectralConfig,
) -> SignalResult<HrSpectralResult> {
    let (n_samples, _) = data.dim();

    // Convert to complex
    let complex_data: Array2<Complex64> = data.mapv(|x| Complex64::new(x, 0.0));

    // Estimate correlation matrix
    let correlation_matrix = estimate_correlation_matrix(&complex_data)?;

    // Add regularization for numerical stability
    let mut regularized_matrix = correlation_matrix.clone();
    for i in 0..n_samples {
        regularized_matrix[[i, i]] += Complex64::new(config.regularization, 0.0);
    }

    // Invert correlation matrix
    let inv_correlation = complex_inverse(&regularized_matrix.view())
        .map_err(|_| SignalError::Compute("Failed to invert correlation matrix".to_string()))?;

    // Compute MVDR spectrum
    let frequencies = create_frequency_grid(config);
    let spectrum = compute_mvdr_spectrum(&inv_correlation, &frequencies)?;

    // Get eigenvalues for diagnostic purposes
    let eig_result = complex_eigh(&correlation_matrix.view())
        .map_err(|e| SignalError::Compute(format!("Eigendecomposition failed: {}", e)))?;
    let eigenvalues = eig_result.eigenvalues;
    let eigenvals: Array1<f64> = eigenvalues.iter().map(|&val| val.norm()).collect();

    Ok(HrSpectralResult {
        frequencies,
        spectrum,
        source_frequencies: None,
        signal_dimension: 0, // MVDR doesn't explicitly estimate signal dimension
        eigenvalues: eigenvals,
    })
}

/// Pisarenko harmonic decomposition
///
/// # Arguments
///
/// * `data` - Input signal vector
/// * `order` - Model order (number of complex exponentials)
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * High-resolution spectral estimate using Pisarenko method
pub fn pisarenko(
    data: &Array1<f64>,
    order: usize,
    config: &HrSpectralConfig,
) -> SignalResult<HrSpectralResult> {
    let n = data.len();

    if order >= n {
        return Err(SignalError::ValueError(
            "Order must be less than data length".to_string(),
        ));
    }

    // Create autocorrelation matrix
    let autocorr_matrix = create_autocorrelation_matrix(data, order + 1)?;

    // Eigendecomposition
    let eig_result = complex_eigh(&autocorr_matrix.view())
        .map_err(|e| SignalError::Compute(format!("Eigendecomposition failed: {}", e)))?;
    let eigenvalues = eig_result.eigenvalues;
    let eigenvectors = eig_result.eigenvectors;

    // Find minimum eigenvalue (noise eigenvalue)
    let min_idx = eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
        .unwrap()
        .0;

    let noise_eigenvector = eigenvectors.column(min_idx);

    // Find roots of the noise eigenvector polynomial
    let polynomial_coeffs: Vec<Complex64> = noise_eigenvector.iter().cloned().collect();
    let roots = find_polynomial_roots(&polynomial_coeffs)?;

    // Extract frequencies from roots on unit circle
    let mut source_freqs: Vec<f64> = roots
        .iter()
        .filter(|&z| (z.norm() - 1.0).abs() < 0.1) // Close to unit circle
        .map(|z| z.arg() / (2.0 * PI))
        .collect();

    // Normalize frequencies
    for freq in &mut source_freqs {
        if *freq < 0.0 {
            *freq += 1.0;
        }
    }
    source_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Create spectrum
    let frequencies = create_frequency_grid(config);
    let spectrum = create_line_spectrum(&frequencies, &source_freqs)?;

    let eigenvals: Array1<f64> = eigenvalues.iter().map(|&val| val.norm()).collect();

    Ok(HrSpectralResult {
        frequencies,
        spectrum,
        source_frequencies: Some(Array1::from_vec(source_freqs)),
        signal_dimension: order,
        eigenvalues: eigenvals,
    })
}

/// Prony's method for exponential modeling
///
/// # Arguments
///
/// * `data` - Input signal vector
/// * `order` - Model order
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Parameters of exponential model and spectral estimate
pub fn prony(
    data: &Array1<f64>,
    order: usize,
    config: &HrSpectralConfig,
) -> SignalResult<HrSpectralResult> {
    let n = data.len();

    if order >= n / 2 {
        return Err(SignalError::ValueError(
            "Order too large for Prony method".to_string(),
        ));
    }

    // Set up linear prediction equations
    let mut prediction_matrix = Array2::zeros((n - order, order));
    let mut observation_vector = Array1::zeros(n - order);

    for i in 0..(n - order) {
        for j in 0..order {
            prediction_matrix[[i, j]] = data[i + order - 1 - j];
        }
        observation_vector[i] = data[i + order];
    }

    // Solve for linear prediction coefficients
    let lp_coeffs = solve_linear_system(&prediction_matrix, &observation_vector)?;

    // Create polynomial from LP coefficients
    let mut polynomial_coeffs = vec![Complex64::new(1.0, 0.0)];
    for &coeff in lp_coeffs.iter() {
        polynomial_coeffs.push(Complex64::new(-coeff, 0.0));
    }

    // Find roots
    let roots = find_polynomial_roots(&polynomial_coeffs)?;

    // Extract frequencies and damping factors
    let mut source_freqs: Vec<f64> = Vec::new();
    for &root in &roots {
        if root.norm() < 1.0 {
            // Stable poles
            let freq = root.arg() / (2.0 * PI);
            let normalized_freq = if freq < 0.0 { freq + 1.0 } else { freq };
            source_freqs.push(normalized_freq);
        }
    }

    source_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Create spectrum
    let frequencies = create_frequency_grid(config);
    let spectrum = create_line_spectrum(&frequencies, &source_freqs)?;

    // Dummy eigenvalues (Prony doesn't use eigendecomposition)
    let eigenvals = Array1::zeros(order);

    Ok(HrSpectralResult {
        frequencies,
        spectrum,
        source_frequencies: Some(Array1::from_vec(source_freqs)),
        signal_dimension: order,
        eigenvalues: eigenvals,
    })
}

// Helper functions

/// Estimate correlation matrix from data
fn estimate_correlation_matrix(data: &Array2<Complex64>) -> SignalResult<Array2<Complex64>> {
    let (n_samples, n_snapshots) = data.dim();
    let mut correlation = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            let mut sum = Complex64::zero();
            for k in 0..n_snapshots {
                sum += data[[i, k]] * data[[j, k]].conj();
            }
            correlation[[i, j]] = sum / n_snapshots as f64;
        }
    }

    // Add small diagonal regularization to ensure numerical stability
    let regularization = 1e-8;
    for i in 0..n_samples {
        correlation[[i, i]] += Complex64::new(regularization, 0.0);
    }

    Ok(correlation)
}

/// Determine signal subspace dimension from eigenvalues
fn determine_signal_dimension(
    eigen_pairs: &[(f64, ArrayView1<Complex64>)],
    config: &HrSpectralConfig,
) -> SignalResult<usize> {
    if let Some(num_sources) = config.num_sources {
        return Ok(num_sources.min(eigen_pairs.len()));
    }

    // Automatic detection based on eigenvalue drop
    let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();

    if eigenvalues.is_empty() {
        return Ok(0);
    }

    let max_eigenvalue = eigenvalues[0];

    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval / max_eigenvalue < config.eigenvalue_threshold {
            return Ok(i);
        }
    }

    // If no clear cutoff, use half the dimensions
    Ok(eigenvalues.len() / 2)
}

/// Create frequency grid for spectrum evaluation
fn create_frequency_grid(config: &HrSpectralConfig) -> Array1<f64> {
    let [start, end] = config.freq_range;
    Array1::linspace(start, end, config.num_freqs)
}

/// Compute MUSIC pseudospectrum
fn compute_music_spectrum(
    noise_eigenvectors: &Array2<Complex64>,
    frequencies: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let (n_samples, n_noise) = noise_eigenvectors.dim();
    let mut spectrum = Array1::zeros(frequencies.len());

    for (k, &freq) in frequencies.iter().enumerate() {
        // Create steering vector
        let steering_vector: Array1<Complex64> = (0..n_samples)
            .map(|n| Complex64::from_polar(1.0, 2.0 * PI * freq * n as f64))
            .collect();

        // Compute projection onto noise subspace
        let mut projection_norm = 0.0;
        for i in 0..n_noise {
            let noise_vec = noise_eigenvectors.column(i);
            let inner_product: Complex64 = steering_vector
                .iter()
                .zip(noise_vec.iter())
                .map(|(&s, &n)| s.conj() * n)
                .sum();
            projection_norm += inner_product.norm_sqr();
        }

        // MUSIC pseudospectrum (reciprocal of projection)
        spectrum[k] = if projection_norm > 1e-12 {
            1.0 / projection_norm
        } else {
            1e12 // Large value for near-zero denominator
        };
    }

    Ok(spectrum)
}

/// Compute MVDR spectrum
fn compute_mvdr_spectrum(
    inv_correlation: &Array2<Complex64>,
    frequencies: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n_samples = inv_correlation.nrows();
    let mut spectrum = Array1::zeros(frequencies.len());

    for (k, &freq) in frequencies.iter().enumerate() {
        // Create steering vector
        let steering_vector: Array1<Complex64> = (0..n_samples)
            .map(|n| Complex64::from_polar(1.0, 2.0 * PI * freq * n as f64))
            .collect();

        // Compute a^H * R^-1 * a
        let mut quadratic_form = Complex64::zero();
        for i in 0..n_samples {
            for j in 0..n_samples {
                quadratic_form +=
                    steering_vector[i].conj() * inv_correlation[[i, j]] * steering_vector[j];
            }
        }

        // MVDR spectrum
        spectrum[k] = if quadratic_form.norm() > 1e-12 {
            1.0 / quadratic_form.re
        } else {
            0.0
        };
    }

    Ok(spectrum)
}

/// Solve ESPRIT equation S2 = S1 * Phi
fn solve_esprit_equation(
    s1: &Array2<Complex64>,
    s2: &Array2<Complex64>,
) -> SignalResult<Array2<Complex64>> {
    // Use least squares: Phi = (S1^H * S1)^-1 * S1^H * S2
    let s1_hermitian = s1.t().mapv(|x| x.conj());
    let s1h_s1 = s1_hermitian.dot(s1);
    let s1h_s2 = s1_hermitian.dot(s2);

    let inv_s1h_s1 = complex_inverse(&s1h_s1.view())
        .map_err(|_| SignalError::Compute("Failed to invert matrix in ESPRIT".to_string()))?;

    Ok(inv_s1h_s1.dot(&s1h_s2))
}

/// Create line spectrum from estimated frequencies
fn create_line_spectrum(
    frequency_grid: &Array1<f64>,
    source_frequencies: &[f64],
) -> SignalResult<Array1<f64>> {
    let mut spectrum = Array1::zeros(frequency_grid.len());

    for &source_freq in source_frequencies {
        // Find closest frequency in grid
        let closest_idx = frequency_grid
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - source_freq).abs())
                    .partial_cmp(&((*b - source_freq).abs()))
                    .unwrap()
            })
            .unwrap()
            .0;

        spectrum[closest_idx] += 1.0;
    }

    Ok(spectrum)
}

/// Create autocorrelation matrix
fn create_autocorrelation_matrix(
    data: &Array1<f64>,
    order: usize,
) -> SignalResult<Array2<Complex64>> {
    let n = data.len();

    if order > n {
        return Err(SignalError::ValueError(
            "Order cannot exceed data length".to_string(),
        ));
    }

    let mut autocorr = Array2::zeros((order, order));

    for i in 0..order {
        for j in 0..order {
            let mut sum = 0.0;
            let lag = (i as i32 - j as i32).unsigned_abs() as usize;

            for k in lag..(n) {
                sum += data[k] * data[k - lag];
            }

            autocorr[[i, j]] = Complex64::new(sum / (n - lag) as f64, 0.0);
        }
    }

    // Add small diagonal regularization to ensure numerical stability
    let regularization = 1e-8;
    for i in 0..order {
        autocorr[[i, i]] += Complex64::new(regularization, 0.0);
    }

    Ok(autocorr)
}

/// Find roots of polynomial (simplified implementation)
fn find_polynomial_roots(coeffs: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    let n = coeffs.len() - 1;

    if n == 0 {
        return Ok(vec![]);
    } else if n == 1 {
        // Linear case: ax + b = 0 => x = -b/a
        if coeffs[0].norm() > 1e-12 {
            return Ok(vec![-coeffs[1] / coeffs[0]]);
        } else {
            return Ok(vec![]);
        }
    }

    // For higher order polynomials, use companion matrix method
    let mut companion = Array2::zeros((n, n));

    // Fill companion matrix
    for i in 0..(n - 1) {
        companion[[i + 1, i]] = Complex64::new(1.0, 0.0);
    }

    for i in 0..n {
        companion[[i, n - 1]] = -coeffs[n - i] / coeffs[0];
    }

    // Find eigenvalues of companion matrix
    let eig_result = complex_eig(&companion.view())
        .map_err(|_| SignalError::Compute("Failed to find polynomial roots".to_string()))?;
    let eigenvalues = eig_result.eigenvalues;

    Ok(eigenvalues.to_vec())
}

/// Solve linear system Ax = b
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // Use scirs2_linalg solve
    compute_solve(&a.view(), &b.view(), None)
        .map_err(|_| SignalError::Compute("Failed to solve linear system".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_music_basic() {
        // Create test signal with two sinusoids
        let n_samples = 64;
        let n_snapshots = 100;
        let freq1 = 0.1;
        let freq2 = 0.15;

        let mut data = Array2::zeros((n_samples, n_snapshots));
        for j in 0..n_snapshots {
            for i in 0..n_samples {
                data[[i, j]] =
                    (2.0 * PI * freq1 * i as f64).sin() + (2.0 * PI * freq2 * i as f64).sin();
            }
        }

        let config = HrSpectralConfig {
            num_sources: Some(2),
            ..Default::default()
        };

        let result = music(&data, &config).unwrap();

        assert_eq!(result.frequencies.len(), config.num_freqs);
        assert_eq!(result.spectrum.len(), config.num_freqs);
        assert_eq!(result.signal_dimension, 2);
    }

    #[test]
    fn test_minimum_variance() {
        // Create test signal
        let n_samples = 32;
        let n_snapshots = 50;
        let freq = 0.2;

        let mut data = Array2::zeros((n_samples, n_snapshots));
        for j in 0..n_snapshots {
            for i in 0..n_samples {
                data[[i, j]] = (2.0 * PI * freq * i as f64).sin();
            }
        }

        let config = HrSpectralConfig::default();
        let result = minimum_variance(&data, &config).unwrap();

        assert_eq!(result.frequencies.len(), config.num_freqs);
        assert_eq!(result.spectrum.len(), config.num_freqs);

        // Check that spectrum has reasonable values
        assert!(result.spectrum.iter().all(|&x| x >= 0.0));
    }

    #[test]
    #[ignore] // TODO: Fix polynomial root finding for degenerate cases
    fn test_pisarenko() {
        // Create test signal with single sinusoid
        let n_samples = 64;
        let freq = 0.1;

        let data: Array1<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64).sin())
            .collect();

        let config = HrSpectralConfig::default();
        let result = pisarenko(&data, 2, &config).unwrap();

        assert_eq!(result.frequencies.len(), config.num_freqs);
        assert_eq!(result.spectrum.len(), config.num_freqs);
        assert!(result.source_frequencies.is_some());
    }

    #[test]
    fn test_frequency_grid() {
        let config = HrSpectralConfig {
            num_freqs: 100,
            freq_range: [0.1, 0.9],
            ..Default::default()
        };

        let freqs = create_frequency_grid(&config);

        assert_eq!(freqs.len(), 100);
        assert_relative_eq!(freqs[0], 0.1, epsilon = 1e-10);
        assert_relative_eq!(freqs[99], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_autocorrelation_matrix() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let autocorr = create_autocorrelation_matrix(&data, 3).unwrap();

        assert_eq!(autocorr.shape(), &[3, 3]);

        // Check symmetry (Hermitian property)
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(autocorr[[i, j]].re, autocorr[[j, i]].re, epsilon = 1e-10);
            }
        }
    }
}
