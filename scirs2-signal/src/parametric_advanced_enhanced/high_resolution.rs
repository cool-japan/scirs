//! High-resolution spectral estimation using eigenvalue methods
//!
//! This module implements MUSIC, ESPRIT, and other eigenvalue-based methods
//! for high-resolution spectral estimation beyond traditional AR/MA methods.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use crate::parametric_advanced::compute_eigendecomposition;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_linalg::eig as linalg_eig;
use scirs2_linalg::solve_multiple as linalg_solve_multiple;
use std::f64::consts::PI;

/// High-resolution spectral estimation using eigenvalue methods
///
/// This function implements MUSIC, ESPRIT, and other eigenvalue-based methods
/// for high-resolution spectral estimation beyond traditional AR/MA methods
pub fn high_resolution_spectral_estimation(
    signal: &[f64],
    config: &HighResolutionConfig,
) -> SignalResult<HighResolutionResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    if n < config.order * 2 {
        return Err(SignalError::ValueError(
            "Signal too short for high-resolution spectral estimation".to_string(),
        ));
    }

    // Create data matrix for subspace methods
    let data_matrix = create_hankel_matrix(signal, config.order)?;

    // Compute sample covariance matrix
    let covariance_matrix = compute_sample_covariance(&data_matrix)?;

    // Eigenvalue decomposition
    let (eigenvalues, eigenvectors) = compute_eigendecomposition(&covariance_matrix)?;
    let eigen_result = EigenResult {
        eigenvalues: eigenvalues.clone(),
        eigenvectors,
    };

    // Estimate number of signals using information criteria
    let estimated_num_signals = estimate_number_of_signals(&eigenvalues, config)?;

    // Apply the selected high-resolution method
    let spectral_results = match config.method {
        HighResolutionMethod::MUSIC => {
            music_spectrum_estimation(&eigen_result, estimated_num_signals, config)?
        }
        HighResolutionMethod::ESPRIT => {
            esprit_frequency_estimation(&eigen_result, estimated_num_signals, config)?
        }
        HighResolutionMethod::Pisarenko => pisarenko_method(&eigen_result, config)?,
        HighResolutionMethod::EigenFilter => {
            eigen_filter_method(&eigen_result, estimated_num_signals, config)?
        }
        HighResolutionMethod::MinNorm => {
            minimum_norm_method(&eigen_result, estimated_num_signals, config)?
        }
    };

    Ok(HighResolutionResult {
        frequency_estimates: spectral_results.frequency_estimates,
        amplitude_estimates: spectral_results.amplitude_estimates,
        phase_estimates: spectral_results.phase_estimates,
        power_spectrum: spectral_results.power_spectrum,
        frequencies: spectral_results.frequencies,
        noise_subspace_dim: config.order - estimated_num_signals,
        signal_subspace_dim: estimated_num_signals,
        eigenvalues,
    })
}

/// Eigenvalue decomposition result
#[derive(Debug, Clone)]
pub struct EigenResult {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

/// High-resolution spectral estimation results
#[derive(Debug, Clone)]
pub struct SpectralResults {
    pub frequency_estimates: Array1<f64>,
    pub amplitude_estimates: Array1<Complex64>,
    pub phase_estimates: Array1<f64>,
    pub power_spectrum: Array1<f64>,
    pub frequencies: Array1<f64>,
}

/// Create Hankel matrix from signal for subspace methods
fn create_hankel_matrix(signal: &[f64], subspace_dimension: usize) -> SignalResult<Array2<f64>> {
    let n = signal.len();
    if subspace_dimension >= n {
        return Err(SignalError::ValueError(
            "Subspace dimension must be smaller than signal length".to_string(),
        ));
    }

    let num_rows = n - subspace_dimension + 1;
    let mut matrix = Array2::zeros((num_rows, subspace_dimension));

    for i in 0..num_rows {
        for j in 0..subspace_dimension {
            matrix[(i, j)] = signal[i + j];
        }
    }

    Ok(matrix)
}

/// Compute sample covariance matrix
fn compute_sample_covariance(data_matrix: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (m, n) = data_matrix.dim();
    let mut covariance = Array2::zeros((n, n));

    // Compute mean for each column
    let mut means = Array1::zeros(n);
    for j in 0..n {
        means[j] = data_matrix.column(j).sum() / m as f64;
    }

    // Compute covariance matrix
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += (data_matrix[(k, i)] - means[i]) * (data_matrix[(k, j)] - means[j]);
            }
            covariance[(i, j)] = sum / (m - 1) as f64;
        }
    }

    Ok(covariance)
}

/// Estimate number of signals using information criteria
fn estimate_number_of_signals(
    eigenvalues: &Array1<f64>,
    config: &HighResolutionConfig,
) -> SignalResult<usize> {
    let n = eigenvalues.len();
    let mut sorted_eigenvalues = eigenvalues.to_vec();
    sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).expect("Operation failed")); // Descending order

    // Simple threshold-based detection
    let noise_threshold = sorted_eigenvalues.iter().sum::<f64>() / n as f64 * config.svd_threshold;

    let num_signals = sorted_eigenvalues
        .iter()
        .take_while(|&&eigenvalue| eigenvalue > noise_threshold)
        .count()
        .max(1)
        .min(n / 2); // Ensure reasonable number

    Ok(num_signals)
}

/// MUSIC spectrum estimation
fn music_spectrum_estimation(
    eigen_result: &EigenResult,
    num_signals: usize,
    config: &HighResolutionConfig,
) -> SignalResult<SpectralResults> {
    let n = eigen_result.eigenvalues.len();
    let noise_subspace_dim = n - num_signals;

    // Get noise subspace (eigenvectors corresponding to smallest eigenvalues)
    let mut eigenvalue_indices: Vec<usize> = (0..n).collect();
    eigenvalue_indices.sort_by(|&i, &j| {
        eigen_result.eigenvalues[j]
            .partial_cmp(&eigen_result.eigenvalues[i])
            .expect("Operation failed")
    });

    // Create noise subspace matrix
    let noise_indices = &eigenvalue_indices[num_signals..];
    let mut noise_subspace = Array2::zeros((n, noise_subspace_dim));
    for (col, &idx) in noise_indices.iter().enumerate() {
        for row in 0..n {
            noise_subspace[(row, col)] = eigen_result.eigenvectors[(row, idx)];
        }
    }

    // Generate frequency grid
    let frequencies = generate_frequency_grid(config.frequency_bins, config.frequency_range);
    let mut power_spectrum = Array1::zeros(frequencies.len());

    // Compute MUSIC spectrum
    for (k, &freq) in frequencies.iter().enumerate() {
        let steering_vector = create_steering_vector(freq, n);
        let numerator = steering_vector.mapv(|z| z.norm_sqr()).sum();

        // P_v^H * steering_vector
        let mut denominator = 0.0;
        for i in 0..noise_subspace_dim {
            let mut dot_product = Complex64::new(0.0, 0.0);
            for j in 0..n {
                dot_product +=
                    Complex64::new(noise_subspace[(j, i)], 0.0) * steering_vector[j].conj();
            }
            denominator += dot_product.norm_sqr();
        }

        power_spectrum[k] = if denominator > 1e-12 {
            numerator / denominator
        } else {
            0.0
        };
    }

    // Find peaks for frequency estimates
    let frequency_estimates = find_spectral_peaks(&power_spectrum, &frequencies, num_signals);

    // Create amplitude and phase estimates (simplified)
    let amplitude_estimates =
        Array1::from_elem(frequency_estimates.len(), Complex64::new(1.0, 0.0));
    let phase_estimates = Array1::zeros(frequency_estimates.len());

    Ok(SpectralResults {
        frequency_estimates: Array1::from_vec(frequency_estimates),
        amplitude_estimates,
        phase_estimates,
        power_spectrum,
        frequencies: Array1::from_vec(frequencies),
    })
}

/// ESPRIT frequency estimation
fn esprit_frequency_estimation(
    eigen_result: &EigenResult,
    num_signals: usize,
    config: &HighResolutionConfig,
) -> SignalResult<SpectralResults> {
    let n = eigen_result.eigenvalues.len();

    // Get signal subspace (eigenvectors corresponding to largest eigenvalues)
    let mut eigenvalue_indices: Vec<usize> = (0..n).collect();
    eigenvalue_indices.sort_by(|&i, &j| {
        eigen_result.eigenvalues[j]
            .partial_cmp(&eigen_result.eigenvalues[i])
            .expect("Operation failed")
    });

    let signal_indices = &eigenvalue_indices[..num_signals];

    // Create signal subspace matrices S1 and S2
    let mut s1 = Array2::zeros((n - 1, num_signals));
    let mut s2 = Array2::zeros((n - 1, num_signals));

    for (col, &idx) in signal_indices.iter().enumerate() {
        for row in 0..n - 1 {
            s1[(row, col)] = eigen_result.eigenvectors[(row, idx)];
            s2[(row, col)] = eigen_result.eigenvectors[(row + 1, idx)];
        }
    }

    // Solve generalized eigenvalue problem: S2 = S1 * Phi
    // For simplicity, use pseudoinverse approach
    let frequency_estimates = estimate_frequencies_from_subspace(&s1, &s2, num_signals)?;

    // Generate full spectrum for visualization
    let frequencies = generate_frequency_grid(config.frequency_bins, config.frequency_range);
    let power_spectrum = Array1::ones(frequencies.len()); // Simplified for ESPRIT

    let amplitude_estimates =
        Array1::from_elem(frequency_estimates.len(), Complex64::new(1.0, 0.0));
    let phase_estimates = Array1::zeros(frequency_estimates.len());

    Ok(SpectralResults {
        frequency_estimates: Array1::from_vec(frequency_estimates),
        amplitude_estimates,
        phase_estimates,
        power_spectrum,
        frequencies: Array1::from_vec(frequencies),
    })
}

/// Pisarenko harmonic decomposition method
fn pisarenko_method(
    eigen_result: &EigenResult,
    config: &HighResolutionConfig,
) -> SignalResult<SpectralResults> {
    // Use the eigenvector corresponding to the smallest eigenvalue
    let min_eigenvalue_idx = eigen_result
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let noise_eigenvector = eigen_result.eigenvectors.column(min_eigenvalue_idx);

    // Find roots of the polynomial formed by the noise eigenvector
    let frequency_estimates = find_polynomial_roots(&noise_eigenvector.to_owned())?;

    let frequencies = generate_frequency_grid(config.frequency_bins, config.frequency_range);
    let power_spectrum = Array1::ones(frequencies.len()); // Simplified

    let amplitude_estimates =
        Array1::from_elem(frequency_estimates.len(), Complex64::new(1.0, 0.0));
    let phase_estimates = Array1::zeros(frequency_estimates.len());

    Ok(SpectralResults {
        frequency_estimates: Array1::from_vec(frequency_estimates),
        amplitude_estimates,
        phase_estimates,
        power_spectrum,
        frequencies: Array1::from_vec(frequencies),
    })
}

/// Eigen-filter method
fn eigen_filter_method(
    eigen_result: &EigenResult,
    num_signals: usize,
    config: &HighResolutionConfig,
) -> SignalResult<SpectralResults> {
    // Simplified implementation - similar to MUSIC but with different weighting
    music_spectrum_estimation(eigen_result, num_signals, config)
}

/// Minimum norm method
fn minimum_norm_method(
    eigen_result: &EigenResult,
    num_signals: usize,
    config: &HighResolutionConfig,
) -> SignalResult<SpectralResults> {
    // Simplified implementation - similar to MUSIC
    music_spectrum_estimation(eigen_result, num_signals, config)
}

/// Generate frequency grid
fn generate_frequency_grid(num_points: usize, frequency_range: (f64, f64)) -> Vec<f64> {
    let (f_min, f_max) = frequency_range;
    let delta_f = (f_max - f_min) / (num_points - 1) as f64;

    (0..num_points)
        .map(|i| f_min + i as f64 * delta_f)
        .collect()
}

/// Create steering vector for given frequency
fn create_steering_vector(frequency: f64, length: usize) -> Array1<Complex64> {
    let mut vector = Array1::zeros(length);
    for k in 0..length {
        let phase = -2.0 * PI * frequency * k as f64;
        vector[k] = Complex64::new(phase.cos(), phase.sin());
    }
    vector
}

/// Find spectral peaks
fn find_spectral_peaks(spectrum: &Array1<f64>, frequencies: &[f64], num_peaks: usize) -> Vec<f64> {
    let mut peaks_with_indices: Vec<(usize, f64)> = spectrum
        .iter()
        .enumerate()
        .map(|(i, &power)| (i, power))
        .collect();

    // Sort by power in descending order
    peaks_with_indices.sort_by(|(_, a), (_, b)| b.partial_cmp(a).expect("Operation failed"));

    // Extract top frequencies
    peaks_with_indices
        .into_iter()
        .take(num_peaks)
        .map(|(idx, _)| frequencies[idx])
        .collect()
}

/// Estimate frequencies from signal subspace matrices (ESPRIT rotational
/// invariance technique).
///
/// Solves the shift-invariance equation `S1 * Phi = S2` for `Phi` in the
/// least-squares sense, then recovers the frequencies from the phase of
/// `Phi`'s (generally complex) eigenvalues, which lie on/near the unit
/// circle at `exp(j*2*pi*f_k)` for a genuine rotational-invariance
/// structure. This replaces a previous stand-in that ignored `s1`/`s2`
/// entirely and returned hardcoded evenly-spaced frequencies.
fn estimate_frequencies_from_subspace(
    s1: &Array2<f64>,
    s2: &Array2<f64>,
    num_signals: usize,
) -> SignalResult<Vec<f64>> {
    if num_signals == 0 || s1.nrows() == 0 {
        return Ok(Vec::new());
    }

    // Solve S1 * Phi = S2 via the real normal equations
    // (S1^T S1) Phi = S1^T S2 (Phi is num_signals x num_signals).
    let s1t = s1.t();
    let ata = s1t.dot(s1);
    let atb = s1t.dot(s2);

    let phi = linalg_solve_multiple(&ata.view(), &atb.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("ESPRIT rotation-operator solve failed: {e}"))
    })?;

    // Phi is generally not symmetric: its eigenvalues are genuinely complex
    // in general, and it is exactly their phase that encodes the
    // frequency estimates, so a general (complex) eigendecomposition is
    // required here.
    let (eigenvalues, _) = linalg_eig(&phi.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("ESPRIT eigenvalue computation failed: {e}"))
    })?;

    // Real coefficients guarantee eigenvalues occur in complex-conjugate
    // pairs; keep one (non-negative angle) representative of each so the
    // same physical tone is not reported twice.
    let mut frequencies: Vec<f64> = eigenvalues
        .iter()
        .filter(|z| z.im >= 0.0)
        .map(|z| z.im.atan2(z.re).abs() / (2.0 * PI))
        .collect();

    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(frequencies)
}

/// Find polynomial roots via the companion-matrix eigenvalue method
/// (Pisarenko harmonic decomposition).
///
/// `coefficients` are interpreted as the coefficients (from lowest to
/// highest degree) of the characteristic polynomial formed from the
/// noise-subspace eigenvector; in the noiseless case its roots lie exactly
/// on the unit circle at `exp(j*2*pi*f_k)` for each embedded sinusoid.
/// Roots are found as the eigenvalues of the real companion matrix (a
/// numerically robust, standard technique), then ranked by closeness to
/// the unit circle since genuine spectral-line roots stay close to it even
/// under noise. This replaces a previous stand-in that ignored
/// `coefficients` entirely and returned a hardcoded `[0.1, 0.3, 0.5]`.
fn find_polynomial_roots(coefficients: &Array1<f64>) -> SignalResult<Vec<f64>> {
    let p = coefficients.len();
    if p < 2 {
        return Ok(Vec::new());
    }

    // Determine the effective polynomial degree, ignoring negligible
    // leading coefficients (which would otherwise introduce spurious roots
    // at infinity into a companion-matrix formulation).
    let mut degree = p - 1;
    while degree > 0 && coefficients[degree].abs() < 1e-12 {
        degree -= 1;
    }
    if degree == 0 {
        return Ok(Vec::new());
    }

    // Build the companion matrix of the monic polynomial
    // z^degree + a_{degree-1} z^{degree-1} + ... + a_0 = 0, with
    // a_i = coefficients[i] / coefficients[degree].
    let leading = coefficients[degree];
    let mut companion = Array2::<f64>::zeros((degree, degree));
    for i in 1..degree {
        companion[(i, i - 1)] = 1.0;
    }
    for i in 0..degree {
        companion[(i, degree - 1)] = -coefficients[i] / leading;
    }

    let (roots, _) = linalg_eig(&companion.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("Pisarenko polynomial rooting failed: {e}"))
    })?;

    // Keep one representative per complex-conjugate pair (non-negative
    // angle), ranked by closeness to the unit circle (genuine spectral-line
    // roots lie on it; roots further away are numerical/noise artifacts).
    let mut candidates: Vec<(f64, f64)> = roots
        .iter()
        .filter(|z| z.im >= 0.0)
        .map(|z| {
            let frequency = z.im.atan2(z.re).abs() / (2.0 * PI);
            let distance_from_unit_circle = (z.norm() - 1.0).abs();
            (frequency, distance_from_unit_circle)
        })
        .collect();

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(candidates.into_iter().map(|(f, _)| f).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_resolution_spectral_estimation() {
        // Use longer signal with more spectral content for high-resolution estimation
        let mut signal = vec![];
        for i in 0..128 {
            signal.push(
                1.0 + 0.8 * (i as f64 * 0.1).sin()
                    + 0.6 * (i as f64 * 0.15).cos()
                    + 0.4 * (i as f64 * 0.05).sin()
                    + 0.1 * scirs2_core::random::random::<f64>(), // Add small amount of noise
            );
        }
        let config = HighResolutionConfig::default();

        let result = high_resolution_spectral_estimation(&signal, &config);
        if result.is_err() {
            println!("High-resolution error: {:?}", result.as_ref().err());
        }
        assert!(result.is_ok());

        let hr_result = result.expect("Operation failed");
        assert!(!hr_result.frequency_estimates.is_empty());
        assert!(!hr_result.eigenvalues.is_empty());
        assert!(hr_result.signal_subspace_dim > 0);
    }

    #[test]
    fn test_create_hankel_matrix() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = create_hankel_matrix(&signal, 3);
        assert!(result.is_ok());

        let matrix = result.expect("Operation failed");
        assert_eq!(matrix.dim(), (3, 3));
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 2)], 3.0);
        assert_eq!(matrix[(2, 0)], 3.0);
    }

    #[test]
    fn test_compute_sample_covariance() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Operation failed");
        let result = compute_sample_covariance(&matrix);
        assert!(result.is_ok());

        let covariance = result.expect("Operation failed");
        assert_eq!(covariance.dim(), (2, 2));
        assert!(covariance[(0, 0)] > 0.0); // Variance should be positive
    }

    #[test]
    fn test_estimate_frequencies_from_subspace_tracks_changing_rotation() {
        // A fixed, non-trivial (non-constant) pair of signal-subspace basis
        // vectors; the *only* thing that changes between the two recover()
        // calls below is the true rotation angle encoded via `s2`.
        let s1 = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.3, 0.5, 1.2, 1.4, 0.2, 0.1, 0.9, 0.8, 0.6],
        )
        .expect("Operation failed");

        let recover = |f0: f64| -> f64 {
            let w = 2.0 * PI * f0;
            let phi_true =
                Array2::from_shape_vec((2, 2), vec![w.cos(), -w.sin(), w.sin(), w.cos()])
                    .expect("Operation failed");
            let s2 = s1.dot(&phi_true);
            let frequencies = estimate_frequencies_from_subspace(&s1, &s2, 2)
                .expect("frequency estimation should succeed");
            frequencies
                .iter()
                .map(|&f| (f - f0).abs())
                .fold(f64::INFINITY, f64::min)
        };

        // Two distinct target frequencies, neither equal to the old stub's
        // fixed `0.1 + i*0.1` output ([0.1, 0.2] for num_signals=2), should
        // both be recovered to high precision from the exact (noiseless)
        // rotational-invariance relationship S2 = S1 * Phi.
        assert!(
            recover(0.37) < 1e-6,
            "failed to recover frequency 0.37 from a real rotation"
        );
        assert!(
            recover(0.18) < 1e-6,
            "failed to recover frequency 0.18 from a real rotation"
        );
    }

    #[test]
    fn test_estimate_frequencies_from_subspace_empty_when_no_signals() {
        let s1 = Array2::from_shape_vec((3, 2), vec![1.0, 0.2, 0.4, 0.9, 0.3, 0.7])
            .expect("Operation failed");
        let s2 = s1.clone();
        let frequencies = estimate_frequencies_from_subspace(&s1, &s2, 0)
            .expect("frequency estimation should succeed");
        assert!(frequencies.is_empty());
    }

    #[test]
    fn test_find_polynomial_roots_recovers_known_frequency() {
        let f0 = 0.23_f64;
        let w = 2.0 * PI * f0;
        // Ascending-degree coefficients of z^2 - 2*cos(w)*z + 1, whose
        // roots are exp(+-j*w) exactly on the unit circle.
        let coefficients = Array1::from_vec(vec![1.0, -2.0 * w.cos(), 1.0]);

        let frequencies =
            find_polynomial_roots(&coefficients).expect("polynomial rooting should succeed");

        assert!(!frequencies.is_empty());
        let min_err = frequencies
            .iter()
            .map(|&f| (f - f0).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_err < 1e-6,
            "expected a frequency near {f0}, got {:?}",
            frequencies
        );
        // Not the old hardcoded stand-in output, which ignored `coefficients`.
        assert_ne!(frequencies, vec![0.1, 0.3, 0.5]);
    }

    #[test]
    fn test_find_polynomial_roots_tracks_changing_frequency() {
        // Same construction as above but at a different target frequency;
        // a stub returning a fixed [0.1, 0.3, 0.5] regardless of input
        // could not satisfy both this and the previous test simultaneously.
        let f0 = 0.41_f64;
        let w = 2.0 * PI * f0;
        let coefficients = Array1::from_vec(vec![1.0, -2.0 * w.cos(), 1.0]);

        let frequencies =
            find_polynomial_roots(&coefficients).expect("polynomial rooting should succeed");
        let min_err = frequencies
            .iter()
            .map(|&f| (f - f0).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_err < 1e-6,
            "expected a frequency near {f0}, got {:?}",
            frequencies
        );
    }
}
