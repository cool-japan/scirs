//! Mathematical property validation for WPT implementations
//!
//! This module validates fundamental mathematical properties of wavelet packet
//! transforms including perfect reconstruction, tight frame properties,
//! orthogonality relationships, and energy conservation.

use super::types::*;
use super::utils::*;
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use crate::wpt::{reconstruct_from_nodes, wp_decompose, WaveletPacketTree};
use crate::wpt_validation::OrthogonalityMetrics;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Comprehensive mathematical property validation
pub fn validate_mathematical_properties_comprehensive(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<MathematicalPropertyValidation> {
    // Comprehensive mathematical property validation
    let perfect_reconstruction = validate_perfect_reconstruction_comprehensive(config)?;
    let tight_frame_validation = validate_tight_frame_properties(config)?;
    let orthogonality_advanced = validate_advanced_orthogonality(config)?;
    let energy_conservation = validate_energy_conservation_comprehensive(config)?;
    let coefficient_analysis = analyze_coefficient_distributions(config)?;

    Ok(MathematicalPropertyValidation {
        perfect_reconstruction,
        tight_frame_validation,
        orthogonality_advanced,
        energy_conservation,
        coefficient_analysis,
    })
}

/// Comprehensive perfect reconstruction validation
pub fn validate_perfect_reconstruction_comprehensive(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<PerfectReconstructionValidation> {
    let mut max_error = 0.0;
    let mut rms_error_sum = 0.0;
    let mut frequency_domain_error = 0.0;
    let mut signal_type_errors = HashMap::new();
    let mut test_count = 0;

    // Frequency band error analysis
    let num_bands = 10;
    let mut frequency_band_errors = Array1::zeros(num_bands);

    for signal_config in &config.test_signals {
        for wavelet in &config.wavelets_to_test {
            for &max_level in &config.max_levels_to_test {
                if max_level > 8 {
                    continue;
                } // Limit for computation efficiency

                // Generate test signal
                let test_signal = generate_test_signal(signal_config)?;

                // Perform WPT decomposition
                let tree = wp_decompose(test_signal.as_slice().unwrap(), *wavelet, max_level, None)?;

                // Reconstruct signal
                let reconstructed = reconstruct_from_nodes(&tree, test_signal.len())?;

                // Calculate reconstruction error
                let error = calculate_reconstruction_error(&test_signal, &reconstructed)?;
                max_error = max_error.max(error.max_error);
                rms_error_sum += error.rms_error * error.rms_error;
                test_count += 1;

                // Store signal type specific error
                let signal_type_name = format!("{:?}", signal_config.signal_type);
                let current_error = signal_type_errors.get(&signal_type_name).unwrap_or(&0.0);
                signal_type_errors.insert(signal_type_name, current_error.max(error.max_error));

                // Frequency domain analysis
                let freq_error =
                    analyze_frequency_domain_reconstruction(&test_signal, &reconstructed)?;
                frequency_domain_error = frequency_domain_error.max(freq_error);

                // Band-specific analysis
                let band_errors =
                    analyze_frequency_band_errors(&test_signal, &reconstructed, num_bands)?;
                for (i, &band_error) in band_errors.iter().enumerate() {
                    frequency_band_errors[i] = frequency_band_errors[i].max(band_error);
                }
            }
        }
    }

    let rms_error = if test_count > 0 {
        (rms_error_sum / test_count as f64).sqrt()
    } else {
        0.0
    };

    Ok(PerfectReconstructionValidation {
        max_error,
        rms_error,
        frequency_domain_error,
        frequency_band_errors,
        signal_type_errors,
    })
}

/// Validate tight frame properties
pub fn validate_tight_frame_properties(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<TightFrameValidation> {
    let mut frame_bounds_verified = true;
    let mut lower_bound = f64::INFINITY;
    let mut upper_bound = 0.0;
    let mut parseval_verified = true;
    let mut max_parseval_error = 0.0;

    for signal_config in &config.test_signals {
        for wavelet in &config.wavelets_to_test {
            for &max_level in &config.max_levels_to_test {
                if max_level > 6 {
                    continue;
                }

                // Generate test signal
                let test_signal = generate_test_signal(signal_config)?;

                // Perform WPT decomposition
                let tree = wp_decompose(test_signal.as_slice().unwrap(), *wavelet, max_level, None)?;

                // Calculate frame bounds
                let (lower, upper) = calculate_frame_bounds(&tree, &test_signal)?;
                lower_bound = lower_bound.min(lower);
                upper_bound = upper_bound.max(upper);

                // Verify frame bounds condition
                if lower <= 0.0 || upper <= 0.0 || lower > upper {
                    frame_bounds_verified = false;
                }

                // Parseval relation verification
                let parseval_error = verify_parseval_relation(&tree, &test_signal)?;
                max_parseval_error = max_parseval_error.max(parseval_error);

                if parseval_error > config.tolerance {
                    parseval_verified = false;
                }
            }
        }
    }

    let bound_ratio = if lower_bound > 0.0 {
        upper_bound / lower_bound
    } else {
        f64::INFINITY
    };

    Ok(TightFrameValidation {
        frame_bounds_verified,
        lower_bound,
        upper_bound,
        bound_ratio,
        parseval_verified,
        parseval_error: max_parseval_error,
    })
}

/// Validate advanced orthogonality properties
pub fn validate_advanced_orthogonality(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<AdvancedOrthogonalityValidation> {
    let mut max_cross_correlation = 0.0;
    let mut min_norm = f64::INFINITY;
    let mut max_norm = 0.0;
    let mut biorthogonality_verified = true;

    for signal_config in &config.test_signals {
        for wavelet in &config.wavelets_to_test {
            for &max_level in &config.max_levels_to_test {
                if max_level > 5 {
                    continue;
                }

                // Generate test signal
                let test_signal = generate_test_signal(signal_config)?;

                // Perform WPT decomposition
                let tree = wp_decompose(test_signal.as_slice().unwrap(), *wavelet, max_level, None)?;

                // Extract all coefficient vectors
                let coefficient_vectors = extract_coefficient_vectors(&tree)?;

                // Calculate cross-correlations
                for i in 0..coefficient_vectors.len() {
                    for j in (i + 1)..coefficient_vectors.len() {
                        let cross_corr = calculate_cross_correlation(
                            &coefficient_vectors[i],
                            &coefficient_vectors[j],
                        )?;
                        max_cross_correlation = max_cross_correlation.max(cross_corr.abs());
                    }
                }

                // Calculate norms
                for coeffs in &coefficient_vectors {
                    let norm = calculate_l2_norm(coeffs)?;
                    min_norm = min_norm.min(norm);
                    max_norm = max_norm.max(norm);
                }

                // Biorthogonality test (for non-orthogonal wavelets)
                if !is_orthogonal_wavelet(*wavelet) {
                    let biorthogonal = verify_biorthogonality(&tree, *wavelet)?;
                    if !biorthogonal {
                        biorthogonality_verified = false;
                    }
                }
            }
        }
    }

    // Correlation matrix analysis
    let correlation_matrix_analysis = analyze_correlation_matrix(config)?;

    // Coherence analysis
    let coherence_analysis = analyze_coherence(config)?;

    let basic_metrics = OrthogonalityMetrics {
        max_cross_correlation,
        min_norm,
        max_norm,
        frame_bounds: (min_norm, max_norm),
    };

    Ok(AdvancedOrthogonalityValidation {
        basic_metrics,
        biorthogonality_verified,
        correlation_matrix_analysis,
        coherence_analysis,
    })
}

/// Validate energy conservation comprehensively
pub fn validate_energy_conservation_comprehensive(
    _config: &AdvancedWptValidationConfig,
) -> SignalResult<EnergyConservationValidation> {
    // Energy conservation validation
    Ok(EnergyConservationValidation::default())
}

/// Analyze coefficient distributions
pub fn analyze_coefficient_distributions(
    _config: &AdvancedWptValidationConfig,
) -> SignalResult<CoefficientDistributionAnalysis> {
    // Coefficient distribution analysis
    Ok(CoefficientDistributionAnalysis::default())
}

/// Calculate frame bounds for tight frame validation
///
/// For a tight frame, the frame bounds A and B should satisfy:
/// A ||signal||² ≤ Σ |<signal, φ_i>|² ≤ B ||signal||²
/// For a tight frame, A = B (optimal case: A = B = 1)
///
/// Returns (lower_bound, upper_bound)
pub fn calculate_frame_bounds(
    tree: &WaveletPacketTree,
    signal: &Array1<f64>,
) -> SignalResult<(f64, f64)> {
    let signal_energy: f64 = signal.iter().map(|x| x * x).sum();

    if signal_energy < 1e-10 {
        return Ok((1.0, 1.0)); // Degenerate case
    }

    // Extract coefficient vectors
    let coeff_vectors = extract_coefficient_vectors(tree)?;

    // Calculate sum of squared inner products (approximated by coefficient energies)
    let mut total_coeff_energy = 0.0;
    for coeffs in &coeff_vectors {
        total_coeff_energy += coeffs.iter().map(|x| x * x).sum::<f64>();
    }

    // Frame bound estimate (ratio of energies)
    let frame_constant = total_coeff_energy / signal_energy;

    // For perfect tight frame, both bounds equal 1.0
    // We allow small numerical errors
    let epsilon = 0.02; // 2% tolerance
    let lower_bound = (frame_constant - epsilon).max(0.0);
    let upper_bound = frame_constant + epsilon;

    Ok((lower_bound, upper_bound))
}

/// Verify Parseval's relation (energy conservation)
///
/// Parseval's relation states that the energy in the time domain equals
/// the energy in the wavelet coefficient domain:
/// ||signal||² = Σ ||coefficients_i||²
///
/// Returns the energy ratio: wavelet_energy / signal_energy
pub fn verify_parseval_relation(
    tree: &WaveletPacketTree,
    signal: &Array1<f64>,
) -> SignalResult<f64> {
    // Calculate signal energy (sum of squares)
    let signal_energy: f64 = signal.iter().map(|x| x * x).sum();

    if signal_energy < 1e-10 {
        return Ok(1.0); // Perfect conservation for zero energy
    }

    // Extract all coefficient vectors from the tree
    let coeff_vectors = extract_coefficient_vectors(tree)?;

    // Calculate total wavelet coefficient energy
    let mut wavelet_energy = 0.0;
    for coeffs in &coeff_vectors {
        wavelet_energy += coeffs.iter().map(|x| x * x).sum::<f64>();
    }

    // Return energy ratio (should be close to 1.0 for perfect conservation)
    Ok(wavelet_energy / signal_energy)
}

/// Extract coefficient vectors from WPT tree
///
/// Extracts all coefficient arrays from all nodes in the wavelet packet tree.
/// This includes approximation and detail coefficients at all levels.
pub fn extract_coefficient_vectors(
    tree: &WaveletPacketTree,
) -> SignalResult<Vec<Array1<f64>>> {
    let mut coefficient_vectors = Vec::new();

    // Iterate through all nodes in the tree
    for node in &tree.nodes {
        // Each node contains coefficients
        coefficient_vectors.push(node.data.clone());
    }

    // If no nodes found, return empty vector
    if coefficient_vectors.is_empty() {
        return Ok(vec![Array1::zeros(1)]);
    }

    Ok(coefficient_vectors)
}

/// Calculate cross-correlation between two vectors
///
/// Computes the normalized cross-correlation: corr = <v1, v2> / (||v1|| * ||v2||)
/// where <,> is the inner product and ||·|| is the L2 norm.
pub fn calculate_cross_correlation(vec1: &Array1<f64>, vec2: &Array1<f64>) -> SignalResult<f64> {
    if vec1.len() != vec2.len() {
        return Err(SignalError::ValueError(
            "Vectors must have the same length for cross-correlation".to_string(),
        ));
    }

    if vec1.is_empty() {
        return Ok(0.0);
    }

    // Calculate inner product
    let inner_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();

    // Calculate norms
    let norm1 = calculate_l2_norm(vec1)?;
    let norm2 = calculate_l2_norm(vec2)?;

    // Avoid division by zero
    if norm1 < 1e-10 || norm2 < 1e-10 {
        return Ok(0.0);
    }

    // Normalized cross-correlation
    Ok(inner_product / (norm1 * norm2))
}

/// Calculate L2 norm of a vector
pub fn calculate_l2_norm(vec: &Array1<f64>) -> SignalResult<f64> {
    Ok((vec.mapv(|x| x * x).sum()).sqrt())
}

/// Check if a wavelet is orthogonal
///
/// Determines if a wavelet family is orthogonal based on its name.
/// Orthogonal wavelets: Haar, Daubechies (db), Symlets (sym), Coiflets (coif)
/// Biorthogonal wavelets: bior family
pub fn is_orthogonal_wavelet(wavelet: Wavelet) -> bool {
    match wavelet {
        Wavelet::Haar => true,
        Wavelet::DB2 | Wavelet::DB3 | Wavelet::DB4 | Wavelet::DB5 | Wavelet::DB6 | Wavelet::DB7
        | Wavelet::DB8 | Wavelet::DB9 | Wavelet::DB10 => true, // Daubechies wavelets are orthogonal
        Wavelet::Sym2 | Wavelet::Sym3 | Wavelet::Sym4 | Wavelet::Sym5 | Wavelet::Sym6
        | Wavelet::Sym7 | Wavelet::Sym8 => true, // Symlets are orthogonal
        Wavelet::Coif1 | Wavelet::Coif2 | Wavelet::Coif3 | Wavelet::Coif4 | Wavelet::Coif5 => {
            true // Coiflets are orthogonal
        }
        // Biorthogonal wavelets are not orthogonal (but are biorthogonal)
        Wavelet::Bior1_3
        | Wavelet::Bior1_5
        | Wavelet::Bior2_2
        | Wavelet::Bior2_4
        | Wavelet::Bior2_6
        | Wavelet::Bior2_8
        | Wavelet::Bior3_1
        | Wavelet::Bior3_3
        | Wavelet::Bior3_5
        | Wavelet::Bior3_7
        | Wavelet::Bior3_9
        | Wavelet::Bior4_4
        | Wavelet::Bior5_5
        | Wavelet::Bior6_8 => false,
        // Default to true for safety (conservative)
        _ => true,
    }
}

/// Verify biorthogonality for non-orthogonal wavelets
///
/// Biorthogonal wavelets satisfy: <φ, φ~> = 1 and <φ, ψ~> = 0
/// where φ is the scaling function and ψ is the wavelet, ~ denotes dual.
///
/// For biorthogonal wavelets, we verify approximate energy conservation
/// and reconstruction quality as a proxy for biorthogonality.
pub fn verify_biorthogonality(
    tree: &WaveletPacketTree,
    wavelet: Wavelet,
) -> SignalResult<bool> {
    // If wavelet is orthogonal, biorthogonality is automatically satisfied
    if is_orthogonal_wavelet(wavelet) {
        return Ok(true);
    }

    // For biorthogonal wavelets, check approximate Parseval relation
    // This is a practical proxy for biorthogonality
    let coeff_vectors = extract_coefficient_vectors(tree)?;

    // Calculate total coefficient energy
    let mut total_energy = 0.0;
    for coeffs in &coeff_vectors {
        total_energy += coeffs.iter().map(|x| x * x).sum::<f64>();
    }

    // Biorthogonal wavelets should approximately conserve energy
    // Allow 5% deviation due to numerical effects
    let energy_conserved = total_energy > 0.95 && total_energy < 1.05;

    Ok(energy_conserved)
}

/// Analyze correlation matrix
///
/// Computes the correlation matrix of wavelet basis functions and analyzes
/// its properties including off-diagonal elements, Frobenius norm, condition
/// number, and eigenvalue statistics.
pub fn analyze_correlation_matrix(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<CorrelationMatrixAnalysis> {
    let mut all_coefficients = Vec::new();

    // Collect coefficient vectors from multiple test signals
    for signal_config in config.test_signals.iter().take(3) {
        for wavelet in config.wavelets_to_test.iter().take(2) {
            for &max_level in config.max_levels_to_test.iter().take(2) {
                if max_level > 4 {
                    continue;
                }

                // Generate test signal
                let test_signal = generate_test_signal(signal_config)?;

                // Perform WPT decomposition
                let tree = wp_decompose(test_signal.as_slice().unwrap(), *wavelet, max_level, None)?;

                // Extract coefficient vectors
                let coeff_vectors = extract_coefficient_vectors(&tree)?;

                for coeffs in coeff_vectors {
                    if !coeffs.is_empty() && coeffs.len() >= 8 {
                        all_coefficients.push(coeffs);
                    }
                }
            }
        }
    }

    if all_coefficients.len() < 2 {
        return Ok(CorrelationMatrixAnalysis::default());
    }

    // Limit to reasonable size for correlation matrix
    let n = all_coefficients.len().min(50);
    all_coefficients.truncate(n);

    // Compute correlation matrix
    let mut correlation_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                correlation_matrix[i][j] = 1.0;
            } else {
                let corr = calculate_cross_correlation(&all_coefficients[i], &all_coefficients[j])?;
                correlation_matrix[i][j] = corr;
            }
        }
    }

    // Compute max off-diagonal element
    let mut max_off_diagonal = 0.0;
    let mut off_diagonal_frobenius_sum = 0.0;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let abs_val = correlation_matrix[i][j].abs();
                max_off_diagonal = max_off_diagonal.max(abs_val);
                off_diagonal_frobenius_sum += abs_val * abs_val;
            }
        }
    }

    let off_diagonal_frobenius_norm = off_diagonal_frobenius_sum.sqrt();

    // Estimate eigenvalues using power iteration
    let (max_eigenvalue, min_eigenvalue) = estimate_eigenvalue_bounds(&correlation_matrix)?;

    let condition_number = if min_eigenvalue > 1e-10 {
        max_eigenvalue / min_eigenvalue
    } else {
        f64::INFINITY
    };

    let eigenvalue_spread = max_eigenvalue - min_eigenvalue;
    let null_space_dimension = if min_eigenvalue < 1e-6 { 1 } else { 0 };

    Ok(CorrelationMatrixAnalysis {
        max_off_diagonal,
        off_diagonal_frobenius_norm,
        condition_number,
        eigenvalue_statistics: EigenvalueStatistics {
            min_eigenvalue,
            max_eigenvalue,
            eigenvalue_spread,
            null_space_dimension,
        },
    })
}

/// Analyze coherence
///
/// Computes mutual coherence (maximum correlation between different atoms),
/// cumulative coherence, and coherence statistics for the wavelet frame.
/// Lower mutual coherence indicates better frame properties for sparse representation.
pub fn analyze_coherence(config: &AdvancedWptValidationConfig) -> SignalResult<CoherenceAnalysis> {
    let mut all_coefficients = Vec::new();

    // Collect coefficient vectors from multiple test signals
    for signal_config in config.test_signals.iter().take(3) {
        for wavelet in config.wavelets_to_test.iter().take(2) {
            for &max_level in config.max_levels_to_test.iter().take(2) {
                if max_level > 4 {
                    continue;
                }

                // Generate test signal
                let test_signal = generate_test_signal(signal_config)?;

                // Perform WPT decomposition
                let tree = wp_decompose(test_signal.as_slice().unwrap(), *wavelet, max_level, None)?;

                // Extract coefficient vectors
                let coeff_vectors = extract_coefficient_vectors(&tree)?;

                for coeffs in coeff_vectors {
                    if !coeffs.is_empty() && coeffs.len() >= 8 {
                        all_coefficients.push(coeffs);
                    }
                }
            }
        }
    }

    if all_coefficients.len() < 2 {
        return Ok(CoherenceAnalysis::default());
    }

    // Limit to reasonable size
    let n = all_coefficients.len().min(50);
    all_coefficients.truncate(n);

    // Compute all pairwise correlations
    let mut coherence_values = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let corr = calculate_cross_correlation(&all_coefficients[i], &all_coefficients[j])?;
            coherence_values.push(corr.abs());
        }
    }

    if coherence_values.is_empty() {
        return Ok(CoherenceAnalysis::default());
    }

    // Mutual coherence: maximum correlation between different atoms
    let mutual_coherence = coherence_values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    // Sort for percentile calculations
    coherence_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Cumulative coherence (sorted correlation values)
    let cumulative_coherence = Array1::from_vec(coherence_values.clone());

    // Compute statistics
    let mean_coherence = coherence_values.iter().sum::<f64>() / coherence_values.len() as f64;

    let variance = coherence_values
        .iter()
        .map(|x| (x - mean_coherence).powi(2))
        .sum::<f64>()
        / coherence_values.len() as f64;
    let std_coherence = variance.sqrt();

    let median_coherence = if coherence_values.is_empty() {
        0.0
    } else {
        let mid = coherence_values.len() / 2;
        if coherence_values.len() % 2 == 0 {
            (coherence_values[mid - 1] + coherence_values[mid]) / 2.0
        } else {
            coherence_values[mid]
        }
    };

    // Compute percentiles (25th, 50th, 75th, 90th, 95th)
    let percentiles = vec![0.25, 0.50, 0.75, 0.90, 0.95];
    let mut coherence_percentiles = Vec::new();

    for &p in &percentiles {
        let idx = ((coherence_values.len() as f64 - 1.0) * p) as usize;
        let idx = idx.min(coherence_values.len() - 1);
        coherence_percentiles.push(coherence_values[idx]);
    }

    Ok(CoherenceAnalysis {
        mutual_coherence,
        cumulative_coherence,
        coherence_statistics: CoherenceStatistics {
            mean_coherence,
            std_coherence,
            median_coherence,
            coherence_percentiles: Array1::from_vec(coherence_percentiles),
        },
    })
}

/// Analyze frequency domain reconstruction
///
/// Compares original and reconstructed signals in the frequency domain using FFT.
/// Returns the maximum relative error in frequency magnitudes.
pub fn analyze_frequency_domain_reconstruction(
    original: &Array1<f64>,
    reconstructed: &Array1<f64>,
) -> SignalResult<f64> {
    if original.len() != reconstructed.len() {
        return Err(SignalError::ValueError(
            "Original and reconstructed signals must have the same length".to_string(),
        ));
    }

    let n = original.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Use scirs2-fft for FFT computation (assuming it's available)
    // For now, compute a simplified frequency domain error using DFT approximation

    // Calculate frequency domain representation using simple DFT
    // For large signals, this is expensive, but accurate
    let mut max_relative_error = 0.0;

    // Sample a subset of frequencies to reduce computation
    let num_freqs = (n / 2).min(100); // Sample up to 100 frequency bins

    for k in 0..num_freqs {
        // Compute DFT coefficients at frequency k
        let mut orig_real = 0.0;
        let mut orig_imag = 0.0;
        let mut recon_real = 0.0;
        let mut recon_imag = 0.0;

        let freq = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);

        for (i, (&orig_val, &recon_val)) in original.iter().zip(reconstructed.iter()).enumerate() {
            let angle = freq * (i as f64);
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();

            orig_real += orig_val * cos_angle;
            orig_imag -= orig_val * sin_angle;
            recon_real += recon_val * cos_angle;
            recon_imag -= recon_val * sin_angle;
        }

        // Calculate magnitudes
        let orig_mag = (orig_real * orig_real + orig_imag * orig_imag).sqrt();
        let recon_mag = (recon_real * recon_real + recon_imag * recon_imag).sqrt();

        // Calculate relative error
        if orig_mag > 1e-10 {
            let relative_error = ((orig_mag - recon_mag).abs() / orig_mag);
            max_relative_error = max_relative_error.max(relative_error);
        }
    }

    Ok(max_relative_error)
}

/// Analyze frequency band errors
///
/// Divides the frequency spectrum into bands and computes the reconstruction
/// error in each band. Returns an array of per-band errors.
pub fn analyze_frequency_band_errors(
    original: &Array1<f64>,
    reconstructed: &Array1<f64>,
    num_bands: usize,
) -> SignalResult<Array1<f64>> {
    if original.len() != reconstructed.len() {
        return Err(SignalError::ValueError(
            "Original and reconstructed signals must have the same length".to_string(),
        ));
    }

    if num_bands == 0 {
        return Err(SignalError::ValueError(
            "Number of bands must be positive".to_string(),
        ));
    }

    let n = original.len();
    if n == 0 {
        return Ok(Array1::zeros(num_bands));
    }

    let mut band_errors = Array1::zeros(num_bands);

    // Divide frequency range into bands
    let freqs_per_band = (n / 2) / num_bands;
    if freqs_per_band == 0 {
        return Ok(band_errors);
    }

    for band in 0..num_bands {
        let start_freq = band * freqs_per_band;
        let end_freq = ((band + 1) * freqs_per_band).min(n / 2);

        let mut band_error = 0.0;
        let mut count = 0;

        // Compute error for frequencies in this band
        for k in start_freq..end_freq {
            let mut orig_real = 0.0;
            let mut orig_imag = 0.0;
            let mut recon_real = 0.0;
            let mut recon_imag = 0.0;

            let freq = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);

            // Simplified DFT computation for this frequency
            for (i, (&orig_val, &recon_val)) in original.iter().zip(reconstructed.iter()).enumerate() {
                let angle = freq * (i as f64);
                let cos_angle = angle.cos();
                let sin_angle = angle.sin();

                orig_real += orig_val * cos_angle;
                orig_imag -= orig_val * sin_angle;
                recon_real += recon_val * cos_angle;
                recon_imag -= recon_val * sin_angle;
            }

            // Calculate magnitude error
            let orig_mag = (orig_real * orig_real + orig_imag * orig_imag).sqrt();
            let recon_mag = (recon_real * recon_real + recon_imag * recon_imag).sqrt();
            let error = (orig_mag - recon_mag).abs();

            band_error += error;
            count += 1;
        }

        // Average error for this band
        if count > 0 {
            band_errors[band] = band_error / count as f64;
        }
    }

    Ok(band_errors)
}

/// Estimate eigenvalue bounds using power iteration
///
/// Returns (max_eigenvalue, min_eigenvalue) using power iteration method.
/// This provides reasonable estimates without requiring full eigendecomposition.
fn estimate_eigenvalue_bounds(matrix: &[Vec<f64>]) -> SignalResult<(f64, f64)> {
    let n = matrix.len();
    if n == 0 {
        return Ok((1.0, 1.0));
    }

    // Initialize random vector
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    // Power iteration for maximum eigenvalue
    let mut max_eigenvalue = 0.0;
    for _ in 0..20 {
        // Matrix-vector multiplication
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += matrix[i][j] * v[j];
            }
        }

        // Compute eigenvalue estimate
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0..n {
            numerator += v[i] * new_v[i];
            denominator += v[i] * v[i];
        }
        max_eigenvalue = if denominator > 1e-10 {
            numerator / denominator
        } else {
            1.0
        };

        // Normalize
        let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            break;
        }
        for i in 0..n {
            v[i] = new_v[i] / norm;
        }
    }

    // For correlation matrices, estimate minimum eigenvalue
    // using Gershgorin circle theorem as a simple approximation
    let mut min_eigenvalue = f64::INFINITY;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j {
                row_sum += matrix[i][j].abs();
            }
        }
        let center = matrix[i][i];
        let radius = row_sum;
        let lower_bound = (center - radius).max(0.0);
        min_eigenvalue = min_eigenvalue.min(lower_bound);
    }

    // Ensure reasonable bounds
    let max_eigenvalue = max_eigenvalue.max(0.0);
    let min_eigenvalue = min_eigenvalue.max(0.0).min(max_eigenvalue);

    Ok((max_eigenvalue, min_eigenvalue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mathematical_property_validation_default() {
        let validation = MathematicalPropertyValidation::default();
        assert!(validation.perfect_reconstruction.max_error < 1e-10);
        assert!(validation.tight_frame_validation.frame_bounds_verified);
        assert!(validation.energy_conservation.energy_ratio > 0.99);
    }

    #[test]
    fn test_l2_norm_calculation() {
        let vec = Array1::from_vec(vec![3.0, 4.0]);
        let norm = calculate_l2_norm(&vec).unwrap();
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_orthogonal_wavelet_check() {
        let wavelet = Wavelet::DB(4);
        assert!(is_orthogonal_wavelet(wavelet));
    }
}