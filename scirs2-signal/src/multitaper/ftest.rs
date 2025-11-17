// F-test for detecting harmonic line components in multitaper spectra
//
// This module implements the F-test for detecting periodic components in signals
// using the multitaper method. The F-test provides a statistical measure to
// identify spectral lines against a background continuum.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_core::ndarray::ArrayStatCompat;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use statrs::statistics::Statistics;

#[allow(unused_imports)]
/// Compute F-test for line components in multitaper spectrum
///
/// The F-test provides a statistical test for the presence of line components
/// (sinusoidal signals) in the spectrum. It compares the power in a given
/// frequency bin concentrated in the first few eigenspectra to the total power.
///
/// # Arguments
///
/// * `eigenspectra` - Array of eigenspectra from multitaper analysis (k x nfreq)
/// * `eigenvalues` - Concentration values of the DPSS tapers
/// * `p_value` - P-value threshold for significance (default = 0.05)
///
/// # Returns
///
/// * `f_statistic` - F-statistic values for each frequency
/// * `p_values` - P-values for each frequency
/// * `significant` - Boolean array indicating significant line components
///
/// # References
///
/// Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
/// Proceedings of the IEEE, 70(9), 1055-1096.
#[allow(dead_code)]
pub fn multitaper_ftest(
    eigenspectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    p_value: Option<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>, Array1<bool>)> {
    let (k, n_freq) = eigenspectra.dim();
    checkshape(eigenspectra, (Some(k), None), "eigenspectra")?;
    checkshape(eigenvalues, (k,), "eigenvalues")?;

    let p_threshold = p_value.unwrap_or(0.05);
    check_positive(p_threshold, "p_value")?;
    if p_threshold >= 1.0 {
        return Err(SignalError::ValueError(
            "p_value must be less than 1.0".to_string(),
        ));
    }

    let mut f_statistic = Array1::zeros(n_freq);
    let mut p_values = Array1::zeros(n_freq);
    let mut significant = Array1::from_elem(n_freq, false);

    // Degrees of freedom for F-distribution
    let dof1 = 2.0; // Numerator DOF (real and imaginary parts)
    let dof2 = 2.0 * (k as f64 - 1.0); // Denominator DOF

    // Create F-distribution
    let f_dist = FisherSnedecor::new(dof1, dof2).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create F-distribution: {}", e))
    })?;

    for j in 0..n_freq {
        // Compute weighted mean spectrum
        let mut s_bar = 0.0;
        let mut lambda_sum = 0.0;

        for i in 0..k {
            s_bar += eigenvalues[i] * eigenspectra[[i, j]];
            lambda_sum += eigenvalues[i];
        }
        s_bar /= lambda_sum;

        // Compute regression coefficients for line component test
        // Model: S_k = μ + U_k * Re(A) + V_k * Im(A) + ε_k
        // where U_k and V_k are the real and imaginary parts of the kth taper

        // For simplicity, we use the approximation that the line component
        // amplitude is proportional to the concentration in the first eigenspectrum
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        // Compute weighted variance estimate
        for i in 0..k {
            let residual = eigenspectra[[i, j]] - s_bar;
            denominator += eigenvalues[i] * residual * residual;
        }

        // Line component estimate from first eigenspectrum
        if k > 0 {
            let line_estimate = eigenspectra[[0, j]] - s_bar;
            numerator = eigenvalues[0] * line_estimate * line_estimate;
        }

        // F-statistic
        if denominator > 1e-10 {
            let f_stat = (numerator * dof2) / (denominator * dof1);
            f_statistic[j] = f_stat;

            // Compute p-_value
            p_values[j] = 1.0 - f_dist.cdf(f_stat);

            // Test for significance
            significant[j] = p_values[j] < p_threshold;
        }
    }

    Ok((f_statistic, p_values, significant))
}

/// Refined F-test using regression approach
///
/// This implements a more sophisticated F-test that properly accounts for
/// the complex nature of the line component estimation.
///
/// # Arguments
///
/// * `eigenspectra` - Complex eigenspectra before taking magnitude squared
/// * `tapers` - DPSS tapers used in the analysis
/// * `eigenvalues` - Concentration values
/// * `p_value` - Significance threshold
///
/// # Returns
///
/// * `f_statistic` - F-statistic values
/// * `line_amplitudes` - Estimated complex amplitudes of line components
/// * `significant` - Boolean significance indicators
#[allow(dead_code)]
pub fn multitaper_ftest_complex(
    eigenspectra: &Array2<scirs2_core::numeric::Complex64>,
    tapers: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    p_value: Option<f64>,
) -> SignalResult<(Array1<f64>, Array1<scirs2_core::numeric::Complex64>, Array1<bool>)> {
    let (k, n_freq) = eigenspectra.dim();
    let (k_taper, n_time) = tapers.dim();

    if k != k_taper {
        return Err(SignalError::ShapeMismatch(
            "Number of eigenspectra must match number of tapers".to_string(),
        ));
    }

    let p_threshold = p_value.unwrap_or(0.05);

    let mut f_statistic = Array1::zeros(n_freq);
    let mut line_amplitudes = Array1::zeros(n_freq);
    let mut significant = Array1::from_elem(n_freq, false);

    // F-distribution for hypothesis testing
    let dof1 = 2.0;
    let dof2 = 2.0 * (k as f64 - 1.0);
    let f_dist = FisherSnedecor::new(dof1, dof2).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create F-distribution: {}", e))
    })?;

    // Compute mean taper values (for DC component regression)
    let taper_means: Vec<f64> = (0..k)
        .map(|i| {
            let row = tapers.row(i);
            row.mean_or(0.0)
        })
        .collect();

    for j in 0..n_freq {
        // Weighted least squares regression for line component
        // Model: Y_k = μ + a * U_k + ε_k
        // where Y_k are the eigenspectra and U_k are the taper DC values

        let mut sum_wy = scirs2_core::numeric::Complex64::new(0.0, 0.0);
        let mut sum_wu = 0.0;
        let mut sum_wuu = 0.0;
        let mut sum_w = 0.0;

        // Compute regression coefficients
        for i in 0..k {
            let w = eigenvalues[i];
            let y = eigenspectra[[i, j]];
            let u = taper_means[i];

            sum_wy += w * y;
            sum_wu += w * u;
            sum_wuu += w * u * u;
            sum_w += w;
        }

        // Mean estimates
        let y_mean = sum_wy / sum_w;
        let u_mean = sum_wu / sum_w;

        // Regression coefficient
        let mut beta = scirs2_core::numeric::Complex64::new(0.0, 0.0);
        let denominator = sum_wuu - sum_wu * u_mean;

        if denominator.abs() > 1e-10 {
            let mut numerator = scirs2_core::numeric::Complex64::new(0.0, 0.0);
            for i in 0..k {
                let w = eigenvalues[i];
                let y_centered = eigenspectra[[i, j]] - y_mean;
                let u_centered = taper_means[i] - u_mean;
                numerator += w * y_centered * u_centered;
            }
            beta = numerator / denominator;
        }

        line_amplitudes[j] = beta;

        // Compute residual sum of squares
        let mut rss = 0.0;
        let mut tss = 0.0;

        for i in 0..k {
            let w = eigenvalues[i];
            let y = eigenspectra[[i, j]];
            let u = taper_means[i];
            let y_pred = y_mean + beta * (u - u_mean);
            let residual = y - y_pred;
            let total_dev = y - y_mean;

            rss += w * residual.norm_sqr();
            tss += w * total_dev.norm_sqr();
        }

        // F-statistic
        if rss > 1e-10 && tss > rss {
            let mss = tss - rss; // Model sum of squares
            let f_stat = (mss / dof1) / (rss / dof2);
            f_statistic[j] = f_stat;

            // P-_value and significance test
            let p_val = 1.0 - f_dist.cdf(f_stat);
            significant[j] = p_val < p_threshold;
        }
    }

    Ok((f_statistic, line_amplitudes, significant))
}

/// Harmonic F-test for detecting multiple harmonics
///
/// Tests for the presence of harmonic series (fundamental + overtones)
/// in the multitaper spectrum.
///
/// # Arguments
///
/// * `eigenspectra` - Eigenspectra from multitaper analysis
/// * `eigenvalues` - Taper concentration values
/// * `frequencies` - Frequency array
/// * `fundamental_range` - Range to search for fundamental frequency
/// * `n_harmonics` - Number of harmonics to test
/// * `p_value` - Significance threshold
///
/// # Returns
///
/// * `fundamental_freq` - Detected fundamental frequency (if any)
/// * `harmonic_amplitudes` - Amplitudes of detected harmonics
/// * `combined_significance` - Overall significance of harmonic series
#[allow(dead_code)]
pub fn harmonic_ftest(
    eigenspectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    frequencies: &[f64],
    fundamental_range: (f64, f64),
    n_harmonics: usize,
    p_value: Option<f64>,
) -> SignalResult<(Option<f64>, Vec<f64>, f64)> {
    let (k, n_freq) = eigenspectra.dim();

    if frequencies.len() != n_freq {
        return Err(SignalError::ShapeMismatch(
            "Frequency array length must match eigenspectra".to_string(),
        ));
    }

    let p_threshold = p_value.unwrap_or(0.05);
    check_positive(n_harmonics as f64, "n_harmonics")?;

    // Find frequency indices within fundamental _range
    let mut fund_start = 0;
    let mut fund_end = n_freq;

    for (i, &f) in frequencies.iter().enumerate() {
        if f >= fundamental_range.0 && fund_start == 0 {
            fund_start = i;
        }
        if f > fundamental_range.1 {
            fund_end = i;
            break;
        }
    }

    if fund_start >= fund_end {
        return Ok((None, vec![], 1.0));
    }

    // Search for fundamental frequency with strongest harmonic series
    let mut best_fundamental = None;
    let mut best_amplitudes = vec![0.0; n_harmonics];
    let mut best_significance = 1.0;

    // Compute F-statistics for all frequencies
    let (f_stats__) = multitaper_ftest(eigenspectra, eigenvalues, Some(p_threshold))?;

    for i in fund_start..fund_end {
        let fundamental = frequencies[i];
        let mut harmonic_amps = vec![0.0; n_harmonics];
        let mut combined_f = 0.0;
        let mut valid_harmonics = 0;

        // Test each harmonic
        for h in 0..n_harmonics {
            let harmonic_freq = fundamental * (h + 1) as f64;

            // Find closest frequency bin
            let mut closest_idx = 0;
            let mut min_diff = f64::INFINITY;

            for (j, &f) in frequencies.iter().enumerate() {
                let diff = (f - harmonic_freq).abs();
                if diff < min_diff {
                    min_diff = diff;
                    closest_idx = j;
                }
            }

            // Only include if within reasonable tolerance
            if min_diff < 0.5 * (frequencies[1] - frequencies[0]) {
                harmonic_amps[h] = f_stats__[closest_idx];
                combined_f += f_stats__[closest_idx];
                valid_harmonics += 1;
            }
        }

        if valid_harmonics > 0 {
            // Combined significance using Fisher's method
            // -2 * sum(ln(p_i)) ~ chi-squared with 2k degrees of freedom
            let avg_f = combined_f / valid_harmonics as f64;

            // Approximate combined p-_value
            let dof1 = 2.0;
            let dof2 = 2.0 * (k as f64 - 1.0);
            let f_dist = FisherSnedecor::new(dof1, dof2).map_err(|e| {
                SignalError::ComputationError(format!("Failed to create F-distribution: {}", e))
            })?;
            let combined_p = 1.0 - f_dist.cdf(avg_f);

            if combined_p < best_significance {
                best_fundamental = Some(fundamental);
                best_amplitudes = harmonic_amps;
                best_significance = combined_p;
            }
        }
    }

    Ok((best_fundamental, best_amplitudes, best_significance))
}
