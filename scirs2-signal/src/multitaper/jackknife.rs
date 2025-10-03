// Jackknife error estimation for multitaper spectral analysis
//
// This module implements jackknife resampling methods to estimate
// confidence intervals and standard errors for multitaper spectral estimates.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::validation::checkshape;
use statrs::distribution::{ContinuousCDF, StudentsT};

#[allow(unused_imports)]
/// Jackknife confidence intervals for multitaper PSD estimate
///
/// Computes confidence intervals using the delete-one jackknife method
/// on the eigenspectra. This provides robust error estimates that account
/// for the correlation structure in the multitaper estimate.
///
/// # Arguments
///
/// * `eigenspectra` - Array of eigenspectra (k x nfreq)
/// * `eigenvalues` - Concentration values of the DPSS tapers
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
/// * `log_transform` - If true, compute CI in log domain (better for wide dynamic range)
///
/// # Returns
///
/// * `psd` - Point estimate of PSD
/// * `lower_ci` - Lower confidence interval
/// * `upper_ci` - Upper confidence interval
/// * `std_error` - Standard error estimate
#[allow(dead_code)]
pub fn jackknife_confidence_intervals(
    eigenspectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    confidence: Option<f64>,
    log_transform: Option<bool>,
) -> SignalResult<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
    let (k, n_freq) = eigenspectra.dim();
    checkshape(eigenspectra, (Some(k), None), "eigenspectra")?;
    checkshape(eigenvalues, (k,), "eigenvalues")?;

    let conf_level = confidence.unwrap_or(0.95);
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(SignalError::ValueError(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    let use_log = log_transform.unwrap_or(true);

    // Compute full PSD estimate
    let mut psd_full = Array1::zeros(n_freq);
    for j in 0..n_freq {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..k {
            weighted_sum += eigenvalues[i] * eigenspectra[[i, j]];
            weight_sum += eigenvalues[i];
        }

        psd_full[j] = weighted_sum / weight_sum;
    }

    // Jackknife resampling - leave one taper out
    let mut jackknife_estimates = Array2::zeros((k, n_freq));

    for i_out in 0..k {
        // Compute PSD without taper i_out
        for j in 0..n_freq {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k {
                if i != i_out {
                    weighted_sum += eigenvalues[i] * eigenspectra[[i, j]];
                    weight_sum += eigenvalues[i];
                }
            }

            if weight_sum > 0.0 {
                let estimate = weighted_sum / weight_sum;
                jackknife_estimates[[i_out, j]] = if use_log { estimate.ln() } else { estimate };
            }
        }
    }

    // Compute jackknife statistics
    let mut psd = Array1::zeros(n_freq);
    let mut lower_ci = Array1::zeros(n_freq);
    let mut upper_ci = Array1::zeros(n_freq);
    let mut std_error = Array1::zeros(n_freq);

    // Student's t-distribution for small samples
    let dof = (k - 1) as f64;
    let t_dist = StudentsT::new(0.0, 1.0, dof).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create t-distribution: {}", e))
    })?;
    let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);

    for j in 0..n_freq {
        // Transform full estimate if using log
        let full_transformed = if use_log {
            psd_full[j].ln()
        } else {
            psd_full[j]
        };

        // Compute mean and variance of jackknife estimates
        let mut jack_mean = 0.0;
        for i in 0..k {
            jack_mean += jackknife_estimates[[i, j]];
        }
        jack_mean /= k as f64;

        let mut jack_var = 0.0;
        for i in 0..k {
            let diff = jackknife_estimates[[i, j]] - jack_mean;
            jack_var += diff * diff;
        }
        jack_var *= (k - 1) as f64 / k as f64;

        // Standard error
        let se = jack_var.sqrt();

        // Bias-corrected estimate
        let bias = (k - 1) as f64 * (jack_mean - full_transformed);
        let corrected_estimate = full_transformed - bias;

        // Confidence intervals
        let margin = t_critical * se;
        let lower = corrected_estimate - margin;
        let upper = corrected_estimate + margin;

        // Transform back if using log
        if use_log {
            psd[j] = corrected_estimate.exp();
            lower_ci[j] = lower.exp();
            upper_ci[j] = upper.exp();
            std_error[j] = se * psd[j]; // Delta method approximation
        } else {
            psd[j] = corrected_estimate;
            lower_ci[j] = lower;
            upper_ci[j] = upper;
            std_error[j] = se;
        }
    }

    Ok((psd, lower_ci, upper_ci, std_error))
}

/// Weighted jackknife for adaptive multitaper estimates
///
/// This implements a weighted jackknife that accounts for the
/// frequency-dependent weights in adaptive multitaper estimates.
///
/// # Arguments
///
/// * `eigenspectra` - Array of eigenspectra
/// * `adaptive_weights` - Frequency-dependent weights from adaptive estimation
/// * `confidence` - Confidence level
///
/// # Returns
///
/// * `psd` - PSD estimate
/// * `lower_ci` - Lower confidence bound
/// * `upper_ci` - Upper confidence bound
#[allow(dead_code)]
pub fn weighted_jackknife(
    eigenspectra: &Array2<f64>,
    adaptive_weights: &Array2<f64>,
    confidence: Option<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let (k, n_freq) = eigenspectra.dim();
    checkshape(
        adaptive_weights,
        (Some(k), Some(n_freq)),
        "adaptive_weights",
    )?;

    let conf_level = confidence.unwrap_or(0.95);

    // Compute full weighted PSD
    let mut psd_full = Array1::zeros(n_freq);
    for j in 0..n_freq {
        let mut weighted_sum = 0.0;

        for i in 0..k {
            weighted_sum += adaptive_weights[[i, j]] * eigenspectra[[i, j]];
        }

        psd_full[j] = weighted_sum;
    }

    // Weighted jackknife estimates
    let mut jackknife_estimates = Array2::zeros((k, n_freq));

    for i_out in 0..k {
        for j in 0..n_freq {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k {
                if i != i_out {
                    // Renormalize _weights
                    weight_sum += adaptive_weights[[i, j]];
                }
            }

            if weight_sum > 0.0 {
                for i in 0..k {
                    if i != i_out {
                        let renorm_weight = adaptive_weights[[i, j]] / weight_sum;
                        weighted_sum += renorm_weight * eigenspectra[[i, j]];
                    }
                }
            }

            jackknife_estimates[[i_out, j]] = weighted_sum.ln(); // Log transform
        }
    }

    // Compute confidence intervals
    let mut psd = Array1::zeros(n_freq);
    let mut lower_ci = Array1::zeros(n_freq);
    let mut upper_ci = Array1::zeros(n_freq);

    let dof = (k - 1) as f64;
    let t_dist = StudentsT::new(0.0, 1.0, dof).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create t-distribution: {}", e))
    })?;
    let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);

    for j in 0..n_freq {
        let full_log = psd_full[j].ln();

        // Jackknife statistics
        let mut jack_mean = 0.0;
        for i in 0..k {
            jack_mean += jackknife_estimates[[i, j]];
        }
        jack_mean /= k as f64;

        let mut jack_var = 0.0;
        for i in 0..k {
            let diff = jackknife_estimates[[i, j]] - jack_mean;
            jack_var += diff * diff;
        }
        jack_var *= (k - 1) as f64 / k as f64;

        let se = jack_var.sqrt();
        let bias = (k - 1) as f64 * (jack_mean - full_log);
        let corrected = full_log - bias;

        let margin = t_critical * se;

        psd[j] = corrected.exp();
        lower_ci[j] = (corrected - margin).exp();
        upper_ci[j] = (corrected + margin).exp();
    }

    Ok((psd, lower_ci, upper_ci))
}

/// Cross-spectrum jackknife confidence intervals
///
/// Estimates confidence intervals for cross-spectral density between two signals.
///
/// # Arguments
///
/// * `eigenspectra_x` - Eigenspectra of first signal
/// * `eigenspectra_y` - Eigenspectra of second signal
/// * `eigenvalues` - Taper concentration values
/// * `confidence` - Confidence level
///
/// # Returns
///
/// * `coherence` - Magnitude squared coherence
/// * `coherence_ci` - Confidence intervals for coherence
/// * `phase` - Cross-spectrum phase
/// * `phase_ci` - Confidence intervals for phase
#[allow(dead_code)]
pub fn cross_spectrum_jackknife(
    eigenspectra_x: &Array2<scirs2_core::numeric::Complex64>,
    eigenspectra_y: &Array2<scirs2_core::numeric::Complex64>,
    eigenvalues: &Array1<f64>,
    confidence: Option<f64>,
) -> SignalResult<(
    Array1<f64>,
    (Array1<f64>, Array1<f64>),
    Array1<f64>,
    (Array1<f64>, Array1<f64>),
)> {
    let (k, n_freq) = eigenspectra_x.dim();
    checkshape(eigenspectra_y, (Some(k), Some(n_freq)), "eigenspectra_y")?;
    checkshape(eigenvalues, (k,), "eigenvalues")?;

    let conf_level = confidence.unwrap_or(0.95);

    let mut coherence = Array1::zeros(n_freq);
    let mut coherence_lower = Array1::zeros(n_freq);
    let mut coherence_upper = Array1::zeros(n_freq);
    let mut phase = Array1::zeros(n_freq);
    let mut phase_lower = Array1::zeros(n_freq);
    let mut phase_upper = Array1::zeros(n_freq);

    // Full estimate
    for j in 0..n_freq {
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = scirs2_core::numeric::Complex64::new(0.0, 0.0);
        let mut weight_sum = 0.0;

        for i in 0..k {
            let w = eigenvalues[i];
            sxx += w * eigenspectra_x[[i, j]].norm_sqr();
            syy += w * eigenspectra_y[[i, j]].norm_sqr();
            sxy += w * eigenspectra_x[[i, j]] * eigenspectra_y[[i, j]].conj();
            weight_sum += w;
        }

        sxx /= weight_sum;
        syy /= weight_sum;
        sxy /= weight_sum;

        if sxx > 0.0 && syy > 0.0 {
            coherence[j] = sxy.norm_sqr() / (sxx * syy);
            phase[j] = sxy.arg();
        }
    }

    // Jackknife for confidence intervals
    let mut jack_coh = Array2::zeros((k, n_freq));
    let mut jack_phase = Array2::zeros((k, n_freq));

    for i_out in 0..k {
        for j in 0..n_freq {
            let mut sxx = 0.0;
            let mut syy = 0.0;
            let mut sxy = scirs2_core::numeric::Complex64::new(0.0, 0.0);
            let mut weight_sum = 0.0;

            for i in 0..k {
                if i != i_out {
                    let w = eigenvalues[i];
                    sxx += w * eigenspectra_x[[i, j]].norm_sqr();
                    syy += w * eigenspectra_y[[i, j]].norm_sqr();
                    sxy += w * eigenspectra_x[[i, j]] * eigenspectra_y[[i, j]].conj();
                    weight_sum += w;
                }
            }

            if weight_sum > 0.0 {
                sxx /= weight_sum;
                syy /= weight_sum;
                sxy /= weight_sum;

                if sxx > 0.0 && syy > 0.0 {
                    jack_coh[[i_out, j]] = sxy.norm_sqr() / (sxx * syy);
                    jack_phase[[i_out, j]] = sxy.arg();
                }
            }
        }
    }

    // Compute confidence intervals
    let dof = (k - 1) as f64;
    let t_dist = StudentsT::new(0.0, 1.0, dof).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create t-distribution: {}", e))
    })?;
    let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);

    for j in 0..n_freq {
        // Coherence CI (using Fisher z-transform)
        let mut z_vals = vec![0.0; k];
        for i in 0..k {
            let coh = jack_coh[[i, j]];
            if coh > 0.0 && coh < 1.0 {
                z_vals[i] = 0.5 * ((1.0 + coh.sqrt()) / (1.0 - coh.sqrt())).ln();
            }
        }

        let z_mean: f64 = z_vals.iter().sum::<f64>() / k as f64;
        let z_var: f64 =
            z_vals.iter().map(|&z| (z - z_mean).powi(2)).sum::<f64>() * (k - 1) as f64 / k as f64;
        let z_se = z_var.sqrt();

        let z_lower = z_mean - t_critical * z_se;
        let z_upper = z_mean + t_critical * z_se;

        // Transform back
        coherence_lower[j] = ((2.0 * z_lower).exp() - 1.0) / ((2.0 * z_lower).exp() + 1.0);
        coherence_upper[j] = ((2.0 * z_upper).exp() - 1.0) / ((2.0 * z_upper).exp() + 1.0);
        coherence_lower[j] = coherence_lower[j].powi(2).max(0.0).min(1.0);
        coherence_upper[j] = coherence_upper[j].powi(2).max(0.0).min(1.0);

        // Phase CI
        let mut phase_mean = 0.0;
        for i in 0..k {
            phase_mean += jack_phase[[i, j]];
        }
        phase_mean /= k as f64;

        let mut phase_var = 0.0;
        for i in 0..k {
            let diff = jack_phase[[i, j]] - phase_mean;
            phase_var += diff * diff;
        }
        phase_var *= (k - 1) as f64 / k as f64;
        let phase_se = phase_var.sqrt();

        phase_lower[j] = phase[j] - t_critical * phase_se;
        phase_upper[j] = phase[j] + t_critical * phase_se;
    }

    Ok((
        coherence,
        (coherence_lower, coherence_upper),
        phase,
        (phase_lower, phase_upper),
    ))
}
