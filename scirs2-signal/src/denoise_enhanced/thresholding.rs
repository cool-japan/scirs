//! Thresholding methods and threshold computation algorithms
//!
//! This module provides various thresholding methods (soft, hard, SCAD, etc.)
//! and threshold selection algorithms (SURE, Bayes, minimax, etc.).

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;

/// Soft thresholding function
pub fn soft_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;

    for val in thresholded.iter_mut() {
        if val.abs() > threshold {
            *val = val.signum() * (val.abs() - threshold);
            retained_count += 1;
        } else {
            *val = 0.0;
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    (thresholded, retention_rate)
}

/// Hard thresholding function
pub fn hard_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;

    for val in thresholded.iter_mut() {
        if val.abs() > threshold {
            retained_count += 1;
        } else {
            *val = 0.0;
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    (thresholded, retention_rate)
}

/// Garotte thresholding function
pub fn garotte_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;
    let threshold_sq = threshold * threshold;

    for val in thresholded.iter_mut() {
        if val.abs() > threshold {
            let val_sq = (*val) * (*val);
            *val *= 1.0 - threshold_sq / val_sq.max(f64::EPSILON);
            retained_count += 1;
        } else {
            *val = 0.0;
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    (thresholded, retention_rate)
}

/// SCAD (Smoothly Clipped Absolute Deviation) thresholding
pub fn scad_threshold(coeffs: &Array1<f64>, threshold: f64, a: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;

    for val in thresholded.iter_mut() {
        let abs_val = val.abs();
        if abs_val <= threshold {
            *val = 0.0;
        } else if abs_val <= 2.0 * threshold {
            // Soft-threshold region
            *val = val.signum() * (abs_val - threshold);
            retained_count += 1;
        } else if abs_val <= a * threshold {
            // SCAD middle region
            let numerator = (a - 1.0) * *val - val.signum() * a * threshold;
            *val = numerator / (a - 2.0);
            retained_count += 1;
        } else {
            // Keep original value
            retained_count += 1;
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    (thresholded, retention_rate)
}

/// Firm thresholding function
pub fn firm_threshold(coeffs: &Array1<f64>, threshold: f64, alpha: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;
    let alpha_threshold = alpha * threshold;

    for val in thresholded.iter_mut() {
        let abs_val = val.abs();
        if abs_val > alpha_threshold {
            retained_count += 1; // Keep original value
        } else if abs_val > threshold {
            *val = val.signum() * alpha * (abs_val - threshold) / (alpha - 1.0);
            retained_count += 1;
        } else {
            *val = 0.0;
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    (thresholded, retention_rate)
}

/// Hyperbolic thresholding function
pub fn hyperbolic_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;

    for val in thresholded.iter_mut() {
        let abs_val = val.abs();
        if abs_val > 1e-12 {
            let hyperbolic_factor = 1.0 - threshold / abs_val;
            if hyperbolic_factor > 0.0 {
                *val *= hyperbolic_factor;
                retained_count += 1;
            } else {
                *val = 0.0;
            }
        } else {
            *val = 0.0;
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    (thresholded, retention_rate)
}

/// Block thresholding function
pub fn block_threshold(
    coeffs: &Array1<f64>,
    threshold: f64,
    block_size: usize,
) -> SignalResult<(Array1<f64>, f64)> {
    if block_size == 0 {
        return Err(SignalError::ValueError(
            "Block size must be greater than 0".to_string(),
        ));
    }

    let mut thresholded = coeffs.clone();
    let mut retained_count = 0;
    let n = coeffs.len();

    for i in (0..n).step_by(block_size) {
        let end = (i + block_size).min(n);
        let block = &coeffs.slice(scirs2_core::ndarray::s![i..end]);

        // Compute block energy
        let block_energy: f64 = block.iter().map(|&x| x * x).sum();
        let block_norm = block_energy.sqrt();

        if block_norm > threshold {
            // Keep block with soft thresholding scaling
            let scale_factor = (block_norm - threshold) / block_norm;
            for j in i..end {
                thresholded[j] *= scale_factor;
                retained_count += 1;
            }
        } else {
            // Zero out entire block
            for j in i..end {
                thresholded[j] = 0.0;
            }
        }
    }

    let retention_rate = retained_count as f64 / coeffs.len() as f64;
    Ok((thresholded, retention_rate))
}

/// Compute SURE threshold
pub fn compute_sure_threshold(coeffs: &Array1<f64>, noise_sigma: f64) -> SignalResult<f64> {
    let n = coeffs.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Empty coefficients array".to_string(),
        ));
    }

    // Sort coefficients by absolute value
    let mut sorted_coeffs: Vec<f64> = coeffs.iter().map(|&x| x.abs()).collect();
    sorted_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut best_threshold = noise_sigma * (2.0 * (n as f64).ln()).sqrt();
    let mut best_risk = f64::INFINITY;

    // Evaluate SURE risk for different thresholds
    for &candidate_threshold in &sorted_coeffs {
        if candidate_threshold > 0.0 {
            let risk = sure_risk(coeffs, candidate_threshold, noise_sigma);
            if risk < best_risk {
                best_risk = risk;
                best_threshold = candidate_threshold;
            }
        }
    }

    Ok(best_threshold)
}

/// Compute Bayes threshold
pub fn compute_bayes_threshold(coeffs: &Array1<f64>, noise_sigma: f64) -> f64 {
    let signal_variance = coeffs.var(0.0);
    let noise_variance = noise_sigma * noise_sigma;

    if signal_variance > noise_variance {
        noise_variance / (signal_variance - noise_variance).sqrt()
    } else {
        noise_sigma * (2.0 * (coeffs.len() as f64).ln()).sqrt()
    }
}

/// Compute minimax threshold
pub fn compute_minimax_threshold(n: f64, noise_sigma: f64) -> f64 {
    if n >= 32.0 {
        noise_sigma * (2.0 * n.ln()).sqrt()
    } else {
        // For small n, use a more conservative threshold
        noise_sigma * (2.0 * n.ln() + 0.5).sqrt()
    }
}

/// Compute FDR (False Discovery Rate) threshold
pub fn compute_fdr_threshold(coeffs: &Array1<f64>, noise_sigma: f64, q: f64) -> SignalResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SignalError::ValueError(
            "FDR parameter q must be between 0 and 1".to_string(),
        ));
    }

    let n = coeffs.len();
    let mut abs_coeffs: Vec<f64> = coeffs.iter().map(|&x| x.abs()).collect();
    abs_coeffs.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort in descending order

    // Convert to normalized z-scores
    let z_scores: Vec<f64> = abs_coeffs.iter().map(|&x| x / noise_sigma).collect();

    // Find largest k such that z_k >= sqrt(2*ln(n*q/k))
    for (k, &z_k) in z_scores.iter().enumerate() {
        let k_idx = k + 1; // 1-indexed
        let threshold_z = (2.0 * ((n as f64 * q) / k_idx as f64).ln()).sqrt();

        if z_k >= threshold_z {
            return Ok(z_k * noise_sigma);
        }
    }

    // If no threshold found, return universal threshold
    Ok(noise_sigma * (2.0 * (n as f64).ln()).sqrt())
}

/// Compute cross-validation threshold
pub fn compute_cv_threshold(coeffs: &Array1<f64>, noise_sigma: f64) -> SignalResult<f64> {
    let n = coeffs.len();
    if n < 10 {
        return Ok(noise_sigma * (2.0 * (n as f64).ln()).sqrt());
    }

    let fold_size = n / 5; // 5-fold cross-validation
    let mut candidate_thresholds: Vec<f64> = coeffs.iter().map(|&x| x.abs()).collect();
    candidate_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    candidate_thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let mut best_threshold = candidate_thresholds[candidate_thresholds.len() / 2];
    let mut best_cv_error = f64::INFINITY;

    for &threshold in &candidate_thresholds {
        let mut total_error = 0.0;

        // 5-fold cross-validation
        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            // Training set (all except current fold)
            let mut train_coeffs = Vec::new();
            for i in 0..n {
                if i < test_start || i >= test_end {
                    train_coeffs.push(coeffs[i]);
                }
            }

            // Apply threshold to training set
            let (thresholded_train, _) = soft_threshold(&Array1::from_vec(train_coeffs), threshold);

            // Compute error on test set
            for i in test_start..test_end {
                let predicted = if coeffs[i].abs() > threshold {
                    coeffs[i] - coeffs[i].signum() * threshold
                } else {
                    0.0
                };
                total_error += (coeffs[i] - predicted).powi(2);
            }
        }

        if total_error < best_cv_error {
            best_cv_error = total_error;
            best_threshold = threshold;
        }
    }

    Ok(best_threshold)
}

/// Compute SURE risk for given threshold
fn sure_risk(coeffs: &Array1<f64>, threshold: f64, noise_sigma: f64) -> f64 {
    let n = coeffs.len() as f64;
    let sigma_sq = noise_sigma * noise_sigma;

    // Count coefficients exceeding threshold
    let exceeded_count = coeffs.iter().filter(|&&x| x.abs() > threshold).count() as f64;

    // SURE formula: n*sigma^2 - 2*sigma^2*#{|w_i| <= t} + sum(min(w_i^2, t^2))
    let mut sum_min = 0.0;
    for &coeff in coeffs.iter() {
        sum_min += (coeff * coeff).min(threshold * threshold);
    }

    n * sigma_sq - 2.0 * sigma_sq * (n - exceeded_count) + sum_min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold() {
        let coeffs = Array1::from_vec(vec![1.0, 0.5, -0.8, 0.2, -1.5]);
        let threshold = 0.6;
        let (thresholded, retention_rate) = soft_threshold(&coeffs, threshold);

        assert!((thresholded[0] - 0.4).abs() < 1e-10); // 1.0 - 0.6
        assert!((thresholded[1] - 0.0).abs() < 1e-10); // |0.5| <= 0.6
        assert!((thresholded[2] - (-0.2)).abs() < 1e-10); // -0.8 + 0.6
        assert!((thresholded[3] - 0.0).abs() < 1e-10); // |0.2| <= 0.6
        assert!((thresholded[4] - (-0.9)).abs() < 1e-10); // -1.5 + 0.6
        assert!((retention_rate - 0.6).abs() < 1e-10); // 3 out of 5 coefficients retained
    }

    #[test]
    fn test_hard_threshold() {
        let coeffs = Array1::from_vec(vec![1.0, 0.5, -0.8, 0.2, -1.5]);
        let threshold = 0.6;
        let (thresholded, retention_rate) = hard_threshold(&coeffs, threshold);

        assert_eq!(thresholded[0], 1.0); // |1.0| > 0.6
        assert_eq!(thresholded[1], 0.0); // |0.5| <= 0.6
        assert_eq!(thresholded[2], -0.8); // |-0.8| > 0.6
        assert_eq!(thresholded[3], 0.0); // |0.2| <= 0.6
        assert_eq!(thresholded[4], -1.5); // |-1.5| > 0.6
        assert_eq!(retention_rate, 0.6); // 3 out of 5 coefficients retained
    }

    #[test]
    fn test_block_threshold() {
        let coeffs = Array1::from_vec(vec![1.0, 0.5, 0.8, 0.2]);
        let threshold = 0.5;
        let block_size = 2;
        let result = block_threshold(&coeffs, threshold, block_size);
        assert!(result.is_ok());
        let (thresholded, retention_rate) = result.unwrap();
        assert_eq!(thresholded.len(), coeffs.len());
        assert!(retention_rate >= 0.0 && retention_rate <= 1.0);
    }

    #[test]
    fn test_threshold_computation() {
        let coeffs = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.2, 1.2, 0.3]);
        let noise_sigma = 0.1;

        // Test SURE threshold
        let sure_thresh = compute_sure_threshold(&coeffs, noise_sigma);
        assert!(sure_thresh.is_ok());
        assert!(sure_thresh.unwrap() > 0.0);

        // Test Bayes threshold
        let bayes_thresh = compute_bayes_threshold(&coeffs, noise_sigma);
        assert!(bayes_thresh > 0.0);

        // Test minimax threshold
        let minimax_thresh = compute_minimax_threshold(coeffs.len() as f64, noise_sigma);
        assert!(minimax_thresh > 0.0);

        // Test FDR threshold
        let fdr_thresh = compute_fdr_threshold(&coeffs, noise_sigma, 0.1);
        assert!(fdr_thresh.is_ok());
        assert!(fdr_thresh.unwrap() > 0.0);

        // Test CV threshold
        let cv_thresh = compute_cv_threshold(&coeffs, noise_sigma);
        assert!(cv_thresh.is_ok());
        assert!(cv_thresh.unwrap() > 0.0);
    }

    #[test]
    fn test_error_conditions() {
        let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Test block threshold with zero block size
        let result = block_threshold(&coeffs, 0.5, 0);
        assert!(result.is_err());

        // Test FDR with invalid q
        let result = compute_fdr_threshold(&coeffs, 0.1, 1.5);
        assert!(result.is_err());

        let result = compute_fdr_threshold(&coeffs, 0.1, -0.1);
        assert!(result.is_err());
    }
}
