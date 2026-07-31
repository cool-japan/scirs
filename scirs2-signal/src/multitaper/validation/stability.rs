//! Numerical stability testing functions
//!
//! This module provides validation functions for numerical stability
//! including condition number analysis and extreme input testing.

use super::types::NumericalStabilityMetrics;
use crate::error::{SignalError, SignalResult};
use crate::multitaper::psd::pmtm;
use crate::multitaper::windows::dpss;

/// Test numerical stability of multitaper operations.
///
/// Actually runs the multitaper implementation (DPSS taper generation and
/// `pmtm` spectral estimation) on very short/long signals and extreme
/// NW/taper-count configurations, checking for NaN/Inf outputs, and
/// computes the condition number and precision-loss metrics from genuine
/// DPSS taper properties -- replacing a previous stand-in that hardcoded
/// "always stable" values (`condition_number: 100.0`, `numerical_issues:
/// 0`, four sub-checks each hardcoded `true`) regardless of the actual
/// implementation's behavior.
///
/// NOTE: this crate's DPSS computation uses a dense (Jacobi/QR)
/// eigendecomposition whose cost grows very steeply with signal length
/// (a pre-existing performance characteristic of `dpss`/`pmtm` themselves,
/// not something this validation module controls), so the sizes used
/// below are deliberately kept small to keep this validation function
/// itself usable, while still genuinely exercising each scenario.
pub fn test_numerical_stability_enhanced() -> SignalResult<NumericalStabilityMetrics> {
    let mut numerical_issues = 0usize;

    let very_short_ok = test_very_short_signals(&mut numerical_issues);
    let very_long_ok = test_very_long_signals(&mut numerical_issues);
    let extreme_nw_ok = test_extreme_nw_values(&mut numerical_issues);
    let many_tapers_ok = test_many_tapers(&mut numerical_issues);
    let extreme_input_stable = very_short_ok && very_long_ok && extreme_nw_ok && many_tapers_ok;

    let condition_number = estimate_condition_number_multitaper()?;
    let precision_loss = estimate_precision_loss()?;

    Ok(NumericalStabilityMetrics {
        condition_number,
        precision_loss,
        numerical_issues,
        extreme_input_stable,
    })
}

/// Condition number of a representative DPSS taper set: the ratio of the
/// largest to smallest concentration eigenvalue (lambda), a genuine measure
/// of how ill-conditioned the multitaper spectral weighting actually is for
/// this crate's own DPSS implementation.
fn estimate_condition_number_multitaper() -> SignalResult<f64> {
    let (_, eigenvalues_opt) = dpss(32, 4.0, 5, true)?;
    let eigenvalues = eigenvalues_opt.ok_or_else(|| {
        SignalError::ComputationError("DPSS eigenvalues were not returned".to_string())
    })?;

    let max_eig = eigenvalues.iter().cloned().fold(f64::MIN, f64::max);
    let min_eig = eigenvalues
        .iter()
        .cloned()
        .fold(f64::MAX, f64::min)
        .max(1e-300);
    Ok((max_eig / min_eig).max(1.0))
}

/// Precision loss: the actual deviation from orthogonality of a computed
/// DPSS taper set (root-mean-square of the off-diagonal entries of
/// `tapers^T * tapers`, which is exactly zero for perfectly orthogonal
/// tapers and grows with genuine accumulated floating-point/algorithmic
/// error).
fn estimate_precision_loss() -> SignalResult<f64> {
    let (tapers, _) = dpss(32, 4.0, 5, false)?;
    let gram = tapers.t().dot(&tapers);
    let n = gram.nrows();
    if n == 0 {
        return Ok(0.0);
    }

    let mut off_diag_sq = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                off_diag_sq += gram[[i, j]].powi(2);
            }
        }
    }
    let off_diag_count = (n * n - n).max(1);
    Ok((off_diag_sq / off_diag_count as f64).sqrt())
}

/// Test with very short signals: the minimal viable signal length for a
/// small taper count.
fn test_very_short_signals(issues: &mut usize) -> bool {
    let signal: Vec<f64> = (0..8).map(|i| (i as f64 * 0.3).sin()).collect();
    match pmtm(
        &signal,
        Some(1.0),
        Some(1.5),
        Some(2),
        None,
        Some(true),
        Some(false),
    ) {
        Ok((_, psd, _, _)) => {
            let bad = psd.iter().any(|&p| !p.is_finite());
            if bad {
                *issues += 1;
            }
            !bad
        }
        Err(_) => {
            *issues += 1;
            false
        }
    }
}

/// Test with a signal longer than the "very short" case above.
fn test_very_long_signals(issues: &mut usize) -> bool {
    let n = 64; // must be a power of 2: `pmtm`'s FFT requires it
    let signal: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.05).sin() + 0.1 * (i as f64 * 0.31).cos())
        .collect();
    match pmtm(
        &signal,
        Some(1.0),
        Some(3.0),
        Some(3),
        None,
        Some(true),
        Some(false),
    ) {
        Ok((_, psd, _, _)) => {
            let bad = psd.iter().any(|&p| !p.is_finite());
            if bad {
                *issues += 1;
            }
            !bad
        }
        Err(_) => {
            *issues += 1;
            false
        }
    }
}

/// Test with extreme (very small and very large, but still valid)
/// time-bandwidth product NW values.
fn test_extreme_nw_values(issues: &mut usize) -> bool {
    let n = 32;

    let small_ok = match dpss(n, 0.55, 1, false) {
        Ok((tapers, _)) => {
            let bad = tapers.iter().any(|&t| !t.is_finite());
            if bad {
                *issues += 1;
            }
            !bad
        }
        Err(_) => {
            *issues += 1;
            false
        }
    };

    let large_nw = (n as f64 / 2.0 - 1.0).max(1.0);
    let large_ok = match dpss(n, large_nw, 1, false) {
        Ok((tapers, _)) => {
            let bad = tapers.iter().any(|&t| !t.is_finite());
            if bad {
                *issues += 1;
            }
            !bad
        }
        Err(_) => {
            *issues += 1;
            false
        }
    };

    small_ok && large_ok
}

/// Test with a large number of tapers relative to the signal length.
fn test_many_tapers(issues: &mut usize) -> bool {
    let n = 32;
    let nw = 4.0;
    // Request close to the practical maximum usable number of tapers
    // (2*NW - 1).
    let k = ((2.0 * nw) as usize).saturating_sub(1).min(n - 1).max(1);

    match dpss(n, nw, k, true) {
        Ok((tapers, eigenvalues)) => {
            let tapers_bad = tapers.iter().any(|&t| !t.is_finite());
            let eig_bad = eigenvalues
                .as_ref()
                .map(|e| e.iter().any(|&v| !v.is_finite()))
                .unwrap_or(true);
            if tapers_bad || eig_bad {
                *issues += 1;
            }
            !tapers_bad && !eig_bad
        }
        Err(_) => {
            *issues += 1;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_stability() {
        // A single comprehensive pass (rather than re-running the same
        // expensive DPSS/pmtm computations across several separate tests)
        // covering all of `NumericalStabilityMetrics`'s genuinely-computed
        // fields.
        let result = test_numerical_stability_enhanced();
        assert!(result.is_ok());

        let metrics = result.expect("Operation failed");

        // Genuinely computed, not the old hardcoded 100.0/0.0 constants:
        // both metrics react to the real DPSS taper properties.
        assert!(metrics.condition_number.is_finite());
        assert!(metrics.condition_number >= 1.0);
        assert!(metrics.precision_loss.is_finite());
        assert!(metrics.precision_loss >= 0.0);
        assert!(
            metrics.precision_loss < 0.5,
            "precision_loss unexpectedly large: {}",
            metrics.precision_loss
        );

        assert!(metrics.extreme_input_stable);
        assert_eq!(metrics.numerical_issues, 0);
    }

    #[test]
    fn test_stability_checks_detect_genuine_failure() {
        // A DPSS call with an invalid (zero) time-bandwidth product must
        // be counted as a genuine numerical issue, not silently reported
        // as `true` regardless of outcome (as the old stub did). This is
        // intentionally cheap (fails validation immediately, without
        // running the expensive eigensolver).
        let mut issues = 0usize;
        let ok = match dpss(64, 0.0, 1, false) {
            Ok((tapers, _)) => !tapers.iter().any(|&t| !t.is_finite()),
            Err(_) => {
                issues += 1;
                false
            }
        };
        assert!(!ok);
        assert_eq!(issues, 1);
    }
}
