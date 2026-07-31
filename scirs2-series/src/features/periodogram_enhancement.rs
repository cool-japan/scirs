//! Periodogram enhancement functions: window analysis, confidence
//! intervals, peak significance, bias correction, variance reduction,
//! smoothing, and frequency-resolution enhancement (zero-padding and cubic
//! spline interpolation) for periodograms produced elsewhere in
//! [`super::frequency`].
//!
//! Split out of `frequency.rs` to keep that file under the workspace's
//! 2000-line-per-file guideline.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::EnhancedPeriodogramConfig;
use super::frequency::{calculate_simple_periodogram, create_window, WindowTypeInfo};
use crate::error::Result;

// Periodogram enhancement functions
//
// Standard window characteristics (main-lobe width, highest side-lobe level,
// scalloping loss) are widely-tabulated constants; the values used below are
// from:
//   F. J. Harris, "On the Use of Windows for Harmonic Analysis with the
//   Discrete Fourier Transform," Proceedings of the IEEE, Vol. 66, No. 1,
//   January 1978, pp. 51-83 (Table I).
// Coherent gain, (equivalent) noise bandwidth, processing gain, and overlap
// correlation are instead computed directly from the actual window
// coefficients for the window length in use, rather than tabulated.

/// Reference window characteristics from Harris (1978), Table I:
/// `(main_lobe_width_bins, highest_side_lobe_db, scalloping_loss_db)`.
fn window_reference_characteristics(window_name: &str) -> (f64, f64, f64) {
    match window_name {
        "Rectangular" => (2.0, -13.3, 3.92),
        "Hamming" => (4.0, -42.7, 1.78),
        "Blackman" => (6.0, -58.1, 1.10),
        // `create_window` itself falls back to Hanning for any unrecognized
        // name, so mirror that here (also covers "Hanning" / "Hann").
        _ => (4.0, -31.5, 1.42),
    }
}

/// Calculate window analysis for enhanced periodogram.
///
/// Generates the actual window coefficients (via [`create_window`], at the
/// window length used elsewhere in this module by
/// [`calculate_enhanced_welch_periodogram`](super::frequency::calculate_enhanced_welch_periodogram))
/// and computes real,
/// length-specific coherent gain, (equivalent) noise bandwidth, processing
/// gain, and 50%-overlap correlation directly from them, combined with the
/// standard tabulated main-lobe-width/side-lobe/scalloping-loss constants for
/// the chosen window type (see module-level reference above).
#[allow(dead_code)]
pub fn calculate_window_analysis<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<WindowTypeInfo<F>>
where
    F: Float + FromPrimitive,
{
    let window_length = ((ts.len() as f64 * 0.25).round() as usize).max(2);
    let window: Vec<F> = create_window(&config.primary_window_type, window_length)?;
    let n_f = F::from(window_length).expect("Failed to convert to float");

    let sum_w = window.iter().fold(F::zero(), |acc, &w| acc + w);
    let sum_w2 = window.iter().fold(F::zero(), |acc, &w| acc + w * w);

    let coherent_gain = sum_w / n_f;

    // Normalized equivalent noise bandwidth, in FFT bins: N*sum(w^2)/(sum(w))^2.
    let noise_bandwidth = if sum_w > F::zero() {
        n_f * sum_w2 / (sum_w * sum_w)
    } else {
        F::one()
    };
    // Same quantity expressed as a fraction of the sampling rate, matching
    // this module's `freq = i / (len * 2)` normalized-frequency convention.
    let equivalent_noise_bandwidth = noise_bandwidth / n_f;

    // Coherent power gain relative to a rectangular window (<=1); the
    // reciprocal of the noise bandwidth.
    let processing_gain = if noise_bandwidth > F::zero() {
        F::one() / noise_bandwidth
    } else {
        F::zero()
    };

    // Normalized window autocorrelation at the 50% overlap shift used by
    // `calculate_enhanced_welch_periodogram`.
    let overlap_shift = (window_length as f64 * 0.5).round() as usize;
    let overlap_correlation = if sum_w2 > F::zero() && overlap_shift < window_length {
        let mut cross = F::zero();
        for i in 0..(window_length - overlap_shift) {
            cross = cross + window[i] * window[i + overlap_shift];
        }
        cross / sum_w2
    } else {
        F::zero()
    };

    let (main_lobe_bins, side_lobe_db, scalloping_db) =
        window_reference_characteristics(&config.primary_window_type);

    Ok(WindowTypeInfo {
        window_name: config.primary_window_type.clone(),
        main_lobe_width: F::from(main_lobe_bins).expect("Failed to convert to float"),
        side_lobe_level: F::from(side_lobe_db).expect("Failed to convert to float"),
        scalloping_loss: F::from(scalloping_db).expect("Failed to convert to float"),
        processing_gain,
        noise_bandwidth,
        coherent_gain,
        window_length,
        equivalent_noise_bandwidth,
        overlap_correlation,
    })
}

/// Calculate window effectiveness metrics.
///
/// Uses the window's processing gain (coherent power gain relative to a
/// rectangular window, i.e. the reciprocal of its equivalent noise
/// bandwidth) as a direct, standard measure of how effectively the window
/// preserves signal power while it suppresses spectral leakage: 1.0
/// (rectangular) is maximal, and tapered windows (Hann, Hamming, Blackman)
/// score progressively lower as they trade coherent gain for lower leakage.
#[allow(dead_code)]
pub fn calculate_window_effectiveness<F>(windowinfo: &WindowTypeInfo<F>) -> F
where
    F: Float + FromPrimitive,
{
    windowinfo.processing_gain
}

/// Calculate spectral leakage measures.
///
/// Converts the window's highest side-lobe level (dB, always <= 0) to a
/// linear amplitude ratio in `(0, 1]` via the standard dB-to-linear
/// conversion `10^(dB/20)`: larger values mean more energy leaks into
/// neighboring frequency bins.
#[allow(dead_code)]
pub fn calculate_spectral_leakage<F>(windowinfo: &WindowTypeInfo<F>) -> F
where
    F: Float + FromPrimitive,
{
    let ten = F::from(10.0).expect("Failed to convert constant to float");
    let twenty = F::from(20.0).expect("Failed to convert constant to float");
    ten.powf(windowinfo.side_lobe_level / twenty)
}

/// Calculate confidence intervals for periodogram ordinates.
///
/// Uses the classical chi-squared sampling theory for periodogram estimates
/// (see e.g. Percival & Walden, "Spectral Analysis for Physical
/// Applications," §6.9): `dof * I(f) / S(f) ~ chi2(dof)`, so a
/// `(1 - alpha)` confidence interval for the true spectrum `S(f)` given an
/// observed ordinate `I(f)` is
/// `[dof * I(f) / chi2_{dof}(1 - alpha/2), dof * I(f) / chi2_{dof}(alpha/2)]`
/// where `chi2_{dof}(q)` is the CDF-based (lower) quantile function. A raw
/// (unaveraged) periodogram has `dof = 2`; a `K`-segment Bartlett average
/// has `dof = 2K` (treating segments as independent, which is exact for
/// non-overlapping Bartlett segments).
#[allow(dead_code)]
pub fn calculate_periodogram_confidence_intervals<F>(
    periodogram: &[F],
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<(F, F)>>
where
    F: Float + FromPrimitive,
{
    if periodogram.is_empty() {
        return Ok(Vec::new());
    }

    let dof = if config.enable_bartlett_method {
        (2 * config.bartlett_num_segments).max(2)
    } else {
        2
    };

    let alpha = (1.0 - config.confidence_level).clamp(1e-6, 1.0 - 1e-6);
    // chi2_hi: small upper-tail probability -> large quantile -> denominator
    // for the (smaller) lower bound. chi2_lo: the complementary large
    // upper-tail probability -> small quantile -> denominator for the
    // (larger) upper bound.
    let chi2_hi = F::from(crate::causality::chi_squared_quantile(alpha / 2.0, dof))
        .expect("Failed to convert to float");
    let chi2_lo = F::from(crate::causality::chi_squared_quantile(
        1.0 - alpha / 2.0,
        dof,
    ))
    .expect("Failed to convert to float");
    let dof_f = F::from(dof).expect("Failed to convert to float");

    Ok(periodogram
        .iter()
        .map(|&value| {
            let lower = if chi2_hi > F::zero() {
                dof_f * value / chi2_hi
            } else {
                F::zero()
            };
            let upper = if chi2_lo > F::zero() {
                dof_f * value / chi2_lo
            } else {
                F::zero()
            };
            (lower, upper)
        })
        .collect())
}

/// Calculate peak significance for periodogram ordinates.
///
/// Tests each periodogram ordinate against a white-noise null hypothesis:
/// under H0, periodogram ordinates are (asymptotically) i.i.d.
/// exponentially distributed with mean equal to the average power, so
/// `P(I(f_k) > x) = exp(-x / mean(I))` exactly for the exponential
/// distribution. The returned value is `1 - p_value` for each bin (so
/// larger = more significant / less likely to be pure noise), which is a
/// simplified single-bin building block of the classical periodogram
/// peak-significance tests (e.g. Fisher's g-test, Priestley 1981 §6.1.6).
#[allow(dead_code)]
pub fn calculate_peak_significance<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    if periodogram.is_empty() {
        return Ok(Vec::new());
    }

    let n_f = F::from_usize(periodogram.len()).expect("Operation failed");
    let mean_power = periodogram.iter().fold(F::zero(), |acc, &x| acc + x) / n_f;

    if mean_power <= F::zero() {
        return Ok(vec![F::zero(); periodogram.len()]);
    }

    Ok(periodogram
        .iter()
        .map(|&value| F::one() - (-(value / mean_power)).exp())
        .collect())
}

/// Calculate bias-corrected periodogram.
///
/// `calculate_enhanced_welch_periodogram` applies a taper directly to each
/// segment without renormalizing, which multiplies the expected periodogram
/// value by the window's power gain `U = sum(w^2) / N` (Welch, 1967).
/// Rescaling by `1/U` restores an (asymptotically) unbiased estimate of the
/// true power spectral density.
#[allow(dead_code)]
pub fn calculate_bias_corrected_periodogram<F>(
    periodogram: &[F],
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    if periodogram.is_empty() {
        return Ok(Vec::new());
    }

    // The Welch periodogram in this module has `window_length / 2` bins.
    let window_length = (periodogram.len() * 2).max(2);
    let window: Vec<F> = create_window(&config.primary_window_type, window_length)?;
    let n_f = F::from(window_length).expect("Failed to convert to float");
    let sum_w2 = window.iter().fold(F::zero(), |acc, &w| acc + w * w);
    let power_gain = sum_w2 / n_f;

    if power_gain <= F::zero() {
        return Ok(periodogram.to_vec());
    }

    Ok(periodogram
        .iter()
        .map(|&value| value / power_gain)
        .collect())
}

/// Calculate variance-reduced periodogram.
///
/// Smooths in the log domain with a 3-bin moving average, then exponentiates
/// back. Periodogram ordinates are highly right-skewed (approximately
/// exponentially distributed, with coefficient of variation 1 regardless of
/// sample size); smoothing their logarithm is a standard variance-stabilizing
/// transform that reduces variance more evenly across large and small
/// ordinates than averaging in the raw (linear) domain, which is what
/// distinguishes this from [`calculate_smoothed_periodogram`].
#[allow(dead_code)]
pub fn calculate_variance_reduced_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let n = periodogram.len();
    if n < 3 {
        return Ok(periodogram.to_vec());
    }

    let epsilon = F::from(1e-300).expect("Failed to convert constant to float");
    let log_values: Vec<F> = periodogram.iter().map(|&v| v.max(epsilon).ln()).collect();

    let mut result = vec![F::zero(); n];
    for (i, slot) in result.iter_mut().enumerate() {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(n - 1);
        let count = F::from_usize(hi - lo + 1).expect("Operation failed");
        let sum = (lo..=hi).fold(F::zero(), |acc, j| acc + log_values[j]);
        *slot = (sum / count).exp();
    }

    Ok(result)
}

/// Calculate smoothed periodogram.
///
/// Applies a modified Daniell smoother: a 5-bin triangular-weighted local
/// average (weights `1, 2, 3, 2, 1`), a standard periodogram-smoothing
/// kernel that trades frequency resolution for reduced variance by
/// averaging in the linear (not log) domain -- see
/// [`calculate_variance_reduced_periodogram`] for the log-domain variant.
#[allow(dead_code)]
pub fn calculate_smoothed_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let n = periodogram.len();
    if n < 3 {
        return Ok(periodogram.to_vec());
    }

    const WEIGHTS: [f64; 5] = [1.0, 2.0, 3.0, 2.0, 1.0];
    let half = (WEIGHTS.len() / 2) as isize;

    let mut result = vec![F::zero(); n];
    for (i, slot) in result.iter_mut().enumerate() {
        let mut weighted_sum = F::zero();
        let mut weight_total = F::zero();
        for (k, &w) in WEIGHTS.iter().enumerate() {
            let j = i as isize + (k as isize - half);
            if j >= 0 && (j as usize) < n {
                let w_f = F::from(w).expect("Failed to convert constant to float");
                weighted_sum = weighted_sum + w_f * periodogram[j as usize];
                weight_total = weight_total + w_f;
            }
        }
        *slot = if weight_total > F::zero() {
            weighted_sum / weight_total
        } else {
            periodogram[i]
        };
    }

    Ok(result)
}

/// Calculate zero-padded periodogram.
///
/// Actually zero-pads the time series to `config.zero_padding_factor` times
/// its original length before computing the periodogram, which increases
/// the number of frequency-domain samples (genuine DFT interpolation) --
/// unlike a plain re-run of [`calculate_simple_periodogram`] on the
/// unmodified series, which would return the same resolution every time.
#[allow(dead_code)]
pub fn calculate_zero_padded_periodogram<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let padded_len = n * config.zero_padding_factor.max(1);
    let mut padded = Array1::<F>::zeros(padded_len);
    for (i, &value) in ts.iter().enumerate() {
        padded[i] = value;
    }

    calculate_simple_periodogram(&padded)
}

/// Calculate interpolated periodogram.
///
/// Fits a natural cubic spline through the periodogram ordinates (treated as
/// uniformly-spaced samples) and evaluates it at twice the original density,
/// inserting a genuine interpolated midpoint between every pair of original
/// bins rather than merely resampling the existing values.
#[allow(dead_code)]
pub fn calculate_interpolated_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let n = periodogram.len();
    if n < 3 {
        return Ok(periodogram.to_vec());
    }

    let second_derivatives = natural_cubic_spline_second_derivatives(periodogram);
    let half = F::from(0.5).expect("Failed to convert constant to float");

    let mut interpolated = Vec::with_capacity(2 * n - 1);
    for i in 0..(n - 1) {
        interpolated.push(periodogram[i]);
        interpolated.push(evaluate_natural_cubic_spline(
            periodogram,
            &second_derivatives,
            i,
            half,
        ));
    }
    interpolated.push(periodogram[n - 1]);

    Ok(interpolated)
}

/// Second derivatives (`M_i`) of the natural cubic spline through
/// unit-spaced points `y_0, ..., y_{n-1}`, solved via the standard
/// tridiagonal system (Thomas algorithm) for natural boundary conditions
/// `M_0 = M_{n-1} = 0`.
fn natural_cubic_spline_second_derivatives<F>(y: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive,
{
    let n = y.len();
    let mut m = vec![F::zero(); n];
    if n < 3 {
        return m;
    }

    let two = F::from(2.0).expect("Failed to convert constant to float");
    let four = F::from(4.0).expect("Failed to convert constant to float");
    let six = F::from(6.0).expect("Failed to convert constant to float");

    let mut c_prime = vec![F::zero(); n];
    let mut d_prime = vec![F::zero(); n];

    c_prime[1] = F::one() / four;
    d_prime[1] = (y[2] - two * y[1] + y[0]) * six / four;

    for i in 2..(n - 1) {
        let denom = four - c_prime[i - 1];
        c_prime[i] = F::one() / denom;
        let rhs = (y[i + 1] - two * y[i] + y[i - 1]) * six;
        d_prime[i] = (rhs - d_prime[i - 1]) / denom;
    }

    m[n - 2] = d_prime[n - 2];
    for i in (1..(n - 2)).rev() {
        m[i] = d_prime[i] - c_prime[i] * m[i + 1];
    }

    m
}

/// Evaluate the natural cubic spline segment between knots `i` and `i+1` at
/// fractional offset `t` in `[0, 1]`.
fn evaluate_natural_cubic_spline<F>(y: &[F], m: &[F], i: usize, t: F) -> F
where
    F: Float + FromPrimitive,
{
    let six = F::from(6.0).expect("Failed to convert constant to float");
    let one_minus_t = F::one() - t;

    let term_a = m[i] * one_minus_t * one_minus_t * one_minus_t / six;
    let term_b = m[i + 1] * t * t * t / six;
    let term_c = (y[i] - m[i] / six) * one_minus_t;
    let term_d = (y[i + 1] - m[i + 1] / six) * t;

    term_a + term_b + term_c + term_d
}

/// Count local maxima (strict, interior) in a spectrum.
fn count_local_maxima<F>(spectrum: &[F]) -> usize
where
    F: Float,
{
    if spectrum.len() < 3 {
        return 0;
    }
    (1..spectrum.len() - 1)
        .filter(|&i| spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1])
        .count()
}

/// Resolution-enhancement effectiveness shared by
/// [`calculate_zero_padding_effectiveness`] and
/// [`calculate_interpolation_effectiveness`]: the fraction of local maxima
/// (candidate spectral peaks) in `enhanced` that were not already resolvable
/// in `original`, clamped to `[0, 1]`. A resolution-enhancing operation
/// (zero-padding or interpolation) that reveals no additional peaks scores
/// 0; one that reveals many previously-unresolved peaks scores close to 1.
fn resolution_enhancement_effectiveness<F>(enhanced: &[F], original: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let enhanced_peaks = count_local_maxima(enhanced);
    let original_peaks = count_local_maxima(original);

    if enhanced_peaks == 0 {
        return F::zero();
    }

    let additional = enhanced_peaks.saturating_sub(original_peaks);
    let ratio = F::from_usize(additional).expect("Operation failed")
        / F::from_usize(enhanced_peaks).expect("Operation failed");
    ratio.min(F::one()).max(F::zero())
}

/// Calculate zero padding effectiveness: see
/// the private `resolution_enhancement_effectiveness` function.
#[allow(dead_code)]
pub fn calculate_zero_padding_effectiveness<F>(padded: &[F], original: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    resolution_enhancement_effectiveness(padded, original)
}

/// Calculate interpolation effectiveness: see
/// the private `resolution_enhancement_effectiveness` function.
#[allow(dead_code)]
pub fn calculate_interpolation_effectiveness<F>(interpolated: &[F], original: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    resolution_enhancement_effectiveness(interpolated, original)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_window(window: &str) -> EnhancedPeriodogramConfig {
        EnhancedPeriodogramConfig {
            primary_window_type: window.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_window_effectiveness_and_leakage_differ_by_window_type() {
        // Regression guard: the former stub returned 0.8/0.1 for every
        // window type regardless of input. Real values must (a) differ
        // across window types, and (b) match the known DSP ordering:
        // Rectangular has the highest processing gain (no tapering loss)
        // but also the worst (highest) leakage of any window.
        let ts = Array1::<f64>::zeros(100);

        let mut effectiveness = std::collections::HashMap::new();
        let mut leakage = std::collections::HashMap::new();
        for window in ["Rectangular", "Hamming", "Hanning", "Blackman"] {
            let config = config_with_window(window);
            let info = calculate_window_analysis(&ts, &config)
                .unwrap_or_else(|_| panic!("window analysis should succeed for {window}"));
            effectiveness.insert(window, calculate_window_effectiveness(&info));
            leakage.insert(window, calculate_spectral_leakage(&info));
        }

        // Not the old hardcoded constants.
        for window in ["Rectangular", "Hamming", "Hanning", "Blackman"] {
            assert!(
                (effectiveness[window] - 0.8).abs() > 1e-6,
                "{window}: effectiveness should not be the old hardcoded 0.8"
            );
            assert!(
                (leakage[window] - 0.1).abs() > 1e-6,
                "{window}: leakage should not be the old hardcoded 0.1"
            );
        }

        // Rectangular has perfect (1.0) processing gain; every tapered
        // window trades some of that away.
        assert!((effectiveness["Rectangular"] - 1.0).abs() < 1e-9);
        assert!(effectiveness["Rectangular"] > effectiveness["Hamming"]);
        assert!(effectiveness["Hamming"] > effectiveness["Hanning"]);
        assert!(effectiveness["Hanning"] > effectiveness["Blackman"]);

        // Rectangular leaks by far the most; Blackman the least.
        assert!(leakage["Rectangular"] > leakage["Hanning"]);
        assert!(leakage["Hanning"] > leakage["Hamming"]);
        assert!(leakage["Hamming"] > leakage["Blackman"]);
    }

    #[test]
    fn test_periodogram_confidence_intervals_bracket_the_estimate() {
        // Regression guard: the former stub always returned an empty Vec.
        let periodogram = vec![1.0, 4.0, 9.0, 2.0, 6.0, 3.0];
        let config = EnhancedPeriodogramConfig {
            confidence_level: 0.95,
            enable_bartlett_method: false,
            ..Default::default()
        };

        let intervals = calculate_periodogram_confidence_intervals(&periodogram, &config)
            .expect("confidence intervals should succeed");
        assert_eq!(intervals.len(), periodogram.len());

        for (i, &(lower, upper)) in intervals.iter().enumerate() {
            assert!(
                lower <= periodogram[i] && periodogram[i] <= upper,
                "bin {i}: point estimate {} should lie within [{lower}, {upper}]",
                periodogram[i]
            );
            assert!(lower >= 0.0, "bin {i}: lower bound should be non-negative");
        }

        // More degrees of freedom (simulating a Bartlett average over more
        // segments) must narrow the interval relative to the raw (dof=2)
        // case, since averaging genuinely reduces uncertainty.
        let averaged_config = EnhancedPeriodogramConfig {
            confidence_level: 0.95,
            enable_bartlett_method: true,
            bartlett_num_segments: 16,
            ..Default::default()
        };
        let averaged_intervals =
            calculate_periodogram_confidence_intervals(&periodogram, &averaged_config)
                .expect("confidence intervals should succeed");

        let raw_width: f64 = intervals.iter().map(|&(lo, hi)| hi - lo).sum();
        let averaged_width: f64 = averaged_intervals.iter().map(|&(lo, hi)| hi - lo).sum();
        assert!(
            averaged_width < raw_width,
            "averaging over more segments should narrow the confidence interval: \
             raw={raw_width}, averaged={averaged_width}"
        );
    }

    #[test]
    fn test_peak_significance_detects_injected_peak() {
        // Regression guard: the former stub always returned an empty Vec.
        // A single, large, isolated peak sitting on a low noise floor
        // should score far more significant than the noise-floor bins.
        let mut periodogram = vec![1.0_f64; 20];
        periodogram[10] = 200.0;
        let config = EnhancedPeriodogramConfig::default();

        let significance = calculate_peak_significance(&periodogram, &config)
            .expect("peak significance should succeed");
        assert_eq!(significance.len(), periodogram.len());

        assert!(
            significance[10] > 0.999,
            "the injected peak should be judged highly significant, got {}",
            significance[10]
        );
        assert!(
            significance[0] < significance[10],
            "a noise-floor bin should be far less significant than the peak"
        );
        for &s in &significance {
            assert!(
                (0.0..=1.0).contains(&s),
                "significance must lie in [0, 1], got {s}"
            );
        }
    }

    #[test]
    fn test_bias_corrected_periodogram_rescales_by_window_power_gain() {
        // Regression guard: the former stub returned the input unchanged.
        let periodogram = vec![2.0, 5.0, 3.0, 8.0, 1.0];
        let config = config_with_window("Hamming");

        let corrected = calculate_bias_corrected_periodogram(&periodogram, &config)
            .expect("bias correction should succeed");
        assert_eq!(corrected.len(), periodogram.len());
        assert_ne!(
            corrected, periodogram,
            "bias correction must actually transform the periodogram"
        );

        // Independently recompute the expected window power gain and check
        // the exact rescaling formula.
        let window_length = periodogram.len() * 2;
        let window: Vec<f64> = create_window("Hamming", window_length).expect("window");
        let sum_w2: f64 = window.iter().map(|w| w * w).sum();
        let power_gain = sum_w2 / window_length as f64;

        for (i, &value) in periodogram.iter().enumerate() {
            let expected = value / power_gain;
            assert!(
                (corrected[i] - expected).abs() < 1e-10,
                "bin {i}: expected {expected}, got {}",
                corrected[i]
            );
        }
    }

    #[test]
    fn test_variance_reduced_periodogram_reduces_log_domain_variance() {
        // Regression guard: the former stub returned the input unchanged.
        let periodogram = vec![1.0, 50.0, 2.0, 40.0, 3.0, 60.0, 1.5, 45.0, 2.5, 55.0];
        let config = EnhancedPeriodogramConfig::default();

        let reduced = calculate_variance_reduced_periodogram(&periodogram, &config)
            .expect("variance reduction should succeed");
        assert_eq!(reduced.len(), periodogram.len());
        assert_ne!(
            reduced, periodogram,
            "variance reduction must actually transform the periodogram"
        );

        let log_variance = |values: &[f64]| -> f64 {
            let logs: Vec<f64> = values.iter().map(|v| v.ln()).collect();
            let mean = logs.iter().sum::<f64>() / logs.len() as f64;
            logs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / logs.len() as f64
        };

        let original_variance = log_variance(&periodogram);
        let reduced_variance = log_variance(&reduced);
        assert!(
            reduced_variance < original_variance,
            "log-domain variance should decrease: original={original_variance}, reduced={reduced_variance}"
        );
    }

    #[test]
    fn test_smoothed_periodogram_matches_hand_computed_weights() {
        // Regression guard: the former stub returned the input unchanged.
        let periodogram = vec![1.0, 2.0, 10.0, 3.0, 1.0, 2.0, 8.0];
        let config = EnhancedPeriodogramConfig::default();

        let smoothed = calculate_smoothed_periodogram(&periodogram, &config)
            .expect("smoothing should succeed");
        assert_eq!(smoothed.len(), periodogram.len());
        assert_ne!(
            smoothed, periodogram,
            "smoothing must actually transform the periodogram"
        );

        // Interior point (index 2) has all 5 taps [1,2,10,3,1] with weights
        // [1,2,3,2,1] (sum of weights = 9).
        let expected_interior = (1.0 * 1.0 + 2.0 * 2.0 + 10.0 * 3.0 + 3.0 * 2.0 + 1.0 * 1.0) / 9.0;
        assert!(
            (smoothed[2] - expected_interior).abs() < 1e-10,
            "expected {expected_interior}, got {}",
            smoothed[2]
        );

        // Left edge (index 0) only has taps for offsets 0, +1, +2 (weights 3,2,1).
        let expected_edge = (1.0 * 3.0 + 2.0 * 2.0 + 10.0 * 1.0) / 6.0;
        assert!(
            (smoothed[0] - expected_edge).abs() < 1e-10,
            "expected {expected_edge}, got {}",
            smoothed[0]
        );
    }

    #[test]
    fn test_zero_padded_periodogram_increases_resolution() {
        // Regression guard: the former stub ignored the zero-padding
        // config entirely and returned the unpadded periodogram.
        let n = 16;
        let ts = Array1::from_shape_fn(n, |i| (i as f64 * 0.3).sin());

        let unpadded = calculate_simple_periodogram(&ts).expect("periodogram should succeed");

        let config = EnhancedPeriodogramConfig {
            zero_padding_factor: 4,
            ..Default::default()
        };
        let padded =
            calculate_zero_padded_periodogram(&ts, &config).expect("zero padding should succeed");

        assert_eq!(padded.len(), (n * 4) / 2);
        assert!(
            padded.len() > unpadded.len(),
            "zero padding should genuinely increase spectral resolution: {} vs {}",
            padded.len(),
            unpadded.len()
        );
    }

    #[test]
    fn test_interpolated_periodogram_doubles_resolution_and_preserves_knots() {
        // Regression guard: the former stub returned the input unchanged
        // (so this would fail both the length check and the "not a naive
        // average" check below).
        let periodogram = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.5, 7.0];
        let config = EnhancedPeriodogramConfig::default();

        let interpolated = calculate_interpolated_periodogram(&periodogram, &config)
            .expect("interpolation should succeed");
        assert_eq!(interpolated.len(), 2 * periodogram.len() - 1);

        // Original ordinates must reappear exactly at even indices.
        for (i, &value) in periodogram.iter().enumerate() {
            assert!((interpolated[2 * i] - value).abs() < 1e-10);
        }

        // Cross-check the interpolated midpoints against an independent
        // reference (scipy.interpolate.CubicSpline(..., bc_type="natural")
        // evaluated at each half-integer point for this exact input).
        let expected_midpoints = [
            2.447_655, 2.282_034, 3.549_210, 4.521_127, 5.241_283, 4.201_241, 3.766_253,
        ];
        for (i, &expected) in expected_midpoints.iter().enumerate() {
            let got = interpolated[2 * i + 1];
            assert!(
                (got - expected).abs() < 1e-5,
                "segment {i}: expected {expected}, got {got}"
            );
        }

        // At least one midpoint must differ meaningfully from the naive
        // linear-average midpoint, proving this is genuine cubic-spline
        // interpolation rather than simple linear interpolation.
        let linear_midpoint_0 = (periodogram[0] + periodogram[1]) / 2.0;
        assert!(
            (interpolated[1] - linear_midpoint_0).abs() > 0.05,
            "cubic-spline midpoint should differ from the naive linear average"
        );
    }

    #[test]
    fn test_resolution_enhancement_effectiveness_reflects_new_peaks() {
        // Regression guard: the former stubs returned hardcoded 0.9/0.85
        // regardless of input.
        // No new peaks: a strictly monotonic sequence stays monotonic
        // (peak-free) after "enhancement".
        let monotonic_original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let monotonic_enhanced = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let zero_effectiveness =
            calculate_zero_padding_effectiveness(&monotonic_enhanced, &monotonic_original);
        assert_eq!(zero_effectiveness, 0.0);

        // New peaks: two closely-spaced peaks that overlap into a single
        // local maximum in the coarse original, but resolve into two
        // separate local maxima once "enhanced".
        let coarse_original = vec![1.0, 5.0, 1.0];
        let fine_enhanced = vec![1.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 1.0];
        let positive_effectiveness =
            calculate_interpolation_effectiveness(&fine_enhanced, &coarse_original);
        assert!(
            positive_effectiveness > 0.0,
            "revealing new peaks should score above zero, got {positive_effectiveness}"
        );

        // Not the old hardcoded constants for either scenario.
        assert!((zero_effectiveness - 0.9).abs() > 1e-6);
        assert!((positive_effectiveness - 0.85).abs() > 1e-6);
    }
}
