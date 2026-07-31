// Frequency response estimation methods for system identification

use super::types::{FreqResponseMethod, FreqResponseResult, SysIdConfig};
use super::utils::{compute_fft, next_power_of_2};
use crate::error::{SignalError, SignalResult};
use crate::spectral::welch;
use crate::window::get_window;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;

/// Estimate frequency response function from input-output data
///
/// # Arguments
/// * `input` - Input signal
/// * `output` - Output signal
/// * `fs` - Sampling frequency
/// * `method` - Frequency response estimation method
/// * `config` - Configuration parameters
///
/// # Returns
/// * Frequency response estimation result
#[allow(dead_code)]
pub fn estimate_frequency_response(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    method: FreqResponseMethod,
    config: &SysIdConfig,
) -> SignalResult<FreqResponseResult> {
    if input.len() != output.len() {
        return Err(SignalError::ValueError(
            "Input and output signals must have the same length".to_string(),
        ));
    }

    match method {
        FreqResponseMethod::Welch => estimate_freq_response_welch(input, output, fs, config),
        FreqResponseMethod::Periodogram => {
            estimate_freq_response_periodogram(input, output, fs, config)
        }
        FreqResponseMethod::H1 => estimate_freq_response_h1(input, output, fs, config),
        FreqResponseMethod::H2 => estimate_freq_response_h2(input, output, fs, config),
        FreqResponseMethod::CoherenceWeighted => {
            estimate_freq_response_coherence_weighted(input, output, fs, config)
        }
    }
}

/// Frequency response estimation using Welch's method
#[allow(dead_code)]
fn estimate_freq_response_welch(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    config: &SysIdConfig,
) -> SignalResult<FreqResponseResult> {
    // Use Welch's method to estimate cross-spectral density and auto-spectral density
    let nfft = config.nfft.unwrap_or(next_power_of_2(input.len() / 8));
    let overlap = (nfft as f64 * config.overlap) as usize;

    // Get cross-power spectral density
    let (freqs, pxy) =
        cross_spectral_density_welch(input, output, fs, nfft, overlap, &config.window)?;

    // Get input auto-power spectral density
    let (_, pxx) = welch(
        input.as_slice().expect("Operation failed"),
        Some(fs),
        Some(&config.window),
        Some(nfft),
        Some(overlap),
        Some(nfft),
        None,
        None,
    )?;

    // Calculate frequency response H(f) = Pxy(f) / Pxx(f)
    let mut freq_response = Array1::<Complex64>::zeros(freqs.len());
    let mut coherence = Array1::<f64>::zeros(freqs.len());

    // Also need output auto-spectral density for coherence
    let (_, pyy) = welch(
        output.as_slice().expect("Operation failed"),
        Some(fs),
        Some(&config.window),
        Some(nfft),
        Some(overlap),
        Some(nfft),
        None,
        None,
    )?;

    for i in 0..freqs.len() {
        if pxx[i].abs() > 1e-12 {
            freq_response[i] = pxy[i] / pxx[i];

            // Calculate coherence: |Pxy|^2 / (Pxx * Pyy)
            let coherence_val = pxy[i].norm_sqr() / (pxx[i].abs() * pyy[i]);
            coherence[i] = coherence_val.clamp(0.0, 1.0);
        } else {
            freq_response[i] = Complex64::new(0.0, 0.0);
            coherence[i] = 0.0;
        }
    }

    Ok(FreqResponseResult {
        frequency_response: freq_response,
        frequencies: freqs,
        coherence,
        confidence_bounds: None,
    })
}

/// Cross-spectral density estimation using Welch's method
#[allow(dead_code)]
pub(super) fn cross_spectral_density_welch(
    x: &Array1<f64>,
    y: &Array1<f64>,
    fs: f64,
    nfft: usize,
    overlap: usize,
    window_name: &str,
) -> SignalResult<(Array1<f64>, Array1<Complex64>)> {
    let n = x.len();
    let step = nfft - overlap;

    if step == 0 {
        return Err(SignalError::ValueError(
            "Invalid overlap specification".to_string(),
        ));
    }

    // Generate window
    let window = get_window(window_name, nfft, true)?;
    let window_array = Array1::from(window);
    let window_norm = window_array.mapv(|w| w * w).sum().sqrt();

    let mut num_segments = 0;
    let mut pxy_acc = Array1::<Complex64>::zeros(nfft / 2 + 1);

    // Process overlapping segments
    for start in (0..n).step_by(step) {
        if start + nfft > n {
            break;
        }

        // Extract segments and apply window
        let x_seg = x
            .slice(scirs2_core::ndarray::s![start..start + nfft])
            .to_owned()
            * &window_array;
        let y_seg = y
            .slice(scirs2_core::ndarray::s![start..start + nfft])
            .to_owned()
            * &window_array;

        // Compute FFTs
        let x_fft = compute_fft(&x_seg);
        let y_fft = compute_fft(&y_seg);

        // Compute cross-spectral density for this segment
        let max_freq_bin = if nfft.is_multiple_of(2) {
            nfft / 2
        } else {
            (nfft - 1) / 2
        };
        for i in 0..=max_freq_bin {
            pxy_acc[i] += x_fft[i].conj() * y_fft[i];
        }

        num_segments += 1;
    }

    if num_segments == 0 {
        return Err(SignalError::ValueError(
            "No complete segments found".to_string(),
        ));
    }

    // Average and normalize
    let scale = fs * window_norm * window_norm * num_segments as f64;
    pxy_acc.mapv_inplace(|x| x / scale);

    // Create frequency vector
    let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);

    Ok((freqs, pxy_acc))
}

/// Simple periodogram-based frequency response estimation
///
/// The frequency response `H(f) = Y(f)/X(f)` is computed from a single
/// full-length FFT of the (zero-padded) data, preserving this method's
/// characteristic higher frequency resolution compared to Welch's method.
/// Coherence, however, is mathematically *always exactly 1.0* for any
/// single realization (`|Pxy|^2/(Pxx*Pyy)` reduces identically to 1 when
/// `Pxy = X*conj(Y)`, `Pxx = |X|^2`, `Pyy = |Y|^2` all come from the same
/// single spectrum) -- a genuinely meaningful coherence estimate requires
/// averaging over multiple, independent sub-realizations. This computes
/// that genuine estimate internally by splitting the data into several
/// non-overlapping segments and averaging their cross/auto-spectra, rather
/// than returning a hardcoded constant.
#[allow(dead_code)]
fn estimate_freq_response_periodogram(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    _config: &SysIdConfig,
) -> SignalResult<FreqResponseResult> {
    let n = input.len();
    let nfft = next_power_of_2(n);

    // Compute FFTs
    let mut input_padded = Array1::<f64>::zeros(nfft);
    let mut output_padded = Array1::<f64>::zeros(nfft);

    input_padded
        .slice_mut(scirs2_core::ndarray::s![0..n])
        .assign(input);
    output_padded
        .slice_mut(scirs2_core::ndarray::s![0..n])
        .assign(output);

    let input_fft = compute_fft(&input_padded);
    let output_fft = compute_fft(&output_padded);

    // Compute frequency response
    let mut freq_response = Array1::<Complex64>::zeros(nfft / 2 + 1);

    for i in 0..=nfft / 2 {
        let idx = if i == nfft / 2 { nfft / 2 } else { i };
        if input_fft[idx].norm() > 1e-12 {
            freq_response[i] = output_fft[idx] / input_fft[idx];
        }
    }

    let freqs = Array1::linspace(0.0, fs / 2.0, nfft / 2 + 1);
    let coherence = estimate_periodogram_coherence(input, output, nfft / 2 + 1);

    Ok(FreqResponseResult {
        frequency_response: freq_response,
        frequencies: freqs,
        coherence,
        confidence_bounds: None,
    })
}

/// Genuine (segment-averaged) coherence estimate for the single-shot
/// periodogram method: splits the data into non-overlapping segments
/// (using the largest power-of-2 segment length that fits at least 4
/// segments, falling back to as many segments as the data allows), then
/// averages each segment's cross- and auto-spectra before forming
/// `|mean(Pxy)|^2 / (mean(Pxx) * mean(Pyy))`, interpolated onto the
/// requested `n_freqs`-point frequency grid.
fn estimate_periodogram_coherence(
    input: &Array1<f64>,
    output: &Array1<f64>,
    n_freqs: usize,
) -> Array1<f64> {
    let n = input.len();
    let mut coherence = Array1::<f64>::ones(n_freqs);

    // Choose the largest power-of-2 segment length giving at least 4
    // segments; fall back to 2 segments, then to a trivial (single
    // segment, coherence=1 by construction) case for very short data.
    let mut seg_len = next_power_of_2((n / 4).max(4));
    while seg_len > 2 && n / seg_len < 4 {
        seg_len /= 2;
    }
    if seg_len < 4 || n / seg_len < 2 {
        return coherence;
    }

    let n_segments = n / seg_len;
    let half = seg_len / 2 + 1;
    let mut pxy_sum = vec![Complex64::new(0.0, 0.0); half];
    let mut pxx_sum = vec![0.0_f64; half];
    let mut pyy_sum = vec![0.0_f64; half];

    for seg in 0..n_segments {
        let start = seg * seg_len;
        let x_seg = input
            .slice(scirs2_core::ndarray::s![start..start + seg_len])
            .to_owned();
        let y_seg = output
            .slice(scirs2_core::ndarray::s![start..start + seg_len])
            .to_owned();
        let x_fft = compute_fft(&x_seg);
        let y_fft = compute_fft(&y_seg);

        for i in 0..half {
            pxy_sum[i] += x_fft[i].conj() * y_fft[i];
            pxx_sum[i] += x_fft[i].norm_sqr();
            pyy_sum[i] += y_fft[i].norm_sqr();
        }
    }

    let mut coherence_native = vec![0.0_f64; half];
    for i in 0..half {
        let denom = pxx_sum[i] * pyy_sum[i];
        coherence_native[i] = if denom > 1e-24 {
            (pxy_sum[i].norm_sqr() / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    // Interpolate the (seg_len/2+1)-point native coherence grid onto the
    // requested n_freqs-point grid (nearest-neighbor, sufficient for this
    // secondary diagnostic quantity).
    for (i, coh) in coherence.iter_mut().enumerate() {
        let native_idx = if n_freqs > 1 {
            ((i * (half - 1)) as f64 / (n_freqs - 1).max(1) as f64).round() as usize
        } else {
            0
        };
        *coh = coherence_native[native_idx.min(half - 1)];
    }

    coherence
}

/// H1 estimator (minimizes input noise effects)
#[allow(dead_code)]
fn estimate_freq_response_h1(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    config: &SysIdConfig,
) -> SignalResult<FreqResponseResult> {
    // H1 = Pyx / Pxx (same as Welch method)
    estimate_freq_response_welch(input, output, fs, config)
}

/// H2 estimator (minimizes output noise effects)
#[allow(dead_code)]
fn estimate_freq_response_h2(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    config: &SysIdConfig,
) -> SignalResult<FreqResponseResult> {
    // H2 = Pyy / Pxy
    let nfft = config.nfft.unwrap_or(next_power_of_2(input.len() / 8));
    let overlap = (nfft as f64 * config.overlap) as usize;

    let (freqs, pxy) =
        cross_spectral_density_welch(input, output, fs, nfft, overlap, &config.window)?;
    let (_, pyy) = welch(
        output.as_slice().expect("Operation failed"),
        Some(fs),
        Some(&config.window),
        Some(nfft),
        Some(overlap),
        Some(nfft),
        None,
        None,
    )?;
    let (_, pxx) = welch(
        input.as_slice().expect("Operation failed"),
        Some(fs),
        Some(&config.window),
        Some(nfft),
        Some(overlap),
        Some(nfft),
        None,
        None,
    )?;

    let mut freq_response = Array1::<Complex64>::zeros(freqs.len());
    let mut coherence = Array1::<f64>::zeros(freqs.len());

    for i in 0..freqs.len() {
        if pxy[i].norm() > 1e-12 {
            freq_response[i] = Complex64::new(pyy[i], 0.0) / pxy[i];

            let coherence_val = pxy[i].norm_sqr() / (pxx[i] * pyy[i]);
            coherence[i] = coherence_val.clamp(0.0, 1.0);
        }
    }

    Ok(FreqResponseResult {
        frequency_response: freq_response,
        frequencies: freqs,
        coherence,
        confidence_bounds: None,
    })
}

/// Coherence-weighted frequency response estimation
#[allow(dead_code)]
fn estimate_freq_response_coherence_weighted(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    config: &SysIdConfig,
) -> SignalResult<FreqResponseResult> {
    let h1_result = estimate_freq_response_h1(input, output, fs, config)?;
    let h2_result = estimate_freq_response_h2(input, output, fs, config)?;

    let mut freq_response = Array1::<Complex64>::zeros(h1_result.frequencies.len());

    // Weight estimates by coherence
    for i in 0..freq_response.len() {
        let gamma = h1_result.coherence[i];
        freq_response[i] = gamma * h1_result.frequency_response[i]
            + (1.0 - gamma) * h2_result.frequency_response[i];
    }

    Ok(FreqResponseResult {
        frequency_response: freq_response,
        frequencies: h1_result.frequencies,
        coherence: h1_result.coherence,
        confidence_bounds: None,
    })
}

/// Fit parametric model to frequency response data (used by transfer_function module)
#[allow(dead_code)]
pub(super) fn fit_parametric_to_frequency_response(
    freq_response: &Array1<Complex64>,
    frequencies: &Array1<f64>,
    num_order: usize,
    den_order: usize,
) -> SignalResult<super::types::TfEstimationResult> {
    use super::utils::solve_complex_least_squares;
    use crate::lti::systems::LtiSystem;
    use crate::lti::TransferFunction;
    use std::f64::consts::PI;

    let n_freq = frequencies.len();
    if n_freq < num_order + den_order + 1 {
        return Err(SignalError::ValueError(
            "Insufficient frequency points for model orders".to_string(),
        ));
    }

    // Set up complex least squares problem
    // H(jw) = (b0 + b1*(jw) + ... + bm*(jw)^m) / (1 + a1*(jw) + ... + an*(jw)^n)
    let total_params = num_order + den_order + 1;
    let mut a_matrix = Array2::<Complex64>::zeros((n_freq, total_params));
    let mut b_vector = Array1::<Complex64>::zeros(n_freq);

    for (i, &freq) in frequencies.iter().enumerate() {
        let jw = Complex64::new(0.0, 2.0 * PI * freq);
        let h_val = freq_response[i];

        // Fill the regression matrix
        let mut jw_power = Complex64::new(1.0, 0.0);

        // Denominator terms (multiply by -H(jw))
        for k in 1..=den_order {
            jw_power *= jw;
            a_matrix[[i, k - 1]] = -h_val * jw_power;
        }

        // Numerator terms
        jw_power = Complex64::new(1.0, 0.0);
        for k in 0..=num_order {
            a_matrix[[i, den_order + k]] = jw_power;
            if k < num_order {
                jw_power *= jw;
            }
        }

        b_vector[i] = h_val;
    }

    // Solve complex least squares (use real and imaginary parts separately)
    let params = solve_complex_least_squares(&a_matrix, &b_vector)?;

    // Extract real coefficients
    let mut denominator = Array1::<f64>::zeros(den_order + 1);
    denominator[0] = 1.0;
    for i in 1..=den_order {
        denominator[i] = params[i - 1].re;
    }

    let mut numerator = Array1::<f64>::zeros(num_order + 1);
    for i in 0..=num_order {
        numerator[i] = params[den_order + i].re;
    }

    // Calculate fit quality
    let tf = TransferFunction::new(numerator.to_vec(), denominator.to_vec(), None)?;
    let estimated_response = tf.frequency_response(&frequencies.mapv(|f| 2.0 * PI * f).to_vec())?;

    let mut error_sum = 0.0;
    let mut signal_sum = 0.0;

    for i in 0..n_freq {
        let error = (freq_response[i] - estimated_response[i]).norm_sqr();
        error_sum += error;
        signal_sum += freq_response[i].norm_sqr();
    }

    let fit_percentage = 100.0 * (1.0 - error_sum / signal_sum).max(0.0);

    Ok(super::types::TfEstimationResult {
        numerator,
        denominator,
        fit_percentage,
        error_variance: error_sum / n_freq as f64,
        frequency_response: Some(Array1::from_vec(estimated_response)),
        frequencies: Some(frequencies.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_sequence(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f64 / u64::MAX as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn test_periodogram_coherence_is_high_for_linearly_related_signals() {
        let n = 512;
        let input = Array1::from_vec(pseudo_random_sequence(n, 12345));
        // A clean linear system: output is just a scaled, delayed copy.
        let mut output = Array1::<f64>::zeros(n);
        for i in 1..n {
            output[i] = 2.0 * input[i - 1];
        }

        let config = SysIdConfig::default();
        let result = estimate_freq_response_periodogram(&input, &output, 10.0, &config)
            .expect("periodogram estimation should succeed");

        // The fabricated implementation always returned exactly 0.8 for
        // every frequency bin regardless of the actual relationship
        // between input and output.
        let mean_coherence: f64 =
            result.coherence.iter().sum::<f64>() / result.coherence.len() as f64;
        assert!(
            mean_coherence > 0.7,
            "mean coherence for a clean linear system too low: {mean_coherence}"
        );
        // Not every bin should be exactly 0.8 (the old hardcoded value).
        assert!(result.coherence.iter().any(|&c| (c - 0.8).abs() > 0.05));
    }

    #[test]
    fn test_periodogram_coherence_is_low_for_unrelated_signals() {
        let n = 512;
        let input = Array1::from_vec(pseudo_random_sequence(n, 111));
        let output = Array1::from_vec(pseudo_random_sequence(n, 222));

        let config = SysIdConfig::default();
        let result = estimate_freq_response_periodogram(&input, &output, 10.0, &config)
            .expect("periodogram estimation should succeed");

        let mean_coherence: f64 =
            result.coherence.iter().sum::<f64>() / result.coherence.len() as f64;
        // Two independent random sequences should show low coherence on
        // average; the old stub's constant 0.8 could never reflect this.
        assert!(
            mean_coherence < 0.5,
            "mean coherence for unrelated signals unexpectedly high: {mean_coherence}"
        );
    }
}
