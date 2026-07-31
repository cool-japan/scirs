//! Signal Resampling
//!
//! This module provides comprehensive sample rate conversion capabilities:
//!
//! - **Polyphase rational resampling**: Efficient P/Q rational rate conversion
//! - **Anti-aliasing filter design**: Automatic low-pass filter for resampling
//! - **Sample rate conversion**: High-level API with configurable quality
//! - **Fractional delay filters**: Lagrange and windowed-sinc interpolation
//! - **Arbitrary ratio resampling**: Continuous-time interpolation for any ratio
//! - **Perfect-reconstruction filter banks & arbitrary-ratio multirate
//!   conversion** (see [`multirate`]): polyphase PR filter banks and a
//!   combined upsample/filter/downsample rate converter.

pub mod multirate;
pub use multirate::{
    FilterBankProperties, MultirateConverter, PerfectReconstructionConfig,
    PerfectReconstructionFilterBank, PrFilterDesign,
};

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quality preset for sample rate conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResamplingQuality {
    /// Fast, lower quality (shorter filter, linear interpolation fallback).
    Draft,
    /// Standard quality (windowed-sinc with moderate length).
    Standard,
    /// High quality (longer windowed-sinc filter).
    High,
    /// Maximum quality (very long filter, minimal aliasing).
    Audiophile,
}

/// Configuration for the resampler.
#[derive(Debug, Clone)]
pub struct ResamplingConfig {
    /// Quality preset.
    pub quality: ResamplingQuality,
    /// Filter half-length (number of zero-crossings on each side).
    /// If `None`, determined by quality preset.
    pub filter_half_length: Option<usize>,
    /// Anti-aliasing filter cutoff as fraction of Nyquist (0..1).
    /// Default is 0.9.
    pub cutoff: f64,
    /// Transition bandwidth as fraction of Nyquist (0..1).
    /// Only used for FIR design. Default is 0.1.
    pub transition_bw: f64,
    /// Window type for the anti-aliasing filter.
    pub window: WindowType,
}

impl Default for ResamplingConfig {
    fn default() -> Self {
        Self {
            quality: ResamplingQuality::Standard,
            cutoff: 0.9,
            transition_bw: 0.1,
            filter_half_length: None,
            window: WindowType::Kaiser { beta: 6.0 },
        }
    }
}

/// Window type for filter design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Kaiser window with parameter beta.
    Kaiser { beta: f64 },
}

// ---------------------------------------------------------------------------
// Anti-aliasing filter design
// ---------------------------------------------------------------------------

/// Design an anti-aliasing low-pass FIR filter.
///
/// # Arguments
/// * `length` - Filter length (must be odd for Type I FIR)
/// * `cutoff` - Normalised cutoff frequency (0..1, fraction of Nyquist)
/// * `window` - Window type
///
/// # Returns
/// Filter coefficients.
pub fn design_anti_alias_filter(
    length: usize,
    cutoff: f64,
    window: WindowType,
) -> SignalResult<Vec<f64>> {
    if length == 0 {
        return Err(SignalError::InvalidArgument(
            "Filter length must be > 0".into(),
        ));
    }
    if cutoff <= 0.0 || cutoff > 1.0 {
        return Err(SignalError::InvalidArgument(
            "Cutoff must be in (0, 1]".into(),
        ));
    }

    let len = if length % 2 == 0 { length + 1 } else { length };
    let half = len / 2;
    let fc = cutoff * 0.5; // normalised to sample rate

    // Ideal sinc filter
    let mut h = vec![0.0; len];
    for i in 0..len {
        let n = i as f64 - half as f64;
        if n.abs() < f64::EPSILON {
            h[i] = 2.0 * fc;
        } else {
            h[i] = (2.0 * PI * fc * n).sin() / (PI * n);
        }
    }

    // Apply window
    let w = compute_window(len, window);
    for (hi, &wi) in h.iter_mut().zip(w.iter()) {
        *hi *= wi;
    }

    // Normalise for unity gain at DC
    let sum: f64 = h.iter().sum();
    if sum.abs() > f64::EPSILON {
        for hi in &mut h {
            *hi /= sum;
        }
    }

    Ok(h)
}

fn compute_window(len: usize, window: WindowType) -> Vec<f64> {
    let n_minus_1 = (len.max(1) - 1).max(1) as f64;
    match window {
        WindowType::Rectangular => vec![1.0; len],
        WindowType::Hann => (0..len)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n_minus_1).cos()))
            .collect(),
        WindowType::Hamming => (0..len)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / n_minus_1).cos())
            .collect(),
        WindowType::Blackman => (0..len)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / n_minus_1;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
        WindowType::Kaiser { beta } => (0..len)
            .map(|i| {
                let alpha = n_minus_1 / 2.0;
                let arg = 1.0 - ((i as f64 - alpha) / alpha).powi(2);
                let arg = arg.max(0.0);
                bessel_i0(beta * arg.sqrt()) / bessel_i0(beta)
            })
            .collect(),
    }
}

/// Approximation of the modified Bessel function of the first kind, order 0.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half_sq = (x / 2.0) * (x / 2.0);
    for k in 1..50 {
        term *= x_half_sq / (k as f64 * k as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Polyphase Rational Resampling
// ---------------------------------------------------------------------------

/// Resample a signal by a rational factor P/Q.
///
/// This implements the efficient polyphase decomposition:
/// 1. Upsample by factor P (insert zeros)
/// 2. Filter with anti-aliasing FIR
/// 3. Downsample by factor Q (decimate)
///
/// The polyphase structure avoids computing samples that will be discarded.
///
/// # Arguments
/// * `signal` - Input signal
/// * `up` - Upsampling factor (P)
/// * `down` - Downsampling factor (Q)
/// * `config` - Resampling configuration
///
/// # Returns
/// Resampled signal.
pub fn resample_poly(
    signal: &[f64],
    up: usize,
    down: usize,
    config: Option<&ResamplingConfig>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidArgument(
            "up and down must be > 0".into(),
        ));
    }

    // Simplify the ratio
    let g = gcd(up, down);
    let p = up / g;
    let q = down / g;

    if p == 1 && q == 1 {
        return Ok(signal.to_vec());
    }

    let default_config = ResamplingConfig::default();
    let cfg = config.unwrap_or(&default_config);

    // Determine filter length
    let filter_half = cfg.filter_half_length.unwrap_or_else(|| match cfg.quality {
        ResamplingQuality::Draft => 4,
        ResamplingQuality::Standard => 10,
        ResamplingQuality::High => 20,
        ResamplingQuality::Audiophile => 40,
    });

    // Design anti-aliasing filter at the lower of the two rates
    let cutoff = cfg.cutoff / p.max(q) as f64;
    let cutoff = cutoff.min(1.0).max(0.01);
    let filter_len = 2 * filter_half * p.max(1) + 1;
    let filter = design_anti_alias_filter(filter_len, cutoff, cfg.window)?;

    // Scale filter for upsampling gain
    let filter_scaled: Vec<f64> = filter.iter().map(|&h| h * p as f64).collect();

    // Output length
    let n_in = signal.len();
    let n_out = (n_in as u64 * p as u64 + q as u64 - 1) / q as u64;
    let n_out = n_out as usize;

    let half_len = filter_scaled.len() / 2;

    let mut output = Vec::with_capacity(n_out);

    for out_idx in 0..n_out {
        // Which sample of the upsampled (by P) signal this corresponds to
        let up_idx = out_idx as u64 * q as u64;
        let phase = (up_idx % p as u64) as usize;
        let in_base = (up_idx / p as u64) as i64;

        let mut acc = 0.0;
        // Polyphase filter: only compute at positions where the upsampled signal
        // is non-zero (i.e. every P-th sample)
        let mut k = phase;
        while k < filter_scaled.len() {
            let in_idx = in_base - (half_len as i64 - k as i64) / p as i64;
            let filter_idx = k;
            // Check if this filter tap aligns with a non-zero input
            let rem = (half_len as i64 - k as i64).rem_euclid(p as i64) as usize;
            if rem == 0 {
                if in_idx >= 0 && (in_idx as usize) < n_in {
                    acc += signal[in_idx as usize] * filter_scaled[filter_idx];
                }
            }
            k += 1;
        }

        // Alternative: direct polyphase computation
        // For each output sample, iterate over the polyphase branch
        if acc == 0.0 {
            acc = polyphase_sample(signal, &filter_scaled, p, out_idx, q);
        }

        output.push(acc);
    }

    Ok(output)
}

/// Direct polyphase computation for one output sample.
fn polyphase_sample(signal: &[f64], filter: &[f64], up: usize, out_idx: usize, down: usize) -> f64 {
    let filt_len = filter.len();
    let half = filt_len / 2;
    let n_in = signal.len();

    // The virtual upsampled index
    let virtual_idx = out_idx as i64 * down as i64;

    let mut acc = 0.0;
    // For each filter coefficient
    for k in 0..filt_len {
        let virtual_in = virtual_idx - (k as i64 - half as i64);
        // Only non-zero if virtual_in is a multiple of up
        if virtual_in >= 0 {
            let vi = virtual_in as usize;
            if vi % up == 0 {
                let real_in = vi / up;
                if real_in < n_in {
                    acc += signal[real_in] * filter[k];
                }
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// High-level sample rate conversion
// ---------------------------------------------------------------------------

/// Convert sample rate from `fs_in` to `fs_out`.
///
/// Automatically determines the rational P/Q ratio and applies polyphase
/// resampling.
///
/// # Arguments
/// * `signal` - Input signal
/// * `fs_in` - Input sampling frequency
/// * `fs_out` - Output sampling frequency
/// * `config` - Optional configuration
pub fn resample(
    signal: &[f64],
    fs_in: f64,
    fs_out: f64,
    config: Option<&ResamplingConfig>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if fs_in <= 0.0 || fs_out <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequencies must be positive".into(),
        ));
    }

    if (fs_in - fs_out).abs() < 1e-6 {
        return Ok(signal.to_vec());
    }

    // Find rational approximation P/Q ~ fs_out/fs_in
    let (p, q) = rational_approximation(fs_out / fs_in, 1000);

    resample_poly(signal, p, q, config)
}

/// Resample to a specific number of output samples.
///
/// Uses arbitrary-ratio resampling via sinc interpolation.
pub fn resample_to_length(
    signal: &[f64],
    output_length: usize,
    config: Option<&ResamplingConfig>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if output_length == 0 {
        return Err(SignalError::InvalidArgument(
            "Output length must be > 0".into(),
        ));
    }

    if output_length == signal.len() {
        return Ok(signal.to_vec());
    }

    let default_config = ResamplingConfig::default();
    let cfg = config.unwrap_or(&default_config);

    let ratio = signal.len() as f64 / output_length as f64;
    let half_width = match cfg.quality {
        ResamplingQuality::Draft => 2,
        ResamplingQuality::Standard => 6,
        ResamplingQuality::High => 12,
        ResamplingQuality::Audiophile => 24,
    };

    // Use sinc interpolation with anti-aliasing
    let cutoff = if ratio > 1.0 {
        cfg.cutoff / ratio
    } else {
        cfg.cutoff
    };

    sinc_resample(signal, output_length, half_width, cutoff, cfg.window)
}

// ---------------------------------------------------------------------------
// Sinc interpolation (arbitrary ratio)
// ---------------------------------------------------------------------------

/// Resample using windowed-sinc interpolation.
///
/// This supports arbitrary (non-rational) ratios by computing exact
/// sinc interpolation values at each output sample position.
fn sinc_resample(
    signal: &[f64],
    output_length: usize,
    half_width: usize,
    cutoff: f64,
    window: WindowType,
) -> SignalResult<Vec<f64>> {
    let n_in = signal.len();
    let n_out = output_length;
    let ratio = n_in as f64 / n_out as f64;

    let mut output = Vec::with_capacity(n_out);

    for i in 0..n_out {
        // Position in the input signal
        let pos = (i as f64 + 0.5) * ratio - 0.5;
        let int_pos = pos.floor() as i64;

        let mut acc = 0.0;
        let mut weight_sum = 0.0;

        for k in -(half_width as i64)..=(half_width as i64) {
            let idx = int_pos + k;
            if idx < 0 || idx >= n_in as i64 {
                continue;
            }

            let delta = pos - idx as f64;
            let sinc_val = if delta.abs() < f64::EPSILON {
                1.0
            } else {
                let arg = PI * delta * cutoff;
                arg.sin() / arg
            };

            // Window the sinc
            let win_pos = (delta / half_width as f64 + 1.0) / 2.0;
            let win_val = window_at(win_pos, window);

            let w = sinc_val * win_val;
            acc += signal[idx as usize] * w;
            weight_sum += w;
        }

        if weight_sum.abs() > f64::EPSILON {
            output.push(acc / weight_sum);
        } else {
            output.push(0.0);
        }
    }

    Ok(output)
}

/// Evaluate a window function at normalised position x in [0, 1].
fn window_at(x: f64, window: WindowType) -> f64 {
    if x < 0.0 || x > 1.0 {
        return 0.0;
    }
    match window {
        WindowType::Rectangular => 1.0,
        WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * x).cos()),
        WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
        WindowType::Blackman => 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos(),
        WindowType::Kaiser { beta } => {
            let arg = 1.0 - (2.0 * x - 1.0).powi(2);
            let arg = arg.max(0.0);
            bessel_i0(beta * arg.sqrt()) / bessel_i0(beta)
        }
    }
}

// ---------------------------------------------------------------------------
// Fractional Delay Filters
// ---------------------------------------------------------------------------

/// Design a Lagrange fractional delay filter.
///
/// A Lagrange interpolation filter of order N provides exact interpolation
/// for polynomials up to degree N.
///
/// # Arguments
/// * `delay` - Fractional delay in samples (0 < delay < order)
/// * `order` - Filter order (number of taps = order + 1)
///
/// # Returns
/// Filter coefficients.
pub fn lagrange_delay_filter(delay: f64, order: usize) -> SignalResult<Vec<f64>> {
    if order == 0 {
        return Err(SignalError::InvalidArgument("Order must be > 0".into()));
    }
    if delay < 0.0 {
        return Err(SignalError::InvalidArgument(
            "Delay must be non-negative".into(),
        ));
    }

    let n = order + 1;
    let mut h = vec![0.0; n];

    for k in 0..n {
        let mut prod = 1.0;
        for m in 0..n {
            if m != k {
                let denom = k as f64 - m as f64;
                if denom.abs() < f64::EPSILON {
                    continue;
                }
                prod *= (delay - m as f64) / denom;
            }
        }
        h[k] = prod;
    }

    Ok(h)
}

/// Design a windowed-sinc fractional delay filter.
///
/// This provides better frequency-domain characteristics than Lagrange
/// interpolation for longer filters.
///
/// # Arguments
/// * `delay` - Fractional delay in samples
/// * `half_length` - Half-length of the filter (total length = 2*half_length + 1)
/// * `window` - Window type
///
/// # Returns
/// Filter coefficients.
pub fn sinc_delay_filter(
    delay: f64,
    half_length: usize,
    window: WindowType,
) -> SignalResult<Vec<f64>> {
    if half_length == 0 {
        return Err(SignalError::InvalidArgument(
            "Half-length must be > 0".into(),
        ));
    }

    let len = 2 * half_length + 1;
    let center = half_length as f64 + delay;

    let mut h = vec![0.0; len];
    for i in 0..len {
        let n = i as f64;
        let delta = n - center;
        let sinc_val = if delta.abs() < f64::EPSILON {
            1.0
        } else {
            (PI * delta).sin() / (PI * delta)
        };

        let win_pos = i as f64 / (len.max(1) - 1).max(1) as f64;
        let win_val = window_at(win_pos, window);

        h[i] = sinc_val * win_val;
    }

    // Normalise
    let sum: f64 = h.iter().sum();
    if sum.abs() > f64::EPSILON {
        for hi in &mut h {
            *hi /= sum;
        }
    }

    Ok(h)
}

/// Apply a fractional delay to a signal using a delay filter.
///
/// # Arguments
/// * `signal` - Input signal
/// * `delay` - Fractional delay in samples
/// * `method` - "lagrange" or "sinc"
/// * `order` - Filter order / half-length
pub fn fractional_delay(
    signal: &[f64],
    delay: f64,
    method: &str,
    order: usize,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }

    let filter = match method {
        "lagrange" => lagrange_delay_filter(delay, order)?,
        "sinc" => sinc_delay_filter(delay, order, WindowType::Kaiser { beta: 6.0 })?,
        other => {
            return Err(SignalError::InvalidArgument(format!(
                "Unknown delay method: {}. Use 'lagrange' or 'sinc'.",
                other
            )));
        }
    };

    // Apply FIR filter
    apply_fir(signal, &filter)
}

// ---------------------------------------------------------------------------
// Upsampling and downsampling primitives
// ---------------------------------------------------------------------------

/// Upsample a signal by inserting zeros.
///
/// Inserts `factor - 1` zeros between each sample.
pub fn upsample(signal: &[f64], factor: usize) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if factor == 0 {
        return Err(SignalError::InvalidArgument("Factor must be > 0".into()));
    }
    if factor == 1 {
        return Ok(signal.to_vec());
    }

    let n = signal.len();
    let mut out = vec![0.0; n * factor];
    for (i, &s) in signal.iter().enumerate() {
        out[i * factor] = s;
    }
    Ok(out)
}

/// Downsample a signal by keeping every `factor`-th sample.
///
/// No anti-aliasing is applied; use `decimate` for proper downsampling.
pub fn downsample(signal: &[f64], factor: usize) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if factor == 0 {
        return Err(SignalError::InvalidArgument("Factor must be > 0".into()));
    }
    if factor == 1 {
        return Ok(signal.to_vec());
    }

    let out: Vec<f64> = signal.iter().step_by(factor).copied().collect();
    Ok(out)
}

/// Decimate: apply anti-aliasing filter then downsample.
pub fn decimate(
    signal: &[f64],
    factor: usize,
    filter_order: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if factor == 0 {
        return Err(SignalError::InvalidArgument("Factor must be > 0".into()));
    }
    if factor == 1 {
        return Ok(signal.to_vec());
    }

    let order = filter_order.unwrap_or(8 * factor + 1);
    let cutoff = 1.0 / factor as f64;
    let filter = design_anti_alias_filter(order, cutoff, WindowType::Kaiser { beta: 8.0 })?;

    let filtered = apply_fir(signal, &filter)?;
    downsample(&filtered, factor)
}

/// Interpolate: upsample then apply anti-aliasing filter.
pub fn interpolate(
    signal: &[f64],
    factor: usize,
    filter_order: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if factor == 0 {
        return Err(SignalError::InvalidArgument("Factor must be > 0".into()));
    }
    if factor == 1 {
        return Ok(signal.to_vec());
    }

    let up = upsample(signal, factor)?;
    let order = filter_order.unwrap_or(8 * factor + 1);
    let cutoff = 1.0 / factor as f64;
    let filter = design_anti_alias_filter(order, cutoff, WindowType::Kaiser { beta: 8.0 })?;

    // Scale by factor to compensate for zero-insertion
    let filter_scaled: Vec<f64> = filter.iter().map(|&h| h * factor as f64).collect();

    apply_fir(&up, &filter_scaled)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply a FIR filter to a signal (linear convolution, same length output).
fn apply_fir(signal: &[f64], filter: &[f64]) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let m = filter.len();
    let half = m / 2;

    let mut output = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..m {
            let idx = i as i64 + k as i64 - half as i64;
            if idx >= 0 && (idx as usize) < n {
                acc += signal[idx as usize] * filter[k];
            }
        }
        output[i] = acc;
    }
    Ok(output)
}

/// Greatest common divisor (Euclidean algorithm).
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Find a rational approximation P/Q to a real number using continued fractions.
fn rational_approximation(x: f64, max_denominator: usize) -> (usize, usize) {
    if x <= 0.0 {
        return (0, 1);
    }

    let mut p0: i64 = 0;
    let mut q0: i64 = 1;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;
    let mut xi = x;

    for _ in 0..50 {
        let a = xi.floor() as i64;
        let p2 = a * p1 + p0;
        let q2 = a * q1 + q0;

        if q2 > max_denominator as i64 || q2 < 0 {
            break;
        }

        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;

        let frac = xi - a as f64;
        if frac.abs() < 1e-10 {
            break;
        }
        xi = 1.0 / frac;
    }

    let p = p1.unsigned_abs() as usize;
    let q = q1.unsigned_abs() as usize;
    if q == 0 {
        (1, 1)
    } else {
        (p.max(1), q.max(1))
    }
}

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_tone(n: usize, freq: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_design_anti_alias_filter() {
        let h = design_anti_alias_filter(31, 0.5, WindowType::Hann).expect("filter");
        assert_eq!(h.len(), 31);
        // DC gain should be ~1
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.01,
            "DC gain should be ~1, got {}",
            dc_gain
        );
        // Symmetric
        for i in 0..h.len() / 2 {
            assert!(
                (h[i] - h[h.len() - 1 - i]).abs() < 1e-10,
                "Filter should be symmetric"
            );
        }
    }

    #[test]
    fn test_design_filter_kaiser() {
        let h =
            design_anti_alias_filter(51, 0.4, WindowType::Kaiser { beta: 8.0 }).expect("filter");
        assert_eq!(h.len(), 51);
        let dc_gain: f64 = h.iter().sum();
        assert!((dc_gain - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_upsample() {
        let signal = vec![1.0, 2.0, 3.0];
        let up = upsample(&signal, 3).expect("up");
        assert_eq!(up.len(), 9);
        assert!((up[0] - 1.0).abs() < 1e-10);
        assert!((up[1] - 0.0).abs() < 1e-10);
        assert!((up[2] - 0.0).abs() < 1e-10);
        assert!((up[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_downsample() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let down = downsample(&signal, 2).expect("down");
        assert_eq!(down.len(), 3);
        assert!((down[0] - 1.0).abs() < 1e-10);
        assert!((down[1] - 3.0).abs() < 1e-10);
        assert!((down[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_poly_identity() {
        let signal = make_tone(128, 10.0, 128.0);
        let result = resample_poly(&signal, 1, 1, None).expect("resample");
        assert_eq!(result.len(), signal.len());
        for (a, b) in result.iter().zip(signal.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_resample_poly_upsample() {
        let signal = make_tone(64, 5.0, 64.0);
        let result = resample_poly(&signal, 2, 1, None).expect("resample up");
        // Output should be approximately 128 samples
        assert!(
            result.len() >= 120 && result.len() <= 140,
            "Expected ~128 samples, got {}",
            result.len()
        );
    }

    #[test]
    fn test_resample_poly_downsample() {
        let signal = make_tone(128, 5.0, 128.0);
        let result = resample_poly(&signal, 1, 2, None).expect("resample down");
        assert!(
            result.len() >= 60 && result.len() <= 68,
            "Expected ~64 samples, got {}",
            result.len()
        );
    }

    #[test]
    fn test_resample_sample_rate() {
        let signal = make_tone(1000, 100.0, 8000.0);
        let result = resample(&signal, 8000.0, 16000.0, None).expect("resample");
        assert!(
            result.len() >= 1900 && result.len() <= 2100,
            "Expected ~2000 samples, got {}",
            result.len()
        );
    }

    #[test]
    fn test_resample_to_length() {
        let signal = make_tone(256, 10.0, 256.0);
        let result = resample_to_length(&signal, 128, None).expect("resample");
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn test_resample_to_length_upsample() {
        let signal = make_tone(128, 10.0, 128.0);
        let result = resample_to_length(&signal, 256, None).expect("resample up");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_lagrange_delay_filter_integer() {
        // Integer delay: should be a single 1 at the right position
        let h = lagrange_delay_filter(1.0, 3).expect("lagrange");
        assert_eq!(h.len(), 4);
        assert!(
            (h[1] - 1.0).abs() < 1e-10,
            "h[1] should be 1.0, got {:?}",
            h
        );
    }

    #[test]
    fn test_lagrange_delay_filter_half() {
        let h = lagrange_delay_filter(0.5, 3).expect("lagrange");
        assert_eq!(h.len(), 4);
        // Sum should be ~1 (partition of unity)
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum should be 1, got {}", sum);
    }

    #[test]
    fn test_sinc_delay_filter() {
        let h = sinc_delay_filter(0.3, 8, WindowType::Hann).expect("sinc delay");
        assert_eq!(h.len(), 17);
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Sum should be ~1, got {}", sum);
    }

    #[test]
    fn test_fractional_delay_lagrange() {
        let signal = make_tone(256, 10.0, 256.0);
        let delayed = fractional_delay(&signal, 0.5, "lagrange", 3).expect("delay");
        assert_eq!(delayed.len(), signal.len());
    }

    #[test]
    fn test_fractional_delay_sinc() {
        let signal = make_tone(256, 10.0, 256.0);
        let delayed = fractional_delay(&signal, 0.5, "sinc", 8).expect("delay");
        assert_eq!(delayed.len(), signal.len());
    }

    #[test]
    fn test_decimate() {
        let signal = make_tone(1024, 10.0, 1024.0);
        let decimated = decimate(&signal, 4, None).expect("decimate");
        assert_eq!(decimated.len(), 256);
    }

    #[test]
    fn test_interpolate() {
        let signal = make_tone(256, 10.0, 256.0);
        let interp = interpolate(&signal, 4, None).expect("interpolate");
        assert_eq!(interp.len(), 1024);
    }

    #[test]
    fn test_rational_approximation() {
        // 44100/48000 = 441/480 = 147/160
        let (p, q) = rational_approximation(44100.0 / 48000.0, 1000);
        let ratio = p as f64 / q as f64;
        let expected = 44100.0 / 48000.0;
        assert!(
            (ratio - expected).abs() < 0.001,
            "Expected ~{}, got {}/{} = {}",
            expected,
            p,
            q,
            ratio
        );
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(48000, 44100), 300);
    }

    #[test]
    fn test_bessel_i0() {
        // I0(0) = 1
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-10);
        // I0(1) ~ 1.2660658777520084
        assert!((bessel_i0(1.0) - 1.2660658777520084).abs() < 1e-8);
    }

    #[test]
    fn test_resample_errors() {
        assert!(resample_poly(&[], 2, 1, None).is_err());
        assert!(resample_poly(&[1.0], 0, 1, None).is_err());
        assert!(upsample(&[], 2).is_err());
        assert!(downsample(&[], 2).is_err());
    }

    #[test]
    fn test_resample_quality_presets() {
        let signal = make_tone(256, 10.0, 256.0);
        for quality in &[
            ResamplingQuality::Draft,
            ResamplingQuality::Standard,
            ResamplingQuality::High,
            ResamplingQuality::Audiophile,
        ] {
            let config = ResamplingConfig {
                quality: *quality,
                ..Default::default()
            };
            let result = resample_poly(&signal, 3, 2, Some(&config));
            assert!(
                result.is_ok(),
                "Resampling with {:?} quality failed",
                quality
            );
        }
    }

    #[test]
    fn test_apply_fir_identity() {
        let signal = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let filter = vec![0.0, 1.0, 0.0]; // identity (centered)
        let result = apply_fir(&signal, &filter).expect("fir");
        assert!((result[2] - 1.0).abs() < 1e-10);
    }
}
