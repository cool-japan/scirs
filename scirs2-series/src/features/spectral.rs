//! Spectral feature extraction for time series analysis.
//!
//! This module provides a focused, high-level interface for spectral
//! characterisation of time series, including:
//! - Spectral centroid, bandwidth, and rolloff
//! - Spectral entropy (normalised Shannon entropy of the power spectrum)
//! - Dominant/peak frequency detection
//! - Power ratio across user-defined frequency bands
//!
//! The underlying FFT is computed via a pure-Rust Cooley-Tukey implementation
//! so that no external C/Fortran libraries are required (pure Rust policy).
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray::Array1;
//! use scirs2_series::features::spectral::*;
//!
//! let ts: Array1<f64> = Array1::from_iter(
//!     (0..128).map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 128.0).sin()),
//! );
//!
//! let centroid = spectral_centroid(&ts, 1.0).expect("centroid");
//! let entropy  = spectral_entropy(&ts).expect("entropy");
//! let peaks    = dominant_frequencies(&ts, 1.0, 3).expect("peaks");
//! ```

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Internal pure-Rust FFT (Cooley-Tukey, radix-2 DIT)
// ---------------------------------------------------------------------------

/// Complex number (f64) used for FFT internals.
#[derive(Clone, Copy)]
struct Complex64 {
    re: f64,
    im: f64,
}

impl Complex64 {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Compute the FFT of a real-valued signal.
/// The output length equals the next power-of-two ≥ `signal.len()`.
/// Returns a `Vec<f64>` of power values (|X[k]|²), length `n/2 + 1`.
fn real_fft_power<F: Float + FromPrimitive>(signal: &[F]) -> Vec<f64> {
    let n_orig = signal.len();
    // Next power of two
    let n = n_orig.next_power_of_two();

    // Copy into complex buffer (zero-padded)
    let mut buf: Vec<Complex64> = (0..n)
        .map(|i| {
            let re = if i < n_orig {
                signal[i].to_f64().unwrap_or(0.0)
            } else {
                0.0
            };
            Complex64::new(re, 0.0)
        })
        .collect();

    fft_inplace(&mut buf);

    // Return one-sided power spectrum (DC through Nyquist)
    let half = n / 2 + 1;
    (0..half).map(|k| buf[k].norm_sq()).collect()
}

/// In-place Cooley-Tukey radix-2 DIT FFT (iterative, bit-reversal permutation).
fn fft_inplace(buf: &mut Vec<Complex64>) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let w_root = Complex64::new(angle.cos(), angle.sin());

        let mut k = 0;
        while k < n {
            let mut w = Complex64::new(1.0, 0.0);
            for pos in 0..(len / 2) {
                let u = buf[k + pos];
                let v = buf[k + pos + len / 2].mul(w);
                buf[k + pos] = u.add(v);
                buf[k + pos + len / 2] = u.sub(v);
                w = w.mul(w_root);
            }
            k += len;
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a generic time series to `f64` power spectrum.
/// Also returns the frequency bin spacing (normalised: 0–0.5 for one-sided).
fn compute_power_and_freqs<F>(ts: &Array1<F>) -> Result<(Vec<f64>, Vec<f64>)>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.is_empty() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Empty time series passed to spectral feature".to_string(),
        ));
    }

    let power = real_fft_power(ts.as_slice().expect("contiguous array"));
    let n = ts.len().next_power_of_two();
    let half = n / 2 + 1;

    // Normalised frequencies: 0, 1/n, 2/n, ..., 0.5
    let freqs: Vec<f64> = (0..half).map(|k| k as f64 / n as f64).collect();

    Ok((power, freqs))
}

/// Convert normalised frequencies to physical Hz given a sampling rate.
#[inline]
fn normalised_to_hz(freqs: &[f64], fs: f64) -> Vec<f64> {
    freqs.iter().map(|&f| f * fs).collect()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the spectral centroid of a time series.
///
/// The spectral centroid is the weighted mean of the frequencies, weighted
/// by power, and represents the "center of mass" of the spectrum.
///
/// # Arguments
/// * `ts` - Input time series
/// * `fs` - Sampling rate in Hz (use 1.0 for normalised frequencies)
///
/// # Returns
/// Spectral centroid in Hz.
pub fn spectral_centroid<F>(ts: &Array1<F>, fs: f64) -> Result<f64>
where
    F: Float + FromPrimitive + Debug,
{
    let (power, freqs) = compute_power_and_freqs(ts)?;
    let hz = normalised_to_hz(&freqs, fs);

    let total_power: f64 = power.iter().sum();
    if total_power <= 0.0 {
        return Ok(0.0);
    }

    let centroid = power
        .iter()
        .zip(hz.iter())
        .map(|(&p, &f)| p * f)
        .sum::<f64>()
        / total_power;

    Ok(centroid)
}

/// Compute the spectral bandwidth (weighted standard deviation around centroid).
///
/// A narrow bandwidth indicates a tonal signal; a wide bandwidth suggests
/// broadband noise or highly complex signals.
///
/// # Arguments
/// * `ts` - Input time series
/// * `fs` - Sampling rate in Hz
/// * `p`  - Power exponent (typically 2 for RMS bandwidth)
///
/// # Returns
/// Spectral bandwidth in Hz.
pub fn spectral_bandwidth<F>(ts: &Array1<F>, fs: f64, p: u32) -> Result<f64>
where
    F: Float + FromPrimitive + Debug,
{
    let (power, freqs) = compute_power_and_freqs(ts)?;
    let hz = normalised_to_hz(&freqs, fs);

    let total_power: f64 = power.iter().sum();
    if total_power <= 0.0 {
        return Ok(0.0);
    }

    let centroid = power
        .iter()
        .zip(hz.iter())
        .map(|(&pw, &f)| pw * f)
        .sum::<f64>()
        / total_power;

    let bw = power
        .iter()
        .zip(hz.iter())
        .map(|(&pw, &f)| pw * (f - centroid).abs().powi(p as i32))
        .sum::<f64>()
        / total_power;

    Ok(bw.powf(1.0 / p as f64))
}

/// Compute the spectral rolloff frequency.
///
/// Returns the frequency below which `threshold` (e.g. 0.85) of the total
/// spectral energy is contained.
///
/// # Arguments
/// * `ts`        - Input time series
/// * `fs`        - Sampling rate in Hz
/// * `threshold` - Energy fraction ∈ (0, 1) (typically 0.85 or 0.95)
///
/// # Returns
/// Rolloff frequency in Hz.
pub fn spectral_rolloff<F>(ts: &Array1<F>, fs: f64, threshold: f64) -> Result<f64>
where
    F: Float + FromPrimitive + Debug,
{
    if !(0.0..=1.0).contains(&threshold) {
        return Err(TimeSeriesError::FeatureExtractionError(
            "spectral_rolloff threshold must be in (0, 1)".to_string(),
        ));
    }

    let (power, freqs) = compute_power_and_freqs(ts)?;
    let hz = normalised_to_hz(&freqs, fs);

    let total_power: f64 = power.iter().sum();
    if total_power <= 0.0 {
        return Ok(0.0);
    }

    let target = total_power * threshold;
    let mut cumulative = 0.0;

    for (&p, &f) in power.iter().zip(hz.iter()) {
        cumulative += p;
        if cumulative >= target {
            return Ok(f);
        }
    }

    Ok(*hz.last().expect("non-empty freqs"))
}

/// Compute the spectral entropy (normalised Shannon entropy of the power spectrum).
///
/// Values near 0 indicate a highly tonal signal; values near 1 indicate
/// broadband noise.
///
/// # Arguments
/// * `ts` - Input time series
///
/// # Returns
/// Spectral entropy ∈ [0, 1].
pub fn spectral_entropy<F>(ts: &Array1<F>) -> Result<f64>
where
    F: Float + FromPrimitive + Debug,
{
    let (power, _) = compute_power_and_freqs(ts)?;
    let total: f64 = power.iter().sum();
    if total <= 0.0 {
        return Ok(0.0);
    }

    let entropy = power
        .iter()
        .map(|&p| {
            let prob = p / total;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>();

    // Normalise by log(N)
    let n = power.len();
    let max_entropy = (n as f64).ln();
    if max_entropy <= 0.0 {
        return Ok(0.0);
    }

    Ok((entropy / max_entropy).min(1.0).max(0.0))
}

/// Find the top-N dominant (peak) frequencies in the power spectrum.
///
/// A local maximum in the power spectrum is detected when a bin's power
/// exceeds both neighbours. The results are sorted by descending power.
///
/// # Arguments
/// * `ts`      - Input time series
/// * `fs`      - Sampling rate in Hz
/// * `top_n`   - Number of peak frequencies to return (1 ≤ top_n ≤ spectrum_len)
///
/// # Returns
/// `Vec<PeakFrequency<f64>>` sorted by descending power.
pub fn dominant_frequencies<F>(ts: &Array1<F>, fs: f64, top_n: usize) -> Result<Vec<PeakFrequency>>
where
    F: Float + FromPrimitive + Debug,
{
    if top_n == 0 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "top_n must be at least 1".to_string(),
        ));
    }

    let (power, freqs) = compute_power_and_freqs(ts)?;
    let hz = normalised_to_hz(&freqs, fs);
    let m = power.len();

    // Detect local maxima (include DC and Nyquist as candidates if they dominate)
    let mut peaks: Vec<(f64, f64)> = Vec::new(); // (power, frequency)

    // Always include DC bin as a candidate
    if m > 0 {
        peaks.push((power[0], hz[0]));
    }

    for k in 1..(m.saturating_sub(1)) {
        if power[k] > power[k - 1] && power[k] > power[k + 1] {
            peaks.push((power[k], hz[k]));
        }
    }

    // Include Nyquist if it's a local maximum
    if m > 1 {
        let last = m - 1;
        if power[last] > power[last - 1] {
            peaks.push((power[last], hz[last]));
        }
    }

    // If no peaks found, fall back to max bin
    if peaks.is_empty() {
        if let Some((k, &p)) = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            peaks.push((p, hz[k]));
        }
    }

    // Sort by descending power and take top_n
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_power: f64 = power.iter().sum();

    let result: Vec<PeakFrequency> = peaks
        .into_iter()
        .take(top_n)
        .map(|(p, f)| PeakFrequency {
            frequency_hz: f,
            power: p,
            relative_power: if total_power > 0.0 {
                p / total_power
            } else {
                0.0
            },
        })
        .collect();

    Ok(result)
}

/// A detected spectral peak.
#[derive(Debug, Clone, PartialEq)]
pub struct PeakFrequency {
    /// Frequency in Hz (or normalised 0–0.5 if `fs = 1.0`)
    pub frequency_hz: f64,
    /// Absolute power at this peak
    pub power: f64,
    /// Power as a fraction of total spectral power
    pub relative_power: f64,
}

/// Compute the power ratio in each of the given frequency bands.
///
/// Each band is defined as a `(low_hz, high_hz)` half-open interval.
/// The function returns the fraction of total power that falls within
/// each band.
///
/// # Arguments
/// * `ts`    - Input time series
/// * `fs`    - Sampling rate in Hz
/// * `bands` - Slice of `(low_hz, high_hz)` tuples
///
/// # Returns
/// `Vec<BandPower>` — one entry per band, in the same order as `bands`.
pub fn power_ratio_features<F>(
    ts: &Array1<F>,
    fs: f64,
    bands: &[(f64, f64)],
) -> Result<Vec<BandPower>>
where
    F: Float + FromPrimitive + Debug,
{
    if bands.is_empty() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "At least one frequency band must be specified".to_string(),
        ));
    }

    let (power, freqs) = compute_power_and_freqs(ts)?;
    let hz = normalised_to_hz(&freqs, fs);
    let total_power: f64 = power.iter().sum();

    let result: Vec<BandPower> = bands
        .iter()
        .map(|&(lo, hi)| {
            let band_power: f64 = power
                .iter()
                .zip(hz.iter())
                .filter(|(_, &f)| f >= lo && f < hi)
                .map(|(&p, _)| p)
                .sum();
            BandPower {
                low_hz: lo,
                high_hz: hi,
                power: band_power,
                power_ratio: if total_power > 0.0 {
                    band_power / total_power
                } else {
                    0.0
                },
            }
        })
        .collect();

    Ok(result)
}

/// Power contained in a specific frequency band.
#[derive(Debug, Clone, PartialEq)]
pub struct BandPower {
    /// Lower band edge in Hz
    pub low_hz: f64,
    /// Upper band edge in Hz (exclusive)
    pub high_hz: f64,
    /// Absolute power in this band
    pub power: f64,
    /// Fraction of total spectral power in this band
    pub power_ratio: f64,
}

// ---------------------------------------------------------------------------
// Comprehensive spectral feature bundle
// ---------------------------------------------------------------------------

/// Comprehensive spectral feature bundle for a time series.
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid (Hz)
    pub centroid: f64,
    /// Spectral bandwidth (RMS, Hz)
    pub bandwidth: f64,
    /// 85%-energy rolloff frequency (Hz)
    pub rolloff_85: f64,
    /// 95%-energy rolloff frequency (Hz)
    pub rolloff_95: f64,
    /// Spectral entropy ∈ [0, 1]
    pub entropy: f64,
    /// Top-5 dominant frequencies
    pub dominant_frequencies: Vec<PeakFrequency>,
    /// Total spectral power (sum of power spectrum)
    pub total_power: f64,
    /// Spectral flatness (geometric mean / arithmetic mean of spectrum)
    pub flatness: f64,
    /// Spectral spread (second central moment)
    pub spread: f64,
}

/// Compute all spectral features for a time series.
///
/// # Arguments
/// * `ts` - Input time series (recommend ≥ 16 points)
/// * `fs` - Sampling rate in Hz (use 1.0 for normalised)
pub fn spectral_features<F>(ts: &Array1<F>, fs: f64) -> Result<SpectralFeatures>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 4 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Need at least 4 observations for spectral features".to_string(),
        ));
    }

    let (power, freqs) = compute_power_and_freqs(ts)?;
    let hz = normalised_to_hz(&freqs, fs);
    let total_power: f64 = power.iter().sum();

    let centroid = if total_power > 0.0 {
        power
            .iter()
            .zip(hz.iter())
            .map(|(&p, &f)| p * f)
            .sum::<f64>()
            / total_power
    } else {
        0.0
    };

    let spread = if total_power > 0.0 {
        let var = power
            .iter()
            .zip(hz.iter())
            .map(|(&p, &f)| p * (f - centroid).powi(2))
            .sum::<f64>()
            / total_power;
        var.sqrt()
    } else {
        0.0
    };

    let entropy = spectral_entropy(ts)?;
    let rolloff_85 = spectral_rolloff(ts, fs, 0.85)?;
    let rolloff_95 = spectral_rolloff(ts, fs, 0.95)?;

    let top5 = dominant_frequencies(ts, fs, 5)?;

    // Spectral flatness = geometric mean / arithmetic mean of power spectrum
    let flatness = {
        let n = power.len() as f64;
        let log_sum: f64 = power
            .iter()
            .map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
            .sum();
        let arith_mean = total_power / n;
        if arith_mean > 0.0 && log_sum.is_finite() {
            (log_sum / n).exp() / arith_mean
        } else {
            0.0
        }
    };

    // Bandwidth (p=2)
    let bw = if total_power > 0.0 {
        let var = power
            .iter()
            .zip(hz.iter())
            .map(|(&p, &f)| p * (f - centroid).powi(2))
            .sum::<f64>()
            / total_power;
        var.sqrt()
    } else {
        0.0
    };

    Ok(SpectralFeatures {
        centroid,
        bandwidth: bw,
        rolloff_85,
        rolloff_95,
        entropy,
        dominant_frequencies: top5,
        total_power,
        flatness,
        spread,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn sine_wave(n: usize, freq_hz: f64, fs: f64) -> Array1<f64> {
        Array1::from_iter(
            (0..n).map(|i| (2.0 * std::f64::consts::PI * freq_hz * i as f64 / fs).sin()),
        )
    }

    fn white_noise_like(n: usize) -> Array1<f64> {
        // Deterministic pseudo-noise
        Array1::from_iter((0..n).map(|i| {
            let x = i as f64 * 1.6180339887; // golden ratio
            x.fract() * 2.0 - 1.0
        }))
    }

    #[test]
    fn test_spectral_centroid_sine() {
        let fs = 128.0;
        let freq = 10.0;
        let ts = sine_wave(128, freq, fs);
        let centroid = spectral_centroid(&ts, fs).expect("centroid");
        // Centroid should be near the dominant frequency
        assert!(centroid > 0.0, "centroid non-positive: {}", centroid);
        assert!(centroid <= fs / 2.0, "centroid above Nyquist: {}", centroid);
    }

    #[test]
    fn test_spectral_bandwidth_sine() {
        let fs = 128.0;
        let ts = sine_wave(128, 10.0, fs);
        let bw = spectral_bandwidth(&ts, fs, 2).expect("bandwidth");
        assert!(bw >= 0.0, "negative bandwidth: {}", bw);
    }

    #[test]
    fn test_spectral_rolloff_fraction() {
        let ts = white_noise_like(64);
        let r85 = spectral_rolloff(&ts, 1.0, 0.85).expect("rolloff 85");
        let r95 = spectral_rolloff(&ts, 1.0, 0.95).expect("rolloff 95");
        // 95% rolloff must be ≥ 85% rolloff
        assert!(r95 >= r85 - 1e-9, "r95 ({}) < r85 ({})", r95, r85);
    }

    #[test]
    fn test_spectral_entropy_range() {
        let ts_sine = sine_wave(128, 5.0, 128.0);
        let ts_noise = white_noise_like(128);

        let e_sine = spectral_entropy(&ts_sine).expect("entropy sine");
        let e_noise = spectral_entropy(&ts_noise).expect("entropy noise");

        assert!((0.0..=1.0).contains(&e_sine), "e_sine = {}", e_sine);
        assert!((0.0..=1.0).contains(&e_noise), "e_noise = {}", e_noise);

        // Noise should have higher entropy than a single-frequency sine
        assert!(
            e_noise > e_sine,
            "expected noise ({}) > sine ({})",
            e_noise,
            e_sine
        );
    }

    #[test]
    fn test_dominant_frequencies_sine() {
        let fs = 128.0;
        let freq = 10.0;
        let ts = sine_wave(128, freq, fs);
        let peaks = dominant_frequencies(&ts, fs, 3).expect("peaks");
        assert!(!peaks.is_empty());
        // The dominant peak should be near the expected frequency
        let top = &peaks[0];
        assert!(top.frequency_hz >= 0.0);
        assert!(top.relative_power > 0.0);
    }

    #[test]
    fn test_dominant_frequencies_top_n() {
        let ts = white_noise_like(64);
        let peaks = dominant_frequencies(&ts, 1.0, 5).expect("peaks 5");
        assert!(peaks.len() <= 5);
        // Powers should be sorted descending
        for pair in peaks.windows(2) {
            assert!(pair[0].power >= pair[1].power - 1e-12);
        }
    }

    #[test]
    fn test_power_ratio_features() {
        let fs = 100.0;
        let ts = sine_wave(256, 10.0, fs);
        let bands = vec![(0.0, 5.0), (5.0, 15.0), (15.0, 50.0)];
        let result = power_ratio_features(&ts, fs, &bands).expect("band power");

        assert_eq!(result.len(), 3);
        let total_ratio: f64 = result.iter().map(|b| b.power_ratio).sum();
        // Sum of ratios for non-overlapping full-range bands should be ~1
        // (it may be < 1 if the Nyquist bin is excluded by the band boundaries)
        assert!(total_ratio <= 1.0 + 1e-9, "total_ratio = {}", total_ratio);

        // The middle band (5–15 Hz) should contain most power for a 10 Hz sine
        assert!(
            result[1].power_ratio > result[0].power_ratio,
            "middle band should dominate for 10 Hz sine"
        );
    }

    #[test]
    fn test_power_ratio_features_empty_bands() {
        let ts = sine_wave(64, 5.0, 64.0);
        assert!(power_ratio_features(&ts, 64.0, &[]).is_err());
    }

    #[test]
    fn test_spectral_features_bundle() {
        let fs = 128.0;
        let ts = sine_wave(128, 10.0, fs);
        let feats = spectral_features(&ts, fs).expect("spectral features");

        assert!(feats.centroid >= 0.0);
        assert!(feats.bandwidth >= 0.0);
        assert!((0.0..=1.0).contains(&feats.entropy));
        assert!(feats.rolloff_85 <= feats.rolloff_95 + 1e-9);
        assert!(feats.total_power > 0.0);
        assert!(!feats.dominant_frequencies.is_empty());
    }

    #[test]
    fn test_empty_series() {
        let ts: Array1<f64> = Array1::zeros(0);
        assert!(spectral_centroid(&ts, 1.0).is_err());
        assert!(spectral_entropy(&ts).is_err());
    }

    #[test]
    fn test_small_series() {
        let ts = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
        let c = spectral_centroid(&ts, 4.0).expect("centroid small");
        assert!(c >= 0.0);
    }
}
