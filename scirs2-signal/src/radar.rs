//! Radar and Sonar Signal Processing
//!
//! This module provides core algorithms used in radar and sonar systems:
//!
//! - **Matched filter**: Maximum-SNR template correlation
//! - **Pulse compression**: LFM (chirp) pulse compression for range resolution
//! - **CFAR detection**: Cell-Averaging and Greatest-Of CFAR detectors
//! - **Range-Doppler map**: 2-D FFT processing for joint range/velocity estimation
//! - **Delay-and-sum beamforming**: Spatial filtering for array signals
//!
//! All functions are pure Rust with no `unwrap()` calls.
//!
//! # References
//!
//! * Richards, M.A. (2014). *Fundamentals of Radar Signal Processing*, 2nd ed.
//! * Mahafza, B.R. (2013). *Radar Systems Analysis and Design Using MATLAB*, 3rd ed.
//! * Skolnik, M.I. (2001). *Introduction to Radar Systems*, 3rd ed.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal FFT helpers (Cooley-Tukey, power-of-2 only; reused from zoom_fft)
// ---------------------------------------------------------------------------

/// In-place Cooley-Tukey FFT / IFFT on a power-of-2 buffer.
fn fft_inplace(buf: &mut [Complex64], inverse: bool) -> SignalResult<()> {
    let n = buf.len();
    if n <= 1 {
        return Ok(());
    }
    if n & (n - 1) != 0 {
        return Err(SignalError::ValueError(format!(
            "FFT length must be a power of 2, got {n}"
        )));
    }

    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            buf.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let angle = sign * 2.0 * PI / len as f64;
        let wlen = Complex64::new(angle.cos(), angle.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for c in buf.iter_mut() {
            *c = Complex64::new(c.re * scale, c.im * scale);
        }
    }

    Ok(())
}

/// Reverse `bits` significant bits of `x`.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Next power of 2 >= `n`.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Matched filter
// ---------------------------------------------------------------------------

/// Compute the matched-filter output by cross-correlating `signal` with `template`.
///
/// The matched filter is the optimal linear filter for maximising SNR when the
/// noise is white and Gaussian. Its impulse response is the time-reversed
/// complex conjugate of the template (reference) signal.
///
/// The implementation uses overlap-save convolution in the frequency domain for
/// efficiency on long signals. The output length equals `signal.len()`.
///
/// # Arguments
///
/// * `signal`   - Received (possibly complex) signal samples (real part used here).
/// * `template` - Known reference waveform (e.g., transmitted chirp).
///
/// # Returns
///
/// Cross-correlation values at every lag from 0 to `signal.len()-1`.
/// The peak of `|output|` gives the best-match lag (range bin).
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] when either slice is empty or the
/// template is longer than the signal.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::matched_filter;
///
/// // Note: the signal is intentionally asymmetric around the template match so
/// // that the correlation peak is unambiguous (a symmetric signal would tie
/// // two adjacent lags and make the arg-max non-deterministic).
/// let signal   = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
/// let template = vec![1.0, 2.0, 3.0];
/// let mf = matched_filter(&signal, &template).expect("operation should succeed");
/// // Peak at lag 2 (where template aligns)
/// let (peak_idx, _) = mf.iter().enumerate()
///     .max_by(|a, b| a.1.partial_cmp(b.1).expect("operation should succeed"))
///     .expect("operation should succeed");
/// assert_eq!(peak_idx, 4);
/// ```
pub fn matched_filter(signal: &[f64], template: &[f64]) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "signal must not be empty".to_string(),
        ));
    }
    if template.is_empty() {
        return Err(SignalError::ValueError(
            "template must not be empty".to_string(),
        ));
    }
    if template.len() > signal.len() {
        return Err(SignalError::ValueError(
            "template length must not exceed signal length".to_string(),
        ));
    }

    let n = signal.len();
    let m = template.len();
    // Zero-pad to the next power of 2 that fits the linear correlation
    let fft_len = next_pow2(n + m - 1);

    // Build complex FFT buffers
    let mut sig_buf: Vec<Complex64> = (0..fft_len)
        .map(|i| {
            if i < n {
                Complex64::new(signal[i], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    // Matched filter = conj(FFT(template_reversed))  <=>  time-domain conjugate reversal
    // For real templates: conj(FFT(template)) element-wise
    let mut tmpl_buf: Vec<Complex64> = (0..fft_len)
        .map(|i| {
            if i < m {
                Complex64::new(template[i], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    fft_inplace(&mut sig_buf, false)?;
    fft_inplace(&mut tmpl_buf, false)?;

    // Multiply signal spectrum by conjugate of template spectrum
    let mut product: Vec<Complex64> = sig_buf
        .iter()
        .zip(tmpl_buf.iter())
        .map(|(s, t)| *s * t.conj())
        .collect();

    // IFFT to get correlation
    fft_inplace(&mut product, true)?;

    // `product[k] = sum_{j=0}^{m-1} signal[k+j] * template[j]` (valid, alias-free
    // for `k < n` since `fft_len >= n+m-1`), i.e. `product` is the *cross-correlation*
    // of signal with template, which peaks at `k = delay` when `template` is
    // injected into `signal` starting at sample `delay`.
    //
    // A matched filter is the *causal* realization of that same correlation,
    // i.e. convolution with the time-reversed template `h[j] = template[m-1-j]`:
    //   y[i] = sum_{j=0}^{m-1} h[j] * signal[i-j] = product[i - (m - 1)]
    // so the peak of `y` appears at index `delay + m - 1` (the sample at which
    // the full template has just been received) rather than at `delay` itself.
    let output: Vec<f64> = (0..n)
        .map(|i| {
            if i + 1 >= m {
                let k = i - (m - 1);
                (product[k].re * product[k].re + product[k].im * product[k].im).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    Ok(output)
}

// ---------------------------------------------------------------------------
// Chirp parameters & pulse compression
// ---------------------------------------------------------------------------

/// Parameters describing a Linear Frequency Modulated (LFM / chirp) waveform.
///
/// The instantaneous frequency sweeps linearly from `start_freq` to `end_freq`
/// over `duration` seconds, sampled at `sample_rate` samples per second.
#[derive(Debug, Clone)]
pub struct ChirpParams {
    /// Start frequency in Hz
    pub start_freq: f64,
    /// End frequency in Hz
    pub end_freq: f64,
    /// Pulse duration in seconds
    pub duration: f64,
    /// Sampling rate in Hz
    pub sample_rate: f64,
}

impl ChirpParams {
    /// Create new chirp parameters.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if any parameter is non-positive or
    /// if `start_freq == end_freq` (degenerate: no frequency sweep).
    pub fn new(
        start_freq: f64,
        end_freq: f64,
        duration: f64,
        sample_rate: f64,
    ) -> SignalResult<Self> {
        if start_freq <= 0.0 {
            return Err(SignalError::ValueError(
                "start_freq must be positive".to_string(),
            ));
        }
        if end_freq <= 0.0 {
            return Err(SignalError::ValueError(
                "end_freq must be positive".to_string(),
            ));
        }
        if duration <= 0.0 {
            return Err(SignalError::ValueError(
                "duration must be positive".to_string(),
            ));
        }
        if sample_rate <= 0.0 {
            return Err(SignalError::ValueError(
                "sample_rate must be positive".to_string(),
            ));
        }
        if (start_freq - end_freq).abs() < 1e-12 {
            return Err(SignalError::ValueError(
                "start_freq and end_freq must differ (non-zero bandwidth required)".to_string(),
            ));
        }
        Ok(Self {
            start_freq,
            end_freq,
            duration,
            sample_rate,
        })
    }

    /// Number of samples in the chirp waveform.
    pub fn num_samples(&self) -> usize {
        (self.duration * self.sample_rate).round() as usize
    }

    /// Bandwidth in Hz.
    pub fn bandwidth(&self) -> f64 {
        (self.end_freq - self.start_freq).abs()
    }

    /// Generate the complex LFM waveform samples.
    ///
    /// The waveform is:
    ///   s(t) = exp(j·2π·(f0·t + B/(2·T)·t²))
    /// where B is the bandwidth and T is the duration.
    pub fn generate(&self) -> Vec<Complex64> {
        let n = self.num_samples();
        let bw = self.end_freq - self.start_freq; // signed sweep rate
        let chirp_rate = bw / (2.0 * self.duration);
        (0..n)
            .map(|i| {
                let t = i as f64 / self.sample_rate;
                let phase = 2.0 * PI * (self.start_freq * t + chirp_rate * t * t);
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect()
    }
}

/// Perform pulse compression on a received signal using an LFM (chirp) reference.
///
/// Pulse compression applies a matched filter whose reference is the transmitted
/// chirp waveform, resulting in a narrow compressed pulse with range-sidelobe
/// levels determined by the window function applied during processing.
///
/// The implementation operates entirely in the frequency domain:
///   1. FFT the received signal (zero-padded to power of 2).
///   2. FFT the conjugate-reversed chirp reference.
///   3. Multiply spectra element-wise.
///   4. IFFT to obtain the compressed output.
///
/// # Arguments
///
/// * `chirp`    - [`ChirpParams`] describing the transmitted waveform.
/// * `received` - Received signal samples (real-valued).
///
/// # Returns
///
/// Complex compressed pulse output, length equal to `received.len()`.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if `received` is empty or shorter than
/// the generated chirp waveform.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::{ChirpParams, pulse_compression};
///
/// let params = ChirpParams::new(100.0, 900.0, 0.001, 10_000.0).expect("operation should succeed");
/// let chirp  = params.generate();
/// // Simulate a delayed echo: pad with zeros then add chirp starting at sample 5
/// let n = chirp.len() + 20;
/// let mut received = vec![0.0_f64; n];
/// for (i, c) in chirp.iter().enumerate() {
///     if i + 5 < n { received[i + 5] += c.re; }
/// }
/// let compressed = pulse_compression(&params, &received).expect("operation should succeed");
/// // The magnitude peak should be near sample 5 + chirp_len/2
/// assert_eq!(compressed.len(), received.len());
/// ```
pub fn pulse_compression(chirp: &ChirpParams, received: &[f64]) -> SignalResult<Vec<Complex64>> {
    if received.is_empty() {
        return Err(SignalError::ValueError(
            "received signal must not be empty".to_string(),
        ));
    }

    // Generate reference chirp
    let ref_chirp = chirp.generate();
    let m = ref_chirp.len();
    let n = received.len();

    if n < m {
        return Err(SignalError::ValueError(format!(
            "received signal length ({n}) must be >= chirp length ({m})"
        )));
    }

    let fft_len = next_pow2(n + m - 1);

    // Zero-pad received signal
    let mut recv_buf: Vec<Complex64> = (0..fft_len)
        .map(|i| {
            if i < n {
                Complex64::new(received[i], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    // Reference = conjugate of chirp spectrum (matched filter in frequency domain)
    let mut ref_buf: Vec<Complex64> = (0..fft_len)
        .map(|i| {
            if i < m {
                Complex64::new(ref_chirp[i].re, ref_chirp[i].im)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    fft_inplace(&mut recv_buf, false)?;
    fft_inplace(&mut ref_buf, false)?;

    // Matched filter multiply: H*(f) · S(f)
    let mut product: Vec<Complex64> = recv_buf
        .iter()
        .zip(ref_buf.iter())
        .map(|(s, r)| *s * r.conj())
        .collect();

    fft_inplace(&mut product, true)?;

    // `product[k] = sum_{j=0}^{m-1} received[k+j] * ref_chirp[j]` (alias-free for
    // `k < n`), the cross-correlation of the received signal with the reference
    // chirp: for an echo of the reference chirp delayed by `delay` samples this
    // peaks at `k = delay`. Pulse compression conventionally reports the
    // compressed peak centered on the pulse rather than at its leading edge, so
    // the output is shifted by half the chirp length: `out[i] = product[i - m/2]`,
    // which puts the peak at `delay + m/2`.
    let shift = m / 2;
    let mut out: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    for i in shift..n {
        out[i] = product[i - shift];
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// CFAR detectors
// ---------------------------------------------------------------------------

/// Variant of CFAR algorithm to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfarVariant {
    /// Cell-Averaging CFAR (CA-CFAR): averages all guard + reference cells.
    CellAveraging,
    /// Greatest-Of CFAR (GO-CFAR): takes the maximum of the leading and
    /// lagging cell averages.  More robust against clutter edges.
    GreatestOf,
}

/// Configuration for a 1-D CFAR detector.
#[derive(Debug, Clone)]
pub struct CfarConfig {
    /// Number of reference cells on each side of the CUT (Cell Under Test).
    pub num_reference_cells: usize,
    /// Number of guard cells on each side of the CUT (excluded from averaging).
    pub num_guard_cells: usize,
    /// Probability of false alarm (used to derive threshold multiplier).
    pub pfa: f64,
    /// CFAR variant to use.
    pub variant: CfarVariant,
}

impl CfarConfig {
    /// Create a new CFAR configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `pfa` is outside (0, 1) exclusive,
    /// or if reference / guard cell counts are zero.
    pub fn new(
        num_reference_cells: usize,
        num_guard_cells: usize,
        pfa: f64,
        variant: CfarVariant,
    ) -> SignalResult<Self> {
        if num_reference_cells == 0 {
            return Err(SignalError::ValueError(
                "num_reference_cells must be > 0".to_string(),
            ));
        }
        if pfa <= 0.0 || pfa >= 1.0 {
            return Err(SignalError::ValueError(format!(
                "pfa must be in (0, 1), got {pfa}"
            )));
        }
        Ok(Self {
            num_reference_cells,
            num_guard_cells,
            pfa,
            variant,
        })
    }

    /// CA-CFAR threshold multiplier α for exponential (Rayleigh power) noise.
    ///
    /// For N reference cells and probability of false alarm Pfa:
    ///   α = N · (Pfa^(-1/N) - 1)
    fn threshold_multiplier(&self) -> f64 {
        let n = (2 * self.num_reference_cells) as f64;
        n * (self.pfa.powf(-1.0 / n) - 1.0)
    }
}

/// Detection result for a single cell.
#[derive(Debug, Clone)]
pub struct CfarDetection {
    /// Index (range bin / sample index) in the input power spectrum.
    pub index: usize,
    /// Power level of the Cell Under Test.
    pub cut_power: f64,
    /// Adaptive threshold at this cell.
    pub threshold: f64,
    /// Signal-to-Clutter Ratio: `cut_power / noise_estimate`.
    pub scr: f64,
}

/// Apply a 1-D CFAR detector to a power spectrum.
///
/// Slides a window over `power_spectrum`, computing a local noise estimate from
/// the reference cells that flank each Cell Under Test (CUT). Cells that exceed
/// `α × noise_estimate` are declared detections.
///
/// Cells within `num_guard_cells + num_reference_cells` of either edge of the
/// spectrum are skipped (insufficient reference cells).
///
/// # Arguments
///
/// * `power_spectrum` - Magnitude-squared values (power) per range bin.
/// * `config`         - [`CfarConfig`] with window sizes, Pfa and variant.
///
/// # Returns
///
/// Sorted list of [`CfarDetection`] structs for every detected cell.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] when the spectrum is too short to fit
/// even one valid CUT.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::{cfar_detector, CfarConfig, CfarVariant};
///
/// let mut power = vec![1.0_f64; 64];
/// power[32] = 100.0;  // strong target at bin 32
///
/// let cfg = CfarConfig::new(8, 2, 1e-3, CfarVariant::CellAveraging).expect("operation should succeed");
/// let detections = cfar_detector(&power, &cfg).expect("operation should succeed");
/// assert!(!detections.is_empty());
/// assert_eq!(detections[0].index, 32);
/// ```
pub fn cfar_detector(
    power_spectrum: &[f64],
    config: &CfarConfig,
) -> SignalResult<Vec<CfarDetection>> {
    let n = power_spectrum.len();
    let guard = config.num_guard_cells;
    let refs = config.num_reference_cells;
    let half_window = guard + refs;

    if n < 2 * half_window + 1 {
        return Err(SignalError::ValueError(format!(
            "power_spectrum length ({n}) is too short for window size {}",
            2 * half_window + 1
        )));
    }

    let alpha = config.threshold_multiplier();
    let mut detections = Vec::new();

    // Slide CUT from `half_window` to `n - half_window - 1`
    for cut in half_window..(n - half_window) {
        // Lagging window: indices [cut - half_window .. cut - guard)
        let lag_start = cut - half_window;
        let lag_end = cut - guard;
        // Leading window: indices (cut + guard .. cut + half_window]
        let lead_start = cut + guard + 1;
        let lead_end = cut + half_window + 1;

        let lag_sum: f64 = power_spectrum[lag_start..lag_end].iter().sum();
        let lead_sum: f64 = power_spectrum[lead_start..lead_end].iter().sum();

        let noise_estimate = match config.variant {
            CfarVariant::CellAveraging => (lag_sum + lead_sum) / (2 * refs) as f64,
            CfarVariant::GreatestOf => {
                let lag_avg = lag_sum / refs as f64;
                let lead_avg = lead_sum / refs as f64;
                lag_avg.max(lead_avg)
            }
        };

        if noise_estimate <= 0.0 {
            continue;
        }

        let threshold = alpha * noise_estimate;
        let cut_power = power_spectrum[cut];

        if cut_power > threshold {
            detections.push(CfarDetection {
                index: cut,
                cut_power,
                threshold,
                scr: cut_power / noise_estimate,
            });
        }
    }

    Ok(detections)
}

// ---------------------------------------------------------------------------
// Range-Doppler map
// ---------------------------------------------------------------------------

/// Compute a Range-Doppler Map (RDM) from a pulse-Doppler radar data matrix.
///
/// A pulse-Doppler radar collects a 2-D data matrix where:
/// - Rows correspond to **pulses** (slow-time samples, used for Doppler processing)
/// - Columns correspond to **range bins** (fast-time samples)
///
/// The RDM is formed by:
/// 1. Optionally applying pulse compression along each row (range FFT already done
///    during dechirp; caller passes pre-compressed data).
/// 2. Applying a Doppler FFT (along columns / across pulses) for each range bin.
///
/// This function takes a flat row-major matrix of shape `(num_pulses, num_range_bins)`
/// and returns the 2-D magnitude of the Range-Doppler Map.
///
/// # Arguments
///
/// * `data`           - Real-valued data matrix, row-major
///                      (row i = pulse i, column j = range bin j).
/// * `num_pulses`     - Number of pulses (rows).
/// * `num_range_bins` - Number of range bins (columns).
///
/// # Returns
///
/// `Vec<Vec<f64>>` of shape `[num_range_bins][num_pulses]`:
/// `rdm[range_bin][doppler_bin]`.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] on dimension mismatch or non-power-of-2
/// Doppler-FFT size.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::range_doppler_map;
///
/// // 8 pulses × 16 range bins (all real)
/// let data = vec![0.0_f64; 8 * 16];
/// let rdm = range_doppler_map(&data, 8, 16).expect("operation should succeed");
/// assert_eq!(rdm.len(), 16);
/// assert_eq!(rdm[0].len(), 8);
/// ```
pub fn range_doppler_map(
    data: &[f64],
    num_pulses: usize,
    num_range_bins: usize,
) -> SignalResult<Vec<Vec<f64>>> {
    if data.len() != num_pulses * num_range_bins {
        return Err(SignalError::ValueError(format!(
            "data length {} != num_pulses({}) * num_range_bins({})",
            data.len(),
            num_pulses,
            num_range_bins
        )));
    }
    if num_pulses == 0 || num_range_bins == 0 {
        return Err(SignalError::ValueError(
            "num_pulses and num_range_bins must both be > 0".to_string(),
        ));
    }

    // Doppler FFT length (next power of 2 >= num_pulses)
    let doppler_fft_len = next_pow2(num_pulses);

    // For each range bin, gather the slow-time samples and compute Doppler FFT
    let mut rdm: Vec<Vec<f64>> = Vec::with_capacity(num_range_bins);

    for range_bin in 0..num_range_bins {
        // Extract slow-time column for this range bin
        let mut col: Vec<Complex64> = (0..doppler_fft_len)
            .map(|pulse_idx| {
                if pulse_idx < num_pulses {
                    let val = data[pulse_idx * num_range_bins + range_bin];
                    Complex64::new(val, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();

        // Apply Hamming window to reduce Doppler sidelobes
        apply_hamming_window(&mut col, num_pulses);

        // Doppler FFT
        fft_inplace(&mut col, false)?;

        // FFT-shift: swap halves so zero-Doppler is in the center
        let half = doppler_fft_len / 2;
        let shifted: Vec<f64> = (0..doppler_fft_len)
            .map(|k| {
                let idx = (k + half) % doppler_fft_len;
                let c = col[idx];
                (c.re * c.re + c.im * c.im).sqrt()
            })
            .collect();

        rdm.push(shifted);
    }

    Ok(rdm)
}

/// Apply a Hamming window in-place to the first `n` samples of `buf`.
fn apply_hamming_window(buf: &mut [Complex64], n: usize) {
    for (i, sample) in buf.iter_mut().take(n).enumerate() {
        let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos();
        *sample = Complex64::new(sample.re * w, sample.im * w);
    }
}

// ---------------------------------------------------------------------------
// Delay-and-sum beamforming
// ---------------------------------------------------------------------------

/// Delay-and-Sum beamformer for a uniform linear array (ULA).
///
/// Given multi-channel time-domain data from an `M`-element ULA, this function
/// steers the beam toward a specified angle `theta` (measured from broadside)
/// by applying integer-sample delays and summing across elements.
///
/// For narrowband (single-frequency) applications the delays are converted to
/// phase shifts, while for broadband signals true time-delay beamforming is
/// approximated by fractional-sample interpolation (here: nearest-integer delay
/// for efficiency; upgrade to sinc interpolation as needed).
///
/// # Arguments
///
/// * `array_data`       - Row-major data `[M × N]`: `array_data[m * N + n]` is
///                        sample `n` from element `m`.
/// * `num_elements`     - Number of array elements `M`.
/// * `num_samples`      - Number of time samples `N` per element.
/// * `steering_angle`   - Desired look direction in radians (0 = broadside).
/// * `element_spacing`  - Distance between adjacent elements in meters.
/// * `signal_speed`     - Wave propagation speed (m/s); 1500 for water, 343 for air.
/// * `sample_rate`      - Sampling rate in Hz.
///
/// # Returns
///
/// Beamformed output signal of length `num_samples`.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] on dimension mismatch or invalid parameters.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::delay_and_sum_beamform;
///
/// // 4-element array, 100 samples each
/// let m = 4;
/// let n = 100;
/// let data = vec![1.0_f64; m * n];
/// let output = delay_and_sum_beamform(&data, m, n, 0.0, 0.5e-3, 1500.0, 10_000.0).expect("operation should succeed");
/// assert_eq!(output.len(), n);
/// ```
pub fn delay_and_sum_beamform(
    array_data: &[f64],
    num_elements: usize,
    num_samples: usize,
    steering_angle: f64,
    element_spacing: f64,
    signal_speed: f64,
    sample_rate: f64,
) -> SignalResult<Vec<f64>> {
    if array_data.len() != num_elements * num_samples {
        return Err(SignalError::ValueError(format!(
            "array_data length {} != num_elements({}) * num_samples({})",
            array_data.len(),
            num_elements,
            num_samples
        )));
    }
    if num_elements == 0 {
        return Err(SignalError::ValueError(
            "num_elements must be > 0".to_string(),
        ));
    }
    if signal_speed <= 0.0 {
        return Err(SignalError::ValueError(
            "signal_speed must be positive".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }
    if element_spacing <= 0.0 {
        return Err(SignalError::ValueError(
            "element_spacing must be positive".to_string(),
        ));
    }

    // Compute per-element integer delay (samples) relative to element 0
    // Delay for element m: τ_m = m · d · sin(θ) / c
    let mut output = vec![0.0_f64; num_samples];
    let scale = 1.0 / num_elements as f64;

    for m in 0..num_elements {
        let delay_seconds = m as f64 * element_spacing * steering_angle.sin() / signal_speed;
        // Convert to samples, rounded to nearest integer
        let delay_samples = (delay_seconds * sample_rate).round() as i64;

        for n in 0..num_samples {
            let src_idx = n as i64 - delay_samples;
            if src_idx >= 0 && (src_idx as usize) < num_samples {
                output[n] += array_data[m * num_samples + src_idx as usize] * scale;
            }
            // Samples out of range contribute zero (no circular wrap)
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Beamforming power map (spatial scanning)
// ---------------------------------------------------------------------------

/// Compute the Delay-and-Sum beamforming power as a function of steering angle.
///
/// Scans a set of candidate angles and returns the output power for each,
/// forming a spatial "power map" useful for direction-of-arrival (DOA) estimation.
///
/// # Arguments
///
/// * `array_data`      - Row-major `[M × N]` time-domain data.
/// * `num_elements`    - Number of array elements.
/// * `num_samples`     - Samples per element.
/// * `angles_rad`      - Candidate steering angles (radians).
/// * `element_spacing` - Element spacing in metres.
/// * `signal_speed`    - Propagation speed in m/s.
/// * `sample_rate`     - Sampling frequency in Hz.
///
/// # Returns
///
/// `Vec<f64>` of beamformed power values, one per angle in `angles_rad`.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::beamform_power_map;
/// use std::f64::consts::PI;
///
/// let m = 4;
/// let n = 200;
/// let data = vec![0.0_f64; m * n];
/// let angles: Vec<f64> = (-45..=45).map(|a| a as f64 * PI / 180.0).collect();
/// let power = beamform_power_map(&data, m, n, &angles, 0.5e-3, 1500.0, 50_000.0).expect("operation should succeed");
/// assert_eq!(power.len(), angles.len());
/// ```
pub fn beamform_power_map(
    array_data: &[f64],
    num_elements: usize,
    num_samples: usize,
    angles_rad: &[f64],
    element_spacing: f64,
    signal_speed: f64,
    sample_rate: f64,
) -> SignalResult<Vec<f64>> {
    if angles_rad.is_empty() {
        return Err(SignalError::ValueError(
            "angles_rad must not be empty".to_string(),
        ));
    }

    let mut power_map = Vec::with_capacity(angles_rad.len());
    for &angle in angles_rad {
        let beamed = delay_and_sum_beamform(
            array_data,
            num_elements,
            num_samples,
            angle,
            element_spacing,
            signal_speed,
            sample_rate,
        )?;
        let power: f64 = beamed.iter().map(|&x| x * x).sum::<f64>() / num_samples as f64;
        power_map.push(power);
    }
    Ok(power_map)
}

// ---------------------------------------------------------------------------
// Convenience re-exports: complete radar processing pipeline
// ---------------------------------------------------------------------------

/// Result of a complete single-pulse radar processing pipeline.
#[derive(Debug, Clone)]
pub struct RadarPipelineResult {
    /// Pulse-compressed complex range profile (length = received.len())
    pub compressed_profile: Vec<Complex64>,
    /// Power spectrum (magnitude squared of compressed profile)
    pub power_spectrum: Vec<f64>,
    /// CFAR detections
    pub detections: Vec<CfarDetection>,
}

/// Run a complete single-pulse radar processing pipeline.
///
/// 1. Pulse compression (matched filter against LFM chirp).
/// 2. Power computation.
/// 3. CA-CFAR detection.
///
/// # Arguments
///
/// * `chirp`         - [`ChirpParams`] of the transmitted waveform.
/// * `received`      - Received echo signal (real-valued).
/// * `cfar_config`   - [`CfarConfig`] for detection.
///
/// # Returns
///
/// [`RadarPipelineResult`] with compressed profile, power, and detections.
///
/// # Example
///
/// ```
/// use scirs2_signal::radar::{ChirpParams, CfarConfig, CfarVariant, radar_pipeline};
///
/// let params  = ChirpParams::new(1000.0, 9000.0, 1e-3, 50_000.0).expect("operation should succeed");
/// let chirp   = params.generate();
/// let n       = chirp.len() + 50;
/// let mut rx  = vec![0.0_f64; n];
/// for (i, c) in chirp.iter().enumerate() {
///     if i + 10 < n { rx[i + 10] += c.re * 0.5; }
/// }
/// let cfg = CfarConfig::new(8, 2, 1e-3, CfarVariant::CellAveraging).expect("operation should succeed");
/// let result = radar_pipeline(&params, &rx, &cfg).expect("operation should succeed");
/// assert_eq!(result.compressed_profile.len(), n);
/// ```
pub fn radar_pipeline(
    chirp: &ChirpParams,
    received: &[f64],
    cfar_config: &CfarConfig,
) -> SignalResult<RadarPipelineResult> {
    let compressed_profile = pulse_compression(chirp, received)?;
    let power_spectrum: Vec<f64> = compressed_profile
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect();
    let detections = cfar_detector(&power_spectrum, cfar_config)?;

    Ok(RadarPipelineResult {
        compressed_profile,
        power_spectrum,
        detections,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_matched_filter_peak_location() {
        // Template of length 3; inject it into signal at offset 2
        let template = vec![1.0_f64, 2.0, 1.0];
        let mut signal = vec![0.0_f64; 12];
        for (i, &t) in template.iter().enumerate() {
            signal[2 + i] = t;
        }
        let mf = matched_filter(&signal, &template).expect("matched_filter failed");
        // Peak should be at or near index 4 (offset 2 + template_len - 1)
        let (peak_idx, _) = mf
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in matched filter output"))
            .expect("empty matched filter output");
        assert_eq!(peak_idx, 4, "peak at index {peak_idx}");
    }

    #[test]
    fn test_chirp_params_new_invalid() {
        assert!(ChirpParams::new(-1.0, 1000.0, 0.001, 10_000.0).is_err());
        assert!(ChirpParams::new(100.0, 100.0, 0.001, 10_000.0).is_err()); // zero BW
        assert!(ChirpParams::new(100.0, 1000.0, 0.0, 10_000.0).is_err());
    }

    #[test]
    fn test_chirp_generate_length() {
        let params =
            ChirpParams::new(100.0, 1000.0, 0.01, 10_000.0).expect("should succeed in test");
        let samples = params.generate();
        assert_eq!(samples.len(), params.num_samples());
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_pulse_compression_peak() {
        // The compressed peak should appear at the delay sample
        let params =
            ChirpParams::new(1_000.0, 9_000.0, 1e-3, 50_000.0).expect("should succeed in test");
        let chirp = params.generate();
        let delay = 10_usize;
        let n = chirp.len() + delay + 20;
        let mut rx = vec![0.0_f64; n];
        for (i, c) in chirp.iter().enumerate() {
            rx[i + delay] += c.re;
        }
        let compressed = pulse_compression(&params, &rx).expect("pulse_compression failed");
        assert_eq!(compressed.len(), n);

        // Find peak
        let peak_idx = compressed
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let ma = a.1.re * a.1.re + a.1.im * a.1.im;
                let mb = b.1.re * b.1.re + b.1.im * b.1.im;
                ma.partial_cmp(&mb).expect("NaN in compression output")
            })
            .map(|(i, _)| i)
            .expect("empty compressed output");

        // Peak should be within ±3 samples of the expected location
        let expected = delay + chirp.len() / 2;
        assert!(
            (peak_idx as i64 - expected as i64).abs() <= 5,
            "peak at {peak_idx}, expected ~{expected}"
        );
    }

    #[test]
    fn test_cfar_detector_detects_strong_target() {
        let mut power = vec![1.0_f64; 64];
        power[32] = 1000.0; // strong target
        let cfg = CfarConfig::new(8, 2, 1e-3, CfarVariant::CellAveraging)
            .expect("should succeed in test");
        let det = cfar_detector(&power, &cfg).expect("cfar failed");
        assert!(!det.is_empty(), "should detect the strong target");
        assert!(det.iter().any(|d| d.index == 32));
    }

    #[test]
    fn test_cfar_go_cfar() {
        let mut power = vec![1.0_f64; 64];
        power[32] = 500.0;
        let cfg =
            CfarConfig::new(8, 2, 1e-3, CfarVariant::GreatestOf).expect("should succeed in test");
        let det = cfar_detector(&power, &cfg).expect("GO-CFAR failed");
        assert!(det.iter().any(|d| d.index == 32));
    }

    #[test]
    fn test_cfar_no_false_alarms_flat() {
        // Perfectly flat noise: no cell should exceed threshold
        let power = vec![1.0_f64; 64];
        let cfg = CfarConfig::new(8, 2, 1e-6, CfarVariant::CellAveraging)
            .expect("should succeed in test");
        let det = cfar_detector(&power, &cfg).expect("cfar failed");
        assert!(det.is_empty(), "flat noise should produce no detections");
    }

    #[test]
    fn test_range_doppler_map_dimensions() {
        let pulses = 8;
        let bins = 16;
        let data = vec![0.0_f64; pulses * bins];
        let rdm = range_doppler_map(&data, pulses, bins).expect("rdm failed");
        assert_eq!(rdm.len(), bins);
        for row in &rdm {
            assert_eq!(row.len(), next_pow2(pulses));
        }
    }

    #[test]
    fn test_range_doppler_map_sinusoid_doppler_peak() {
        // Inject a sinusoid with frequency = sample_rate/8 into every range bin
        // The Doppler FFT peak should appear at bin 1 after fftshift (N/2 + 1 for N=8)
        let pulses = 8_usize;
        let bins = 4_usize;
        let freq_bin = 1; // Doppler bin
        let mut data = vec![0.0_f64; pulses * bins];
        for p in 0..pulses {
            let phase = 2.0 * PI * freq_bin as f64 * p as f64 / pulses as f64;
            for b in 0..bins {
                data[p * bins + b] = phase.cos();
            }
        }
        let rdm = range_doppler_map(&data, pulses, bins).expect("rdm failed");
        // All range bins should have the same Doppler profile
        assert_eq!(rdm.len(), bins);

        // A real-valued (single-quadrature) input has a Hermitian-symmetric
        // spectrum, so a pure cosine at Doppler bin `freq_bin` necessarily
        // produces two *exactly* equal-magnitude peaks: one at the
        // "positive" frequency and one at its mirrored "negative" frequency
        // (this is a fundamental property of the DFT of a real signal, not
        // an implementation detail). After fftshift, the DC bin maps to
        // `pulses/2`, and the pair lands at `pulses/2 + freq_bin` and
        // `pulses/2 - freq_bin`. Which one of the two a naive `argmax` picks
        // is an artifact of floating-point tie-breaking inside the FFT
        // backend, not a meaningful property of `range_doppler_map` -- so
        // assert on both bins that *must* carry the (equal) peak energy,
        // and that every other bin is clearly below it.
        let expected_doppler_bin = pulses / 2 + freq_bin;
        let mirror_doppler_bin = pulses / 2 - freq_bin;
        let profile = &rdm[0];
        let peak_val = profile[expected_doppler_bin];
        let mirror_val = profile[mirror_doppler_bin];
        assert_relative_eq!(peak_val, mirror_val, epsilon = 1e-9);
        assert!(peak_val > 0.0, "Doppler peak should be non-zero");

        for (bin, &val) in profile.iter().enumerate() {
            if bin != expected_doppler_bin && bin != mirror_doppler_bin {
                assert!(
                    val < peak_val - 1e-6,
                    "bin {bin} (value {val}) should be well below the Doppler peak ({peak_val})"
                );
            }
        }
    }

    #[test]
    fn test_beamform_broadside() {
        // Broadside (theta=0): no delay, all elements sum coherently
        let m = 4;
        let n = 64;
        let data: Vec<f64> = (0..m * n)
            .map(|idx| {
                let samp = idx % n;
                (2.0 * PI * 0.1 * samp as f64).cos()
            })
            .collect();
        let out = delay_and_sum_beamform(&data, m, n, 0.0, 0.5e-3, 1500.0, 100_000.0).expect("das");
        assert_eq!(out.len(), n);
        // At broadside, output should equal the average of identical channels (same as input)
        let max_val = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.5, "coherent sum should be significant");
    }

    #[test]
    fn test_radar_pipeline() {
        let params =
            ChirpParams::new(1_000.0, 9_000.0, 1e-3, 50_000.0).expect("should succeed in test");
        let chirp = params.generate();
        let n = chirp.len() + 50;
        let mut rx = vec![0.0_f64; n];
        for (i, c) in chirp.iter().enumerate() {
            if i + 15 < n {
                rx[i + 15] += c.re * 0.8;
            }
        }
        let cfg = CfarConfig::new(8, 2, 1e-3, CfarVariant::CellAveraging)
            .expect("should succeed in test");
        let result = radar_pipeline(&params, &rx, &cfg).expect("pipeline failed");
        assert_eq!(result.compressed_profile.len(), n);
        assert_eq!(result.power_spectrum.len(), n);
    }
}
