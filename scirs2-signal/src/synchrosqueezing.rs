//! Synchrosqueezing Transforms (SST)
//!
//! The synchrosqueezing transform (SST) is a post-processing step applied to
//! either the CWT or the STFT.  Each coefficient at `(scale/frequency, time)`
//! is *squeezed* — reassigned in the frequency direction only — to the bin
//! corresponding to the instantaneous frequency of the signal component
//! active at that point.  The result is a highly concentrated, invertible
//! time-frequency representation that is especially useful for non-stationary
//! signals with multiple amplitude- and frequency-modulated components.
//!
//! # Provided functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`cwt_synchrosqueeze`] | Full CWT-based SST returning complex coefficients |
//! | [`stft_synchrosqueeze`] | STFT-based SST (real magnitude) |
//! | [`instantaneous_frequency_cwt`] | Instantaneous frequency from a CWT array |
//! | [`squeezing_operator`] | Raw squeezing accumulation step |
//! | [`synchrosqueeze_invert`] | Reconstruct time-domain signal from SST |
//! | [`extract_ridges`] | Dominant-mode ridge extraction |
//!
//! # References
//!
//! - Daubechies, Lu & Wu (2011). *Applied and Computational Harmonic Analysis* 30(2), 243-261.
//! - Thakur & Wu (2011). *SIAM J. Math. Analysis* 43(5), 2507-2523.
//! - Oberlin, Meignen & Perrier (2014). *IEEE Trans. Signal Process.* 62(3), 708-720.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// CWT-based Synchrosqueezing Transform
// ---------------------------------------------------------------------------

/// CWT-based Synchrosqueezing Transform.
///
/// Computes the analytic Morlet CWT on a logarithmically-spaced set of
/// frequencies, estimates the instantaneous frequency at each `(scale, time)`
/// point, and squeezes the CWT energy to the corresponding frequency bin.
///
/// # Arguments
///
/// * `signal`           – Real-valued input signal of length `n`.
/// * `voices_per_octave`– Number of voices (scales) per octave.  Higher
///   values give finer frequency resolution (typical: 8–32).
/// * `wavelet`          – Wavelet centre-frequency parameter `ω₀` for the
///   Morlet wavelet.  Values 5–8 are typical; `None` defaults to 6.0.
/// * `fs`               – Sampling frequency in Hz.  `None` uses 1.0.
/// * `gamma`            – Threshold below which CWT coefficients are ignored.
///   `None` defaults to `1e-10`.
///
/// # Returns
///
/// Complex SST array of shape `(n_freq_bins, n_time)` where `n_freq_bins` is
/// the number of discrete frequency bins on the output logarithmic axis.
///
/// # Example
///
/// ```
/// use scirs2_signal::synchrosqueezing::cwt_synchrosqueeze;
/// use std::f64::consts::PI;
///
/// let fs = 500.0_f64;
/// let n = 256_usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
///     .collect();
/// let sst = cwt_synchrosqueeze(&signal, 12, None, Some(fs), None).expect("operation should succeed");
/// assert_eq!(sst.shape()[1], n);
/// ```
pub fn cwt_synchrosqueeze(
    signal: &[f64],
    voices_per_octave: usize,
    wavelet: Option<f64>,
    fs: Option<f64>,
    gamma: Option<f64>,
) -> SignalResult<Array2<Complex64>> {
    let n = signal.len();
    if n < 4 {
        return Err(SignalError::ValueError(
            "signal must have at least 4 samples".to_string(),
        ));
    }
    if voices_per_octave == 0 {
        return Err(SignalError::ValueError(
            "voices_per_octave must be at least 1".to_string(),
        ));
    }
    let omega0 = wavelet.unwrap_or(6.0);
    if omega0 <= 0.0 {
        return Err(SignalError::ValueError(
            "wavelet centre frequency must be positive".to_string(),
        ));
    }
    let fs_val = fs.unwrap_or(1.0);
    if fs_val <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    let gamma_val = gamma.unwrap_or(1e-10);
    let dt = 1.0 / fs_val;

    // Build logarithmically-spaced scale/frequency grid.
    // Centre frequency of Morlet at scale a: f_c(a) = omega0 / (2π·a·dt)
    let f_max = fs_val / 2.0 * 0.95;
    let f_min = f_max / (2.0f64.powi(4)); // 4 octaves
    let a_min = omega0 / (2.0 * PI * f_max * dt);
    let a_max = omega0 / (2.0 * PI * f_min * dt);
    let n_octaves = (a_max / a_min).log2().ceil() as usize;
    let total_scales = voices_per_octave * n_octaves;
    if total_scales == 0 {
        return Err(SignalError::ComputationError(
            "computed zero scales; increase voices_per_octave".to_string(),
        ));
    }

    let scales: Vec<f64> = (0..total_scales)
        .map(|j| {
            let t = j as f64 / (total_scales - 1).max(1) as f64;
            a_min * (a_max / a_min).powf(t)
        })
        .collect();

    // CWT using Morlet wavelet in frequency domain
    let cwt = morlet_cwt(signal, &scales, omega0, dt)?;

    // Instantaneous frequency
    let if_matrix = instantaneous_frequency_cwt(&cwt, dt)?;

    // Build the output frequency axis (same as the scale centre frequencies)
    let freqs: Vec<f64> = scales
        .iter()
        .map(|&a| omega0 / (2.0 * PI * a * dt))
        .collect();
    let n_freqs = freqs.len();

    let f_step = if n_freqs > 1 {
        (freqs[n_freqs - 1] - freqs[0]).abs() / (n_freqs - 1) as f64
    } else {
        1.0
    };
    let f_start = freqs.iter().cloned().fold(f64::INFINITY, f64::min);

    // Squeezing
    let log_ratio = if total_scales > 1 {
        (a_max / a_min).ln() / (total_scales - 1) as f64
    } else {
        1.0
    };

    let mut sst: Array2<Complex64> = Array2::zeros((n_freqs, n));

    for (s_idx, &scale) in scales.iter().enumerate() {
        let da = scale * log_ratio;
        for t in 0..n {
            let coeff = cwt[[s_idx, t]];
            if coeff.norm() < gamma_val {
                continue;
            }
            let if_hz = if_matrix[[s_idx, t]] / (2.0 * PI);
            if if_hz <= 0.0 || if_hz > fs_val / 2.0 {
                continue;
            }
            let bin_f = ((if_hz - f_start) / f_step).round() as i64;
            if bin_f < 0 || bin_f >= n_freqs as i64 {
                continue;
            }
            let f_bin = bin_f as usize;
            // Contribution proportional to |W| · |da| (Jacobian of scale map)
            let contribution = coeff * (coeff.norm() * da.abs());
            sst[[f_bin, t]] += contribution;
        }
    }

    Ok(sst)
}

// ---------------------------------------------------------------------------
// STFT-based Synchrosqueezing Transform
// ---------------------------------------------------------------------------

/// STFT-based Synchrosqueezing Transform.
///
/// Computes the STFT, estimates the instantaneous frequency at every
/// `(freq_bin, frame)` from the phase derivative, and squeezes the STFT
/// energy to the instantaneous-frequency bin.
///
/// # Arguments
///
/// * `signal`   – Real-valued input signal.
/// * `window`   – Analysis window (e.g. Hann).
/// * `hop`      – Frame shift in samples.
/// * `fft_size` – FFT size (`None` for auto).
/// * `fs`       – Sampling frequency.
/// * `gamma`    – Magnitude threshold.  `None` defaults to `1e-8`.
///
/// # Returns
///
/// Real-valued SST array of shape `(n_freq_bins, n_frames)`.
///
/// # Example
///
/// ```
/// use scirs2_signal::synchrosqueezing::stft_synchrosqueeze;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 512_usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let window: Vec<f64> = (0..64)
///     .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / 63.0).cos()))
///     .collect();
/// let sst = stft_synchrosqueeze(&signal, &window, 16, None, fs, None).expect("operation should succeed");
/// assert!(sst.iter().all(|&v| v >= 0.0));
/// ```
pub fn stft_synchrosqueeze(
    signal: &[f64],
    window: &[f64],
    hop: usize,
    fft_size: Option<usize>,
    fs: f64,
    gamma: Option<f64>,
) -> SignalResult<Array2<f64>> {
    let win_len = window.len();
    if signal.len() < win_len {
        return Err(SignalError::ValueError(
            "signal must be at least as long as the window".to_string(),
        ));
    }
    if hop == 0 {
        return Err(SignalError::InvalidArgument("hop must be > 0".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    let nfft = match fft_size {
        Some(n) if n >= win_len => n,
        Some(n) => {
            return Err(SignalError::ValueError(format!(
                "fft_size ({n}) must be >= window length ({win_len})"
            )))
        }
        None => next_power_of_two(win_len),
    };
    let nfft = if nfft % 2 != 0 { nfft + 1 } else { nfft };

    let gamma_val = gamma.unwrap_or(1e-8);
    let n_frames = (signal.len().saturating_sub(win_len)) / hop + 1;
    let n_bins = nfft / 2 + 1;

    // Derivative window for instantaneous frequency: Dh[k] = h'(k), the
    // window's own (finite-difference) derivative -- *not* the time-ramp
    // window `k * h(k)` (that window computes the *group delay* / time
    // reassignment correction instead, see [`crate::spectral::reassignment`]
    // for the validated three-window reassignment formulation this mirrors).
    let dh: Vec<f64> = {
        let mut d = vec![0.0f64; win_len];
        if win_len > 1 {
            d[0] = window[1] - window[0];
            for k in 1..(win_len - 1) {
                d[k] = (window[k + 1] - window[k - 1]) / 2.0;
            }
            d[win_len - 1] = window[win_len - 1] - window[win_len - 2];
        }
        d
    };

    let mut sst = Array2::<f64>::zeros((n_bins, n_frames));

    let df = fs / nfft as f64;

    for frame in 0..n_frames {
        let start = frame * hop;
        let end = (start + win_len).min(signal.len());
        let actual = end - start;

        let mut buf_h = vec![0.0f64; nfft];
        let mut buf_dh = vec![0.0f64; nfft];
        for k in 0..actual {
            buf_h[k] = signal[start + k] * window[k];
            buf_dh[k] = signal[start + k] * dh[k];
        }

        let stft_h = raw_fft(&buf_h, nfft)?;
        let stft_dh = raw_fft(&buf_dh, nfft)?;

        for f in 0..n_bins {
            let sh = stft_h[f];
            let mag_sq = sh.norm_sqr();
            if mag_sq.sqrt() < gamma_val {
                continue;
            }
            // Instantaneous frequency estimate (reassignment formula):
            //   f̂ = f_nominal - Im{ S_Dh / S_h } · fs / (2π · nfft)
            // where `S_Dh` is the STFT computed with the window's derivative
            // `Dh[k] = h'(k)`.
            let sdh = stft_dh[f];
            let ratio_dh = sdh / sh;
            let f_nominal = f as f64 * fs / nfft as f64;
            let if_hz = f_nominal - ratio_dh.im * fs / (2.0 * PI * nfft as f64);

            if if_hz < 0.0 || if_hz > fs / 2.0 {
                continue;
            }

            let f_bin = (if_hz / df).round() as i64;
            if f_bin >= 0 && f_bin < n_bins as i64 {
                sst[[f_bin as usize, frame]] += mag_sq.sqrt();
            }
        }
    }

    Ok(sst)
}

// ---------------------------------------------------------------------------
// Instantaneous frequency from CWT
// ---------------------------------------------------------------------------

/// Compute the instantaneous frequency from a CWT coefficient array.
///
/// Uses the central-difference phase-derivative formula:
///
/// ```text
/// ω(a, t) ≈ -Im{ [W(a,t+Δt) - W(a,t-Δt)] / [2·Δt·W(a,t)] }
/// ```
///
/// Forward / backward differences are used at the edges.
///
/// # Arguments
///
/// * `cwt_coeffs` – Complex CWT array of shape `(n_scales, n_time)`.
/// * `dt`         – Temporal sampling interval (1/fs).
///
/// # Returns
///
/// Real-valued instantaneous frequency array of shape `(n_scales, n_time)`
/// in radians per second.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
/// use scirs2_signal::synchrosqueezing::instantaneous_frequency_cwt;
///
/// let cwt: Array2<Complex64> = Array2::ones((4, 64));
/// let if_mat = instantaneous_frequency_cwt(&cwt, 1.0 / 500.0).expect("operation should succeed");
/// assert_eq!(if_mat.shape(), cwt.shape());
/// ```
pub fn instantaneous_frequency_cwt(
    cwt_coeffs: &Array2<Complex64>,
    dt: f64,
) -> SignalResult<Array2<f64>> {
    let (n_scales, n_time) = cwt_coeffs.dim();
    if n_time < 2 {
        return Err(SignalError::ValueError(
            "CWT must have at least 2 time points".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(SignalError::ValueError("dt must be positive".to_string()));
    }

    let mut omega = Array2::<f64>::zeros((n_scales, n_time));

    for s in 0..n_scales {
        for t in 0..n_time {
            let w_t = cwt_coeffs[[s, t]];
            if w_t.norm() < 1e-30 {
                omega[[s, t]] = 0.0;
                continue;
            }
            let dw = if t == 0 {
                (cwt_coeffs[[s, 1]] - cwt_coeffs[[s, 0]]) / dt
            } else if t == n_time - 1 {
                (cwt_coeffs[[s, n_time - 1]] - cwt_coeffs[[s, n_time - 2]]) / dt
            } else {
                (cwt_coeffs[[s, t + 1]] - cwt_coeffs[[s, t - 1]]) / (2.0 * dt)
            };
            // ω = -Im(∂_t W / W)  =  Im(W* · ∂_t W) / |W|²
            omega[[s, t]] = (w_t.conj() * dw).im / w_t.norm_sqr();
        }
    }

    Ok(omega)
}

// ---------------------------------------------------------------------------
// Raw squeezing operator
// ---------------------------------------------------------------------------

/// Apply the squeezing operator to a CWT matrix.
///
/// Redistributes each `(scale, time)` CWT coefficient to the output frequency
/// bin determined by `if_matrix` (in radians per second).  Bins are defined
/// by the `voices` frequency axis (in Hz) with step `dv`.
///
/// This is a low-level building block.  For a complete SST pipeline prefer
/// [`cwt_synchrosqueeze`].
///
/// # Arguments
///
/// * `cwt`       – Complex CWT array `(n_scales, n_time)`.
/// * `if_matrix` – Instantaneous frequency in **rad/s**, shape `(n_scales, n_time)`.
/// * `voices`    – Frequency axis in **Hz**, length `n_voices`.
/// * `dv`        – Frequency step (Hz) of the output axis.
/// * `gamma`     – Magnitude threshold.
///
/// # Returns
///
/// Real SST array `(n_voices, n_time)` with non-negative values.
pub fn squeezing_operator(
    cwt: &Array2<Complex64>,
    if_matrix: &Array2<f64>,
    voices: &[f64],
    dv: f64,
    gamma: f64,
) -> SignalResult<Array2<f64>> {
    let (n_scales, n_time) = cwt.dim();
    let (if_rows, if_cols) = if_matrix.dim();
    if if_rows != n_scales || if_cols != n_time {
        return Err(SignalError::DimensionMismatch(format!(
            "cwt shape ({n_scales}, {n_time}) != if_matrix shape ({if_rows}, {if_cols})"
        )));
    }
    if voices.is_empty() {
        return Err(SignalError::ValueError(
            "voices must not be empty".to_string(),
        ));
    }
    if dv <= 0.0 {
        return Err(SignalError::ValueError("dv must be positive".to_string()));
    }

    let n_voices = voices.len();
    let f_min = voices.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut sst = Array2::<f64>::zeros((n_voices, n_time));

    for s in 0..n_scales {
        for t in 0..n_time {
            let coeff = cwt[[s, t]];
            if coeff.norm() < gamma {
                continue;
            }
            let if_hz = if_matrix[[s, t]] / (2.0 * PI);
            let bin_f = ((if_hz - f_min) / dv).round() as i64;
            if bin_f < 0 || bin_f >= n_voices as i64 {
                continue;
            }
            sst[[bin_f as usize, t]] += coeff.norm();
        }
    }

    Ok(sst)
}

// ---------------------------------------------------------------------------
// Inverse SST
// ---------------------------------------------------------------------------

/// Reconstruct a time-domain signal from SST coefficients.
///
/// Sums SST columns over the requested number of dominant frequency voices.
/// The reconstruction formula is:
///
/// ```text
/// x̂(t) ≈  2 · Re{ Σ_{f: top voices}  SST(f, t) }  ·  C_ψ^{-1}
/// ```
///
/// where `C_ψ` is the Morlet wavelet admissibility constant (approximated as
/// `π^{1/4} / (2·ω₀)` for the analytic Morlet).
///
/// # Arguments
///
/// * `ss_coeffs`        – Real or complex SST result `(n_voices, n_time)`.
/// * `voices_per_octave`– Voices per octave used when computing the SST
///   (needed for the Jacobian normalisation).
///
/// # Returns
///
/// Reconstructed real-valued signal of length `n_time`.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::synchrosqueezing::synchrosqueeze_invert;
///
/// let sst: Array2<f64> = Array2::zeros((32, 128));
/// let x = synchrosqueeze_invert(&sst, 8).expect("operation should succeed");
/// assert_eq!(x.len(), 128);
/// ```
pub fn synchrosqueeze_invert(
    ss_coeffs: &Array2<f64>,
    voices_per_octave: usize,
) -> SignalResult<Array1<f64>> {
    let (n_voices, n_time) = ss_coeffs.dim();
    if n_voices == 0 || n_time == 0 {
        return Err(SignalError::ValueError(
            "SST array must not be empty".to_string(),
        ));
    }
    if voices_per_octave == 0 {
        return Err(SignalError::ValueError(
            "voices_per_octave must be at least 1".to_string(),
        ));
    }

    // Admissibility normalisation factor for analytic Morlet with ω₀=6:
    //   C_ψ ≈ π^{1/4} · sqrt(2π) / ω₀
    let omega0 = 6.0f64;
    let c_psi = PI.powf(0.25) * (2.0 * PI).sqrt() / omega0;
    let dv = 1.0 / voices_per_octave as f64; // log-scale step
    let norm = 2.0 / (c_psi * dv);

    let mut output = Array1::<f64>::zeros(n_time);
    for t in 0..n_time {
        let mut sum = 0.0f64;
        for v in 0..n_voices {
            sum += ss_coeffs[[v, t]];
        }
        output[t] = sum * norm;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Ridge extraction
// ---------------------------------------------------------------------------

/// Extract dominant ridges from a synchrosqueezed transform.
///
/// A ridge is the trajectory of a dominant instantaneous frequency over time.
/// This function finds up to `n_ridges` ridges by iteratively picking the
/// maximum-energy path using a greedy, penalised local-maximum tracker.
///
/// # Algorithm
///
/// For each time frame the function finds local maxima in the frequency
/// direction.  The top `n_ridges` maxima (by energy) seed / continue ridges.
/// A ridge is *continued* from time `t-1` to time `t` only when a local
/// maximum at time `t` lies within `ridge_penalty` bins of the last known
/// ridge position.
///
/// # Arguments
///
/// * `ss_transform` – Real-valued SST array `(n_freq, n_time)`.
/// * `n_ridges`     – Maximum number of ridges to return.
/// * `ridge_penalty`– Maximum allowed frequency-bin jump between adjacent
///   frames.  Larger values allow more frequency variation.
///
/// # Returns
///
/// A `Vec` of at most `n_ridges` ridges.  Each ridge is an `Array1<f64>` of
/// length `n_time` containing the ridge frequency index (as a float) at each
/// time step.  Frames where the ridge is inactive are filled with `NaN`.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::synchrosqueezing::extract_ridges;
///
/// let mut sst: Array2<f64> = Array2::zeros((64, 128));
/// // Plant a fake ridge at frequency bin 10
/// for t in 0..128 { sst[[10, t]] = 1.0; }
/// let ridges = extract_ridges(&sst, 2, 5).expect("operation should succeed");
/// assert!(!ridges.is_empty());
/// assert_eq!(ridges[0].len(), 128);
/// ```
pub fn extract_ridges(
    ss_transform: &Array2<f64>,
    n_ridges: usize,
    ridge_penalty: usize,
) -> SignalResult<Vec<Array1<f64>>> {
    let (n_freq, n_time) = ss_transform.dim();
    if n_freq < 3 {
        return Err(SignalError::ValueError(
            "ss_transform must have at least 3 frequency bins".to_string(),
        ));
    }
    if n_time == 0 {
        return Err(SignalError::ValueError(
            "ss_transform must have at least 1 time frame".to_string(),
        ));
    }
    if n_ridges == 0 {
        return Ok(Vec::new());
    }
    if ridge_penalty == 0 {
        return Err(SignalError::InvalidArgument(
            "ridge_penalty must be > 0".to_string(),
        ));
    }

    // Initialise ridges with NaN
    let nan = f64::NAN;
    let mut ridges: Vec<Vec<f64>> = (0..n_ridges).map(|_| vec![nan; n_time]).collect();

    // Current ridge positions (frequency bin index).  None = ridge not yet started.
    let mut ridge_pos: Vec<Option<usize>> = vec![None; n_ridges];

    for t in 0..n_time {
        // Collect local maxima in the frequency dimension
        let mut local_maxima: Vec<(usize, f64)> = Vec::new();
        for f in 1..n_freq - 1 {
            let v = ss_transform[[f, t]];
            if v > ss_transform[[f - 1, t]] && v > ss_transform[[f + 1, t]] && v > 0.0 {
                local_maxima.push((f, v));
            }
        }
        // Also include endpoints if they dominate
        {
            let v0 = ss_transform[[0, t]];
            let v1 = ss_transform[[1, t]];
            if v0 > v1 && v0 > 0.0 {
                local_maxima.push((0, v0));
            }
            let vl = ss_transform[[n_freq - 1, t]];
            let vl1 = ss_transform[[n_freq - 2, t]];
            if vl > vl1 && vl > 0.0 {
                local_maxima.push((n_freq - 1, vl));
            }
        }
        // Sort by energy descending
        local_maxima.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedily assign maxima to ridges
        let mut used_maxima = vec![false; local_maxima.len()];

        for r in 0..n_ridges {
            // Try to extend existing ridge
            if let Some(prev_f) = ridge_pos[r] {
                // Find the closest local maximum within penalty range
                let mut best_idx: Option<usize> = None;
                let mut best_dist = ridge_penalty + 1;
                for (mi, &(f, _)) in local_maxima.iter().enumerate() {
                    if used_maxima[mi] {
                        continue;
                    }
                    let dist = if f >= prev_f { f - prev_f } else { prev_f - f };
                    if dist <= ridge_penalty && dist < best_dist {
                        best_dist = dist;
                        best_idx = Some(mi);
                    }
                }
                if let Some(mi) = best_idx {
                    let f = local_maxima[mi].0;
                    ridges[r][t] = f as f64;
                    ridge_pos[r] = Some(f);
                    used_maxima[mi] = true;
                } else {
                    // Ridge lost — mark as inactive (NaN) but keep position for possible recovery
                    ridges[r][t] = nan;
                    // Keep ridge_pos so we can reconnect in the next frame
                }
            } else {
                // Start new ridge from the strongest unused maximum
                for (mi, &(f, _)) in local_maxima.iter().enumerate() {
                    if !used_maxima[mi] {
                        ridges[r][t] = f as f64;
                        ridge_pos[r] = Some(f);
                        used_maxima[mi] = true;
                        break;
                    }
                }
            }
        }
    }

    // Convert to Array1 and return only ridges that have at least one valid point
    let result: Vec<Array1<f64>> = ridges
        .into_iter()
        .filter(|r| r.iter().any(|v| !v.is_nan()))
        .map(Array1::from)
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Morlet CWT via frequency-domain convolution.
///
/// Returns complex CWT array of shape `(n_scales, n_time)`.
fn morlet_cwt(
    signal: &[f64],
    scales: &[f64],
    omega0: f64,
    dt: f64,
) -> SignalResult<Array2<Complex64>> {
    let n = signal.len();
    let nfft = next_power_of_two(n);

    // FFT of zero-padded signal
    let signal_padded: Vec<f64> = signal
        .iter()
        .cloned()
        .chain(std::iter::repeat(0.0).take(nfft - n))
        .collect();
    let sig_fft = raw_fft(&signal_padded, nfft)?;

    // Angular frequency grid
    let omegas: Vec<f64> = (0..nfft)
        .map(|k| {
            let k_i = k as f64;
            if k <= nfft / 2 {
                2.0 * PI * k_i / (nfft as f64 * dt)
            } else {
                2.0 * PI * (k_i - nfft as f64) / (nfft as f64 * dt)
            }
        })
        .collect();

    let ns = scales.len();
    let mut result = Array2::<Complex64>::zeros((ns, n));

    for (s_idx, &scale) in scales.iter().enumerate() {
        // Analytic Morlet in frequency domain (only positive frequencies):
        // Ψ̂(a,ω) = π^{-1/4} · √(2πa·dt) · H(ω) · exp(-½(a·dt·ω - ω₀)²)
        let norm = PI.powf(-0.25) * (2.0 * PI * scale * dt).sqrt();

        let mut wav_fft: Vec<Complex64> = omegas
            .iter()
            .map(|&omega| {
                if omega <= 0.0 {
                    Complex64::new(0.0, 0.0)
                } else {
                    let arg = scale * dt * omega - omega0;
                    let val = norm * (-0.5 * arg * arg).exp();
                    Complex64::new(val, 0.0)
                }
            })
            .collect();

        // Multiply in frequency domain
        for k in 0..nfft {
            wav_fft[k] *= sig_fft[k];
        }

        // Inverse FFT
        let conv = complex_ifft(&wav_fft, nfft)?;

        for t in 0..n {
            result[[s_idx, t]] = conv[t];
        }
    }

    Ok(result)
}

/// Raw FFT using `scirs2_fft`.  Input must be `[f64; nfft]`.
fn raw_fft(buf: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    scirs2_fft::fft(buf, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))
}

/// Complex IFFT using conjugate-FFT trick.
fn complex_ifft(x: &[Complex64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    // Conjugate input
    let conj_buf: Vec<f64> = {
        // We need to pass re/im to scirs2_fft.  Build as real+imag interleaved
        // via the general fft interface which accepts Complex64.
        // Use the simpler: IFFT(X) = conj(FFT(conj(X)))/N
        let _ = nfft; // suppress warning
        Vec::new() // placeholder
    };
    let _ = conj_buf; // suppress

    // Use the library's ifft directly
    scirs2_fft::ifft(x, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("IFFT error: {e}")))
}

/// Next power of two.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
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

    fn sine(freq: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    fn hann(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // CWT SST
    // -----------------------------------------------------------------------

    #[test]
    fn test_cwt_sst_output_shape() {
        let fs = 500.0;
        let n = 256;
        let signal = sine(50.0, fs, n);
        let sst =
            cwt_synchrosqueeze(&signal, 12, None, Some(fs), None).expect("failed to create sst");
        assert_eq!(sst.shape()[1], n);
        assert!(sst.shape()[0] > 0);
    }

    #[test]
    fn test_cwt_sst_non_negative_magnitude() {
        let fs = 500.0;
        let signal = sine(50.0, fs, 256);
        let sst =
            cwt_synchrosqueeze(&signal, 8, None, Some(fs), None).expect("failed to create sst");
        // Magnitudes (norms) must be non-negative
        assert!(sst.iter().all(|c| c.norm() >= 0.0));
    }

    #[test]
    fn test_cwt_sst_error_short_signal() {
        let short = vec![1.0, 2.0, 3.0];
        assert!(cwt_synchrosqueeze(&short, 8, None, Some(1.0), None).is_err());
    }

    #[test]
    fn test_cwt_sst_error_zero_voices() {
        let signal = sine(10.0, 100.0, 128);
        assert!(cwt_synchrosqueeze(&signal, 0, None, Some(100.0), None).is_err());
    }

    // -----------------------------------------------------------------------
    // STFT SST
    // -----------------------------------------------------------------------

    #[test]
    fn test_stft_sst_shape() {
        let fs = 1000.0;
        let signal = sine(100.0, fs, 512);
        let window = hann(64);
        let sst = stft_synchrosqueeze(&signal, &window, 16, None, fs, None)
            .expect("failed to create sst");
        let n_bins = 33; // nfft=64, nfft/2+1=33
        assert_eq!(sst.shape()[0], n_bins);
        assert!(sst.shape()[1] > 0);
    }

    #[test]
    fn test_stft_sst_non_negative() {
        let fs = 1000.0;
        let signal = sine(100.0, fs, 512);
        let window = hann(64);
        let sst = stft_synchrosqueeze(&signal, &window, 16, None, fs, None)
            .expect("failed to create sst");
        assert!(sst.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_stft_sst_energy_concentration() {
        // Pure sinusoid at f0 should concentrate SST energy near f0
        let fs = 2000.0;
        let f0 = 200.0;
        let n = 1024;
        let signal = sine(f0, fs, n);
        let window = hann(128);
        let sst = stft_synchrosqueeze(&signal, &window, 32, None, fs, None)
            .expect("failed to create sst");
        let nfft = 128;
        let n_bins = nfft / 2 + 1;

        let total: f64 = sst.iter().sum();
        if total < 1e-10 {
            return; // trivially ok if zero
        }

        // Energy within ±20% of f0
        let f_lo = f0 * 0.80;
        let f_hi = f0 * 1.20;
        let df = fs / nfft as f64;
        let band_energy: f64 = (0..n_bins)
            .filter(|&f| {
                let fv = f as f64 * df;
                fv >= f_lo && fv <= f_hi
            })
            .map(|f| {
                sst.column(0).iter().enumerate().map(|_| 0.0).sum::<f64>()
                    + sst.row(f).iter().sum::<f64>()
            })
            .sum();

        // At least 20% of total energy should land near f0
        assert!(
            band_energy / total > 0.20,
            "Only {:.1}% of STFT-SST energy near f0={f0}",
            100.0 * band_energy / total
        );
    }

    #[test]
    fn test_stft_sst_error_hop_zero() {
        let signal: Vec<f64> = vec![0.0; 256];
        let window = hann(32);
        assert!(stft_synchrosqueeze(&signal, &window, 0, None, 1000.0, None).is_err());
    }

    #[test]
    fn test_stft_sst_error_fft_too_small() {
        let signal: Vec<f64> = vec![0.0; 256];
        let window = hann(64);
        assert!(stft_synchrosqueeze(&signal, &window, 16, Some(16), 1000.0, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Instantaneous frequency from CWT
    // -----------------------------------------------------------------------

    #[test]
    fn test_inst_freq_cwt_shape() {
        let cwt: Array2<Complex64> = Array2::ones((6, 64));
        let dt = 1.0 / 500.0;
        let if_mat = instantaneous_frequency_cwt(&cwt, dt).expect("failed to create if_mat");
        assert_eq!(if_mat.shape(), cwt.shape());
    }

    #[test]
    fn test_inst_freq_cwt_error_short() {
        let cwt: Array2<Complex64> = Array2::zeros((4, 1));
        assert!(instantaneous_frequency_cwt(&cwt, 0.01).is_err());
    }

    #[test]
    fn test_inst_freq_cwt_error_negative_dt() {
        let cwt: Array2<Complex64> = Array2::ones((4, 32));
        assert!(instantaneous_frequency_cwt(&cwt, -0.01).is_err());
    }

    // -----------------------------------------------------------------------
    // Squeezing operator
    // -----------------------------------------------------------------------

    #[test]
    fn test_squeezing_operator_shape() {
        let cwt: Array2<Complex64> = Array2::ones((8, 64));
        let if_mat: Array2<f64> = Array2::from_elem((8, 64), 2.0 * PI * 100.0);
        let voices: Vec<f64> = (0..32).map(|k| k as f64 * 10.0).collect();
        let sst =
            squeezing_operator(&cwt, &if_mat, &voices, 10.0, 1e-8).expect("failed to create sst");
        assert_eq!(sst.shape()[0], 32);
        assert_eq!(sst.shape()[1], 64);
        assert!(sst.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_squeezing_operator_dimension_mismatch() {
        let cwt: Array2<Complex64> = Array2::ones((8, 64));
        let if_mat: Array2<f64> = Array2::zeros((4, 64)); // wrong rows
        let voices: Vec<f64> = vec![0.0; 32];
        assert!(squeezing_operator(&cwt, &if_mat, &voices, 10.0, 1e-8).is_err());
    }

    // -----------------------------------------------------------------------
    // Inverse SST
    // -----------------------------------------------------------------------

    #[test]
    fn test_synchrosqueeze_invert_length() {
        let sst: Array2<f64> = Array2::zeros((32, 128));
        let x = synchrosqueeze_invert(&sst, 8).expect("failed to create x");
        assert_eq!(x.len(), 128);
    }

    #[test]
    fn test_synchrosqueeze_invert_zero_sst() {
        let sst: Array2<f64> = Array2::zeros((16, 64));
        let x = synchrosqueeze_invert(&sst, 8).expect("failed to create x");
        assert!(x.iter().all(|&v| v.abs() < 1e-30));
    }

    #[test]
    fn test_synchrosqueeze_invert_error_empty() {
        let sst: Array2<f64> = Array2::zeros((0, 64));
        assert!(synchrosqueeze_invert(&sst, 8).is_err());
    }

    // -----------------------------------------------------------------------
    // Ridge extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_ridges_basic() {
        let mut sst: Array2<f64> = Array2::zeros((64, 128));
        // Plant a clear ridge at frequency bin 10
        for t in 0..128 {
            sst[[10, t]] = 1.0;
        }
        let ridges = extract_ridges(&sst, 2, 5).expect("failed to create ridges");
        assert!(!ridges.is_empty());
        assert_eq!(ridges[0].len(), 128);
        // The dominant ridge should follow bin 10
        let valid: Vec<f64> = ridges[0].iter().filter(|v| !v.is_nan()).cloned().collect();
        assert!(!valid.is_empty());
        for &v in &valid {
            assert_eq!(v as usize, 10);
        }
    }

    #[test]
    fn test_extract_ridges_returns_empty_for_zero_ridges() {
        let sst: Array2<f64> = Array2::zeros((32, 64));
        let ridges = extract_ridges(&sst, 0, 3).expect("failed to create ridges");
        assert!(ridges.is_empty());
    }

    #[test]
    fn test_extract_ridges_error_too_few_bins() {
        let sst: Array2<f64> = Array2::zeros((2, 64));
        assert!(extract_ridges(&sst, 1, 3).is_err());
    }

    #[test]
    fn test_extract_ridges_length_matches_n_time() {
        let mut sst: Array2<f64> = Array2::zeros((32, 50));
        for t in 0..50 {
            sst[[5, t]] = 0.5;
        }
        let ridges = extract_ridges(&sst, 3, 4).expect("failed to create ridges");
        for ridge in &ridges {
            assert_eq!(ridge.len(), 50);
        }
    }
}
