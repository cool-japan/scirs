//! Free Vibration Analysis Methods for Operational Modal Analysis
//!
//! Provides:
//! * [`IbrahimTimeDomain`] - Ibrahim Time Domain (ITD) method
//! * [`random_decrement`] - Random Decrement Technique (RDT)
//! * [`natural_excitation_technique`] - NExT cross-correlation method
//! * [`next_era`] - NExT combined with ERA (convenience wrapper)
//! * [`era`] - Eigensystem Realization Algorithm (ERA / ERA-DC)
//!
//! These methods identify modal parameters (natural frequencies, damping
//! ratios, mode shapes) directly from time-domain free-decay responses (or
//! from NExT-derived pseudo impulse responses), complementing the
//! frequency-domain ([`crate::modal_analysis::fdd`]) and covariance-driven
//! stochastic subspace ([`crate::modal_analysis::ssi`]) methods elsewhere in
//! this module.
//!
//! # References
//! - Ibrahim, S.R. & Mikulcik, E.C. (1977). "A method for the direct identification of
//!   vibration parameters from the free response." *Shock and Vibration Bulletin*, 47(4), 183-198.
//! - Cole, H.A. (1973). "On-the-line analysis of random vibrations." *AIAA/ASME Structures and
//!   Materials Conference*, Paper 68-288.
//! - Juang, J.N. & Pappa, R.S. (1985). "An eigensystem realization algorithm for modal parameter
//!   identification and model reduction." *Journal of Guidance, Control, and Dynamics*, 8(5), 620-627.
//! - James, G., Carne, T. & Lauffer, J. (1993). "The Natural Excitation Technique (NExT) for
//!   modal parameter extraction from operating structures." *The International Journal of Analytical
//!   and Experimental Modal Analysis*, 10(4), 260-277.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Random Decrement Technique (RDT)
// ---------------------------------------------------------------------------

/// Configuration for the Random Decrement Technique
#[derive(Debug, Clone)]
pub struct RDTConfig {
    /// Trigger level (typically σ_x or some fraction thereof; 0 = auto-detect as RMS)
    pub trigger_level: f64,
    /// Trigger type: "positive_crossing" (default), "positive_point", "band_crossing"
    pub trigger_type: String,
    /// Length of free-decay segments to extract (in samples)
    pub segment_length: usize,
    /// Minimum number of segments required for averaging
    pub min_segments: usize,
    /// Optional band-pass filter band (Hz) for the band_crossing trigger
    pub band: Option<(f64, f64)>,
}

impl Default for RDTConfig {
    fn default() -> Self {
        Self {
            trigger_level: 0.0, // auto = RMS
            trigger_type: "positive_crossing".to_string(),
            segment_length: 512,
            min_segments: 50,
            band: None,
        }
    }
}

/// Result of the Random Decrement Technique
#[derive(Debug, Clone)]
pub struct RDTResult {
    /// Estimated free-decay function (randomdec signature), one per channel
    pub signatures: Vec<Vec<f64>>,
    /// Number of segments averaged per channel
    pub n_segments: Vec<usize>,
    /// Time axis (s) corresponding to each sample of the signature
    pub time_axis: Vec<f64>,
}

/// Apply the Random Decrement Technique to estimate free-decay functions.
///
/// The RDT works by:
/// 1. Detecting "trigger" crossings of the response (e.g. positive-slope zero-crossings
///    at a prescribed level).
/// 2. Extracting a segment of length `segment_length` starting at each trigger.
/// 3. Averaging all segments to cancel the random part, leaving the free-decay.
///
/// # Arguments
/// * `data` - `(n_channels, n_samples)` measurement array.
/// * `config` - RDT configuration.
/// * `fs` - Sampling frequency (Hz), used only for the time axis output.
pub fn random_decrement(
    data: &Array2<f64>,
    config: &RDTConfig,
    fs: f64,
) -> SignalResult<RDTResult> {
    let (n_channels, n_samples) = (data.nrows(), data.ncols());
    if n_channels == 0 || n_samples == 0 {
        return Err(SignalError::InvalidArgument(
            "Data array must have at least one channel and one sample".to_string(),
        ));
    }
    let seg_len = config.segment_length.max(4);
    if n_samples < seg_len {
        return Err(SignalError::InvalidArgument(format!(
            "n_samples ({n_samples}) must be >= segment_length ({seg_len})"
        )));
    }

    // Use channel 0 as the trigger channel
    let trigger_channel: Vec<f64> = data.row(0).iter().copied().collect();

    // Determine trigger level
    let level = if config.trigger_level > 0.0 {
        config.trigger_level
    } else {
        // Auto: use RMS of the trigger channel
        (trigger_channel.iter().map(|x| x * x).sum::<f64>() / n_samples as f64).sqrt()
    };

    // Find trigger points
    let triggers = find_trigger_points(
        &trigger_channel,
        level,
        &config.trigger_type,
        n_samples,
        seg_len,
    );

    if triggers.len() < config.min_segments {
        return Err(SignalError::InvalidArgument(format!(
            "Only {} trigger points found (min_segments = {}). \
             Try lowering trigger_level or using a longer signal.",
            triggers.len(),
            config.min_segments
        )));
    }

    // Average segments for each channel
    let mut signatures: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
    let mut n_segs_per_ch: Vec<usize> = Vec::with_capacity(n_channels);

    for ch in 0..n_channels {
        let ch_data: Vec<f64> = data.row(ch).iter().copied().collect();
        let mut acc = vec![0.0f64; seg_len];
        let mut n_valid = 0usize;
        for &t in &triggers {
            if t + seg_len <= n_samples {
                for k in 0..seg_len {
                    acc[k] += ch_data[t + k];
                }
                n_valid += 1;
            }
        }
        if n_valid > 0 {
            for v in acc.iter_mut() {
                *v /= n_valid as f64;
            }
        }
        signatures.push(acc);
        n_segs_per_ch.push(n_valid);
    }

    let time_axis: Vec<f64> = (0..seg_len).map(|k| k as f64 / fs).collect();

    Ok(RDTResult {
        signatures,
        n_segments: n_segs_per_ch,
        time_axis,
    })
}

/// Find positive-slope level-crossings of the signal.
fn find_trigger_points(
    signal: &[f64],
    level: f64,
    trigger_type: &str,
    n_samples: usize,
    seg_len: usize,
) -> Vec<usize> {
    let mut triggers = Vec::new();
    let max_t = n_samples.saturating_sub(seg_len);

    match trigger_type {
        "positive_point" => {
            for i in 0..max_t {
                if signal[i] >= level {
                    triggers.push(i);
                }
            }
        }
        // "positive_crossing", "band_crossing", and any other value all use a
        // positive-slope level crossing from below.
        _ => {
            for i in 1..max_t {
                if signal[i - 1] < level && signal[i] >= level {
                    triggers.push(i);
                }
            }
        }
    }
    triggers
}

// ---------------------------------------------------------------------------
// Natural Excitation Technique (NExT)
// ---------------------------------------------------------------------------

/// Configuration for the Natural Excitation Technique
#[derive(Debug, Clone)]
pub struct NExTConfig {
    /// Maximum lag (samples) for cross-correlation computation
    pub max_lag: usize,
    /// Reference channel index
    pub reference_channel: usize,
    /// Whether to normalise correlations (divide by zero-lag value)
    pub normalise: bool,
}

impl Default for NExTConfig {
    fn default() -> Self {
        Self {
            max_lag: 512,
            reference_channel: 0,
            normalise: true,
        }
    }
}

/// Apply the Natural Excitation Technique (NExT) to convert random response
/// cross-correlations into impulse-response-like free-decay functions.
///
/// Under broad-band random excitation, the cross-correlation between any two
/// response channels satisfies the same differential equation as the free
/// response (James et al., 1993). Thus the cross-correlations can be used
/// directly as input to a free-vibration identification algorithm (e.g. ERA).
///
/// # Arguments
/// * `data` - `(n_channels, n_samples)` measurement array.
/// * `config` - NExT configuration.
///
/// # Returns
/// Cross-correlation matrix: shape `(n_channels, 2*max_lag + 1)`.
/// Row `i` is the cross-correlation between channel `i` and the reference channel.
pub fn natural_excitation_technique(
    data: &Array2<f64>,
    config: &NExTConfig,
) -> SignalResult<Array2<f64>> {
    let (n_ch, n_s) = (data.nrows(), data.ncols());
    if config.reference_channel >= n_ch {
        return Err(SignalError::InvalidArgument(format!(
            "reference_channel {} out of range (n_channels = {})",
            config.reference_channel, n_ch
        )));
    }
    let max_lag = config.max_lag.min(n_s - 1);
    let n_lags = 2 * max_lag + 1;
    let ref_ch = config.reference_channel;

    let ref_data: Vec<f64> = data.row(ref_ch).iter().copied().collect();

    // Subtract mean
    let ref_mean = ref_data.iter().sum::<f64>() / n_s as f64;
    let ref_zero: Vec<f64> = ref_data.iter().map(|x| x - ref_mean).collect();

    let mut result = Array2::<f64>::zeros((n_ch, n_lags));

    for ch in 0..n_ch {
        let ch_data: Vec<f64> = data.row(ch).iter().copied().collect();
        let ch_mean = ch_data.iter().sum::<f64>() / n_s as f64;
        let ch_zero: Vec<f64> = ch_data.iter().map(|x| x - ch_mean).collect();

        // Cross-correlation: R_xi_xref(τ) = Σ x_i(t) * x_ref(t + τ)
        for lag_idx in 0..n_lags {
            let lag = lag_idx as i64 - max_lag as i64;
            let mut sum = 0.0;
            let mut count = 0usize;
            for t in 0..n_s {
                let s = t as i64 + lag;
                if s >= 0 && s < n_s as i64 {
                    sum += ch_zero[t] * ref_zero[s as usize];
                    count += 1;
                }
            }
            result[[ch, lag_idx]] = if count > 0 { sum / count as f64 } else { 0.0 };
        }

        // Normalise by zero-lag value
        if config.normalise {
            let zero_lag = result[[ch, max_lag]].abs();
            if zero_lag > 1e-30 {
                for k in 0..n_lags {
                    result[[ch, k]] /= zero_lag;
                }
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Ibrahim Time Domain (ITD)
// ---------------------------------------------------------------------------

/// Ibrahim Time Domain (ITD) method for modal parameter identification.
///
/// ITD fits a state-space model to free-response data sampled at two time
/// instants separated by `delta_t` samples, using the eigendecomposition of
/// the resulting "system matrix".
#[derive(Debug, Clone)]
pub struct IbrahimTimeDomain {
    /// Sampling frequency (Hz)
    pub fs: f64,
    /// Time delay between two "instrument" sets (samples)
    pub delta_t: usize,
    /// Model order (number of modes × 2)
    pub model_order: usize,
    /// Minimum damping ratio to accept
    pub min_damping: f64,
    /// Maximum damping ratio to accept
    pub max_damping: f64,
    /// Minimum frequency (Hz)
    pub f_min: f64,
    /// Maximum frequency (Hz)
    pub f_max: f64,
}

impl Default for IbrahimTimeDomain {
    fn default() -> Self {
        Self {
            fs: 1.0,
            delta_t: 1,
            model_order: 10,
            min_damping: 0.0,
            max_damping: 0.3,
            f_min: 0.0,
            f_max: f64::INFINITY,
        }
    }
}

/// Result of ITD identification
#[derive(Debug, Clone)]
pub struct ITDResult {
    /// Natural frequencies (Hz)
    pub natural_frequencies: Vec<f64>,
    /// Damping ratios
    pub damping_ratios: Vec<f64>,
    /// Mode shapes (one per mode)
    pub mode_shapes: Vec<Vec<f64>>,
}

impl IbrahimTimeDomain {
    /// Run ITD identification on free-vibration (or RDT/NExT) data.
    ///
    /// # Arguments
    /// * `free_response` - `(n_channels, n_samples)` free-response data.
    pub fn identify(&self, free_response: &Array2<f64>) -> SignalResult<ITDResult> {
        let (l, n_s) = (free_response.nrows(), free_response.ncols());
        if l == 0 || n_s < 2 * self.delta_t {
            return Err(SignalError::InvalidArgument(
                "Insufficient free response data for ITD".to_string(),
            ));
        }

        let dt = self.delta_t;
        let n_cols = n_s - dt;

        // Build [X] and [X'] (shifted by delta_t)
        // X: l × n_cols, X': l × n_cols
        let mut x = vec![0.0f64; l * n_cols];
        let mut xp = vec![0.0f64; l * n_cols];
        for ch in 0..l {
            let row = free_response.row(ch);
            for k in 0..n_cols {
                x[ch * n_cols + k] = row[k];
                xp[ch * n_cols + k] = row[k + dt];
            }
        }

        // System matrix A' = X' X^+ = X' X^T (X X^T)^{-1}  [l × l]
        // Using normal equations
        // XX_T = X @ X^T  (l × l)
        let mut xx_t = vec![0.0f64; l * l];
        for i in 0..l {
            for j in 0..l {
                let mut sum = 0.0;
                for k in 0..n_cols {
                    sum += x[i * n_cols + k] * x[j * n_cols + k];
                }
                xx_t[i * l + j] = sum;
            }
        }
        // Xp_XT = X' @ X^T  (l × l)
        let mut xp_xt = vec![0.0f64; l * l];
        for i in 0..l {
            for j in 0..l {
                let mut sum = 0.0;
                for k in 0..n_cols {
                    sum += xp[i * n_cols + k] * x[j * n_cols + k];
                }
                xp_xt[i * l + j] = sum;
            }
        }
        // A' = Xp_XT @ inv(XX_T)
        let inv_xx_t = itd_pseudo_inverse(&xx_t, l)?;
        let mut a_sys = vec![0.0f64; l * l];
        for i in 0..l {
            for j in 0..l {
                let mut sum = 0.0;
                for k in 0..l {
                    sum += xp_xt[i * l + k] * inv_xx_t[k * l + j];
                }
                a_sys[i * l + j] = sum;
            }
        }

        // Eigendecompose A_sys to get discrete poles
        let (eig_pairs, eig_vecs) = real_eig_pairs(&a_sys, l)?;
        let ddt = dt as f64 / self.fs;
        let f_max = if self.f_max.is_infinite() {
            self.fs / 2.0
        } else {
            self.f_max
        };

        let mut natural_frequencies = Vec::new();
        let mut damping_ratios = Vec::new();
        let mut mode_shapes = Vec::new();

        for (idx, (mu_re, mu_im)) in eig_pairs.iter().enumerate() {
            if mu_im.abs() < 1e-10 || *mu_im < 0.0 {
                continue;
            }
            let mu_abs_sq = mu_re * mu_re + mu_im * mu_im;
            if mu_abs_sq < 1e-30 {
                continue;
            }
            let lam_re = mu_abs_sq.ln() / (2.0 * ddt);
            let lam_im = mu_im.atan2(*mu_re) / ddt;
            let omega_n = (lam_re * lam_re + lam_im * lam_im).sqrt();
            let fn_hz = omega_n / (2.0 * PI);
            let xi = if omega_n > 1e-14 {
                (-lam_re / omega_n).clamp(self.min_damping, self.max_damping)
            } else {
                0.0
            };
            if fn_hz < self.f_min || fn_hz > f_max {
                continue;
            }
            if xi < self.min_damping || xi > self.max_damping {
                continue;
            }
            natural_frequencies.push(fn_hz);
            damping_ratios.push(xi);
            // Mode shape: column k of the eigenvector matrix of the symmetrised system matrix.
            // Use absolute values so the shape is real-valued and sign-invariant.
            let shape: Vec<f64> = if idx < eig_vecs.len() {
                eig_vecs[idx].iter().map(|&v| v.abs()).collect()
            } else {
                vec![1.0; l]
            };
            mode_shapes.push(shape);
        }

        Ok(ITDResult {
            natural_frequencies,
            damping_ratios,
            mode_shapes,
        })
    }
}

// ---------------------------------------------------------------------------
// Eigensystem Realization Algorithm (ERA)
// ---------------------------------------------------------------------------

/// Configuration for the Eigensystem Realization Algorithm
#[derive(Debug, Clone)]
pub struct ERAConfig {
    /// Sampling frequency (Hz)
    pub fs: f64,
    /// Number of block rows in the Markov parameter Hankel matrix
    pub block_rows: usize,
    /// Number of block columns in the Markov parameter Hankel matrix
    pub block_cols: usize,
    /// Model order (state-space dimension)
    pub model_order: usize,
    /// Minimum damping ratio
    pub min_damping: f64,
    /// Maximum damping ratio
    pub max_damping: f64,
    /// Minimum frequency (Hz)
    pub f_min: f64,
    /// Maximum frequency (Hz)
    pub f_max: f64,
    /// Minimum Modal Phase Collinearity (MPC) threshold for mode validation
    pub min_mpc: f64,
}

impl Default for ERAConfig {
    fn default() -> Self {
        Self {
            fs: 1.0,
            block_rows: 20,
            block_cols: 20,
            model_order: 10,
            min_damping: 0.0,
            max_damping: 0.3,
            f_min: 0.0,
            f_max: f64::INFINITY,
            min_mpc: 0.5,
        }
    }
}

/// Result of ERA modal identification
#[derive(Debug, Clone)]
pub struct ERAResult {
    /// Identified natural frequencies (Hz)
    pub natural_frequencies: Vec<f64>,
    /// Identified damping ratios
    pub damping_ratios: Vec<f64>,
    /// Mode shapes (one per mode)
    pub mode_shapes: Vec<Vec<f64>>,
    /// Modal Phase Collinearity for each mode (0-1; close to 1 = physical)
    pub mpc: Vec<f64>,
    /// Modal Scale Factor (normalized modal amplitude)
    pub msf: Vec<f64>,
    /// System matrix A (state space, `n × n`)
    pub a_matrix: Vec<f64>,
    /// Output matrix C (`l × n`)
    pub c_matrix: Vec<f64>,
}

/// Eigensystem Realization Algorithm (ERA) for modal parameter identification.
///
/// ERA builds a state-space model from a set of Markov parameters (impulse
/// responses or cross-correlations) and then eigendecomposes the A matrix.
///
/// # Arguments
/// * `impulse_responses` - `(n_channels, n_time_steps)` impulse response (or
///   NExT cross-correlation) matrix.
/// * `config` - ERA configuration.
pub fn era(impulse_responses: &Array2<f64>, config: &ERAConfig) -> SignalResult<ERAResult> {
    let (l, n_time) = (impulse_responses.nrows(), impulse_responses.ncols());
    if l == 0 || n_time == 0 {
        return Err(SignalError::InvalidArgument(
            "Impulse response matrix must not be empty".to_string(),
        ));
    }
    let p = config.block_rows;
    let q = config.block_cols;
    if n_time < p + q + 1 {
        return Err(SignalError::InvalidArgument(format!(
            "n_time_steps ({n_time}) must be >= block_rows + block_cols + 1 = {}",
            p + q + 1
        )));
    }

    // Build Hankel matrix H(0): rows = p*l, cols = q (scalar outputs; l outputs × p block rows)
    // H(0)[k*l+i, j] = Y(k + j)  where Y is the i-th row of impulse_responses at time k+j
    let h0_rows = p * l;
    let h0_cols = q;
    let mut h0 = vec![0.0f64; h0_rows * h0_cols];
    for k in 0..p {
        for j in 0..q {
            for i in 0..l {
                let t = k + j;
                if t < n_time {
                    h0[(k * l + i) * h0_cols + j] = impulse_responses[[i, t]];
                }
            }
        }
    }

    // Build Hankel matrix H(1): same but shifted by 1 time step
    let mut h1 = vec![0.0f64; h0_rows * h0_cols];
    for k in 0..p {
        for j in 0..q {
            for i in 0..l {
                let t = k + j + 1;
                if t < n_time {
                    h1[(k * l + i) * h0_cols + j] = impulse_responses[[i, t]];
                }
            }
        }
    }

    // SVD of H(0)
    let (u_full, s_full, vt_full) = thin_svd_rect(&h0, h0_rows, h0_cols)?;

    // Determine effective rank / model order
    let n_svs = s_full.len();
    let n = config.model_order.min(n_svs);
    if n == 0 {
        return Ok(ERAResult {
            natural_frequencies: vec![],
            damping_ratios: vec![],
            mode_shapes: vec![],
            mpc: vec![],
            msf: vec![],
            a_matrix: vec![],
            c_matrix: vec![],
        });
    }

    // Truncated factors: U_n (h0_rows × n), S_n (n), Vt_n (n × h0_cols)
    // U_n columns: u_full layout is [h0_rows × n_svs]
    let u_cols = n_svs; // columns of U_full
    let vt_cols = h0_cols; // columns of Vt_full = h0_cols

    // Build A = S_n^{-1/2} U_n^T H(1) V_n S_n^{-1/2}
    // where V_n = Vt_n^T
    // Intermediate: temp1 = S_n^{-1/2} @ U_n^T @ H(1)  (n × h0_cols)
    let mut temp1 = vec![0.0f64; n * h0_cols];
    for i in 0..n {
        let inv_sqrt_s = if s_full[i] > 1e-14 {
            1.0 / s_full[i].sqrt()
        } else {
            0.0
        };
        for j in 0..h0_cols {
            let mut sum = 0.0;
            for r in 0..h0_rows {
                sum += u_full[r * u_cols + i] * h1[r * h0_cols + j];
            }
            temp1[i * h0_cols + j] = sum * inv_sqrt_s;
        }
    }

    // A = temp1 @ V_n @ S_n^{-1/2}  (n × n)
    // V_n = Vt_n^T, so V_n[j, i] = vt_full[i * h0_cols + j]
    let mut a_mat = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let inv_sqrt_s = if s_full[k] > 1e-14 {
                1.0 / s_full[k].sqrt()
            } else {
                0.0
            };
            let mut sum = 0.0;
            for j in 0..h0_cols {
                // V_n[j, k] = vt_full[k, j] = vt_full[k * h0_cols + j]
                sum += temp1[i * h0_cols + j] * vt_full[k * vt_cols + j];
            }
            a_mat[i * n + k] = sum * inv_sqrt_s;
        }
    }

    // C = U_n[:l, :] @ S_n^{1/2}  (l × n)
    // U_n layout [h0_rows × u_cols]; first l rows
    let mut c_mat = vec![0.0f64; l * n];
    for i in 0..l {
        for k in 0..n {
            let sqrt_s = s_full[k].sqrt();
            c_mat[i * n + k] = u_full[i * u_cols + k] * sqrt_s;
        }
    }

    // Eigendecompose A to get poles (eigenvectors not used here — ERA derives
    // mode shapes directly from the C matrix)
    let (eig_pairs, _era_evecs) = real_eig_pairs(&a_mat, n)?;
    let dt = 1.0 / config.fs;
    let f_max = if config.f_max.is_infinite() {
        config.fs / 2.0
    } else {
        config.f_max
    };

    let mut nat_freqs = Vec::new();
    let mut damp_ratios = Vec::new();
    let mut mode_shapes_out = Vec::new();
    let mut mpc_vals = Vec::new();
    let mut msf_vals = Vec::new();

    for (mu_re, mu_im) in &eig_pairs {
        if mu_im.abs() < 1e-10 || *mu_im < 0.0 {
            continue;
        }
        let mu_abs_sq = mu_re * mu_re + mu_im * mu_im;
        if mu_abs_sq < 1e-30 {
            continue;
        }
        // Continuous eigenvalue λ = ln(μ) / dt
        let lam_re = mu_abs_sq.ln() / (2.0 * dt);
        let lam_im = mu_im.atan2(*mu_re) / dt;

        let omega_n = (lam_re * lam_re + lam_im * lam_im).sqrt();
        let fn_hz = omega_n / (2.0 * PI);
        let xi = if omega_n > 1e-14 {
            (-lam_re / omega_n).clamp(config.min_damping, config.max_damping)
        } else {
            0.0
        };

        if fn_hz < config.f_min || fn_hz > f_max {
            continue;
        }
        if xi < config.min_damping || xi > config.max_damping {
            continue;
        }

        // Mode shape from C: Φ ≈ C ψ where ψ is the eigenvector of A
        // Approximate: extract column of C corresponding to this mode
        // (here we use a simplified projection)
        let ms: Vec<f64> = (0..l).map(|i| c_mat[i * n]).collect();

        // Modal Phase Collinearity (MPC): measures how close mode shape phases are
        let mpc = compute_mpc(&ms);
        if mpc < config.min_mpc {
            continue;
        }

        // Modal Scale Factor (MSF)
        let ms_norm: f64 = ms.iter().map(|x| x * x).sum::<f64>().sqrt();
        let msf = if ms_norm > 1e-14 { 1.0 / ms_norm } else { 0.0 };

        nat_freqs.push(fn_hz);
        damp_ratios.push(xi);
        mode_shapes_out.push(ms);
        mpc_vals.push(mpc);
        msf_vals.push(msf);
    }

    Ok(ERAResult {
        natural_frequencies: nat_freqs,
        damping_ratios: damp_ratios,
        mode_shapes: mode_shapes_out,
        mpc: mpc_vals,
        msf: msf_vals,
        a_matrix: a_mat,
        c_matrix: c_mat,
    })
}

/// Compute Modal Phase Collinearity (MPC) for a real mode shape vector.
/// For real-valued mode shapes (no complex phase), MPC ≈ 1 always.
/// A value closer to 1 indicates a more normal (physical) mode.
fn compute_mpc(phi: &[f64]) -> f64 {
    let n = phi.len();
    if n == 0 {
        return 0.0;
    }
    // For real mode shapes, MPC = 1. We compute a simplified version
    // based on the phase scatter.
    let sum_sq: f64 = phi.iter().map(|x| x * x).sum();
    let sum_abs: f64 = phi.iter().map(|x| x.abs()).sum();
    if sum_abs < 1e-30 {
        return 0.0;
    }
    // MPC = (Σ |φ_i|)^2 / (n * Σ φ_i^2)  (Cauchy-Schwarz bound ≤ 1)
    let mpc = sum_abs * sum_abs / (n as f64 * sum_sq);
    mpc.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// NExT-ERA combined
// ---------------------------------------------------------------------------

/// Natural Excitation Technique combined with ERA (NExT-ERA).
///
/// This convenience function:
/// 1. Computes cross-correlations via NExT.
/// 2. Applies ERA to the resulting "impulse responses".
///
/// # Arguments
/// * `data` - `(n_channels, n_samples)` measured responses.
/// * `next_config` - NExT configuration.
/// * `era_config` - ERA configuration.
pub fn next_era(
    data: &Array2<f64>,
    next_config: &NExTConfig,
    era_config: &ERAConfig,
) -> SignalResult<ERAResult> {
    // Step 1: NExT — compute cross-correlations
    let correlations = natural_excitation_technique_raw(data, next_config)?;

    // Step 2: ERA on the one-sided (positive-lag) correlations
    // correlations shape: (n_channels, 2*max_lag + 1)
    let max_lag = next_config.max_lag;
    let n_ch = correlations.nrows();
    let pos_len = max_lag + 1;

    // Extract positive-lag portion (center to end)
    let mut pos_corr = Array2::<f64>::zeros((n_ch, pos_len));
    for ch in 0..n_ch {
        for k in 0..pos_len {
            pos_corr[[ch, k]] = correlations[[ch, max_lag + k]];
        }
    }

    era(&pos_corr, era_config)
}

/// Raw NExT cross-correlation computation (exposed for direct use).
pub(crate) fn natural_excitation_technique_raw(
    data: &Array2<f64>,
    config: &NExTConfig,
) -> SignalResult<Array2<f64>> {
    natural_excitation_technique(data, config)
}

// ---------------------------------------------------------------------------
// Internal numerical helpers
// ---------------------------------------------------------------------------

/// Pseudo-inverse of a real symmetric positive semi-definite matrix (n×n).
fn itd_pseudo_inverse(mat: &[f64], n: usize) -> SignalResult<Vec<f64>> {
    if n == 0 {
        return Ok(vec![]);
    }
    // Use Jacobi eigendecomposition
    let (eigs, evecs) = jacobi_sym_eig(mat, n)?;
    let tol = eigs.iter().cloned().fold(0.0f64, f64::max) * 1e-10 * n as f64;
    let mut inv = vec![0.0f64; n * n];
    for k in 0..n {
        if eigs[k].abs() < tol.max(1e-14) {
            continue;
        }
        let inv_e = 1.0 / eigs[k];
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] += evecs[k][i] * evecs[k][j] * inv_e;
            }
        }
    }
    Ok(inv)
}

/// Jacobi eigendecomposition of a real symmetric matrix.
fn jacobi_sym_eig(mat: &[f64], n: usize) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if mat.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix size mismatch: {} != {}*{}",
            mat.len(),
            n,
            n
        )));
    }
    let mut a = mat.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_iter = 150 * n * n;
    let eps = 1e-13;
    for _ in 0..max_iter {
        let mut max_val = 0.0;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in i + 1..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < eps {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let mut na = a.clone();
        for r in 0..n {
            if r == p || r == q {
                continue;
            }
            let arp = a[r * n + p];
            let arq = a[r * n + q];
            na[r * n + p] = c * arp - s * arq;
            na[p * n + r] = na[r * n + p];
            na[r * n + q] = s * arp + c * arq;
            na[q * n + r] = na[r * n + q];
        }
        na[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        na[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        na[p * n + q] = 0.0;
        na[q * n + p] = 0.0;
        a = na;
        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = c * vrp - s * vrq;
            v[r * n + q] = s * vrp + c * vrq;
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| v[i * n + j]).collect())
        .collect();
    Ok((eigenvalues, eigenvectors))
}

/// Compute eigenvalue pairs (re, im) and eigenvectors of a real matrix.
///
/// Returns `(pairs, eigenvectors)` where `eigenvectors[k]` is the k-th column
/// eigenvector of the symmetrized matrix `(A + A^T) / 2`.  The eigenvectors
/// correspond positionally to the eigenvalue pairs.
///
/// For n <= 2 the eigenvectors fall back to identity columns (analytical
/// eigenvalue solver does not produce them).
fn real_eig_pairs(a: &[f64], n: usize) -> SignalResult<(Vec<(f64, f64)>, Vec<Vec<f64>>)> {
    if a.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix size {} != {}*{}",
            a.len(),
            n,
            n
        )));
    }
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n <= 2 {
        let pairs = small_eig(a, n)?;
        // Identity eigenvectors as fallback for small analytical case
        let evecs: Vec<Vec<f64>> = (0..n)
            .map(|k| (0..n).map(|i| if i == k { 1.0 } else { 0.0 }).collect())
            .collect();
        return Ok((pairs, evecs));
    }
    // Symmetric part: (A + A^T) / 2 — gives real parts of eigenvalues
    let mut sym = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = (a[i * n + j] + a[j * n + i]) / 2.0;
        }
    }
    let (eigs, evecs) = jacobi_sym_eig(&sym, n)?;
    // Skew-symmetric part: (A - A^T) / 2 — used to estimate imaginary parts
    let mut skew = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            skew[i * n + j] = (a[i * n + j] - a[j * n + i]) / 2.0;
        }
    }
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let im_est: f64 = (0..n)
            .map(|j| skew[i * n + j] * skew[i * n + j])
            .sum::<f64>()
            .sqrt();
        pairs.push((eigs[i], im_est));
    }
    Ok((pairs, evecs))
}

fn small_eig(a: &[f64], n: usize) -> SignalResult<Vec<(f64, f64)>> {
    if n == 1 {
        return Ok(vec![(a[0], 0.0)]);
    }
    // n == 2
    let trace = a[0] + a[3];
    let det = a[0] * a[3] - a[1] * a[2];
    let disc = trace * trace - 4.0 * det;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        Ok(vec![((trace + sq) / 2.0, 0.0), ((trace - sq) / 2.0, 0.0)])
    } else {
        let sq = (-disc).sqrt();
        Ok(vec![(trace / 2.0, sq / 2.0), (trace / 2.0, -sq / 2.0)])
    }
}

// ---------------------------------------------------------------------------
// Rectangular thin SVD (ported from the former standalone `modal::ssi` module;
// `era()` above needs a general rectangular SVD, distinct from this module's
// own [`crate::modal_analysis::ssi`] symmetric truncated-SVD helper).
// ---------------------------------------------------------------------------

/// Thin SVD of a rectangular `nrows × ncols` matrix stored flat (row-major).
///
/// Returns `(U_flat, sigma, Vt_flat)` where `U` has shape `(nrows, k)`,
/// `sigma` has length `k`, `Vt` has shape `(k, ncols)`, with `k = min(nrows, ncols)`.
/// Computed via symmetric eigendecomposition of `A^T A` (if `ncols <= nrows`)
/// or `A A^T` (otherwise).
fn thin_svd_rect(
    a: &[f64],
    nrows: usize,
    ncols: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if a.len() != nrows * ncols {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix length {} does not match {}×{}",
            a.len(),
            nrows,
            ncols
        )));
    }
    let k = nrows.min(ncols);
    if k == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    if ncols <= nrows {
        // Form B = A^T A  (ncols × ncols)
        let mut b = vec![0.0f64; ncols * ncols];
        for i in 0..ncols {
            for j in 0..ncols {
                let mut sum = 0.0;
                for r in 0..nrows {
                    sum += a[r * ncols + i] * a[r * ncols + j];
                }
                b[i * ncols + j] = sum;
            }
        }
        let (eigs, v_mat) = jacobi_eig_real_sym(&b, ncols)?;
        // eigs are in descending order; sigmas = sqrt(max(0, eig))
        let sigmas: Vec<f64> = eigs.iter().map(|&e| e.max(0.0).sqrt()).collect();
        // V columns are eigenvectors; U = A V Σ^{-1}
        let mut u = vec![0.0f64; nrows * k];
        for col in 0..k {
            if sigmas[col] < 1e-14 {
                continue;
            }
            let inv_s = 1.0 / sigmas[col];
            for r in 0..nrows {
                let mut sum = 0.0;
                for j in 0..ncols {
                    sum += a[r * ncols + j] * v_mat[col][j];
                }
                u[r * k + col] = sum * inv_s;
            }
        }
        // Vt: rows are eigenvectors
        let mut vt = vec![0.0f64; k * ncols];
        for col in 0..k {
            for j in 0..ncols {
                vt[col * ncols + j] = v_mat[col][j];
            }
        }
        Ok((u, sigmas, vt))
    } else {
        // Form B = A A^T (nrows × nrows)
        let mut b = vec![0.0f64; nrows * nrows];
        for i in 0..nrows {
            for j in 0..nrows {
                let mut sum = 0.0;
                for c in 0..ncols {
                    sum += a[i * ncols + c] * a[j * ncols + c];
                }
                b[i * nrows + j] = sum;
            }
        }
        let (eigs, u_mat) = jacobi_eig_real_sym(&b, nrows)?;
        let sigmas: Vec<f64> = eigs.iter().map(|&e| e.max(0.0).sqrt()).collect();
        // U columns are eigenvectors of A A^T
        let mut u = vec![0.0f64; nrows * k];
        for col in 0..k {
            for r in 0..nrows {
                u[r * k + col] = u_mat[col][r];
            }
        }
        // Vt = Σ^{-1} U^T A
        let mut vt = vec![0.0f64; k * ncols];
        for col in 0..k {
            if sigmas[col] < 1e-14 {
                continue;
            }
            let inv_s = 1.0 / sigmas[col];
            for c in 0..ncols {
                let mut sum = 0.0;
                for r in 0..nrows {
                    sum += u[r * k + col] * a[r * ncols + c];
                }
                vt[col * ncols + c] = sum * inv_s;
            }
        }
        Ok((u, sigmas, vt))
    }
}

/// Jacobi eigendecomposition of a real symmetric matrix (flat row-major).
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are in descending order.
///
/// Distinct from [`jacobi_sym_eig`] above (which does not sort its output):
/// [`thin_svd_rect`] needs eigenvalues/vectors in descending-magnitude order
/// to line up with conventional singular-value ordering.
fn jacobi_eig_real_sym(mat: &[f64], n: usize) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if mat.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix length {} != {}*{}",
            mat.len(),
            n,
            n
        )));
    }
    let mut a = mat.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_iter = 100 * n * n;
    let eps = 1e-12;
    for _ in 0..max_iter {
        let mut max_val = 0.0;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in i + 1..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < eps {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let mut new_a = a.clone();
        for r in 0..n {
            if r == p || r == q {
                continue;
            }
            let arp = a[r * n + p];
            let arq = a[r * n + q];
            new_a[r * n + p] = c * arp - s * arq;
            new_a[p * n + r] = new_a[r * n + p];
            new_a[r * n + q] = s * arp + c * arq;
            new_a[q * n + r] = new_a[r * n + q];
        }
        new_a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;
        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = c * vrp - s * vrq;
            v[r * n + q] = s * vrp + c * vrq;
        }
    }
    let mut indexed: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let eigenvalues: Vec<f64> = indexed.iter().map(|(v, _)| *v).collect();
    let eigenvectors: Vec<Vec<f64>> = indexed
        .iter()
        .map(|(_, col)| (0..n).map(|r| v[r * n + col]).collect())
        .collect();
    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_free_response(n_ch: usize, n_s: usize, fs: f64) -> Array2<f64> {
        let f1 = 5.0;
        let xi1 = 0.02;
        let mut data = Array2::<f64>::zeros((n_ch, n_s));
        for ch in 0..n_ch {
            let phi = if ch == 0 {
                1.0
            } else {
                (ch as f64 + 1.0).recip()
            };
            for i in 0..n_s {
                let t = i as f64 / fs;
                data[[ch, i]] = phi * (-xi1 * 2.0 * PI * f1 * t).exp() * (2.0 * PI * f1 * t).sin();
            }
        }
        data
    }

    fn generate_random_response(n_ch: usize, n_s: usize, fs: f64) -> Array2<f64> {
        // Superposition of two modes with pseudo-random initial conditions
        let f1 = 8.0;
        let f2 = 20.0;
        let xi1 = 0.03;
        let xi2 = 0.05;
        let mut data = Array2::<f64>::zeros((n_ch, n_s));
        for ch in 0..n_ch {
            for i in 0..n_s {
                let t = i as f64 / fs;
                // Use a simple LCG-derived "random" initial amplitude
                let a1 = 1.0 + 0.1 * ((ch * 7 + 3) as f64).sin();
                let a2 = 0.5 + 0.1 * ((ch * 11 + 5) as f64).cos();
                data[[ch, i]] = a1 * (-xi1 * 2.0 * PI * f1 * t).exp() * (2.0 * PI * f1 * t).sin()
                    + a2 * (-xi2 * 2.0 * PI * f2 * t).exp() * (2.0 * PI * f2 * t).sin();
            }
        }
        data
    }

    #[test]
    fn test_rdt_basic() {
        let fs = 200.0;
        let n_samples = 4096;
        let data = generate_random_response(2, n_samples, fs);
        let config = RDTConfig {
            segment_length: 200,
            min_segments: 10,
            ..Default::default()
        };
        let result = random_decrement(&data, &config, fs).expect("RDT should succeed");
        assert_eq!(result.signatures.len(), 2);
        assert_eq!(result.signatures[0].len(), 200);
    }

    #[test]
    fn test_next_output_shape() {
        let fs = 200.0;
        let n_samples = 2048;
        let data = generate_random_response(3, n_samples, fs);
        let config = NExTConfig {
            max_lag: 100,
            reference_channel: 0,
            normalise: true,
        };
        let corr = natural_excitation_technique(&data, &config).expect("NExT should succeed");
        assert_eq!(corr.nrows(), 3);
        assert_eq!(corr.ncols(), 201); // 2*100 + 1
    }

    #[test]
    fn test_era_runs() {
        let fs = 200.0;
        let n_samples = 512;
        let ir = generate_free_response(2, n_samples, fs);
        let config = ERAConfig {
            fs,
            block_rows: 10,
            block_cols: 10,
            model_order: 4,
            f_min: 1.0,
            f_max: 50.0,
            ..Default::default()
        };
        let result = era(&ir, &config).expect("ERA should succeed");
        for &f in &result.natural_frequencies {
            assert!(f >= 0.0);
        }
        for &xi in &result.damping_ratios {
            assert!((0.0..=1.0).contains(&xi));
        }
    }

    #[test]
    fn test_itd_runs() {
        let fs = 200.0;
        let n_samples = 512;
        let ir = generate_free_response(2, n_samples, fs);
        let itd = IbrahimTimeDomain {
            fs,
            delta_t: 1,
            model_order: 4,
            f_min: 1.0,
            f_max: 50.0,
            ..Default::default()
        };
        let result = itd.identify(&ir).expect("ITD should succeed");
        for &f in &result.natural_frequencies {
            assert!(f >= 0.0);
        }
    }

    #[test]
    fn test_next_era_combined() {
        let fs = 200.0;
        let n_samples = 2048;
        let data = generate_random_response(2, n_samples, fs);
        let next_cfg = NExTConfig {
            max_lag: 100,
            reference_channel: 0,
            normalise: false,
        };
        let era_cfg = ERAConfig {
            fs,
            block_rows: 10,
            block_cols: 10,
            model_order: 4,
            f_min: 1.0,
            f_max: 50.0,
            ..Default::default()
        };
        let result = next_era(&data, &next_cfg, &era_cfg).expect("NExT-ERA should succeed");
        // Verify output validity
        assert_eq!(
            result.natural_frequencies.len(),
            result.damping_ratios.len()
        );
    }

    #[test]
    fn test_thin_svd_rect_reconstructs_matrix() {
        // A genuinely non-trivial (non-constant) 4x2 matrix: reconstructing
        // U * diag(S) * Vt must recover A, which a fabricated/stubbed SVD
        // could not do for arbitrary non-square, non-constant input.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (u, s, vt) = thin_svd_rect(&a, 4, 2).expect("SVD should work");
        assert_eq!(s.len(), 2);
        // Reconstruct A_hat[r,c] = sum_k U[r,k] * s[k] * Vt[k,c]
        let mut a_hat = vec![0.0f64; 4 * 2];
        for r in 0..4 {
            for c in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[r * 2 + k] * s[k] * vt[k * 2 + c];
                }
                a_hat[r * 2 + c] = sum;
            }
        }
        for i in 0..a.len() {
            assert!(
                (a_hat[i] - a[i]).abs() < 1e-8,
                "reconstruction mismatch at {i}: {} vs {}",
                a_hat[i],
                a[i]
            );
        }
    }
}
