//! TimesNet: Temporal 2D-Variation Modeling
//!
//! Implementation of *"TimesNet: Temporal 2D-Variation Modeling for General Time Series
//! Analysis"* (Wu et al., 2023). The key idea is to discover **dominant periods** in the
//! time series via FFT and reshape the 1D temporal signal into 2D representations
//! (period × frequency), enabling 2D convolutions to capture both intra-period and
//! inter-period variations.
//!
//! Algorithm per TimesBlock:
//! 1. FFT the input → find top-k periods by amplitude
//! 2. For each period p: reshape 1D `(seq_len,)` → 2D `(p, ⌈seq_len/p⌉)`
//! 3. Apply 2D inception-like convolution (multiple kernel sizes)
//! 4. Reshape back to 1D, aggregate across periods (amplitude-weighted sum)
//!
//! Architecture: Embedding → N × TimesBlock → Projection.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the TimesNet model.
#[derive(Debug, Clone)]
pub struct TimesNetConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of input variates (channels).
    pub n_channels: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Number of TimesBlock layers.
    pub n_layers: usize,
    /// Number of top periods to discover via FFT.
    pub top_k_periods: usize,
    /// Number of kernel sizes in the inception block.
    pub num_kernels: usize,
    /// Random seed for weight initialization.
    pub seed: u32,
}

impl Default for TimesNetConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            d_model: 64,
            d_ff: 128,
            n_layers: 2,
            top_k_periods: 5,
            num_kernels: 3,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// FFT period discovery
// ---------------------------------------------------------------------------

/// Discover top-k periods from a 1-D signal using FFT (DFT).
///
/// Returns `(periods, amplitudes)` sorted by amplitude (descending).
/// Each period is clamped to `[2, seq_len]`.
fn discover_periods<F: Float + FromPrimitive + Debug>(
    signal: &Array1<F>,
    top_k: usize,
) -> (Vec<usize>, Vec<F>) {
    let n = signal.len();
    if n < 2 || top_k == 0 {
        return (vec![n.max(2)], vec![F::one()]);
    }

    // Compute DFT amplitudes (magnitude of frequency bins)
    // We only need the first n/2 + 1 bins (positive frequencies)
    let half_n = n / 2 + 1;
    let mut amplitudes_vec: Vec<(usize, F)> = Vec::with_capacity(half_n.saturating_sub(1));

    let n_f = n as f64;
    let two_pi = std::f64::consts::PI * 2.0;

    // Skip DC component (freq_idx=0) and Nyquist
    for freq_idx in 1..half_n {
        let mut re = F::zero();
        let mut im = F::zero();
        let freq_f = freq_idx as f64;

        for t in 0..n {
            let angle = two_pi * freq_f * (t as f64) / n_f;
            let cos_val = F::from(angle.cos()).unwrap_or_else(|| F::zero());
            let sin_val = F::from(angle.sin()).unwrap_or_else(|| F::zero());
            re = re + signal[t] * cos_val;
            im = im - signal[t] * sin_val;
        }

        let magnitude = (re * re + im * im).sqrt();
        // period = n / freq_idx
        let period = n / freq_idx;
        if period >= 2 {
            amplitudes_vec.push((period, magnitude));
        }
    }

    if amplitudes_vec.is_empty() {
        return (vec![n.max(2)], vec![F::one()]);
    }

    // Sort by amplitude descending
    amplitudes_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate periods (keep highest amplitude for each period)
    let mut seen_periods = std::collections::HashSet::new();
    let mut unique: Vec<(usize, F)> = Vec::new();
    for (p, amp) in &amplitudes_vec {
        if seen_periods.insert(*p) {
            unique.push((*p, *amp));
        }
        if unique.len() >= top_k {
            break;
        }
    }

    let periods: Vec<usize> = unique.iter().map(|(p, _)| *p).collect();
    let amps: Vec<F> = unique.iter().map(|(_, a)| *a).collect();

    (periods, amps)
}

/// Reshape a 1-D signal into a 2-D representation based on a period.
///
/// Input: `(seq_len,)` with period `p`.
/// Output: 2-D array of shape `(p, ceil(seq_len / p))`.
/// Pads with zeros if `seq_len` is not divisible by `p`.
fn reshape_to_2d<F: Float>(signal: &Array1<F>, period: usize) -> Array2<F> {
    let seq_len = signal.len();
    let n_cols = (seq_len + period - 1) / period; // ceil division
    let mut matrix = Array2::zeros((period, n_cols));

    for t in 0..seq_len {
        let row = t % period;
        let col = t / period;
        matrix[[row, col]] = signal[t];
    }

    matrix
}

/// Reshape a 2-D representation back to 1-D, truncating to `target_len`.
fn reshape_to_1d<F: Float>(matrix: &Array2<F>, target_len: usize) -> Array1<F> {
    let (period, n_cols) = matrix.dim();
    let mut output = Array1::zeros(target_len);

    for t in 0..target_len {
        let row = t % period;
        let col = t / period;
        if col < n_cols {
            output[t] = matrix[[row, col]];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// 2D Convolution (inception-style with multiple kernels)
// ---------------------------------------------------------------------------

/// A single 2D convolution kernel with weights.
#[derive(Debug)]
struct Conv2DKernel<F: Float> {
    /// Kernel weights: (kh, kw, d_in, d_out)
    /// Stored as a flat representation for simplicity.
    kernel_h: usize,
    kernel_w: usize,
    d_in: usize,
    d_out: usize,
    weights: Array2<F>, // (d_out, d_in * kh * kw)
    bias: Array1<F>,    // (d_out,)
}

impl<F: Float + FromPrimitive + Debug> Conv2DKernel<F> {
    fn new(kernel_h: usize, kernel_w: usize, d_in: usize, d_out: usize, seed: u32) -> Self {
        let fan_in = d_in * kernel_h * kernel_w;
        Self {
            kernel_h,
            kernel_w,
            d_in,
            d_out,
            weights: nn_utils::xavier_matrix(d_out, fan_in, seed),
            bias: nn_utils::zero_bias(d_out),
        }
    }

    /// Apply 2D convolution with same-padding.
    ///
    /// Input: `(d_in, height, width)` stored as Vec of Array2.
    /// Output: `(d_out, height, width)` stored as Vec of Array2.
    fn forward(&self, input: &[Array2<F>]) -> Vec<Array2<F>> {
        if input.is_empty() {
            return Vec::new();
        }
        let (height, width) = input[0].dim();
        let pad_h = self.kernel_h / 2;
        let pad_w = self.kernel_w / 2;

        let mut output = Vec::with_capacity(self.d_out);
        for _ in 0..self.d_out {
            output.push(Array2::zeros((height, width)));
        }

        for o in 0..self.d_out {
            for i in 0..self.d_in.min(input.len()) {
                for r in 0..height {
                    for c in 0..width {
                        let mut acc = F::zero();
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let in_r = r as isize + kh as isize - pad_h as isize;
                                let in_c = c as isize + kw as isize - pad_w as isize;
                                if in_r >= 0
                                    && in_r < height as isize
                                    && in_c >= 0
                                    && in_c < width as isize
                                {
                                    let w_idx =
                                        i * self.kernel_h * self.kernel_w + kh * self.kernel_w + kw;
                                    acc = acc
                                        + input[i][[in_r as usize, in_c as usize]]
                                            * self.weights[[o, w_idx]];
                                }
                            }
                        }
                        output[o][[r, c]] = output[o][[r, c]] + acc;
                    }
                }
            }
            // Add bias
            for r in 0..height {
                for c in 0..width {
                    output[o][[r, c]] = output[o][[r, c]] + self.bias[o];
                }
            }
        }

        output
    }
}

/// Inception-style block with multiple kernel sizes for 2D convolution.
#[derive(Debug)]
struct InceptionBlock<F: Float> {
    kernels: Vec<Conv2DKernel<F>>,
    /// Projection after concatenation: (d_model, d_model * num_kernels)
    w_proj: Array2<F>,
    b_proj: Array1<F>,
    d_model: usize,
}

impl<F: Float + FromPrimitive + Debug> InceptionBlock<F> {
    fn new(d_model: usize, num_kernels: usize, seed: u32) -> Self {
        let kernel_sizes: Vec<usize> = (0..num_kernels).map(|i| 2 * i + 1).collect();
        // Kernel sizes: 1, 3, 5, 7, ... (odd sizes for same-padding)

        let mut kernels = Vec::with_capacity(num_kernels);
        for (i, &ks) in kernel_sizes.iter().enumerate() {
            // Each kernel: d_model input channels -> d_model output channels
            // But for efficiency, each produces d_model / num_kernels or d_model outputs
            kernels.push(Conv2DKernel::new(
                ks,
                ks,
                d_model,
                d_model,
                seed.wrapping_add(i as u32 * 200),
            ));
        }

        let concat_dim = d_model * num_kernels;
        Self {
            kernels,
            w_proj: nn_utils::xavier_matrix(d_model, concat_dim, seed.wrapping_add(5000)),
            b_proj: nn_utils::zero_bias(d_model),
            d_model,
        }
    }

    /// Apply inception block on a multi-channel 2D input.
    ///
    /// Input: Vec of Array2 (d_model channels, each of shape (height, width)).
    /// Output: Vec of Array2 (d_model channels, same spatial dims).
    fn forward(&self, input: &[Array2<F>]) -> Vec<Array2<F>> {
        if input.is_empty() {
            return Vec::new();
        }
        let (height, width) = input[0].dim();

        // Apply each kernel and collect outputs
        let mut all_outputs: Vec<Vec<Array2<F>>> = Vec::with_capacity(self.kernels.len());
        for kernel in &self.kernels {
            let out = kernel.forward(input);
            // Apply GELU activation
            let activated: Vec<Array2<F>> = out
                .iter()
                .map(|arr| {
                    let half = F::from(0.5).unwrap_or_else(|| F::zero());
                    let sqrt_2_pi = F::from(0.7978845608).unwrap_or_else(|| F::one());
                    let c = F::from(0.044715).unwrap_or_else(|| F::zero());
                    arr.mapv(|v| half * v * (F::one() + (sqrt_2_pi * (v + c * v * v * v)).tanh()))
                })
                .collect();
            all_outputs.push(activated);
        }

        // Concatenate along channel dimension and project back
        let n_kernels = all_outputs.len();
        let concat_dim = self.d_model * n_kernels;

        let mut result = Vec::with_capacity(self.d_model);
        for _ in 0..self.d_model {
            result.push(Array2::zeros((height, width)));
        }

        // For each spatial position, concatenate features and project
        for r in 0..height {
            for c in 0..width {
                // Build concatenated feature vector
                let mut concat_feat = Array1::zeros(concat_dim);
                for (ki, kernel_out) in all_outputs.iter().enumerate() {
                    for ch in 0..self.d_model.min(kernel_out.len()) {
                        concat_feat[ki * self.d_model + ch] = kernel_out[ch][[r, c]];
                    }
                }

                // Project: (d_model, concat_dim) * concat_feat + bias
                let projected =
                    nn_utils::dense_forward_vec(&concat_feat, &self.w_proj, &self.b_proj);

                for ch in 0..self.d_model {
                    result[ch][[r, c]] = projected[ch];
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// TimesBlock
// ---------------------------------------------------------------------------

/// A single TimesBlock: period discovery → 2D reshape → inception conv → aggregate.
#[derive(Debug)]
struct TimesBlock<F: Float> {
    top_k: usize,
    seq_len: usize,
    d_model: usize,
    inception: InceptionBlock<F>,
    /// Layer norm
    ln_gamma: Array1<F>,
    ln_beta: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> TimesBlock<F> {
    fn new(seq_len: usize, d_model: usize, top_k: usize, num_kernels: usize, seed: u32) -> Self {
        Self {
            top_k,
            seq_len,
            d_model,
            inception: InceptionBlock::new(d_model, num_kernels, seed),
            ln_gamma: Array1::ones(d_model),
            ln_beta: Array1::zeros(d_model),
        }
    }

    /// Forward pass for a single channel/feature.
    ///
    /// Input: `(seq_len, d_model)` -> Output: `(seq_len, d_model)`.
    fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (sl, dm) = input.dim();

        // Average across d_model to get a representative 1-D signal for period discovery
        let mut avg_signal = Array1::zeros(sl);
        let dm_f = F::from(dm as f64).unwrap_or_else(|| F::one());
        for t in 0..sl {
            let mut sum = F::zero();
            for d in 0..dm {
                sum = sum + input[[t, d]];
            }
            avg_signal[t] = sum / dm_f;
        }

        // Discover periods
        let (periods, amplitudes) = discover_periods(&avg_signal, self.top_k);

        if periods.is_empty() {
            return Ok(input.clone());
        }

        // Softmax over amplitudes for weighting
        let amp_weights = softmax_vec(&amplitudes);

        // For each period: reshape to 2D, apply inception, reshape back
        let mut aggregated = Array2::zeros((sl, dm));

        for (idx, (&period, &weight)) in periods.iter().zip(amp_weights.iter()).enumerate() {
            let clamped_period = period.max(2).min(sl);

            // For each feature dimension, reshape to 2D and apply convolution
            let n_cols = (sl + clamped_period - 1) / clamped_period;

            // Build multi-channel 2D input (d_model channels)
            let mut channels_2d: Vec<Array2<F>> = Vec::with_capacity(dm);
            for d in 0..dm {
                let mut col_1d = Array1::zeros(sl);
                for t in 0..sl {
                    col_1d[t] = input[[t, d]];
                }
                channels_2d.push(reshape_to_2d(&col_1d, clamped_period));
            }

            // Apply inception convolution
            let conv_out = self.inception.forward(&channels_2d);

            // Reshape back to 1D and weight
            for d in 0..dm.min(conv_out.len()) {
                let col_1d = reshape_to_1d(&conv_out[d], sl);
                for t in 0..sl {
                    aggregated[[t, d]] = aggregated[[t, d]] + col_1d[t] * weight;
                }
            }
        }

        // Residual connection + layer norm
        let residual = add_2d(input, &aggregated);
        let normed = nn_utils::layer_norm(&residual, &self.ln_gamma, &self.ln_beta);

        Ok(normed)
    }
}

/// Softmax over a vector of floats.
fn softmax_vec<F: Float>(v: &[F]) -> Vec<F> {
    if v.is_empty() {
        return Vec::new();
    }
    let max_val = v.iter().cloned().fold(F::neg_infinity(), F::max);
    let mut exps: Vec<F> = v.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: F = exps.iter().cloned().fold(F::zero(), |a, b| a + b);
    if sum > F::zero() {
        for e in &mut exps {
            *e = *e / sum;
        }
    }
    exps
}

/// Element-wise addition of two identically shaped 2-D arrays.
fn add_2d<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (rows, cols) = a.dim();
    let mut out = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            out[[r, c]] = a[[r, c]] + b[[r, c]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// TimesNet model
// ---------------------------------------------------------------------------

/// TimesNet model for multivariate time series forecasting.
///
/// Uses FFT-based period discovery and 2D inception convolutions to capture
/// temporal variations at multiple scales.
///
/// Input shape: `[batch, n_channels, seq_len]`
/// Output shape: `[batch, n_channels, pred_len]`
#[derive(Debug)]
pub struct TimesNetModel<F: Float + Debug> {
    config: TimesNetConfig,
    /// Input embedding: (d_model, 1) per channel
    w_embed: Array2<F>,
    b_embed: Array1<F>,
    /// TimesBlock layers
    blocks: Vec<TimesBlock<F>>,
    /// Output projection: (pred_len, d_model)
    w_proj: Array2<F>,
    b_proj: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> TimesNetModel<F> {
    /// Create a new TimesNet model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration parameters are invalid.
    pub fn new(config: TimesNetConfig) -> Result<Self> {
        if config.seq_len == 0 || config.pred_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "seq_len and pred_len must be positive".to_string(),
            ));
        }
        if config.n_channels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_channels must be positive".to_string(),
            ));
        }
        if config.d_model == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "d_model must be positive".to_string(),
            ));
        }
        if config.top_k_periods == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "top_k_periods must be positive".to_string(),
            ));
        }
        if config.num_kernels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "num_kernels must be positive".to_string(),
            ));
        }

        let seed = config.seed;
        let dm = config.d_model;
        let total_len = config.seq_len + config.pred_len;

        // Input embedding: project each scalar to d_model
        let w_embed = nn_utils::xavier_matrix(dm, 1, seed);
        let b_embed = nn_utils::zero_bias(dm);

        // TimesBlock layers operate on (total_len, d_model)
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(TimesBlock::new(
                total_len,
                dm,
                config.top_k_periods,
                config.num_kernels,
                seed.wrapping_add(1000 + i as u32 * 1000),
            ));
        }

        // Output projection: d_model -> 1 (per time step)
        let w_proj = nn_utils::xavier_matrix(1, dm, seed.wrapping_add(9000));
        let b_proj = nn_utils::zero_bias(1);

        Ok(Self {
            config,
            w_embed,
            b_embed,
            blocks,
            w_proj,
            b_proj,
        })
    }

    /// Forward pass for a single channel.
    ///
    /// Input: `(seq_len,)` -> Output: `(pred_len,)`.
    fn forward_channel(&self, channel_data: &Array1<F>) -> Result<Array1<F>> {
        let sl = self.config.seq_len;
        let pl = self.config.pred_len;
        let total_len = sl + pl;
        let dm = self.config.d_model;

        // Pad with zeros for prediction horizon
        let mut padded = Array1::zeros(total_len);
        for t in 0..sl {
            padded[t] = channel_data[t];
        }

        // Embed: each scalar -> d_model vector
        // (total_len, 1) * W^T + b -> (total_len, d_model)
        let mut embedded = Array2::zeros((total_len, dm));
        for t in 0..total_len {
            let mut scalar = Array2::zeros((1, 1));
            scalar[[0, 0]] = padded[t];
            let proj = nn_utils::dense_forward(&scalar, &self.w_embed, &self.b_embed);
            for d in 0..dm {
                embedded[[t, d]] = proj[[0, d]];
            }
        }

        // Apply TimesBlock layers
        let mut hidden = embedded;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // Output projection: for each time step in prediction horizon,
        // project d_model -> 1
        let mut output = Array1::zeros(pl);
        for t in 0..pl {
            let mut h_vec = Array1::zeros(dm);
            for d in 0..dm {
                h_vec[d] = hidden[[sl + t, d]];
            }
            let proj = nn_utils::dense_forward_vec(&h_vec, &self.w_proj, &self.b_proj);
            output[t] = proj[0];
        }

        Ok(output)
    }

    /// Forward pass for a batch of multivariate time series.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, n_channels, seq_len]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, n_channels, pred_len]`.
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions don't match the configuration.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        let (batch, n_ch, sl) = x.dim();

        if sl != self.config.seq_len {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.seq_len,
                actual: sl,
            });
        }
        if n_ch != self.config.n_channels {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.n_channels,
                actual: n_ch,
            });
        }

        let mut output = Array3::zeros((batch, n_ch, self.config.pred_len));

        for b in 0..batch {
            for ch in 0..n_ch {
                let mut channel_data = Array1::zeros(sl);
                for t in 0..sl {
                    channel_data[t] = x[[b, ch, t]];
                }

                let pred = self.forward_channel(&channel_data)?;

                for t in 0..self.config.pred_len {
                    output[[b, ch, t]] = pred[t];
                }
            }
        }

        Ok(output)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &TimesNetConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use std::f64::consts::PI;

    #[test]
    fn test_default_config_produces_valid_model() {
        let config = TimesNetConfig::default();
        let model = TimesNetModel::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_shape() {
        let config = TimesNetConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 3,
            d_model: 16,
            d_ff: 32,
            n_layers: 1,
            top_k_periods: 3,
            num_kernels: 2,
            seed: 42,
        };
        let model = TimesNetModel::<f64>::new(config).expect("model creation failed");

        let x = Array3::zeros((2, 3, 48));
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (2, 3, 12));
    }

    #[test]
    fn test_period_discovery_sine_wave() {
        // Create a pure sine wave with period 8
        let n = 64;
        let period = 8.0;
        let mut signal = Array1::zeros(n);
        for t in 0..n {
            signal[t] = (2.0 * PI * t as f64 / period).sin();
        }

        let (periods, _amplitudes) = discover_periods(&signal, 3);

        // The dominant period should be 8
        assert!(!periods.is_empty(), "Should discover at least one period");
        assert_eq!(
            periods[0], 8,
            "Dominant period should be 8, got {}",
            periods[0]
        );
    }

    #[test]
    fn test_period_discovery_multi_frequency() {
        // Two sine waves: period 8 (strong) and period 16 (weaker)
        let n = 64;
        let mut signal = Array1::zeros(n);
        for t in 0..n {
            signal[t] =
                2.0 * (2.0 * PI * t as f64 / 8.0).sin() + 0.5 * (2.0 * PI * t as f64 / 16.0).sin();
        }

        let (periods, amplitudes) = discover_periods(&signal, 5);

        assert!(periods.len() >= 2, "Should find at least 2 periods");
        // First period should have higher amplitude
        if amplitudes.len() >= 2 {
            assert!(
                amplitudes[0] >= amplitudes[1],
                "Amplitudes should be sorted descending"
            );
        }
    }

    #[test]
    fn test_2d_reshape_dimensions() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Period 3: 3 × 2
        let m = reshape_to_2d(&signal, 3);
        assert_eq!(m.dim(), (3, 2));

        // Period 4: 4 × 2 (with padding)
        let m = reshape_to_2d(&signal, 4);
        assert_eq!(m.dim(), (4, 2));

        // Period 6: 6 × 1
        let m = reshape_to_2d(&signal, 6);
        assert_eq!(m.dim(), (6, 1));
    }

    #[test]
    fn test_2d_reshape_roundtrip() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let period = 3;
        let m = reshape_to_2d(&signal, period);
        let recovered = reshape_to_1d(&m, signal.len());

        for i in 0..signal.len() {
            assert!(
                (signal[i] - recovered[i]).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                signal[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_top_k_amplitude_ordering() {
        // Create signal with three known frequencies
        let n = 128;
        let mut signal = Array1::zeros(n);
        for t in 0..n {
            signal[t] = 3.0 * (2.0 * PI * t as f64 / 16.0).sin()
                + 1.0 * (2.0 * PI * t as f64 / 8.0).sin()
                + 0.5 * (2.0 * PI * t as f64 / 32.0).sin();
        }

        let (_periods, amplitudes) = discover_periods(&signal, 5);

        // Verify amplitude ordering is descending
        for i in 1..amplitudes.len() {
            assert!(
                amplitudes[i - 1] >= amplitudes[i],
                "Amplitudes not sorted: {} < {} at index {}",
                amplitudes[i - 1],
                amplitudes[i],
                i
            );
        }
    }

    #[test]
    fn test_zero_input_finite_output() {
        let config = TimesNetConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            d_model: 16,
            d_ff: 32,
            n_layers: 1,
            top_k_periods: 3,
            num_kernels: 2,
            seed: 42,
        };
        let model = TimesNetModel::<f64>::new(config).expect("model creation failed");

        let x = Array3::zeros((1, 2, 32));
        let out = model.forward(&x).expect("forward failed");

        for ch in 0..2 {
            for t in 0..8 {
                assert!(
                    out[[0, ch, t]].is_finite(),
                    "Non-finite at ch={}, t={}",
                    ch,
                    t
                );
            }
        }
    }

    #[test]
    fn test_batch_dimension_preserved() {
        let config = TimesNetConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            d_model: 16,
            d_ff: 32,
            n_layers: 1,
            top_k_periods: 3,
            num_kernels: 2,
            seed: 42,
        };
        let model = TimesNetModel::<f64>::new(config).expect("model creation failed");

        for batch_size in [1, 3] {
            let x = Array3::zeros((batch_size, 2, 32));
            let out = model.forward(&x).expect("forward failed");
            assert_eq!(out.dim().0, batch_size);
        }
    }

    #[test]
    fn test_invalid_config_errors() {
        let config = TimesNetConfig {
            seq_len: 0,
            ..TimesNetConfig::default()
        };
        assert!(TimesNetModel::<f64>::new(config).is_err());

        let config = TimesNetConfig {
            top_k_periods: 0,
            ..TimesNetConfig::default()
        };
        assert!(TimesNetModel::<f64>::new(config).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = TimesNetConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            d_model: 16,
            d_ff: 32,
            n_layers: 1,
            top_k_periods: 3,
            num_kernels: 2,
            seed: 42,
        };
        let model = TimesNetModel::<f64>::new(config).expect("model creation failed");

        // Wrong seq_len
        let x = Array3::zeros((1, 2, 64));
        assert!(model.forward(&x).is_err());

        // Wrong n_channels
        let x = Array3::zeros((1, 5, 32));
        assert!(model.forward(&x).is_err());
    }
}
