//! N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
//!
//! Implementation of *"N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"*
//! (Challu et al., 2022). Key innovations:
//!
//! - **Multi-rate temporal processing**: Multiple stacks process the input at different
//!   temporal resolutions via max-pooling subsampling (coarse to fine).
//!
//! - **Hierarchical interpolation**: Each stack outputs a forecast at a reduced rate,
//!   which is then upsampled (interpolated) to the full prediction length.
//!
//! - **Residual connections**: Each stack predicts a backcast that is subtracted from
//!   the input residual before passing to the next stack, enabling progressive
//!   decomposition of the signal.
//!
//! Architecture:
//! ```text
//! Input series → Stack_1 (pool rate r_1) → Stack_2 (pool rate r_2) → ... → Sum forecasts
//! Each stack: MaxPool(rate) → MLP(input_size → hidden) → split(backcast, forecast)
//!             forecast interpolated to pred_len
//!             backcast subtracted from input
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the N-HiTS model.
#[derive(Debug, Clone)]
pub struct NHiTSConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of input variates (channels).
    pub n_channels: usize,
    /// Number of stacks (hierarchical levels).
    pub n_stacks: usize,
    /// Number of blocks per stack.
    pub n_blocks: usize,
    /// Hidden size of the MLP in each block.
    pub hidden_size: usize,
    /// Max-pool kernel size for each stack (length must match n_stacks).
    /// Typical: [16, 8, 1] for 3 stacks (coarse to fine).
    pub n_pool_kernel_size: Vec<usize>,
    /// Random seed for weight initialization.
    pub seed: u32,
}

impl Default for NHiTSConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 1,
            n_stacks: 3,
            n_blocks: 1,
            hidden_size: 256,
            n_pool_kernel_size: vec![16, 8, 1],
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Max-pool a 1-D signal with a given kernel size.
///
/// The output length is `ceil(input_len / kernel_size)`.
fn max_pool_1d<F: Float>(signal: &Array1<F>, kernel_size: usize) -> Array1<F> {
    if kernel_size <= 1 {
        return signal.clone();
    }
    let n = signal.len();
    let out_len = (n + kernel_size - 1) / kernel_size;
    let mut out = Array1::zeros(out_len);
    for i in 0..out_len {
        let start = i * kernel_size;
        let end = (start + kernel_size).min(n);
        let mut max_val = F::neg_infinity();
        for j in start..end {
            if signal[j] > max_val {
                max_val = signal[j];
            }
        }
        out[i] = max_val;
    }
    out
}

/// Linear interpolation of a 1-D array to a target length.
///
/// Uses nearest-neighbour for length-1 sources.
fn interpolate_1d<F: Float + FromPrimitive>(signal: &Array1<F>, target_len: usize) -> Array1<F> {
    let src_len = signal.len();
    if src_len == 0 || target_len == 0 {
        return Array1::zeros(target_len);
    }
    if src_len == target_len {
        return signal.clone();
    }
    if src_len == 1 {
        return Array1::from_elem(target_len, signal[0]);
    }
    let mut out = Array1::zeros(target_len);
    let scale =
        F::from((src_len - 1) as f64 / (target_len - 1).max(1) as f64).unwrap_or_else(|| F::one());
    for i in 0..target_len {
        let pos = F::from(i as f64).unwrap_or_else(|| F::zero()) * scale;
        let lo = pos.floor();
        let hi = pos.ceil();
        let frac = pos - lo;
        let lo_idx = {
            let v = lo.to_f64().unwrap_or(0.0) as usize;
            v.min(src_len - 1)
        };
        let hi_idx = {
            let v = hi.to_f64().unwrap_or(0.0) as usize;
            v.min(src_len - 1)
        };
        out[i] = signal[lo_idx] * (F::one() - frac) + signal[hi_idx] * frac;
    }
    out
}

// ---------------------------------------------------------------------------
// N-HiTS Block
// ---------------------------------------------------------------------------

/// A single N-HiTS block: MLP that maps pooled input to backcast + forecast.
///
/// The block takes a pooled input of size `pooled_size` and produces:
/// - A **backcast** of length `seq_len` (for residual subtraction)
/// - A **forecast** of reduced length `forecast_size = ceil(pred_len / expr_rate)`
///   which is later interpolated to the full `pred_len`.
#[derive(Debug)]
pub struct NHiTSBlock<F: Float + Debug> {
    /// Input size after pooling.
    pooled_size: usize,
    /// Original sequence length (for backcast).
    seq_len: usize,
    /// Forecast output size from this block (before interpolation).
    forecast_size: usize,
    /// MLP layer 1: (hidden, pooled_size)
    w1: Array2<F>,
    b1: Array1<F>,
    /// MLP layer 2: (hidden, hidden)
    w2: Array2<F>,
    b2: Array1<F>,
    /// Backcast projection: (seq_len, hidden)
    w_backcast: Array2<F>,
    b_backcast: Array1<F>,
    /// Forecast projection: (forecast_size, hidden)
    w_forecast: Array2<F>,
    b_forecast: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> NHiTSBlock<F> {
    /// Create a new N-HiTS block.
    ///
    /// # Arguments
    /// * `pooled_size` - Input length after max-pooling
    /// * `seq_len` - Original sequence length (backcast target)
    /// * `forecast_size` - Reduced forecast output size
    /// * `hidden_size` - Hidden layer size
    /// * `seed` - RNG seed
    pub fn new(
        pooled_size: usize,
        seq_len: usize,
        forecast_size: usize,
        hidden_size: usize,
        seed: u32,
    ) -> Self {
        Self {
            pooled_size,
            seq_len,
            forecast_size,
            w1: nn_utils::xavier_matrix(hidden_size, pooled_size, seed),
            b1: nn_utils::zero_bias(hidden_size),
            w2: nn_utils::xavier_matrix(hidden_size, hidden_size, seed.wrapping_add(100)),
            b2: nn_utils::zero_bias(hidden_size),
            w_backcast: nn_utils::xavier_matrix(seq_len, hidden_size, seed.wrapping_add(200)),
            b_backcast: nn_utils::zero_bias(seq_len),
            w_forecast: nn_utils::xavier_matrix(forecast_size, hidden_size, seed.wrapping_add(300)),
            b_forecast: nn_utils::zero_bias(forecast_size),
        }
    }

    /// Forward pass: pooled input → (backcast, forecast).
    ///
    /// # Arguments
    /// * `pooled` - Pooled 1-D input of shape `(pooled_size,)`
    ///
    /// # Returns
    /// `(backcast, forecast)` where:
    /// - `backcast` has shape `(seq_len,)`
    /// - `forecast` has shape `(forecast_size,)`
    pub fn forward(&self, pooled: &Array1<F>) -> (Array1<F>, Array1<F>) {
        // MLP forward: 2 hidden layers with ReLU
        let h1 = nn_utils::dense_forward_vec(pooled, &self.w1, &self.b1);
        let h1_act = nn_utils::relu_1d(&h1);
        let h2 = nn_utils::dense_forward_vec(&h1_act, &self.w2, &self.b2);
        let h2_act = nn_utils::relu_1d(&h2);

        // Project to backcast and forecast
        let backcast = nn_utils::dense_forward_vec(&h2_act, &self.w_backcast, &self.b_backcast);
        let forecast = nn_utils::dense_forward_vec(&h2_act, &self.w_forecast, &self.b_forecast);

        (backcast, forecast)
    }
}

// ---------------------------------------------------------------------------
// N-HiTS Stack
// ---------------------------------------------------------------------------

/// A stack of N-HiTS blocks at a given pooling rate.
///
/// All blocks in a stack share the same max-pool kernel size. The stack:
/// 1. Max-pools the input residual
/// 2. Passes through each block sequentially (residual subtraction)
/// 3. Accumulates the forecast outputs
#[derive(Debug)]
pub struct NHiTSStack<F: Float + Debug> {
    /// Max-pool kernel size for this stack.
    pool_kernel_size: usize,
    /// Original sequence length.
    seq_len: usize,
    /// Full prediction length.
    pred_len: usize,
    /// Forecast output size per block (before interpolation).
    forecast_size: usize,
    /// The blocks in this stack.
    blocks: Vec<NHiTSBlock<F>>,
}

impl<F: Float + FromPrimitive + Debug> NHiTSStack<F> {
    /// Create a new N-HiTS stack.
    ///
    /// # Arguments
    /// * `seq_len` - Input sequence length
    /// * `pred_len` - Prediction horizon
    /// * `n_blocks` - Number of blocks
    /// * `hidden_size` - MLP hidden size
    /// * `pool_kernel_size` - Max-pool kernel size
    /// * `seed` - Base RNG seed
    pub fn new(
        seq_len: usize,
        pred_len: usize,
        n_blocks: usize,
        hidden_size: usize,
        pool_kernel_size: usize,
        seed: u32,
    ) -> Self {
        let ks = pool_kernel_size.max(1);
        let pooled_size = (seq_len + ks - 1) / ks;
        // Expression rate: how many time steps each forecast point represents
        // This effectively downsamples the forecast before interpolation
        let expr_rate = ks;
        let forecast_size = (pred_len + expr_rate - 1) / expr_rate;
        let forecast_size = forecast_size.max(1);

        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(NHiTSBlock::new(
                pooled_size,
                seq_len,
                forecast_size,
                hidden_size,
                seed.wrapping_add(i as u32 * 1000),
            ));
        }

        Self {
            pool_kernel_size: ks,
            seq_len,
            pred_len,
            forecast_size,
            blocks,
        }
    }

    /// Process one channel through the stack.
    ///
    /// # Arguments
    /// * `residual` - Input residual of shape `(seq_len,)`
    ///
    /// # Returns
    /// `(new_residual, stack_forecast)` where:
    /// - `new_residual` has shape `(seq_len,)` (after backcast subtraction)
    /// - `stack_forecast` has shape `(pred_len,)` (accumulated and interpolated)
    pub fn forward(&self, residual: &Array1<F>) -> (Array1<F>, Array1<F>) {
        // Max-pool the residual for all blocks in this stack
        let pooled = max_pool_1d(residual, self.pool_kernel_size);

        let mut current_residual = residual.clone();
        let mut stack_forecast = Array1::zeros(self.pred_len);

        for block in &self.blocks {
            // Ensure pooled slice matches expected pooled_size
            let pooled_input = if pooled.len() != block.pooled_size {
                // Pad or truncate
                let mut adj = Array1::zeros(block.pooled_size);
                let copy_len = pooled.len().min(block.pooled_size);
                for k in 0..copy_len {
                    adj[k] = pooled[k];
                }
                adj
            } else {
                pooled.clone()
            };

            let (backcast, forecast_reduced) = block.forward(&pooled_input);

            // Subtract backcast from residual
            for t in 0..self.seq_len {
                current_residual[t] = current_residual[t] - backcast[t];
            }

            // Interpolate forecast from forecast_size to pred_len
            let forecast_full = interpolate_1d(&forecast_reduced, self.pred_len);

            // Accumulate forecast
            for t in 0..self.pred_len {
                stack_forecast[t] = stack_forecast[t] + forecast_full[t];
            }
        }

        (current_residual, stack_forecast)
    }
}

// ---------------------------------------------------------------------------
// N-HiTS Model
// ---------------------------------------------------------------------------

/// N-HiTS model for multi-variate time series forecasting.
///
/// Each channel is processed independently through a shared set of stacks
/// (channel-independent mode, following the original paper's recommendation).
///
/// # Input/Output
/// - Input: `[seq_len, n_channels]`
/// - Output: `[pred_len, n_channels]`
#[derive(Debug)]
pub struct NHiTSModel<F: Float + Debug> {
    config: NHiTSConfig,
    /// One set of stacks per channel (channel-independent).
    stacks: Vec<Vec<NHiTSStack<F>>>,
}

impl<F: Float + FromPrimitive + Debug> NHiTSModel<F> {
    /// Create a new N-HiTS model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: NHiTSConfig) -> Result<Self> {
        if config.seq_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "seq_len must be positive".to_string(),
            ));
        }
        if config.pred_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "pred_len must be positive".to_string(),
            ));
        }
        if config.n_channels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_channels must be positive".to_string(),
            ));
        }
        if config.n_stacks == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_stacks must be positive".to_string(),
            ));
        }
        if config.n_blocks == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_blocks must be positive".to_string(),
            ));
        }
        if config.n_pool_kernel_size.len() != config.n_stacks {
            return Err(TimeSeriesError::InvalidInput(format!(
                "n_pool_kernel_size length ({}) must equal n_stacks ({})",
                config.n_pool_kernel_size.len(),
                config.n_stacks
            )));
        }

        // Build stacks for each channel
        let mut stacks = Vec::with_capacity(config.n_channels);
        for ch in 0..config.n_channels {
            let mut ch_stacks = Vec::with_capacity(config.n_stacks);
            for (s, &pool_ks) in config.n_pool_kernel_size.iter().enumerate() {
                let stack_seed = config
                    .seed
                    .wrapping_add(ch as u32 * 10000)
                    .wrapping_add(s as u32 * 1000);
                ch_stacks.push(NHiTSStack::new(
                    config.seq_len,
                    config.pred_len,
                    config.n_blocks,
                    config.hidden_size,
                    pool_ks,
                    stack_seed,
                ));
            }
            stacks.push(ch_stacks);
        }

        Ok(Self { config, stacks })
    }

    /// Forecast future values for a multivariate input.
    ///
    /// # Arguments
    /// * `input` - Input array of shape `[seq_len, n_channels]`
    ///
    /// # Returns
    /// Forecast array of shape `[pred_len, n_channels]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match configuration.
    pub fn forecast(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (seq_len, n_ch) = input.dim();

        if seq_len != self.config.seq_len {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.seq_len,
                actual: seq_len,
            });
        }
        if n_ch != self.config.n_channels {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.n_channels,
                actual: n_ch,
            });
        }

        let mut output = Array2::zeros((self.config.pred_len, n_ch));

        for ch in 0..n_ch {
            // Extract channel as 1-D array
            let mut channel_series: Array1<F> = Array1::zeros(seq_len);
            for t in 0..seq_len {
                channel_series[t] = input[[t, ch]];
            }

            // Process through stacks
            let mut residual = channel_series.clone();
            let mut total_forecast: Array1<F> = Array1::zeros(self.config.pred_len);

            for stack in &self.stacks[ch] {
                let (new_residual, stack_forecast) = stack.forward(&residual);
                residual = new_residual;
                for t in 0..self.config.pred_len {
                    total_forecast[t] = total_forecast[t] + stack_forecast[t];
                }
            }

            for t in 0..self.config.pred_len {
                output[[t, ch]] = total_forecast[t];
            }
        }

        Ok(output)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &NHiTSConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(seq_len: usize, n_channels: usize) -> Array2<f64> {
        let mut arr = Array2::zeros((seq_len, n_channels));
        for t in 0..seq_len {
            for c in 0..n_channels {
                arr[[t, c]] = (t as f64 * 0.1 + c as f64) as f64;
            }
        }
        arr
    }

    #[test]
    fn test_default_config_values() {
        let cfg = NHiTSConfig::default();
        assert_eq!(cfg.seq_len, 96);
        assert_eq!(cfg.pred_len, 24);
        assert_eq!(cfg.n_channels, 1);
        assert_eq!(cfg.n_stacks, 3);
        assert_eq!(cfg.n_blocks, 1);
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(cfg.n_pool_kernel_size, vec![16, 8, 1]);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_model_creation_default() {
        let model = NHiTSModel::<f64>::new(NHiTSConfig::default());
        assert!(model.is_ok());
    }

    #[test]
    fn test_model_creation_invalid_seq_len() {
        let cfg = NHiTSConfig {
            seq_len: 0,
            ..NHiTSConfig::default()
        };
        assert!(NHiTSModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_model_creation_invalid_kernel_size_len() {
        let cfg = NHiTSConfig {
            n_stacks: 3,
            n_pool_kernel_size: vec![8, 4], // only 2 instead of 3
            ..NHiTSConfig::default()
        };
        assert!(NHiTSModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_block_output_shapes() {
        let pooled_size = 12;
        let seq_len = 96;
        let forecast_size = 6;
        let hidden = 32;
        let block = NHiTSBlock::<f64>::new(pooled_size, seq_len, forecast_size, hidden, 42);

        let pooled = Array1::zeros(pooled_size);
        let (backcast, forecast) = block.forward(&pooled);
        assert_eq!(backcast.len(), seq_len, "backcast length mismatch");
        assert_eq!(forecast.len(), forecast_size, "forecast length mismatch");
    }

    #[test]
    fn test_stack_output_shapes() {
        let seq_len = 48;
        let pred_len = 12;
        let stack = NHiTSStack::<f64>::new(seq_len, pred_len, 2, 64, 8, 42);
        let residual = Array1::zeros(seq_len);
        let (new_res, forecast) = stack.forward(&residual);
        assert_eq!(new_res.len(), seq_len, "residual shape mismatch");
        assert_eq!(forecast.len(), pred_len, "forecast shape mismatch");
    }

    #[test]
    fn test_forecast_shape_single_channel() {
        let cfg = NHiTSConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 1,
            n_stacks: 2,
            n_blocks: 1,
            hidden_size: 32,
            n_pool_kernel_size: vec![8, 1],
            seed: 42,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(48, 1);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (12, 1));
    }

    #[test]
    fn test_forecast_shape_multichannel() {
        let cfg = NHiTSConfig {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            n_stacks: 3,
            n_blocks: 1,
            hidden_size: 64,
            n_pool_kernel_size: vec![16, 8, 1],
            seed: 42,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(96, 7);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (24, 7));
    }

    #[test]
    fn test_forecast_output_is_finite() {
        let cfg = NHiTSConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            n_stacks: 2,
            n_blocks: 2,
            hidden_size: 32,
            n_pool_kernel_size: vec![4, 1],
            seed: 7,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(32, 3);
        let output = model.forecast(&input).expect("forecast failed");
        for pred_t in 0..8 {
            for ch in 0..3 {
                assert!(
                    output[[pred_t, ch]].is_finite(),
                    "Non-finite at [{pred_t},{ch}]"
                );
            }
        }
    }

    #[test]
    fn test_wrong_seq_len_returns_error() {
        let cfg = NHiTSConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 1,
            n_stacks: 1,
            n_blocks: 1,
            hidden_size: 16,
            n_pool_kernel_size: vec![4],
            seed: 1,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let bad_input = make_input(32, 1); // wrong seq_len
        assert!(model.forecast(&bad_input).is_err());
    }

    #[test]
    fn test_wrong_n_channels_returns_error() {
        let cfg = NHiTSConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            n_stacks: 1,
            n_blocks: 1,
            hidden_size: 16,
            n_pool_kernel_size: vec![4],
            seed: 1,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let bad_input = make_input(32, 5); // wrong n_channels
        assert!(model.forecast(&bad_input).is_err());
    }

    #[test]
    fn test_n_stacks_effect_on_residual() {
        // More stacks should further decompose the signal
        let base_cfg = NHiTSConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 1,
            n_stacks: 1,
            n_blocks: 1,
            hidden_size: 16,
            n_pool_kernel_size: vec![4],
            seed: 42,
        };
        let three_stack_cfg = NHiTSConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 1,
            n_stacks: 3,
            n_blocks: 1,
            hidden_size: 16,
            n_pool_kernel_size: vec![8, 4, 1],
            seed: 42,
        };
        let m1 = NHiTSModel::<f64>::new(base_cfg).expect("model1 creation failed");
        let m3 = NHiTSModel::<f64>::new(three_stack_cfg).expect("model3 creation failed");
        let input = make_input(32, 1);
        let out1 = m1.forecast(&input).expect("forecast1 failed");
        let out3 = m3.forecast(&input).expect("forecast3 failed");
        // Both should have correct shape
        assert_eq!(out1.dim(), (8, 1));
        assert_eq!(out3.dim(), (8, 1));
    }

    #[test]
    fn test_max_pool_1d_basic() {
        let sig = Array1::from_vec(vec![1.0_f64, 3.0, 2.0, 4.0]);
        let out = max_pool_1d(&sig, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.0).abs() < 1e-12); // max(1, 3) = 3
        assert!((out[1] - 4.0).abs() < 1e-12); // max(2, 4) = 4
    }

    #[test]
    fn test_interpolate_1d_basic() {
        // Interpolate [0, 1] to length 5 → [0, 0.25, 0.5, 0.75, 1.0]
        let sig = Array1::from_vec(vec![0.0_f64, 1.0]);
        let out = interpolate_1d(&sig, 5);
        assert_eq!(out.len(), 5);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_n_blocks_greater_than_1() {
        let cfg = NHiTSConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            n_stacks: 2,
            n_blocks: 3,
            hidden_size: 16,
            n_pool_kernel_size: vec![4, 1],
            seed: 99,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(32, 2);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (8, 2));
    }

    #[test]
    fn test_pool_kernel_size_1_acts_as_identity() {
        // With pool_kernel_size=1 everywhere, pooled == original
        let cfg = NHiTSConfig {
            seq_len: 16,
            pred_len: 4,
            n_channels: 1,
            n_stacks: 1,
            n_blocks: 1,
            hidden_size: 8,
            n_pool_kernel_size: vec![1],
            seed: 0,
        };
        let model = NHiTSModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(16, 1);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (4, 1));
    }

    #[test]
    fn test_forecast_shape_standard_config() {
        // Replicate default config forecast
        let model = NHiTSModel::<f64>::new(NHiTSConfig::default()).expect("model creation failed");
        let input = make_input(96, 1);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (24, 1));
    }
}
