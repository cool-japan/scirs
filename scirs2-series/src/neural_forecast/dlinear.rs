//! DLinear and NLinear: Simple Linear Models for Time Series Forecasting
//!
//! Implementation of *"Are Transformers Effective for Time Series Forecasting?"*
//! (Zeng et al., 2022). Despite their simplicity, these linear models often
//! outperform complex Transformer-based methods on standard benchmarks.
//!
//! ## DLinear
//! Decomposes the series into trend and seasonal components via a moving average,
//! then applies two separate linear layers (one per component).
//!
//! ```text
//! trend    = moving_average(input, kernel_size)
//! seasonal = input - trend
//! forecast = Linear_trend(trend[-seq_len..]) + Linear_seasonal(seasonal[-seq_len..])
//! ```
//!
//! ## NLinear
//! Subtracts the last observed value (simple normalization), applies a linear
//! mapping, then adds the last value back.
//!
//! ```text
//! last       = input[-1]    (last time step, broadcast across channels)
//! normalized = input - last
//! forecast   = Linear(normalized) + last  (broadcast)
//! ```
//!
//! Both models support:
//! - **Shared weights** (`individual=false`): one linear layer shared across all channels
//! - **Per-channel weights** (`individual=true`): separate linear layer per channel

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// DLinear Configuration
// ---------------------------------------------------------------------------

/// Configuration for the DLinear model.
#[derive(Debug, Clone)]
pub struct DLinearConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of input variates (channels).
    pub n_channels: usize,
    /// Moving average kernel size for trend extraction (must be odd for symmetric MA).
    pub kernel_size: usize,
    /// If true, use per-channel linear weights; otherwise share weights across channels.
    pub individual: bool,
    /// Random seed for weight initialization.
    pub seed: u32,
}

impl Default for DLinearConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            kernel_size: 25,
            individual: false,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// NLinear Configuration
// ---------------------------------------------------------------------------

/// Configuration for the NLinear model.
#[derive(Debug, Clone)]
pub struct NLinearConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of input variates (channels).
    pub n_channels: usize,
    /// If true, use per-channel linear weights; otherwise share weights across channels.
    pub individual: bool,
}

impl Default for NLinearConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            individual: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Moving average helper
// ---------------------------------------------------------------------------

/// Compute symmetric (causal) moving average of a 1-D signal.
///
/// Uses a centred moving average of the given kernel size (must be >= 1).
/// Boundary effects are handled by clamping the window to valid indices,
/// equivalent to "same" padding with edge repetition.
fn moving_average<F: Float + FromPrimitive>(signal: &Array1<F>, kernel_size: usize) -> Array1<F> {
    let n = signal.len();
    if kernel_size <= 1 || n == 0 {
        return signal.clone();
    }
    let half = kernel_size / 2;
    let mut out = Array1::zeros(n);
    let ks_f = F::from(kernel_size as f64).unwrap_or_else(|| F::one());
    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let window_len = end - start;
        let mut sum = F::zero();
        for j in start..end {
            sum = sum + signal[j];
        }
        // Normalize by actual kernel_size to stay consistent with centred MA
        // (divide by kernel_size, not actual window length, to avoid boundary scaling)
        let actual_ks = F::from(window_len as f64).unwrap_or_else(|| ks_f);
        out[i] = sum / actual_ks;
    }
    out
}

// ---------------------------------------------------------------------------
// DLinear Model
// ---------------------------------------------------------------------------

/// DLinear model: decomposition-based linear forecasting.
///
/// The model decomposes the input into trend and seasonal components,
/// then uses two separate linear projections to forecast each component.
///
/// # Input/Output
/// - Input: `[seq_len, n_channels]`
/// - Output: `[pred_len, n_channels]`
#[derive(Debug)]
pub struct DLinearModel {
    config: DLinearConfig,
    /// Shared trend linear weight: (pred_len, seq_len)
    w_trend_shared: Option<Array2<f64>>,
    b_trend_shared: Option<Array1<f64>>,
    /// Shared seasonal linear weight: (pred_len, seq_len)
    w_seasonal_shared: Option<Array2<f64>>,
    b_seasonal_shared: Option<Array1<f64>>,
    /// Per-channel trend weights: Vec<(pred_len, seq_len)>  — one per channel
    w_trend_individual: Option<Vec<Array2<f64>>>,
    b_trend_individual: Option<Vec<Array1<f64>>>,
    /// Per-channel seasonal weights
    w_seasonal_individual: Option<Vec<Array2<f64>>>,
    b_seasonal_individual: Option<Vec<Array1<f64>>>,
}

impl DLinearModel {
    /// Create a new DLinear model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration parameters are invalid.
    pub fn new(config: DLinearConfig) -> Result<Self> {
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
        if config.kernel_size == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "kernel_size must be positive".to_string(),
            ));
        }

        let seed = config.seed;
        let sl = config.seq_len;
        let pl = config.pred_len;
        let nc = config.n_channels;

        if config.individual {
            let mut wt = Vec::with_capacity(nc);
            let mut bt = Vec::with_capacity(nc);
            let mut ws = Vec::with_capacity(nc);
            let mut bs = Vec::with_capacity(nc);
            for ch in 0..nc {
                let ch_seed = seed.wrapping_add(ch as u32 * 1000);
                wt.push(nn_utils::xavier_matrix(pl, sl, ch_seed));
                bt.push(nn_utils::zero_bias(pl));
                ws.push(nn_utils::xavier_matrix(pl, sl, ch_seed.wrapping_add(500)));
                bs.push(nn_utils::zero_bias(pl));
            }
            Ok(Self {
                config,
                w_trend_shared: None,
                b_trend_shared: None,
                w_seasonal_shared: None,
                b_seasonal_shared: None,
                w_trend_individual: Some(wt),
                b_trend_individual: Some(bt),
                w_seasonal_individual: Some(ws),
                b_seasonal_individual: Some(bs),
            })
        } else {
            Ok(Self {
                config,
                w_trend_shared: Some(nn_utils::xavier_matrix(pl, sl, seed)),
                b_trend_shared: Some(nn_utils::zero_bias(pl)),
                w_seasonal_shared: Some(nn_utils::xavier_matrix(pl, sl, seed.wrapping_add(500))),
                b_seasonal_shared: Some(nn_utils::zero_bias(pl)),
                w_trend_individual: None,
                b_trend_individual: None,
                w_seasonal_individual: None,
                b_seasonal_individual: None,
            })
        }
    }

    /// Decompose a channel's time series into (trend, seasonal) components.
    pub fn decompose(&self, channel: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let trend = moving_average(channel, self.config.kernel_size);
        let seasonal = channel - &trend;
        (trend, seasonal)
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
    pub fn forecast(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
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
            // Extract channel
            let channel: Array1<f64> = (0..seq_len).map(|t| input[[t, ch]]).collect();

            // Decompose into trend and seasonal
            let (trend, seasonal) = self.decompose(&channel);

            if self.config.individual {
                // Per-channel weights
                let wt = self
                    .w_trend_individual
                    .as_ref()
                    .expect("individual weights must be set");
                let bt = self
                    .b_trend_individual
                    .as_ref()
                    .expect("individual bias must be set");
                let ws = self
                    .w_seasonal_individual
                    .as_ref()
                    .expect("individual weights must be set");
                let bse = self
                    .b_seasonal_individual
                    .as_ref()
                    .expect("individual bias must be set");

                let trend_pred = nn_utils::dense_forward_vec(&trend, &wt[ch], &bt[ch]);
                let seasonal_pred = nn_utils::dense_forward_vec(&seasonal, &ws[ch], &bse[ch]);

                for t in 0..self.config.pred_len {
                    output[[t, ch]] = trend_pred[t] + seasonal_pred[t];
                }
            } else {
                // Shared weights
                let wt = self
                    .w_trend_shared
                    .as_ref()
                    .expect("shared trend weights must be set");
                let bt = self
                    .b_trend_shared
                    .as_ref()
                    .expect("shared trend bias must be set");
                let ws = self
                    .w_seasonal_shared
                    .as_ref()
                    .expect("shared seasonal weights must be set");
                let bse = self
                    .b_seasonal_shared
                    .as_ref()
                    .expect("shared seasonal bias must be set");

                let trend_pred = nn_utils::dense_forward_vec(&trend, wt, bt);
                let seasonal_pred = nn_utils::dense_forward_vec(&seasonal, ws, bse);

                for t in 0..self.config.pred_len {
                    output[[t, ch]] = trend_pred[t] + seasonal_pred[t];
                }
            }
        }

        Ok(output)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &DLinearConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// NLinear Model
// ---------------------------------------------------------------------------

/// NLinear model: last-value normalization + linear forecasting.
///
/// Subtracts the last observed value (per channel), applies a linear layer,
/// then adds the last value back — acting as a simple distribution-shift
/// correction.
///
/// # Input/Output
/// - Input: `[seq_len, n_channels]`
/// - Output: `[pred_len, n_channels]`
#[derive(Debug)]
pub struct NLinearModel {
    config: NLinearConfig,
    /// Shared linear weight: (pred_len, seq_len)
    w_shared: Option<Array2<f64>>,
    b_shared: Option<Array1<f64>>,
    /// Per-channel linear weights
    w_individual: Option<Vec<Array2<f64>>>,
    b_individual: Option<Vec<Array1<f64>>>,
}

impl NLinearModel {
    /// Create a new NLinear model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration parameters are invalid.
    pub fn new(config: NLinearConfig) -> Result<Self> {
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

        let sl = config.seq_len;
        let pl = config.pred_len;
        let nc = config.n_channels;

        // NLinear: weights initialized near identity/zero mapping
        // seed=0 means zero-like initialization; we use xavier for the shared case
        if config.individual {
            let mut w_ind = Vec::with_capacity(nc);
            let mut b_ind = Vec::with_capacity(nc);
            for ch in 0..nc {
                let seed = ch as u32 * 1000 + 1;
                w_ind.push(nn_utils::xavier_matrix(pl, sl, seed));
                b_ind.push(nn_utils::zero_bias(pl));
            }
            Ok(Self {
                config,
                w_shared: None,
                b_shared: None,
                w_individual: Some(w_ind),
                b_individual: Some(b_ind),
            })
        } else {
            Ok(Self {
                config,
                w_shared: Some(nn_utils::xavier_matrix(pl, sl, 42)),
                b_shared: Some(nn_utils::zero_bias(pl)),
                w_individual: None,
                b_individual: None,
            })
        }
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
    pub fn forecast(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
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
            let channel: Array1<f64> = (0..seq_len).map(|t| input[[t, ch]]).collect();

            // Last-value normalization: subtract input[-1]
            let last_val = channel[seq_len - 1];
            let normalized = channel.mapv(|v| v - last_val);

            // Linear projection
            let pred = if self.config.individual {
                let w = self
                    .w_individual
                    .as_ref()
                    .expect("individual weights must be set");
                let b = self
                    .b_individual
                    .as_ref()
                    .expect("individual bias must be set");
                nn_utils::dense_forward_vec(&normalized, &w[ch], &b[ch])
            } else {
                let w = self.w_shared.as_ref().expect("shared weights must be set");
                let b = self.b_shared.as_ref().expect("shared bias must be set");
                nn_utils::dense_forward_vec(&normalized, w, b)
            };

            // Add last value back (de-normalize)
            for t in 0..self.config.pred_len {
                output[[t, ch]] = pred[t] + last_val;
            }
        }

        Ok(output)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &NLinearConfig {
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
                arr[[t, c]] = t as f64 * 0.1 + c as f64;
            }
        }
        arr
    }

    // DLinear tests

    #[test]
    fn test_dlinear_default_config() {
        let cfg = DLinearConfig::default();
        assert_eq!(cfg.seq_len, 96);
        assert_eq!(cfg.pred_len, 24);
        assert_eq!(cfg.n_channels, 7);
        assert_eq!(cfg.kernel_size, 25);
        assert!(!cfg.individual);
        assert_eq!(cfg.seed, 0);
    }

    #[test]
    fn test_dlinear_model_creation() {
        let model = DLinearModel::new(DLinearConfig::default());
        assert!(model.is_ok());
    }

    #[test]
    fn test_dlinear_forecast_shape_shared() {
        let cfg = DLinearConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 4,
            kernel_size: 5,
            individual: false,
            seed: 42,
        };
        let model = DLinearModel::new(cfg).expect("model creation failed");
        let input = make_input(48, 4);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (12, 4));
    }

    #[test]
    fn test_dlinear_forecast_shape_individual() {
        let cfg = DLinearConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 4,
            kernel_size: 5,
            individual: true,
            seed: 42,
        };
        let model = DLinearModel::new(cfg).expect("model creation failed");
        let input = make_input(48, 4);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (12, 4));
    }

    #[test]
    fn test_dlinear_decomposition_correctness() {
        let cfg = DLinearConfig {
            seq_len: 20,
            pred_len: 4,
            n_channels: 1,
            kernel_size: 3,
            individual: false,
            seed: 0,
        };
        let model = DLinearModel::new(cfg).expect("model creation failed");

        // Create a signal where trend = constant = 5.0
        // so seasonal should be ~ 0
        let constant = Array1::from_elem(20, 5.0_f64);
        let (trend, seasonal) = model.decompose(&constant);

        // For a constant signal, moving average = constant
        for t in 0..20 {
            assert!(
                (trend[t] - 5.0).abs() < 1e-10,
                "trend[{t}] = {} != 5.0",
                trend[t]
            );
            assert!(
                seasonal[t].abs() < 1e-10,
                "seasonal[{t}] = {} != 0.0",
                seasonal[t]
            );
        }
    }

    #[test]
    fn test_dlinear_wrong_seq_len_error() {
        let cfg = DLinearConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            kernel_size: 3,
            individual: false,
            seed: 0,
        };
        let model = DLinearModel::new(cfg).expect("model creation failed");
        let bad_input = make_input(24, 2); // wrong seq_len
        assert!(model.forecast(&bad_input).is_err());
    }

    #[test]
    fn test_dlinear_output_finite() {
        let cfg = DLinearConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            kernel_size: 7,
            individual: false,
            seed: 1,
        };
        let model = DLinearModel::new(cfg).expect("model creation failed");
        let input = make_input(32, 3);
        let output = model.forecast(&input).expect("forecast failed");
        for t in 0..8 {
            for c in 0..3 {
                assert!(output[[t, c]].is_finite(), "non-finite at [{t},{c}]");
            }
        }
    }

    // NLinear tests

    #[test]
    fn test_nlinear_default_config() {
        let cfg = NLinearConfig::default();
        assert_eq!(cfg.seq_len, 96);
        assert_eq!(cfg.pred_len, 24);
        assert_eq!(cfg.n_channels, 7);
        assert!(!cfg.individual);
    }

    #[test]
    fn test_nlinear_model_creation() {
        let model = NLinearModel::new(NLinearConfig::default());
        assert!(model.is_ok());
    }

    #[test]
    fn test_nlinear_forecast_shape_shared() {
        let cfg = NLinearConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 4,
            individual: false,
        };
        let model = NLinearModel::new(cfg).expect("model creation failed");
        let input = make_input(48, 4);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (12, 4));
    }

    #[test]
    fn test_nlinear_forecast_shape_individual() {
        let cfg = NLinearConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 4,
            individual: true,
        };
        let model = NLinearModel::new(cfg).expect("model creation failed");
        let input = make_input(48, 4);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (12, 4));
    }

    #[test]
    fn test_nlinear_normalization_and_denorm() {
        // A constant-value series: after normalization (subtract last=C) all values are 0.
        // The linear layer output on zeros (plus bias=0) is 0.
        // Adding back C gives C everywhere in the forecast.
        // This tests the de-normalization logic.
        let cfg = NLinearConfig {
            seq_len: 10,
            pred_len: 3,
            n_channels: 1,
            individual: false,
        };
        let model = NLinearModel::new(cfg).expect("model creation failed");

        let mut constant_input = Array2::zeros((10, 1));
        for t in 0..10 {
            constant_input[[t, 0]] = 7.0;
        }
        let output = model.forecast(&constant_input).expect("forecast failed");
        // With weight=xavier (non-zero) and bias=0, the prediction on all-zero input
        // should be 0 (since dense(zeros)=bias=0). Adding back last_val=7 → all 7.
        for t in 0..3 {
            let expected = 7.0; // last value should be added back
            let actual = output[[t, 0]];
            // The linear output on all-zeros is just the bias (=0), so result = 0 + 7 = 7
            assert!(
                (actual - expected).abs() < 1e-10,
                "nlinear de-norm failed at t={t}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_nlinear_wrong_seq_len_error() {
        let cfg = NLinearConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            individual: false,
        };
        let model = NLinearModel::new(cfg).expect("model creation failed");
        let bad_input = make_input(16, 2);
        assert!(model.forecast(&bad_input).is_err());
    }

    #[test]
    fn test_nlinear_output_finite() {
        let cfg = NLinearConfig {
            seq_len: 24,
            pred_len: 6,
            n_channels: 5,
            individual: true,
        };
        let model = NLinearModel::new(cfg).expect("model creation failed");
        let input = make_input(24, 5);
        let output = model.forecast(&input).expect("forecast failed");
        for t in 0..6 {
            for c in 0..5 {
                assert!(output[[t, c]].is_finite(), "non-finite at [{t},{c}]");
            }
        }
    }
}
